from ast import Call

from torch._ops import OpOverload


"""
A LocalTensor is a tensor subclass which simulates a tensor that is
distributed across SPMD ranks.  A LocalTensor might be size N, but in fact
there are world_size shards/replicas of it stored internally.  When you do a
plain PyTorch operation on it, we apply the operation to each shard; when you
do a collective, we do the mathematically equivalent operation on the local
shards.  A LocalTensor is associated with a list of ranks which specify
which ranks it holds local tensors for.

NB, this is NOT a DataParallel like abstraction where you can run operations
on multiple different GPUs. It is intended purely for *debugging* purposes,
the overhead is almost certainly too high to keep eight GPUs (even the C++
autograd needs multithreading to keep up!)  (It might potentially be possible
to trace through this with torch.compile and then compile it with CUDA graphs
but this is currently a non-goal.)

We do not directly handling MPMD. However in practice even in SPMD you may
encounter divergence in behavior per rank (for example, uneven sharding
across ranks). To support scenarios like this, we provide a helper decorator
that allows you to run a function with no side effects for each LocalTensor
shard and combine results back into LocalTensor or LocalIntNode.

NB: This is a torch dispatch Tensor subclass, as we want to assume that autograd
is SPMD, so we run it once, and dispatch the inner autograd calls to the individual
local shards.

NOTE ABOUT MESH:  This subclass requires collectives that are issued to it to
respect a DeviceMesh like abstraction.  The reason for this is that when
DTensor issues us a collective for a particular rank, you will be asked to do
this on a specific process group which involves some ranks.  However, this
will only be for the LOCAL PG that this particular rank is participating in;
there will be a bunch of other PGs for other nodes that you don't get to see.
We need to be able to reverse engineer all of the collectives that don't
involve the current local rank here to actually issue them.  This can be done
two ways: (1) looking at the participating local ranks in the PG and computing
the complement which specifies all the other collectives you have to run, or
(2) retrieving the device mesh axis corresponding to the PG for this rank, and
then running all the fibers for this.
"""

import contextlib
import copy
import functools
import operator
import os
import sys
import threading
from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
from types import TracebackType
from typing import Any, Optional, Union


try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

import torch
from torch import Size, SymBool, SymInt, Tensor
from torch._C import DispatchKey, DispatchKeySet, ScriptObject
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed import DeviceMesh, ProcessGroup
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.distributed_c10d import _get_default_group
from torch.fx.experimental._constant_symnode import ConstantIntNode
from torch.nested._internal.nested_int import NestedIntNode
from torch.utils import _pytree as pytree
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import return_and_correct_aliasing, TorchDispatchMode
from torch.utils.checkpoint import get_device_states, set_device_states


not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")

# Thread-local storage for tracking the current rank being simulated
_current_rank: threading.local = threading.local()

from . import _c10d


def _is_inplace_op(op: OpOverload | Callable[..., Any]) -> bool:
    return (
        isinstance(op, OpOverload)
        # Not precise heuristic to detect inplace operation
        and op._schema.name[-1] == "_"
        # Strengthen the heuristic to check that the first argument and return value are a write
        and len(op._schema.arguments) > 0
        and op._schema.arguments[0].is_write
        and len(op._schema.returns) > 0
        and op._schema.returns[0].is_write
    )


def _int_on_rank(i: "int | LocalIntNode | ConstantIntNode", r: int) -> int:
    if isinstance(i, LocalIntNode):
        return i._local_ints[r]
    elif isinstance(i, ConstantIntNode):
        return i.val
    elif isinstance(i, int):
        return i
    else:
        raise AssertionError(type(i))


def _check_for_subclass(flat_args: Sequence[object]) -> bool:
    return any(_check_for_subclass_arg(x) for x in flat_args)


def _check_for_subclass_arg(x: object) -> bool:
    return (
        not isinstance(x, LocalTensor)
        and isinstance(x, Tensor)
        and type(x)
        not in (
            Tensor,
            FakeTensor,
            torch.nn.Parameter,
            torch.nn.Buffer,
        )
    )


def _map_to_rank_local_val(val: Any, rank: int) -> Any:
    if isinstance(val, LocalTensor):
        return val._local_tensors[rank]
    if isinstance(val, SymInt):
        if isinstance(val.node, LocalIntNode):
            return val.node._local_ints[rank]
        if isinstance(val.node, ConstantIntNode):
            return val.node.val
    return val


def _collect_cuda_rng_states() -> dict[int, torch.Tensor]:
    """
    Collects RNG state from all available CUDA devices.

    Returns:
        List of RNG state tensors, one for each CUDA device.
        Returns empty list if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return {}

    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()

        with torch.cuda.device(device_idx):
            return {device_idx: torch.cuda.get_rng_state()}

    return {}


def _set_cuda_rng_states(rng_states: dict[int, torch.Tensor]) -> None:
    """
    Sets RNG state for all CUDA devices from a list of states.

    Args:
        rng_states: List of RNG state tensors to restore.
    """
    if not torch.cuda.is_available():
        return

    for device_idx, device_rng_state in rng_states.items():
        with torch.cuda.device(device_idx):
            torch.cuda.set_rng_state(device_rng_state)


def _get_rng_state() -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """
    Gets CPU and CUDA rng states from all devices.
    """
    return (torch.get_rng_state(), _collect_cuda_rng_states())


def _set_rng_state(
    cpu_state: torch.Tensor, cuda_states: dict[int, torch.Tensor]
) -> None:
    """
    Sets CPU and CUDA rng states for all devices. If the list of cuda states
    is shorter than the number of devices only the first len(cuda_states) devices
    will get their rng state set.
    """
    torch.set_rng_state(cpu_state)
    _set_cuda_rng_states(cuda_states)


def _get_device_from_local_tensors(flat_args: list[Any]) -> Optional[torch.device]:
    for arg in flat_args:
        if isinstance(arg, LocalTensor):
            first_tensor = next(iter(arg._local_tensors.values()))
            return first_tensor.device
    return None


def _is_random_op(func: Callable[..., Any]) -> bool:
    from torch._ops import OpOverload
    from torch.distributed.tensor._dispatch import OpDispatcher

    if not isinstance(func, OpOverload):
        return False
    return func in OpDispatcher()._random_ops


def _get_rng_offset_for_rank(rank: int) -> Optional[int]:
    """
    Calculate the RNG offset for a specific rank based on the current DTensorSpec.

    This reuses the offset calculation logic from OffsetBasedRNGTracker to avoid duplication.
    """
    from torch.distributed.tensor import _random as random

    spec = getattr(random._current_dtensor_spec, "spec", None)
    if spec is None:
        return None

    # Check if the RNG tracker exists and has the helper method
    if random._rng_tracker is None or not hasattr(
        random._rng_tracker, "_compute_offset_for_shard"
    ):
        return None

    # Convert rank to mesh coordinate
    mesh = spec.mesh

    # temporarily set the current rank
    # and call the patched get_coordinate method
    saved_rank = getattr(_current_rank, "rank", None)
    try:
        _current_rank.rank = rank
        mesh_coordinate = mesh.get_coordinate()
    finally:
        if saved_rank is not None:
            _current_rank.rank = saved_rank
        else:
            if hasattr(_current_rank, "rank"):
                del _current_rank.rank

    if mesh_coordinate is None:
        return None

    # Use the shared helper method from the RNG tracker
    return random._rng_tracker._compute_offset_for_shard(spec, mesh_coordinate)


def _combine_int_rank_results(rank_results: dict[int, int]) -> int | torch.SymInt:
    any_v = next(iter(rank_results.values()))

    if all(v == any_v for v in rank_results.values()):
        return any_v

    return torch.SymInt(LocalIntNode(rank_results))


def _combine_any_rank_results(rank_results: dict[int, Any]) -> Any:
    any_v = next(iter(rank_results.values()))

    if isinstance(any_v, Tensor):
        # pyrefly: ignore [bad-argument-type, bad-argument-count]
        return LocalTensor(rank_results)

    if isinstance(any_v, int):
        return _combine_int_rank_results(rank_results)

    if isinstance(any_v, (float, bool, complex)):
        return _combine_int_rank_results(rank_results)

    assert all(v == any_v for v in rank_results.values()), (
        "Non Tensor or int rank results must be equal for all ranks"
    )

    return any_v


def _combine_rank_results(rank_results: dict[int, Any], default: Any | None) -> Any:
    rank_ids = rank_results.keys()
    rank_value = rank_results[next(iter(rank_ids))]

    if isinstance(rank_value, (list, tuple)):
        max_rank_result_len = max(len(v) for v in rank_results.values())
        ret_list = []
        for i in range(max_rank_result_len):
            rank_col_results = {
                r: v[i] if i < len(v) else default for r, v in rank_results.items()
            }
            ret_list.append(_combine_any_rank_results(rank_col_results))
        return type(rank_value)(ret_list)
    else:
        return _combine_any_rank_results(rank_results)


def _zero_sized_like(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    tensor_size = list(tensor.size())
    tensor_size[dim] = 0
    empty_tensor = torch.empty(*tensor_size, dtype=tensor.dtype, device=tensor.device)
    return empty_tensor


def _for_each_rank_run_func(
    func: OpOverload | Callable[..., Any],
    ranks: frozenset[int],
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    alias: bool = True,
) -> Any:
    from torch.distributed.tensor import _random as random

    flat_args, args_spec = pytree.tree_flatten((args, kwargs))
    flat_args = [
        a.wait() if isinstance(a, AsyncCollectiveTensor) else a for a in flat_args
    ]

    is_random = _is_random_op(func)
    base_rng_offset = None
    device_for_rng = None

    rng_state = _get_rng_state()

    if is_random:
        if (
            hasattr(random._current_dtensor_spec, "spec")
            and random._current_dtensor_spec.spec is not None
        ):
            device_for_rng = _get_device_from_local_tensors(flat_args)
            if device_for_rng is None:
                spec = random._current_dtensor_spec.spec
                device_for_rng = torch.device(f"{spec.mesh.device_type}:0")

            if device_for_rng is not None and device_for_rng.type in (
                "cuda",
                "xpu",
                "hpu",
            ):
                current_state = random.get_rng_state_for_device(device_for_rng)
                base_rng_offset = int(current_state[8:].view(dtype=torch.int64).item())

    flat_rank_rets = {}
    default_value: Tensor | None = None

    for r in sorted(ranks):
        _current_rank.rank = r
        _set_rng_state(*rng_state)

        # apply per-rank seed if one was stored by manual_seed in LocalTensor mode
        if (
            is_random
            and hasattr(random, "_per_rank_seeds")
            and r in random._per_rank_seeds
        ):
            rank_seed = random._per_rank_seeds[r]
            torch.manual_seed(rank_seed)

        if base_rng_offset is not None and device_for_rng is not None:
            rank_offset = _get_rng_offset_for_rank(r)
            if rank_offset is not None:
                try:
                    if device_for_rng.type in ("cuda", "xpu", "hpu"):
                        current_state = random.get_rng_state_for_device(
                            device_for_rng
                        ).to("cpu")

                        new_offset = base_rng_offset + rank_offset
                        offset_tensor = torch.tensor(
                            [new_offset], dtype=torch.uint64, device="cpu"
                        ).view(torch.uint8)
                        current_state[8:] = offset_tensor

                        random.set_rng_state_for_device(device_for_rng, current_state)
                except Exception:
                    pass

        rank_flat_args = [_map_to_rank_local_val(a, r) for a in flat_args]
        rank_args, rank_kwargs = pytree.tree_unflatten(rank_flat_args, args_spec)
        rank_ret = func(*rank_args, **rank_kwargs)
        flat_rank_rets[r] = rank_ret

        if default_value is None and func is torch.ops.aten.split.Tensor:
            # If split happens over the dimension smaller than the number of chunks
            # it is possible that some ranks will produce shorter lists of chunks.
            # In order to make the result across all ranks of the same length we
            # append empty tensors (zero size on the split dimension).
            tensor = rank_flat_args[0]
            split_dim = 0 if len(rank_flat_args) < 3 else rank_flat_args[2]
            default_value = _zero_sized_like(tensor, split_dim)

    if _is_inplace_op(func):
        alias = False
        # For the in-place ops return self
        ret = args[0]
        if isinstance(func, OpOverload) and torch.Tag.inplace_view in func.tags:
            # Ensure that wrapper tensor size is synchronized with its local tensors
            ret._sync_meta()
    else:
        ret = _combine_rank_results(flat_rank_rets, default_value)

    if alias:
        return return_and_correct_aliasing(func, args, kwargs, ret)
    else:
        return ret


def _get_extra_dispatch_keys(t: torch.Tensor) -> DispatchKeySet:
    extra_dispatch_keys = torch._C.DispatchKeySet.from_raw_repr(0)
    if torch._C._dispatch_keys(t).has(torch._C.DispatchKey.Conjugate):
        extra_dispatch_keys = extra_dispatch_keys.add(torch._C.DispatchKey.Conjugate)
    if torch._C._dispatch_keys(t).has(torch._C.DispatchKey.Negative):
        extra_dispatch_keys = extra_dispatch_keys.add(torch._C.DispatchKey.Negative)
    return extra_dispatch_keys


class LocalIntNode:
    """
    Like a LocalTensor, but for an int.  We can't use a 0D tensor to represent this
    because often only a SymInt is accepted where we wish to use this.
    """

    def __new__(cls, local_ints: dict[int, int]) -> "ConstantIntNode | LocalIntNode":  # type: ignore[misc]
        if len(set(local_ints.values())) == 1:
            return ConstantIntNode(next(iter(local_ints.values())))
        return super().__new__(cls)

    def __init__(self, local_ints: dict[int, int]):
        self._local_ints = local_ints

    def maybe_as_int(self) -> Optional[int]:
        return None

    def is_int(self) -> bool:
        return True

    def is_float(self) -> bool:
        return False

    def is_bool(self) -> bool:
        return False

    def is_nested_int(self) -> bool:
        return False

    def clone(self) -> "LocalIntNode":
        return self

    def _str(self) -> str:
        return f"LocalIntNode({self._local_ints})"

    def int_(self) -> int:
        return self.guard_int("", 0)

    def __str__(self) -> str:
        return self._str()

    def __repr__(self) -> str:
        return self._str()

    def _graph_repr(self) -> str:
        return self._str()

    def is_symbolic(self) -> bool:
        return False

    def is_constant(self) -> bool:
        return False

    def sym_max(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {
                r: max(self._local_ints[r], _int_on_rank(other, r))
                for r in self._local_ints
            }
        )

    def neg(self) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode({r: -self._local_ints[r] for r in self._local_ints})

    def add(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] + _int_on_rank(other, r) for r in self._local_ints}
        )

    def sub(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] - _int_on_rank(other, r) for r in self._local_ints}
        )

    def mul(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] * _int_on_rank(other, r) for r in self._local_ints}
        )

    def floordiv(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] // _int_on_rank(other, r) for r in self._local_ints}
        )

    def mod(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] % _int_on_rank(other, r) for r in self._local_ints}
        )

    def int_floordiv(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] // _int_on_rank(other, r) for r in self._local_ints}
        )

    def eq(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] == _int_on_rank(other, r) for r in self._local_ints}
        return torch._C._get_constant_bool_symnode(len(r) == 1 and next(iter(r)))

    def ne(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] != _int_on_rank(other, r) for r in self._local_ints}
        return torch._C._get_constant_bool_symnode(len(r) > 1 or next(iter(r)))

    def ge(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] >= _int_on_rank(other, r) for r in self._local_ints}
        assert len(r) == 1, (self, other)
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def gt(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] > _int_on_rank(other, r) for r in self._local_ints}
        assert len(r) == 1, (self, other)
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def lt(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] < _int_on_rank(other, r) for r in self._local_ints}
        assert len(r) == 1, (self, other)
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def wrap_int(self, num: int) -> "LocalIntNode | ConstantIntNode":
        return ConstantIntNode(num)

    def guard_int(self, file: str, line: int) -> int:
        current_rank = getattr(_current_rank, "rank", None)

        if current_rank is not None:
            r = self._local_ints[current_rank]
        else:
            unique_values = set(self._local_ints.values())
            if len(unique_values) == 1:
                r = next(iter(unique_values))
            else:
                not_implemented_log.warning(
                    "guard_int called on LocalIntNode with divergent values "
                    "outside rank context: ",
                    self._local_ints,
                    ". Using rank 0's value."
                )
                r = self._local_ints[min(self._local_ints.keys())]

        try:
            return int(r)
        except Exception:
            not_implemented_log.warning("Failed to convert to int: %s", r)
            raise

    def statically_known_true(self, file: str, line: int) -> bool:
        current_rank = getattr(_current_rank, "rank", None)

        if current_rank is not None:
            return bool(self._local_ints[current_rank])
        else:
            unique_values = set(self._local_ints.values())
            if len(unique_values) != 1:
                return False
            return bool(next(iter(unique_values)))


_LOCAL_TENSOR_ATTR_PREFIX = "_local_tensor_"


def _is_local_tensor_attr(attr: str) -> bool:
    return attr.startswith(_LOCAL_TENSOR_ATTR_PREFIX)


def _to_local_tensor_attr(rank: int) -> str:
    return f"{_LOCAL_TENSOR_ATTR_PREFIX}{rank}"


def _from_local_tensor_attr(attr: str) -> int:
    if not _is_local_tensor_attr(attr):
        raise AssertionError(f"Invalid local tensor attr {attr}")
    return int(attr[len(_LOCAL_TENSOR_ATTR_PREFIX) :])


def _all_elements_same(values: list[Any]) -> bool:
    if not values:
        return True
    first_value = values[0]
    return all(value == first_value for value in values)


def _compute_local_tensor_meta(
    local_tensors: dict[int, torch.Tensor],
) -> tuple[
    list[torch.SymInt | int],
    list[torch.SymInt | int],
    torch.device,
    torch.dtype,
    torch.layout,
    DispatchKeySet,
]:
    """
    Computes the meta information for a LocalTensor from its local tensors.
    """
    it = iter(local_tensors.values())
    first_local_tensor = next(it)

    first_shape = first_local_tensor.shape
    first_stride = first_local_tensor.stride()
    dtype = first_local_tensor.dtype
    device = first_local_tensor.device
    layout = first_local_tensor.layout

    extra_dispatch_keys = _get_extra_dispatch_keys(first_local_tensor)

    # Assert that all tensors have the same dtype, layout and dispatch keys. Due
    # to uneven sharding, it is possible that tensors will have different shapes.
    for local_tensor in it:
        assert dtype == local_tensor.dtype, (
            "Tensors representing LocalTensor shards must have the same dtype"
        )
        assert layout == local_tensor.layout, (
            "Tensors representing LocalTensor shards must have the same layout"
        )
        assert extra_dispatch_keys == _get_extra_dispatch_keys(local_tensor), (
            "Tensors representing LocalTensor shards must have the same set of extra dispatch keys"
        )

    # Compute shape/stride.  We allow for non-SPMD'ness here
    local_shapes: dict[int, dict[int, int]] = defaultdict(dict)  # dim => rank => size
    local_strides: dict[int, dict[int, int]] = defaultdict(dict)  # dim => rank => size
    for r, local_tensor in local_tensors.items():
        for d, size in enumerate(local_tensor.shape):
            local_shapes[d][r] = size
            local_strides[d][r] = local_tensor.stride(d)
    shape = [
        (
            first_shape[d]
            if _all_elements_same(list(local_shapes[d].values()))
            else torch.SymInt(LocalIntNode(local_shapes[d]))
        )
        for d in range(len(first_shape))
    ]
    strides = [
        (
            first_stride[d]
            if _all_elements_same(list(local_strides[d].values()))
            else torch.SymInt(LocalIntNode(local_strides[d]))
        )
        for d in range(len(first_shape))
    ]
    return shape, strides, device, dtype, layout, extra_dispatch_keys


class LocalTensor(torch.Tensor):
    """
    LocalTensor is a Tensor subclass that simulates a tensor distributed across multiple SPMD
    (Single Program, Multiple Data) ranks. Each LocalTensor instance internally holds a mapping from
    global rank ids to their corresponding local Tensor shards.Operations performed on a LocalTensor
    are applied independently to each local shard, mimicking distributed computation. Collectives
    and other distributed operations are handled by mapping them to the local shards as appropriate.

    Note:
        This class is primarily intended for debugging and simulating distributed tensor computations
        on a single process.

    """

    # Map from global rank to the local tensor.
    _local_tensors: dict[int, torch.Tensor]
    # Precomputed for speed set of keys from the local tensor map.
    _ranks: frozenset[int]
    _size: list[torch.SymInt | int]
    __slots__ = ["_local_tensors", "_ranks", "_size"]

    @staticmethod
    @torch._disable_dynamo
    def __new__(
        cls,
        local_tensors: dict[int, torch.Tensor],
        requires_grad: bool = False,
    ) -> "LocalTensor":
        if any(t.requires_grad for t in local_tensors.values()):
            raise AssertionError(
                "Internal local_tensors require grad, but we will ignore those autograd graph. "
                "Make a custom autograd function and make sure you detach the inner tensors."
            )

        (shape, strides, device, dtype, layout, extra_dispatch_keys) = (
            _compute_local_tensor_meta(local_tensors)
        )

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            strides=strides,
            dtype=dtype,
            device=device,
            layout=layout,
            # In place ops potentially change local tensor sizes (e.g. resize_). While
            # executing an in-place op the return value must be the same as "self" input
            # otherwise we can introduce errors due to tensor identity changes. Hence we
            # need to be able to update wrapper subclass sizes after in-place ops. This
            # dispatch policy allows us to do that.
            dispatch_sizes_strides_policy="sizes",
            requires_grad=requires_grad,
            _extra_dispatch_keys=extra_dispatch_keys,
        )

        local_tensors = {
            r: v if not isinstance(v, AsyncCollectiveTensor) else v.wait()
            for r, v in local_tensors.items()
        }
        r._local_tensors = local_tensors
        r._ranks = frozenset(local_tensors.keys())
        r._size = shape
        return r

    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    def __deepcopy__(self, memo: dict[Any, Any] | None) -> "LocalTensor":
        local_tensors_copy = {
            r: copy.deepcopy(t, memo) for r, t in self._local_tensors.items()
        }
        return LocalTensor(local_tensors_copy, self.requires_grad)

    def __repr__(self) -> str:  # type: ignore[override]
        parts = []
        for k, v in self._local_tensors.items():
            # pyrefly: ignore [bad-argument-type]
            parts.append(f"  {k}: {v}")
        tensors_str = ",\n".join(parts)
        return f"LocalTensor(\n{tensors_str}\n)"

    def __getattr__(self, name: str) -> Any:
        if _is_local_tensor_attr(name):
            rank = _from_local_tensor_attr(name)
            if rank not in self._ranks:
                raise AttributeError(f"Local tensor has no knowledge of rank {rank}")
            return self._local_tensors[rank]
        return object.__getattribute__(self, name)

    def __tensor_flatten__(self) -> tuple[list[str], tuple[Any, ...]]:
        """
        protocol to inform how to flatten a DTensor to local tensor
        for PT2 tracing
        """
        local_tensor_attrs = [_to_local_tensor_attr(r) for r in self._ranks]
        return local_tensor_attrs, ()

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Any],
        flatten_spec: tuple[Any, ...],
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
    ) -> "LocalTensor":
        assert flatten_spec is not None, (
            "Expecting spec to be not None from `__tensor_flatten__` return value!"
        )
        local_tensors = {
            _from_local_tensor_attr(a): t for a, t in inner_tensors.items()
        }
        return LocalTensor(local_tensors)

    @classmethod
    @torch._disable_dynamo
    def __torch_dispatch__(  # type: ignore[override]
        cls,
        func: Any,
        types: tuple[Any, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        # This is horribly inefficient
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        local_tensor = None
        for arg in flat_args:
            if isinstance(arg, LocalTensor):
                local_tensor = arg
                break

        assert local_tensor is not None, (
            "At least one of the arguments must be a LocalTensor"
        )

        # Check for unrecognized tensor subclasses (but allow regular tensors and scalars)
        has_unrecognized_types = _check_for_subclass(flat_args)
        if has_unrecognized_types:
            unrecognized_types = [
                type(x) for x in flat_args if _check_for_subclass_arg(x)
            ]
            not_implemented_log.debug(
                "LocalTensor unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        with LocalTensorMode(local_tensor._ranks):
            return func(*args, **kwargs)

    def numpy(
        self, *, force: bool = False
    ) -> np.ndarray:  # pyrefly: ignore  # missing-attribute
        if HAS_NUMPY:
            return self.reconcile().numpy(force=force)
        else:
            raise RuntimeError("Numpy is not available")

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> torch.Tensor:
        # pyrefly: ignore [bad-argument-type]
        return LocalTensor(
            # pyrefly: ignore [bad-argument-count]
            {
                r: t.contiguous(memory_format=memory_format)
                for r, t in self._local_tensors.items()
            }
        )

    def is_contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> bool:
        return all(
            t.is_contiguous(memory_format=memory_format)
            for t in self._local_tensors.values()
        )

    def tolist(self) -> list[Any]:
        """
        Reconcile and convert result to list.
        """

        return self.reconcile().tolist()

    def reconcile(self) -> torch.Tensor:
        """
        Reconciles the LocalTensor into a single torch.Tensor by ensuring all local
        shards are identical and returning a detached clone of one of them.

        Note:
            This method is useful for extracting a representative tensor from a LocalTensor
            when all shards are expected to be the same, such as after a collective operation
            that synchronizes all ranks.
        """

        # Force all local tensor shards across ranks to be the same
        it = iter(self._local_tensors.values())
        t1 = next(it)
        for t2 in it:
            assert torch.equal(t1, t2), (
                "LocalTensor shards must be the same to reconcile"
            )
        cl = t1.clone().detach()
        cl.requires_grad_(self.requires_grad)
        return cl

    def _sync_meta(self) -> None:
        with no_dispatch():
            (shape, strides, device, dtype, layout, extra_dispatch_keys) = (
                _compute_local_tensor_meta(self._local_tensors)
            )
            self._size = shape


_LOCAL_TENSOR_MODE: list["LocalTensorMode"] = []


class LocalTensorMode(TorchDispatchMode):
    """
    A TorchDispatchMode that simulates SPMD (Single Program, Multiple Data) execution
    for LocalTensor objects across a set of ranks.

    LocalTensorMode enables PyTorch operations to be transparently applied to each
    local shard of a LocalTensor, as if they were distributed across multiple ranks.
    When active, this mode intercepts tensor operations and dispatches them to each
    rank's local tensor, collecting and wrapping the results as LocalTensors. It also
    handles collective operations by mapping them to local implementations.

    This mode is primarily intended for debugging and simulating distributed tensor
    computations on a single process, rather than for high-performance distributed
    training. It maintains a stack of active modes, patches DeviceMesh coordinate
    resolution, and provides utilities for temporarily disabling the mode or mapping
    functions over ranks.
    """

    # What ranks this local tensor mode is operating over
    def __init__(self, ranks: Union[int, frozenset[int]]):
        if isinstance(ranks, int):
            # assume is world size
            self.ranks = frozenset(range(ranks))
        else:
            assert isinstance(ranks, frozenset)
            self.ranks = ranks
        self._disable = False
        self._old_get_coordinate = None
        self._old_get_local_rank: Any = None

    def __enter__(self) -> "LocalTensorMode":
        self._disable = False
        self._patch_device_mesh()
        _LOCAL_TENSOR_MODE.append(self)

        return super().__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._disable = True
        self._unpatch_device_mesh()
        _LOCAL_TENSOR_MODE.pop()
        super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[Any, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        flat_args, args_spec = pytree.tree_flatten((args, kwargs))

        # Find all LocalTensor arguments to determine ranks
        local_tensors = [a for a in flat_args if isinstance(a, LocalTensor)]

        # Check for unrecognized tensor subclasses (but allow regular tensors and scalars)
        has_unrecognized_types = _check_for_subclass(flat_args)
        if has_unrecognized_types:
            unrecognized_types = [
                type(x) for x in flat_args if _check_for_subclass_arg(x)
            ]
            not_implemented_log.debug(
                "LocalTensorMode unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        # Factory functions convert into LocalTensor, so we don't have to
        # transmute a Tensor into a LocalTensor if mutation happens...
        # But if you do an operation on a Tensor, do NOT wrap it into a
        # LocalTensor.  This helps prevent accidents when you're doing Tensor
        # operations on the inner non-wrapped tensors.
        if not local_tensors:
            if self._disable or any(isinstance(a, Tensor) for a in flat_args):
                return func(*args, **kwargs)

        # For LocalTensors, verify they have compatible ranks
        for a in flat_args:
            if isinstance(a, LocalTensor):
                assert a._ranks <= self.ranks, (
                    f"Input LocalTensor {a} and LocalTensorMode must be configured for the same ranks"
                )

        if func.overloadpacket == torch.ops.aten.dim:
            return len(args[0]._size)
        if func.overloadpacket == torch.ops.aten.sym_size:
            return tuple(args[0]._size)

        if func.namespace == "c10d":
            if func is torch.ops.c10d.allreduce_.default:
                return _c10d._local_all_reduce_(*args, **kwargs)
            elif func is torch.ops.c10d.allreduce_coalesced_.default:
                return _c10d._local_allreduce_coalesced_(*args, **kwargs)
            elif func is torch.ops.c10d.reduce_scatter_tensor_coalesced_.default:
                return _c10d._local_reduce_scatter_tensor_coalesced_(*args, **kwargs)
            elif func is torch.ops.c10d.scatter_.default:
                return _c10d._local_scatter_(*args, **kwargs)
            elif func is torch.ops.c10d.broadcast_.default:
                return _c10d._local_broadcast_(*args, **kwargs)
            elif func is torch.ops.c10d.allgather_.default:
                return _c10d._local_all_gather_(*args, **kwargs)
            elif func is torch.ops.c10d.allgather_into_tensor_coalesced_.default:
                return _c10d._local_allgather_into_tensor_coalesced_(*args, **kwargs)
            elif func is torch.ops.c10d.gather_.default:
                return _c10d._local_gather_(*args, **kwargs)
            elif func is torch.ops.c10d.alltoall_.default:
                return _c10d._local_alltoall_(*args, **kwargs)
            elif func is torch.ops.c10d.alltoall_base_.default:
                return _c10d._local_alltoall_base_(*args, **kwargs)
            elif func is torch.ops.c10d.barrier.default:
                return _c10d._local_barrier(*args, **kwargs)
            elif func is torch.ops.c10d.monitored_barrier_.default:
                return _c10d._local_monitored_barrier_(*args, **kwargs)
            elif func is torch.ops.c10d.send.default:
                return _c10d._local_send(*args, **kwargs)
            elif func is torch.ops.c10d.recv_.default:
                return _c10d._local_recv_(*args, **kwargs)
            elif func is torch.ops.c10d.recv_any_source_.default:
                return _c10d._local_recv_any_source_(*args, **kwargs)
            raise NotImplementedError(f"{func} not implemented")

        if func.namespace == "_c10d_functional" or func.namespace == "_dtensor":
            if func is torch.ops._dtensor.shard_dim_alltoall.default:
                return _c10d._local_functional_shard_dim_alltoall(*args, **kwargs)
            elif func is torch.ops._c10d_functional.all_gather_into_tensor.default:
                return _c10d._local_functional_all_gather_into_tensor(*args, **kwargs)
            elif func is torch.ops._c10d_functional.reduce_scatter_tensor.default:
                return _c10d._local_functional_reduce_scatter_tensor(*args, **kwargs)
            else:
                with LocalTensorMode(self.ranks):
                    return func._op_dk(
                        DispatchKey.CompositeExplicitAutograd, *args, **kwargs
                    )

        if func.namespace == "profiler":
            return func(*args, **kwargs)

        if func.namespace == "_c10d_functional_autograd":
            raise NotImplementedError(f"{func} not implemented")

        if func.namespace == "symm_mem":
            raise NotImplementedError(f"{func} not implemented")

        return _for_each_rank_run_func(func, self.ranks, args, kwargs, alias=True)

    @contextlib.contextmanager
    def disable(self) -> Generator[None, None, None]:
        """
        Disables LocalTensorMode temporarily. Primarily is intended to be used to perform
        rank specific computations and merge results back before enabling LocalTensorMode back.
        """

        old = self._disable
        self._disable = True
        self._unpatch_device_mesh()
        try:
            yield
        finally:
            self._disable = old
            self._patch_device_mesh()

    def rank_map(self, cb: Callable[[int], Tensor]) -> LocalTensor:
        """
        Creates a LocalTensor instance by mapping rank id to ids local shard.
        """

        with self.disable():
            # pyrefly: ignore [bad-argument-type, bad-argument-count]
            return LocalTensor({r: cb(r) for r in self.ranks})

    def _patch_device_mesh(self) -> None:
        assert self._old_get_coordinate is None
        self._old_get_coordinate = DeviceMesh.get_coordinate  # type: ignore[assignment]
        DeviceMesh.get_coordinate = _LocalDeviceMesh.get_coordinate  # type: ignore[method-assign]
        self._old_get_local_rank = DeviceMesh.get_local_rank  # type: ignore[assignment]
        DeviceMesh.get_local_rank = _LocalDeviceMesh.get_local_rank  # type: ignore[assignment]

    def _unpatch_device_mesh(self) -> None:
        assert self._old_get_coordinate is not None
        DeviceMesh.get_coordinate = self._old_get_coordinate
        DeviceMesh.get_local_rank = self._old_get_local_rank  # type: ignore[method-assign]
        # pyrefly: ignore [bad-assignment]
        self._old_get_coordinate = None
        self._old_get_local_rank = None  # pyrefly: ignore [bad-assignment]


class _LocalDeviceMesh:
    """
    Holds implementations of DeviceMesh functionality that must be patched while running
    under LocalTensorMode.
    """

    @staticmethod
    def get_local_rank(
        self: DeviceMesh, mesh_dim: Optional[Union[int, str]] = None
    ) -> Union[int, torch.SymInt]:
        lm = local_tensor_mode()
        assert lm is not None, "not in LocalTensorMode"

        local_ranks: dict[int, int] = {}

        if self.ndim > 1 and mesh_dim is None:
            raise RuntimeError(
                f"Found the DeviceMesh have {len(self._layout)} dimensions",
                "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
            )
        elif mesh_dim is None:
            mesh_dim_idx = 0
        elif isinstance(mesh_dim, str):
            mesh_dim_idx = (
                self.mesh_dim_names.index(mesh_dim) if self.mesh_dim_names else 0
            )
        else:
            mesh_dim_idx = mesh_dim

        try:
            for r in lm.ranks:
                _current_rank.rank = r
                coord_ints = self.get_coordinate()
                if coord_ints is None:
                    continue
                local_ranks[r] = coord_ints[mesh_dim_idx]
        finally:
            if hasattr(_current_rank, "rank"):
                del _current_rank.rank

        if len(set(local_ranks.values())) == 1:
            return next(iter(local_ranks.values()))

        return torch.SymInt(LocalIntNode(local_ranks))

    @staticmethod
    def get_coordinate(self: DeviceMesh) -> Optional[list[int] | None]:
        # NB: In order to support submeshes the code below recreates for each
        # rank submesh with the same mesh dimensions as current mesh. We are
        # doing this because when submesh is created it is created for a particular
        # rank (therefore below we are patching get_rank method). We are trying to
        # limit the invasiveness of local tensor.
        lm = local_tensor_mode()
        assert lm is not None, "Unexpectedly not in LocalTensorMode"

        current_rank = getattr(_current_rank, "rank", None)
        if current_rank is not None:
            rank_tensor = self._layout.remap_to_tensor(self._rank_map)
            rank_coords = (rank_tensor == current_rank).nonzero().tolist()
            assert len(rank_coords) == 1
            return rank_coords[0][1:]

        coords: list[dict[int, int]] = [{} for _ in range(self.ndim)]
        for r in lm.ranks:
            rank_tensor = self._layout.remap_to_tensor(self._rank_map)
            rank_coords = (rank_tensor == r).nonzero().tolist()
            assert len(rank_coords) == 1
            for d, c in enumerate(rank_coords[0][1:]):
                coords[d][r] = c

        out = [torch.SymInt(LocalIntNode(c)) for c in coords]
        # The output contains coordinates for each of the ranks with respect to
        # their meshes formed from root mesh and selecting the same dimensions
        # as the current mesh.
        return out  # type: ignore[return-value]


def reconcile_args(args: Any, kwargs: dict[str, Any] | None = None) -> Any:
    """
    Reconciles arguments by converting any LocalTensor instances in the input
    arguments to their underlying torch.Tensor representation.

    This function is typically used to prepare arguments for functions that
    expect standard torch.Tensor objects, by flattening the input arguments,
    replacing LocalTensor instances with their reconciled (standard tensor)
    versions, and then reconstructing the original argument structure.

    Args:
        args: Positional arguments, possibly containing LocalTensor instances.
        kwargs: Keyword arguments, possibly containing LocalTensor instances.

    Returns:
        Any: The arguments with all LocalTensor instances replaced by their reconciled torch.Tensor equivalents,
             preserving the original structure.
    """
    if kwargs is None:
        kwargs = {}
    flat_args, args_spec = pytree.tree_flatten((args, kwargs))
    reconciled_args = [
        a.reconcile() if isinstance(a, LocalTensor) else a for a in flat_args
    ]
    return pytree.tree_unflatten(reconciled_args, args_spec)


def local_tensor_mode() -> Optional[LocalTensorMode]:
    """
    Returns the current active LocalTensorMode if one exists.

    This function checks the global stack of LocalTensorMode instance. If there
    is at least one LocalTensorMode active, it returns the most recently entered
    (top of the stack) LocalTensorMode. If no LocalTensorMode is active, it returns None.

    Returns:
        Optional[LocalTensorMode]: The current LocalTensorMode if active, else None.
    """
    if len(_LOCAL_TENSOR_MODE) > 0:
        return _LOCAL_TENSOR_MODE[-1]
    return None


def maybe_run_for_local_tensor(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that ensures a function is executed for each local tensor shard
    when running under LocalTensorMode. If not in LocalTensorMode, the function
    is executed normally. When in LocalTensorMode, the function is run for each
    rank, and the results are collected appropriately.

    This decorator is useful for functions that exhibit non-SPMD behavior, such
    as those requiring rank specific actions. For example, a function that computes
    offset into input tensor based on rank.

    Note that the function being decorated must not have any side effects and
    contain operations for a single rank only. For example, wrapping a function
    that performs a collective operation will not work.

    Args:
        func (Callable[..., Any]): The function to be decorated.

    Returns:
        Callable[..., Any]: The wrapped function that handles LocalTensorMode logic.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        lm = local_tensor_mode()
        if lm is None or lm._disable:
            return func(*args, **kwargs)
        ret = None
        with lm.disable():
            ret = _for_each_rank_run_func(func, lm.ranks, args, kwargs, alias=False)

        return ret

    return wrapper


def maybe_disable_local_tensor_mode() -> contextlib.AbstractContextManager:
    """
    Context manager that disables LocalTensorMode for the duration of the context.
    """
    lm = local_tensor_mode()
    return lm.disable() if lm is not None else contextlib.nullcontext()
