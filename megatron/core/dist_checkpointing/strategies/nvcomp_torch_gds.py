# megatron/core/dist_checkpointing/strategies/nvcomp_torch_gds.py
import io
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import torch
from nvidia import nvcomp
from torch.utils.dlpack import from_dlpack
from torch.distributed.checkpoint import (
    FileSystemReader,
    LoadPlan,
    ReadItem,
)
from torch.distributed.checkpoint._traverse import OBJ_PATH
from torch.distributed.checkpoint.metadata import Metadata

from megatron.core.dist_checkpointing.serialization import (
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.base import AsyncSaveShardedStrategy
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    _replace_state_dict_keys_with_sharded_keys,
    mcore_to_pyt_state_dict,
    _replace_sharded_keys_with_state_dict_keys,
    _restore_dict_types,
    TorchShardedTensor,
    ShardedTensor,
    nested_values,
    is_nd_flattened_tensor,
    get_reformulation_metadata,
    apply_nd_flattened_tensors_reformulation,
    restore_nd_flattened_tensors_formulation,
    MCoreLoadPlanner,
    _unwrap_pyt_sharded_tensor,
)


class NvcompTorchDistAsyncSave(AsyncSaveShardedStrategy):
    """
    GPU-compress tensor shards with nvCOMP before persisting. If GDS is available,
    write compressed GPU storages via torch.cuda.gds.GdsFile.
    """

    def __init__(self, codec_name="deflate", codec_opts=None, use_gds=True):
        super().__init__(backend="torch_dist", version=1)
        self._inner = get_default_save_sharded_strategy(backend="torch_dist", version=1)
        self._use_gds = use_gds and hasattr(torch.cuda, "gds")
        self._codec = nvcomp.Codec(**(codec_opts or {"algorithm": codec_name}))

    def _compress_cuda(self, t: torch.Tensor):
        arr = nvcomp.as_array(t)
        enc = self._codec.encode(arr)
        enc_dl = enc.to_dlpack()
        enc_t = from_dlpack(enc_dl)
        return enc_t

    def async_save(self, *args, **kwargs):
        req = self._inner.async_save(*args, **kwargs)
        orig_write = getattr(req, "write_tensor", None)

        def write_tensor_nvcomp(item):
            t: torch.Tensor = item.tensor
            assert t.is_cuda
            enc_t = self._compress_cuda(t.contiguous())
            path = item.path
            offset = item.offset
            if self._use_gds:
                from torch.cuda.gds import GdsFile

                flags = os.O_CREAT | os.O_WRONLY
                f = GdsFile(path, flags)
                try:
                    f.save_storage(enc_t.untyped_storage(), offset=offset)
                finally:
                    f.deregister_handle()
            else:
                enc_cpu = enc_t.detach().to("cpu", non_blocking=True).contiguous()
                item.bytes_payload = memoryview(enc_cpu.numpy())
                return orig_write(item)

        if orig_write is not None:
            req.write_tensor = write_tensor_nvcomp
        return req


class _NvcompGdsFileSystemReader(FileSystemReader):
    """FileSystemReader that decodes nvCOMP-compressed tensor payloads, with optional GDS IO."""

    def __init__(self, path: Union[str, os.PathLike], codec: Optional[nvcomp.Codec] = None, use_gds: bool = True) -> None:
        super().__init__(path=path)
        self._codec = codec or nvcomp.Codec(algorithm="deflate")
        self._use_gds = use_gds and hasattr(torch.cuda, "gds")

    def _read_compressed_cuda(self, relative_path: str, offset: int, length: int) -> torch.Tensor:
        enc_t = torch.empty(length, dtype=torch.uint8, device="cuda")
        if self._use_gds:
            from torch.cuda.gds import GdsFile

            flags = os.O_RDONLY
            f = GdsFile(cast(str, self.fs.concat_path(self.path, relative_path)), flags)
            try:
                f.load_storage(enc_t.untyped_storage(), offset=offset)
            finally:
                f.deregister_handle()
        else:
            # Fallback: CPU read then D2H
            new_path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(new_path, "rb") as stream:
                stream.seek(offset)
                data = stream.read(length)
            cpu = torch.frombuffer(memoryview(data), dtype=torch.uint8)
            enc_t.copy_(cpu, non_blocking=True)
            torch.cuda.synchronize()
        return enc_t

    def read_data(self, plan: LoadPlan, planner: MCoreLoadPlanner):  # type: ignore[override]
        per_file: dict[str, list[ReadItem]] = {}
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            for req in reqs:
                item_md = self.storage_data[req.storage_index]
                if req.type.name == "BYTE_IO":
                    # Delegate to base for BytesIO objects
                    new_path = self.fs.concat_path(self.path, relative_path)
                    with self.fs.create_stream(new_path, "rb") as stream:
                        stream.seek(item_md.offset)
                        read_bytes = io.BytesIO(stream.read(item_md.length))
                        read_bytes.seek(0)
                        planner.load_bytes(req, read_bytes)
                else:
                    enc_t = self._read_compressed_cuda(relative_path, item_md.offset, item_md.length)
                    dec_arr = self._codec.decode(nvcomp.as_array(enc_t))
                    dec_t = from_dlpack(dec_arr.to_dlpack())
                    # Narrow/cast to the requested chunk shape
                    target_tensor = planner.resolve_tensor(req).detach()
                    if dec_t.dtype != target_tensor.dtype:
                        dec_t = dec_t.to(target_tensor.dtype)
                    if dec_t.numel() != target_tensor.numel():
                        dec_t = dec_t.view(target_tensor.size())
                    target_tensor.copy_(dec_t)
                    planner.commit_tensor(req, target_tensor)

        fut: torch.futures.Future = torch.futures.Future()
        fut.set_result(None)
        return fut


class NvcompTorchDistLoadShardedStrategy(TorchDistLoadShardedStrategy):
    """Load strategy that decodes nvCOMP-compressed tensor shards, optionally via GDS."""

    def __init__(self, codec_name: str = "deflate", codec_opts: Optional[Dict[str, Any]] = None, use_gds: bool = True):
        super().__init__()
        self._codec = nvcomp.Codec(**(codec_opts or {"algorithm": codec_name}))
        self._use_gds = use_gds and hasattr(torch.cuda, "gds")

    def load(self, sharded_state_dict, checkpoint_dir: Union[str, Path]):  # type: ignore[override]
        # Preprocess (copied from TorchDistLoadShardedStrategy, but swap reader)
        reformulation_metadata = get_reformulation_metadata(sharded_state_dict, checkpoint_dir)
        sharded_state_dict, formulation_restore_data = apply_nd_flattened_tensors_reformulation(
            sharded_state_dict, reformulation_metadata
        )
        has_legacy_1d_flattened_tensors = False
        for sh_ten in nested_values(sharded_state_dict):
            if is_nd_flattened_tensor(sh_ten) and sh_ten.key not in reformulation_metadata:
                has_legacy_1d_flattened_tensors = True
                break
        flexible_shape_sharded_tensors = [
            sh_ten
            for sh_ten in nested_values(sharded_state_dict)
            if isinstance(sh_ten, ShardedTensor) and not sh_ten.allow_shape_mismatch
        ]
        allow_shape_mismatch_sharded_tensors = {
            sh_ten.key: sh_ten
            for sh_ten in nested_values(sharded_state_dict)
            if isinstance(sh_ten, ShardedTensor) and sh_ten.allow_shape_mismatch
        }
        orig_sharded_state_dict = sharded_state_dict
        (sharded_state_dict, flat_mapping, rename_mapping) = (
            _replace_state_dict_keys_with_sharded_keys(sharded_state_dict)
        )
        pyt_state_dict = mcore_to_pyt_state_dict(
            sharded_state_dict, True, load_legacy_1d_flatten_tensors=has_legacy_1d_flattened_tensors
        )
        fsr = _NvcompGdsFileSystemReader(checkpoint_dir, codec=self._codec, use_gds=self._use_gds)
        # Use MCore planner for validation/FP8 handling
        from megatron.core.dist_checkpointing.strategies.torch import checkpoint
        checkpoint.load_state_dict(
            pyt_state_dict,
            fsr,
            planner=MCoreLoadPlanner(
                shapes_validation_sharded_tensors=flexible_shape_sharded_tensors,
                allow_shape_mismatch_sharded_tensors=allow_shape_mismatch_sharded_tensors,
            ),
        )
        self.cached_global_metadata = fsr.read_metadata()
        pyt_state_dict = cast(Dict[str, Union[TorchShardedTensor, List[io.BytesIO]]], pyt_state_dict)
        mcore_state_dict = {k: _unwrap_pyt_sharded_tensor(v) for k, v in pyt_state_dict.items()}
        mcore_state_dict = _replace_sharded_keys_with_state_dict_keys(
            mcore_state_dict, flat_mapping, rename_mapping  # type: ignore[arg-type]
        )
        _restore_dict_types(mcore_state_dict, orig_sharded_state_dict)
        mcore_state_dict = restore_nd_flattened_tensors_formulation(
            mcore_state_dict, formulation_restore_data
        )
        return mcore_state_dict
