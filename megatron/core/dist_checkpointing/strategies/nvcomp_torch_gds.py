# megatron/core/dist_checkpointing/strategies/nvcomp_torch_gds.py
import os
import torch
from nvidia import nvcomp
from torch.utils.dlpack import from_dlpack
from megatron.core.dist_checkpointing.serialization import (
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.base import AsyncSaveShardedStrategy

class NvcompTorchDistAsyncSave(AsyncSaveShardedStrategy):
    """
    Wrap the default torch_dist v1 async sharded saver, but GPU-compress each
    tensor shard with nvCOMP before it leaves device. If GDS is available,
    write compressed GPU storages via torch.cuda.gds.GdsFile.
    """

    def __init__(self, codec_name="deflate", codec_opts=None, use_gds=True):
        super().__init__(backend="torch_dist", version=1)
        self._inner = get_default_save_sharded_strategy(backend="torch_dist", version=1)
        self._use_gds = use_gds and hasattr(torch.cuda, "gds")
        # Build an nvCOMP codec; you can pass algorithm/level knobs via codec_opts
        self._codec = nvcomp.Codec(**(codec_opts or {"algorithm": codec_name}))

    def _compress_cuda(self, t: torch.Tensor):
        # Assumes t is CUDA and contiguous
        arr = nvcomp.as_array(t)              # zero-copy wrap
        enc = self._codec.encode(arr)         # nvCOMP encodes on the GPU
        # Create a torch CUDA tensor (uint8) from the encoded nvCOMP Array
        enc_dl = enc.to_dlpack()
        enc_t = from_dlpack(enc_dl)           # torch.uint8 tensor on CUDA
        return enc_t

    def async_save(self, *args, **kwargs):
        """
        Delegate planning to the inner strategy, but intercept its per-write
        execution to compress+write. The exact interception point depends on the
        inner strategy’s implementation; we wrap callable(s) it uses to emit
        bytes and swap them with our compressed+GDS path.
        """
        # Ask inner strategy for a request object that exposes per-item saves
        req = self._inner.async_save(*args, **kwargs)

        # Monkey-patch the per-item "write tensor" hook the inner request uses.
        # This is a light wrapper: fetch tensor, compress on GPU, then:
        #  - if GDS: open GdsFile and file.save_storage(compressed.untyped_storage())
        #  - else: enc.cpu().numpy().tobytes() -> original writer
        orig_write = getattr(req, "write_tensor", None)

        def write_tensor_nvcomp(item):
            t: torch.Tensor = item.tensor
            assert t.is_cuda
            t_c = t.contiguous()
            enc_t = self._compress_cuda(t_c)

            # The inner strategy normally knows file path & offsets from the plan:
            path = item.path
            offset = item.offset

            if self._use_gds:
                import os
                from torch.cuda.gds import GdsFile
                # Write compressed CUDA storage directly via cuFile/GDS
                flags = os.O_CREAT | os.O_WRONLY
                f = GdsFile(path, flags)
                try:
                    f.save_storage(enc_t.untyped_storage(), offset=offset)
                finally:
                    f.deregister_handle()
            else:
                # fallback: bring compressed bytes to pinned host and let inner writer persist
                enc_cpu = enc_t.detach().to("cpu", non_blocking=True).contiguous()
                # Depending on inner API, forward a bytes-like buffer
                item.bytes_payload = memoryview(enc_cpu.numpy())
                return orig_write(item)

        if orig_write is not None:
            req.write_tensor = write_tensor_nvcomp

        return req

