# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Common functions used in train_*.py and pretrain_*.py scripts."""

from typing import Callable, Optional, Union

import torch

from megatron.core.models.gpt import GPTModel
from megatron.core.models.mamba import MambaModel
from megatron.training import get_args, print_rank_0

try:
    from megatron.post_training.arguments import modelopt_args_enabled
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

import megatron.legacy.model  # isort: skip

# NOTE: Loading `megatron.legacy.model` earlier fails due to circular import

# NVCOMP / Checkpoint #################################################
# --- add near the top of pretrain_gpt.py ---
from megatron.core.dist_checkpointing.strategies.base import (
    register_default_strategy, StrategyAction,
)
from megatron.core.dist_checkpointing.serialization import (
    get_default_save_sharded_strategy,  # optional sanity check
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper,FullyParallelLoadStrategyWrapper
from megatron.core import parallel_state

# your GPU-compressing strategy (nvCOMP + optional GDS)
from megatron.core.dist_checkpointing.strategies.nvcomp_torch_gds import NvcompTorchDistAsyncSave, NvcompTorchDistLoadShardedStrategy
_NVCOMP_STRATEGY_INSTALLED = False
def _maybe_register_nvcomp_strategy():
    global _NVCOMP_STRATEGY_INSTALLED
    if _NVCOMP_STRATEGY_INSTALLED:
        return
    args = get_args()
    if not getattr(args, "use_dist_ckpt", False):
        return

    # Build your base torch_dist saver (version 1).
    base = NvcompTorchDistAsyncSave(codec_name="deflate", codec_opts=None, use_gds=True)

    # Optional but typical: keep “fully-parallel DP” semantics.
    dp_group = parallel_state.get_data_parallel_group(with_context_parallel=False)
    strategy = FullyParallelSaveStrategyWrapper(base, dp_group, getattr(args, "async_save", True))

    # Make it the default for ('torch_dist', 1) SAVE_SHARDED so Megatron picks it up.
    register_default_strategy(StrategyAction.SAVE_SHARDED, "torch_dist", 1, strategy)

    # Also register the matching LOAD_SHARDED strategy so resumes work.
    register_default_strategy(StrategyAction.LOAD_SHARDED, "torch_dist", 1, NvcompTorchDistLoadShardedStrategy())

    _NVCOMP_STRATEGY_INSTALLED = True
######################################################################


def model_provider(
    model_builder: Callable, pre_process=True, post_process=True, vp_stage: Optional[int] = None
) -> Union[GPTModel, megatron.legacy.model.GPTModel, MambaModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        model_builder: A callable that builds the actual model, its signature is the same as model_provider's with an exception of the first argument which is a builder itself. In addition might take a config passed from outside to skip its own config loading. See gpt_builder or mamba_builder for an example, see _gpt_model_builder in train_rl.py to see how to augment a default gpt builder and pass the config from outside
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel, MambaModel]: The returned model
    """
    _maybe_register_nvcomp_strategy()
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return model_provider_modelopt(pre_process, post_process)

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            dump(
                snapshot,
                open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'),
            )

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    return model_builder(args, pre_process, post_process, vp_stage)


def count_parameters_in_layer(model, layer_name):
    num_params = 0
    for name, param in model.named_parameters():
        if layer_name in name:
            num_params += param.numel()
            print_rank_0(f" - {name}: {param.numel()}")
    return num_params
