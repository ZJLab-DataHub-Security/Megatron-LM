# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import os
import sys

# Add megatron and the multimodal example to the path.
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)
    )
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

import torch
from transformers import AutoModel
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from examples.multimodal.model import model_provider
from examples.multimodal.train import loss_func
from examples.multimodal.multimodal_args import add_multimodal_extra_args
from megatron.training import get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.core.models.multimodal.llava_model import IMAGE_TOKEN,IGNORE_INDEX,pixel_shuffle


def compare(mcore_res, hf_res):
    diff = (mcore_res - hf_res).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    
    mean_value = (mcore_res.abs() + hf_res.abs()).mean().item()
    mean_diff_per = (mean_diff / mean_value) * 100
    print(f"mean diff {mean_diff}, max diff {max_diff}, mean diff percentage {mean_diff_per:.2f}%")
    assert mean_diff < 0.2, "mean output difference is greater than expected"
    assert max_diff < 50, "max output difference is greater than expected"

    print("lgtm")

def build_mcore_model(model_path):
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    # Megatron has some mandatory flags.
    sys.argv = [
        "ignore_me.py",
        "--micro-batch-size=2",
        "--num-layers=24",
        "--vision-model-type=internvit300M",
        "--language-model-type=qwen2.0_0.5B",
        "--tokenizer-prompt-format=qwen2p0",
        "--tokenizer-type=MultimodalTokenizer",
        "--tokenizer-model=Qwen/Qwen2-0.5B-Instruct",
        "--vocab-size=151655",
        "--hidden-size=896",
        "--num-attention-heads=14",
        "--seq-length=4096",
        "--decoder-seq-length=4096",
        "--max-position-embeddings=32768",
        "--bf16",
        "--img-h=448",
        "--img-w=448",
        "--patch-dim=14",
        "--tensor-model-parallel-size=1",
        "--use-te",
        "--apply-layernorm-1p",
        "--attention-softmax-in-fp32",
        "--use-distributed-optimizer",
        "--transformer-impl=transformer_engine",
        "--normalization=RMSNorm",
        "--group-query-attention",
        "--num-query-groups=2",
        "--no-masked-softmax-fusion",
        "--use-flash-attn",
        "--untie-embeddings-and-output-weights",
        "--disable-vision-class-token",
        "--disable-bias-linear",
        "--position-embedding-type=rope",
        "--rotary-percent=1.0",
        "--rotary-base=1000000",
        "--swiglu",
        "--attention-dropout=0.0",
        "--hidden-dropout=0.0",
        "--pipeline-model-parallel-size=1",
        "--max-position-embeddings=32768",
        "--ffn-hidden-size=4864",
        "--eod-mask-loss",
        "--freeze-ViT",
        "--pixel-shuffle",
        "--max-num-tiles=6",
        "--use-thumbnail",
        "--force-system-message",
        "--use-tiling",
        "--ckpt-format=torch",
        f"--pretrained-checkpoint={model_path}",
    ]

    initialize_megatron(extra_args_provider=add_multimodal_extra_args)
    def wrapped_model_provider(pre_process, post_process):
        return model_provider(pre_process, post_process, parallel_output=False)

    # Set up model and load checkpoint.
    model = get_model(wrapped_model_provider, wrap_with_ddp=False)
    
    load_checkpoint(model, None, None)
    
    if isinstance(model, list):
        model = model[0]
    
    model.eval()
    return model

def run_mcore_language(model):
    input_ids = torch.ones((1, 512), dtype=torch.int64, device='cuda')
    attn_mask = torch.ones((1, 512), dtype=torch.bool, device='cuda')
    position_ids = torch.stack([torch.arange(512, dtype=torch.int64, device='cuda') for _ in range(1)])

    output = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attn_mask)
    
    return output

def run_mcore_vision(model):
    """Run mcore vision model."""
    images = torch.ones((4, 3, 448, 448), dtype=torch.bfloat16, device='cuda')
    image_embs = model(images)
    image_embs = image_embs[:, 1:, :]
    image_embs = pixel_shuffle(image_embs)
    
    return image_embs

def run_mcore_vision_projection(model):
    
    hidden_states = torch.ones(256, 4, 4096, dtype=torch.bfloat16,device='cuda')
    hidden_states = model(hidden_states)
    hidden_states = hidden_states.permute(1, 0, 2).contiguous()
    
    return hidden_states

def run_hf_language_model(model):
    input_ids = torch.ones((1, 512), dtype=torch.int64, device='cuda')
    attn_mask = torch.ones((1, 512), dtype=torch.bool, device='cuda')
    position_ids = torch.stack([torch.arange(512, dtype=torch.int64, device='cuda') for _ in range(1)])
    
    output = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attn_mask).logits
    
    return output

def run_hf_vision(model):
    """Run HF vision model."""
    images = torch.ones((4, 3, 448, 448), dtype=torch.bfloat16, device='cuda')
    image_embs = model(pixel_values=images, output_hidden_states=False, return_dict=True).last_hidden_state
    image_embs = image_embs[:, 1:, :]
    image_embs = pixel_shuffle(image_embs)
    
    return image_embs
    
def run_hf_vision_projection(model):
    hidden_states = torch.ones(4, 256, 4096, dtype=torch.bfloat16,device='cuda')
    hidden_states = model(hidden_states)
    
    return hidden_states


def main(mcore_model, hf_model):
    """Compare vision model outputs between mcore and HF given the same fixed input."""
    mcore_model = build_mcore_model(mcore_model)
    hf_model = (
        AutoModel.from_pretrained(hf_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
        .cuda()
        .eval()
    )
    
    print("Starting to compare the result of mcore language model and hf language model")
    mcore_lm_res = run_mcore_language(mcore_model.module.language_model).cpu()
    hf_lm_res = run_hf_language_model(hf_model.language_model).cpu()
    mcore_lm_res = mcore_lm_res[:, :, : hf_lm_res.shape[-1]]
    compare(mcore_lm_res, hf_lm_res)
    
    print("Starting to compare the result of mcore vision model and hf vision model")
    mcore_vm_res = run_mcore_vision(mcore_model.module.vision_model)
    hf_vm_res = run_hf_vision(hf_model.vision_model)
    compare(mcore_vm_res, hf_vm_res)
    
    print("Starting to compare the result of mcore vision proj model and hf vision proj model")
    mcore_vm_proj_res = run_mcore_vision_projection(mcore_model.module.vision_projection)
    hf_vm_proj_res = run_hf_vision_projection(hf_model.mlp1)
    compare(mcore_vm_proj_res, hf_vm_proj_res)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check mcore vision model output vs. HF numerically.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mcore-model", type=str, required=True, help="directory for mcore model weights"
    )
    parser.add_argument("--hf-model", type=str, required=True, help="Model name in HF")

    args = parser.parse_args()

    main(args.mcore_model, args.hf_model)
