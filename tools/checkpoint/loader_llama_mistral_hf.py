# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import torch
try:
    import transformers
except ImportError:
    raise ImportError("The 'transformers' package is not installed.")
import gc
import shutil
from tqdm import tqdm
import types

def add_arguments(parser):
    group = parser.add_argument_group(title='Llama/Mistral loader.')

    # TODO(jbarker): Need assertion to make sure *exactly* one of these is used
    parser.add_argument('--model-size', type=str, required=True,
                        choices=['llama2-7B', 'llama2-13B', 'llama2-70B', 'llama2-7Bf', 'llama2-13Bf', 'llama2-70Bf', 'llama3', 'mistral', 'yi-34B', 'qwen2.5', 'qwen3-30B-A3B'],
                        help='Select model size/type')
    parser.add_argument('--checkpoint-type', type=str, required=True,
                        choices=['meta', 'hf'],
                        help='Type of checkpoint to convert, options are "meta" or "hf"')
    parser.add_argument('--bf16', action='store_true', help='Whether to load weights in bf16.')
    parser.add_argument('--fp16', action='store_true', help='Whether to load weights in fp16.')
    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--tokenizer-model', required=True,
                       help='Tokenizer model file.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument("--make-vocab-size-divisible-by", type=int, default=None, help="Make vocab size divisible by")
    group.add_argument('--loader-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')


def verify_transformers_version():
    major, minor, patch = map(int, transformers.__version__.split('.'))
    assert major >= 4 and minor >= 31


NUM_SHARDS = {
    "llama2-7B": 1,
    "llama2-7Bf": 1,
    "llama2-13B": 2,
    "llama2-13Bf": 2,
    "llama2-70B": 8,
    "llama2-70Bf": 8,
}


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


# This conversion is adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
def convert_to_hf(model_path, input_base_path, model_size, tokenizer_path):
    if "llama2" in model_size:
        from transformers import LlamaConfig as ModelConfig
        from transformers import  LlamaTokenizer, LlamaTokenizerFast
    else:
        raise NotImplementedError(f"converting {model_size} is only supported using HuggingFace weights")

    # for backward compatibility, before you needed the repo to be called `my_repo/model_size`
    if not os.path.isfile(os.path.join(input_base_path, "params.json")):
        input_base_path = os.path.join(input_base_path, model_size)

    os.makedirs(model_path, exist_ok=True)

    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]
    params = params.get("model", params)
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = params.get("rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    if base > 10000.0:
        max_position_embeddings = 32768 if "mistral" in model_size else 16384
    else:
        max_position_embeddings = 4096

    if "llama2" in model_size:
        tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    else:
        raise AttributeError(f"model_size={model_size} not supported")

    if tokenizer_path is not None:
        if "llama2" in model_size:
            tokenizer = tokenizer_class(tokenizer_path)
            tokenizer.save_pretrained(model_path)
            vocab_size = tokenizer.vocab_size if tokenizer_path is not None else 32000
        else:
            raise AttributeError(f"model_size={model_size} is not supported")

    if params.get("n_kv_heads", None) is not None:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_local_key_value_heads = n_heads_per_shard // num_key_value_heads
        key_value_dim = dim // num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    # permute for sliced rotary
    def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # Load weights
    if num_shards == 1:
        # Not sharded
        # (The sharded implementation would also work, but this is simpler.)
        loaded = torch.load(os.path.join(input_base_path, "consolidated.00.pth"), map_location="cpu")
    else:
        # Sharded
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
            for i in range(num_shards)
        ]
    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        if num_shards == 1:
            # Unsharded
            q_proj = loaded[f"layers.{layer_i}.attention.wq.weight"]
            k_proj = loaded[f"layers.{layer_i}.attention.wk.weight"]
            if ("llama2" in model_size) or ("mistral" in model_size):
                q_proj = permute(q_proj)
                k_proj = permute(k_proj)
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": q_proj,
                f"model.layers.{layer_i}.self_attn.k_proj.weight": k_proj,
                f"model.layers.{layer_i}.self_attn.v_proj.weight": loaded[f"layers.{layer_i}.attention.wv.weight"],
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"layers.{layer_i}.attention.wo.weight"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w1.weight"],
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w2.weight"],
                f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w3.weight"],
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[f"layers.{layer_i}.attention_norm.weight"],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[f"layers.{layer_i}.ffn_norm.weight"],
            }
        else:
            # Sharded
            # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
            # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
            # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.

            state_dict = {
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[0][
                    f"layers.{layer_i}.attention_norm.weight"
                ].clone(),
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[0][
                    f"layers.{layer_i}.ffn_norm.weight"
                ].clone(),
            }
            state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(dim, dim)
            )
            state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(
                            num_local_key_value_heads, dims_per_head, dim
                        )
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(key_value_dim, dim),
                num_key_value_heads,
                key_value_dim,
                dim,
            )
            state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(
                        num_local_key_value_heads, dims_per_head, dim
                    )
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(key_value_dim, dim)

            state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
            )

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(model_path, filename))

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    if num_shards == 1:
        # Unsharded
        state_dict = {
            "model.embed_tokens.weight": loaded["tok_embeddings.weight"],
            "model.norm.weight": loaded["norm.weight"],
            "lm_head.weight": loaded["output.weight"],
        }
    else:
        d = 0 if "llama3" in model_size else 1
        state_dict = {
            "model.norm.weight": loaded[0]["norm.weight"],
            "model.embed_tokens.weight": torch.cat(
                [loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=d
            ),
            "lm_head.weight": torch.cat([loaded[i]["output.weight"] for i in range(num_shards)], dim=0),
        }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(model_path, "pytorch_model.bin.index.json"))
    ffn_dim_multiplier = params["ffn_dim_multiplier"] if "ffn_dim_multiplier" in params else 1
    multiple_of = params["multiple_of"] if "multiple_of" in params else 256
    config = ModelConfig(
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(dim, ffn_dim_multiplier, multiple_of),
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        rope_theta=base,
        max_position_embeddings=max_position_embeddings,
    )
    config.save_pretrained(model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    return model_path


def load_args_from_checkpoint(args, model_size):

    # Read Llama args.
    model_args_path = os.path.join(args.load, "config.json")
    with open(model_args_path) as f:
        model_args = json.load(f)

    # Update Megatron args.
    args.seq_length = 4096
    if "llama2" in model_size:
        # Correct bug in earlier conversion script.
        args.max_position_embeddings = 4096
    else:
        args.max_position_embeddings = model_args["max_position_embeddings"]

    args.hidden_size = model_args["hidden_size"]
    args.num_attention_heads = model_args["num_attention_heads"]
    args.num_layers = model_args["num_hidden_layers"]
    args.global_batch_size = 1024
    args.norm_epsilon = model_args["rms_norm_eps"]
    args.iteration = 1 # '0', 'release' don't work
    args.position_embedding_type = "rope"
    args.swiglu = True
    args.normalization = "RMSNorm"
    args.add_bias_linear = False
    args.untie_embeddings_and_output_weights = not model_args.get("tie_word_embeddings", False)
    args.vocab_size = model_args["vocab_size"]
    args.padded_vocab_size = model_args["vocab_size"]
    args.ffn_hidden_size = model_args["intermediate_size"]

    if "num_key_value_heads" in model_args:
        args.group_query_attention = True
        args.num_query_groups = model_args["num_key_value_heads"]
    if "head_dim" in model_args:
        args.kv_channels = model_args["head_dim"]
    if model_size == "qwen3-30B-A3B":
        args.bf16 = True
        args.qk_layernorm = True
        args.moe_grouped_gemm = True
        args.moe_router_score_function = "softmax"
        args.moe_token_dispatcher_type = "alltoall"
        args.moe_layer_freq = ([1]*48)
    if "moe_intermediate_size" in model_args:
        args.moe_ffn_hidden_size = model_args["moe_intermediate_size"]
    if "num_experts_per_tok" in model_args:
        args.moe_router_topk = model_args["num_experts_per_tok"]
    if "num_experts" in model_args:
        args.num_experts = model_args["num_experts"]
    if "router_aux_loss_coef" in model_args:
        args.moe_aux_loss_coeff = model_args["router_aux_loss_coef"]

def set_preprocess_state(args, model, hf_model):
    '''Set embedding params.'''
    model.embedding.word_embeddings.weight.data.copy_(
        hf_model.model.embed_tokens.weight)


def set_postprocess_state(args, model, hf_model):
    '''Set output layer & norm params.'''
    model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    if args.untie_embeddings_and_output_weights:
        model.output_layer.weight.data.copy_(hf_model.lm_head.weight)


def set_attn_state(args, layer, hf_layer):
    '''Set self-attention params.'''

    # Get attention layer & state.
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    # Reshape loaded weights.
    tp = args.tensor_model_parallel_size
    nh = args.num_attention_heads // tp
    ng = (args.num_query_groups if args.group_query_attention \
        else args.num_attention_heads) // tp
    dim = args.kv_channels
    assert nh % ng == 0

    # Copy weights (re-order dimensions for Megatron).
    attn.linear_qkv.weight.data.copy_(torch.cat([
        hf_attn.q_proj.weight.reshape((ng, dim*nh//ng, -1)),
        hf_attn.k_proj.weight.reshape((ng, dim, -1)),
        hf_attn.v_proj.weight.reshape((ng, dim, -1)),
    ], dim=1).reshape((-1, args.hidden_size)))
    if args.add_qkv_bias:
        attn.linear_qkv.bias.data.copy_(torch.cat([
            hf_attn.q_proj.bias.reshape((ng, dim*nh//ng)),
            hf_attn.k_proj.bias.reshape((ng, dim)),
            hf_attn.v_proj.bias.reshape((ng, dim)),
        ], dim=1).reshape(-1))

    attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)
    if args.qk_layernorm:
        attn.q_layernorm.weight.data.copy_(hf_attn.q_norm.weight)
        attn.k_layernorm.weight.data.copy_(hf_attn.k_norm.weight)

def set_mlp_state(args, layer, hf_layer):
    '''Set MLP params.'''

    mlp = layer.mlp
    hf_mlp = hf_layer.mlp

    if getattr(args, "num_experts", 0) == 0:
        mlp.linear_fc1.weight.data.copy_(torch.cat([
            hf_mlp.gate_proj.weight,
            hf_mlp.up_proj.weight,
        ], dim=0))
        mlp.linear_fc2.weight.data.copy_(hf_mlp.down_proj.weight)
    else:
        layer.mlp.router.weight.data.copy_(hf_mlp.gate.weight)
        hf_experts = hf_mlp.experts
        if not args.moe_grouped_gemm:
            mcore_experts = layer.mlp.experts.local_experts
            for expert_idx in range(args.num_experts):
                mcore_experts[expert_idx].linear_fc1.weight.data.copy_(
                    torch.cat([
                        hf_experts[expert_idx].gate_proj.weight,
                        hf_experts[expert_idx].up_proj.weight
                    ], dim=0)
                )
                mcore_experts[expert_idx].linear_fc2.weight.data.copy_(
                    hf_experts[expert_idx].down_proj.weight
                )
        else:
            mcore_experts = layer.mlp.experts

            for expert_idx in range(len(hf_experts)):
                getattr(mcore_experts.linear_fc1,f'weight{expert_idx}').data.copy_(
                    torch.cat([
                        hf_experts[expert_idx].gate_proj.weight,
                        hf_experts[expert_idx].up_proj.weight
                    ], dim=0)
                )
                getattr(mcore_experts.linear_fc2, f'weight{expert_idx}').data.copy_(
                    hf_experts[expert_idx].down_proj.weight
                )


def set_layer_state(args, model, hf_model, layer_idx):
    '''Set transformer layer params.'''

    layer = model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    set_attn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer)
    layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(hf_layer.input_layernorm.weight)
    if getattr(args, "num_experts", 0) == 0:
        layer.mlp.linear_fc1.layer_norm_weight.data.copy_(hf_layer.post_attention_layernorm.weight)
    else:
        layer.pre_mlp_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)


def load_checkpoint_to_model(args):
    '''Set model params.'''

    from pretrain_gpt import model_provider
    from transformers import AutoModelForCausalLM

    # Load Huggingface model.
    hf_model = AutoModelForCausalLM.from_pretrained(args.load, torch_dtype=args.params_dtype, device_map="auto")

    
    # Init Megatron model.
    model = model_provider(True, True).to(args.params_dtype)

    # Set model state.
    set_preprocess_state(args, model, hf_model)
    set_postprocess_state(args, model, hf_model)
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        set_layer_state(args, model, hf_model, layer_idx)

    return model


def _load_checkpoint(queue, args):

    verify_transformers_version()

    # Search in directory above this.
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    # Convert Meta checkpoint to HF format as an intermediate step
    if args.checkpoint_type == "meta":
        model_tmp_path = convert_to_hf(model_path=os.path.join(args.save_dir, 'tmp'), input_base_path=args.load_dir, model_size=args.model_size, tokenizer_path=args.tokenizer_model)
        args.load_dir = model_tmp_path
        args.tokenizer_model = model_tmp_path # point to HF tokenizer model

    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_args, set_global_variables, get_args
        from megatron.legacy.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us.
    sys.argv = ['script.py',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--mock-data', # To pass the "blend data checks" in arguments.py
                '--no-initialization',
                '--load', args.load_dir,
                '--no-one-logger',
                ]

    if args.make_vocab_size_divisible_by is not None:
        sys.argv.extend(["--make-vocab-size-divisible-by", str(args.make_vocab_size_divisible_by)])

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    load_args_from_checkpoint(margs, args.model_size)

    if "llama2" in args.model_size:
        margs.tokenizer_type = "Llama2Tokenizer"
    elif "yi" in args.model_size:
        margs.tokenizer_type = "HuggingFaceTokenizer"
    elif "llama3" in args.model_size:
        margs.tokenizer_type = "HuggingFaceTokenizer"
    elif "mistral" in args.model_size:
        margs.tokenizer_type = "HuggingFaceTokenizer"
    elif "qwen2.5" in args.model_size:
        margs.tokenizer_type = "HuggingFaceTokenizer"
        margs.add_qkv_bias = True
    elif "qwen3-30B-A3B" in args.model_size:
        margs.tokenizer_type = "HuggingFaceTokenizer"
        margs.add_bias_linear = False
        margs.add_qkv_bias = False

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes.
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    margs = validate_args(margs)

    margs.use_legacy_models = False
    margs.transformer_impl = args.loader_transformer_impl

    margs.position_embedding_type = "rope"

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('expert_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('disable_bias_linear', False)
    check_for_arg('params_dtype')
    check_for_arg('swiglu', False)

    # Determine how to make our models.
    assert args.model_type == 'GPT', 'Llama-2, Llama-3 and Mistral are GPT models.'
    margs.model_type = ModelType.encoder_or_decoder
    margs.params_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32

    # Suppress warning about torch.distributed not being initialized.
    module.MegatronModule.embedding_warning_printed = True

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    fused_kernels.load(margs)

    # Short aliases.
    tp_size = margs.tensor_model_parallel_size
    ep_size = margs.expert_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Metadata.
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.qkv_bias = margs.add_qkv_bias
    md.norm_has_bias = False
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_expert_parallel_size = margs.expert_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    md.qk_layernorm = getattr(margs, "qk_layernorm", False)
    md.moe_grouped_gemm = getattr(margs, "moe_grouped_gemm", False)
    

    margs.model_size = args.model_size

    # Get true (non-padded) vocab size
    tokenizer = transformers.AutoTokenizer.from_pretrained(margs.tokenizer_model)
    md.true_vocab_size = tokenizer._tokenizer.get_vocab_size(with_added_tokens=True)

    # Get first pipe stage.
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)
    model = load_checkpoint_to_model(margs)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings.
    message = {
        "word embeddings": model.embedding.word_embeddings.weight.data
    }
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = model.embedding.position_embeddings.weight.data
    else:
        assert not hasattr(model.embedding, 'position_embeddings')

    queue_put("embeddings", message)

    for layer_num in range(margs.num_layers):
        message = {}

        # Get non-parallel tensors from tp_rank 0.
        layer = model.decoder.layers[layer_num]
        message["input norm weight"] = layer.self_attention.linear_qkv.layer_norm_weight.data
        message["post norm weight"] = layer.pre_mlp_layernorm.weight.data if getattr(margs, "num_experts", 0) != 0 else layer.mlp.linear_fc1.layer_norm_weight.data
        if md.linear_bias:
            message["dense bias"] = layer.mlp.linear_fc1.bias.data
            message["mlp l1 bias"] = layer.mlp.linear_fc2.bias.data

        # Grab all parallel tensors for this layer.
        qkv_weight = []
        qkv_bias = []
        dense_weight = []
        
        # layer = model.language_model.encoder.layers[layer_num]
        qkv_weight.append(layer.self_attention.linear_qkv.weight.data)
        dense_weight.append(layer.self_attention.linear_proj.weight.data)
        if md.qkv_bias:
            qkv_bias.append(layer.self_attention.linear_qkv.bias.data)
        message["qkv weight"] = torch.cat(qkv_weight, dim=0)
        message["dense weight"] = torch.cat(dense_weight, dim=1)
        if md.qkv_bias:
            message["qkv bias"] = torch.cat(qkv_bias, dim=0)
        if margs.qk_layernorm:
            message["q norm weight"] = layer.self_attention.q_layernorm.weight.data
            message["k norm weight"] = layer.self_attention.k_layernorm.weight.data
        if getattr(margs, "num_experts", 0) == 0:
            mlp_l0_weight = []
            mlp_l0_bias = []
            mlp_l1_weight = []
            mlp_l0_weight.append(layer.mlp.linear_fc1.weight.data)
            mlp_l1_weight.append(layer.mlp.linear_fc2.weight.data)
            if md.linear_bias:
                mlp_l0_bias.append(layer.mlp.linear_fc1.bias.data)
            # Handle gated linear units.
            if md.swiglu:
                # Concat all the first halves ('W's) and all the second halves ('V's).
                for tp_rank in range(tp_size):
                    mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
                message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
                message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
            else:
                message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)
            message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
            if md.linear_bias: 
                if md.swiglu:
                    for tp_rank in range(tp_size):
                        mlp_l0_bias[tp_rank] = torch.chunk(mlp_l0_bias[tp_rank], 2, dim=0)
                    message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias],dim=0)
                    message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias],dim=0)
                else:
                    message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)
        else:
            message["router weight"] = layer.mlp.router.weight.data
            if not margs.moe_grouped_gemm:
                experts = layer.mlp.experts.local_experts
                if md.swiglu:
                    chunked_mlp_l0_weight =  [torch.chunk(local_expert.linear_fc1.weight.data, 2, dim=0) for local_expert in experts]
                    message["mlp l0 weight W"] = torch.stack([local_weight[0] for local_weight in chunked_mlp_l0_weight], dim=0)
                    message["mlp l0 weight V"] = torch.stack([local_weight[1] for local_weight in chunked_mlp_l0_weight], dim=0)
                else:
                    message["mlp l0 weight"] = torch.stack([local_expert.linear_fc1.weight.data for local_expert in experts])
                message["mlp l1 weight"] = torch.stack([local_expert.linear_fc2.weight.data for local_expert in experts], dim=0)
            else:
                experts = layer.mlp.experts
                if md.swiglu:
                    chunked_mlp_l0_weight =  [torch.chunk(weight.data, 2, dim=0) for weight in experts.linear_fc1._parameters.values()]
                    message["mlp l0 weight W"] = torch.stack([local_weight[0] for local_weight in chunked_mlp_l0_weight], dim=0)
                    message["mlp l0 weight V"] = torch.stack([local_weight[1] for local_weight in chunked_mlp_l0_weight], dim=0)
                else:
                    message["mlp l0 weight"] = torch.stack([weight.data for weight in experts.linear_fc1._parameters.values()])
                message["mlp l1 weight"] = torch.stack([weight.data for weight in experts.linear_fc2._parameters.values()], dim=0)
                    
                
    
        queue_put(f"transformer layer {layer_num}", message)

    # Send final norm from tp_rank 0.
    message = {
        "weight": model.decoder.final_layernorm.weight.data,
    }
    queue_put("final norm", message)

    if md.output_layer:
        message = {
            "weight": model.output_layer.weight.data
        }
        queue_put("output layer", message)

    queue.put("done")

    if args.checkpoint_type == "meta":
        shutil.rmtree(os.path.join(args.load_dir))


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except Exception:
        queue.put("exit")
        raise
