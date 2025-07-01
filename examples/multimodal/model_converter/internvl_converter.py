import argparse
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)
    )
)

import torch
import transformers
from transformers import AutoModel
from internvit300M_converter import convert as convert_vision_model
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding

def convert_qwen2(hf_model, new_state_dicts, args, hidden_size=896, num_heads=14, head_size=64, num_layers=24, num_key_value_heads=2):
    """Convert InternViT HF checkpoint to mcore."""
    tensor_parallel_size = args.tensor_model_parallel_size

    hf_state_dict = hf_model.state_dict()
    prefix = "language_model."
    embed_weight = hf_state_dict[f'model.embed_tokens.weight']
    embed_weights_base_name = f"embedding.word_embeddings.weight"
    
    args.rank = 0
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_model)
    true_vocab_size = tokenizer._tokenizer.get_vocab_size(with_added_tokens=True)
    padded_vocab_size = _vocab_size_with_padding(true_vocab_size, args)
    origin_vocab_size = embed_weight.shape[0]
    if origin_vocab_size > padded_vocab_size:
        full_word_embed = embed_weight[0:padded_vocab_size, :]
    elif origin_vocab_size < padded_vocab_size:
        padding_size = padded_vocab_size - origin_vocab_size
        full_word_embed = torch.cat((
            embed_weight,
            embed_weight[-1].unsqueeze(0).expand(padding_size, -1)))
    # Same size!
    else:
        full_word_embed = embed_weight
    
    embed_weight_tp = torch.chunk(full_word_embed, tensor_parallel_size, dim=0)
    
    for i in range(tensor_parallel_size):
        new_state_dicts[i]["model"][prefix + embed_weights_base_name] = embed_weight_tp[i].clone()
    
    for layer_id in range(num_layers):
        base = f"decoder.layers.{layer_id}"
        
        num_query_groups = num_key_value_heads
        
        old_tensor_shape = hf_state_dict[f'model.layers.{layer_id}.self_attn.q_proj.weight'].size()
        new_q_tensor_shape = (num_heads, head_size) + old_tensor_shape[1:]
        new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]
         
        q = hf_state_dict[f'model.layers.{layer_id}.self_attn.q_proj.weight'].view(*new_q_tensor_shape)
        k = hf_state_dict[f'model.layers.{layer_id}.self_attn.k_proj.weight'].view(*new_kv_tensor_shape)
        v = hf_state_dict[f'model.layers.{layer_id}.self_attn.v_proj.weight'].view(*new_kv_tensor_shape)
        qkv_weights = torch.empty((0, head_size) + old_tensor_shape[1:])
        
        heads_per_group = num_heads // num_query_groups
        
        for i in range(num_query_groups):
            qkv_weights = torch.cat((qkv_weights, q[i * heads_per_group : (i + 1) * heads_per_group, :, :]))
            qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]))
            qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]))
        qkv_weights = qkv_weights.reshape([head_size * (num_heads + 2 * num_query_groups), hidden_size])
        if tensor_parallel_size == 1:
            num_padded_heads = num_query_groups
        else:
            num_padded_heads = ((num_query_groups + tensor_parallel_size - 1) // tensor_parallel_size) * tensor_parallel_size
        
        padded_dim = head_size * num_padded_heads * (2 + heads_per_group)
        qkv_weights_padded = torch.zeros((padded_dim, q.shape[-1]), dtype=qkv_weights.dtype, device=qkv_weights.device)
        qkv_weights_padded[: qkv_weights.shape[0], :] = qkv_weights
        qkv_weights = qkv_weights_padded
        qkv_weights_tp = torch.chunk(qkv_weights, tensor_parallel_size, dim=0)
        
        qkv_weights_base_name = f'{base}.self_attention.linear_qkv.weight'

        new_q_tensor_shape = (num_heads, head_size)
        new_kv_tensor_shape = (num_query_groups, head_size)
        q = hf_state_dict[f'model.layers.{layer_id}.self_attn.q_proj.bias'].view(*new_q_tensor_shape)
        k = hf_state_dict[f'model.layers.{layer_id}.self_attn.k_proj.bias'].view(*new_kv_tensor_shape)
        v = hf_state_dict[f'model.layers.{layer_id}.self_attn.v_proj.bias'].view(*new_kv_tensor_shape)
        qkv_bias = torch.empty((0, head_size))
        
        for i in range(num_query_groups):
            qkv_bias = torch.cat((qkv_bias, q[i * heads_per_group : (i + 1) * heads_per_group, :]))
            qkv_bias = torch.cat((qkv_bias, k[i : i + 1, :]))
            qkv_bias = torch.cat((qkv_bias, v[i : i + 1, :]))
        qkv_bias = qkv_bias.reshape([head_size * (num_heads + 2 * num_query_groups), ])
        qkv_bias_padded = torch.zeros(padded_dim, dtype=qkv_bias.dtype, device=qkv_bias.device)
        qkv_bias_padded[: qkv_bias.shape[0]] = qkv_bias
        qkv_bias = qkv_bias_padded
        qkv_bias_tp = torch.chunk(qkv_bias, tensor_parallel_size, dim=0)
        
        qkv_bias_base_name = f'{base}.self_attention.linear_qkv.bias'
        
        o_weight = hf_state_dict[f'model.layers.{layer_id}.self_attn.o_proj.weight']
        o_weight_tp = [o_weight for _ in range(tensor_parallel_size)]
        
        o_weight_base_name = f'{base}.self_attention.linear_proj.weight'
        
        mlp_down_weight = hf_state_dict[f'model.layers.{layer_id}.mlp.gate_proj.weight']
        mlp_gate_weight = hf_state_dict[f'model.layers.{layer_id}.mlp.up_proj.weight']
        mlp_down_weight = torch.cat((mlp_down_weight, mlp_gate_weight), axis=0)
        mlp_down_weight_tp = [mlp_down_weight for _ in range(tensor_parallel_size)]
        
        mlp_down_base_name = f'{base}.mlp.linear_fc1.weight'
        
        mlp_up_weight = hf_state_dict[f'model.layers.{layer_id}.mlp.down_proj.weight']
        mlp_up_weight_tp = [mlp_up_weight for _ in range(tensor_parallel_size)]
        
        mlp_up_base_name = f'{base}.mlp.linear_fc2.weight'
        
        input_ln_weight = hf_state_dict[f'model.layers.{layer_id}.input_layernorm.weight']
        input_ln_weight_tp = [input_ln_weight for _ in range(tensor_parallel_size)]
        
        input_ln_base_name = f'{base}.self_attention.linear_qkv.layer_norm_weight'
        
        post_attn_ln_weight = hf_state_dict[f'model.layers.{layer_id}.post_attention_layernorm.weight']
        post_attn_ln_weight_tp = [post_attn_ln_weight for _ in range(tensor_parallel_size)]
        
        post_attn_ln_base_name = f'{base}.mlp.linear_fc1.layer_norm_weight'
        
        for i in range(tensor_parallel_size):
            new_state_dicts[i]["model"][prefix + qkv_weights_base_name] = qkv_weights_tp[i].clone()
            new_state_dicts[i]["model"][prefix + qkv_bias_base_name] = qkv_bias_tp[i].clone()
            new_state_dicts[i]["model"][prefix + o_weight_base_name] = o_weight_tp[i].clone()
            new_state_dicts[i]["model"][prefix + mlp_down_base_name] = mlp_down_weight_tp[i].clone()
            new_state_dicts[i]["model"][prefix + mlp_up_base_name] = mlp_up_weight_tp[i].clone()
            new_state_dicts[i]["model"][prefix + input_ln_base_name] = input_ln_weight_tp[i].clone()
            new_state_dicts[i]["model"][prefix + post_attn_ln_base_name] = post_attn_ln_weight_tp[i].clone()
            new_state_dicts[i]["model"][prefix + f"{base}.self_attention.linear_qkv._extra_state"] = None
            new_state_dicts[i]["model"][prefix + f"{base}.self_attention.linear_proj._extra_state"] = None
            new_state_dicts[i]["model"][prefix + f"{base}.mlp.linear_fc1._extra_state"] = None
            new_state_dicts[i]["model"][prefix + f"{base}.mlp.linear_fc2._extra_state"] = None
        
        
    final_ln_weight = hf_state_dict[f'model.norm.weight']
    final_ln_base_name = f'decoder.final_layernorm.weight'
    
    output_layer_weight = hf_state_dict[f'lm_head.weight']
    origin_vocab_size = output_layer_weight.shape[0]
    if origin_vocab_size > padded_vocab_size:
        full_word_embed = output_layer_weight[0:padded_vocab_size, :]
    elif origin_vocab_size < padded_vocab_size:
        padding_size = padded_vocab_size - origin_vocab_size
        full_word_embed = torch.cat((
            output_layer_weight,
            output_layer_weight[-1].unsqueeze(0).expand(padding_size, -1)))
    # Same size!
    else:
        full_word_embed = embed_weight
    
    output_layer_weight_tp = torch.chunk(full_word_embed, tensor_parallel_size, dim=0)
    output_layer_base_name = f'output_layer.weight'
    
    for i in range(tensor_parallel_size):
        new_state_dicts[i]["model"][prefix + final_ln_base_name] = final_ln_weight.clone()
        new_state_dicts[i]["model"][prefix + output_layer_base_name] = output_layer_weight_tp[i].clone()
    
def convert_vision_proj_model(hf_model, tensor_parallel_size, new_state_dicts):
    hf_state_dict = hf_model.state_dict()
    
    prefix = "vision_projection."
    
    ln_weight = hf_state_dict[f'0.weight']
    ln_weight_base_name = f'encoder.linear_fc1.layer_norm_weight'
    
    ln_bias = hf_state_dict[f'0.bias']
    ln_bias_base_name = f'encoder.linear_fc1.layer_norm_bias'
    
    linear_fc1_weight = hf_state_dict[f'1.weight']
    linear_fc1_weight_tp = [linear_fc1_weight for _ in range(tensor_parallel_size)]
    linear_fc1_weight_base_name = f'encoder.linear_fc1.weight'
    
    linear_fc1_bias = hf_state_dict[f'1.bias']
    linear_fc1_bias_tp = [linear_fc1_bias for _ in range(tensor_parallel_size)]
    linear_fc1_bias_base_name = f'encoder.linear_fc1.bias'
    
    linear_fc2_weight = hf_state_dict[f'3.weight']
    linear_fc2_weight_tp = [linear_fc2_weight for _ in range(tensor_parallel_size)]
    linear_fc2_weight_base_name = f'encoder.linear_fc2.weight'
    
    linear_fc2_bias = hf_state_dict[f'3.bias']
    linear_fc2_bias_tp = [linear_fc2_bias for _ in range(tensor_parallel_size)]
    linear_fc2_bias_base_name = f'encoder.linear_fc2.bias'
    
    for i in range(tensor_parallel_size):
        new_state_dicts[i]["model"][prefix + ln_weight_base_name] = ln_weight.clone()
        new_state_dicts[i]["model"][prefix + ln_bias_base_name] = ln_bias.clone()
        new_state_dicts[i]["model"][prefix + linear_fc1_weight_base_name] = linear_fc1_weight_tp[i].clone()
        new_state_dicts[i]["model"][prefix + linear_fc1_bias_base_name] = linear_fc1_bias_tp[i].clone()
        new_state_dicts[i]["model"][prefix + linear_fc2_weight_base_name] = linear_fc2_weight_tp[i].clone()
        new_state_dicts[i]["model"][prefix + linear_fc2_bias_base_name] = linear_fc2_bias_tp[i].clone()
        new_state_dicts[i]["model"][prefix + "encoder.linear_fc1._extra_state"] = None
        new_state_dicts[i]["model"][prefix + "encoder.linear_fc2._extra_state"] = None
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InternVLChat HuggingFace to Mcore converter")
    parser.add_argument("--model-name-or-path", type=str, default=None, help="Model name in HuggingFace")
    parser.add_argument("--tokenizer-model", type=str, default=None, help="Sentencepiece tokenizer model")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for the mcore model.")
    parser.add_argument("--use-te", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--tensor-model-parallel-size", type=int, required=True)
    parser.add_argument("--make-vocab-size-divisible-by", type=int, required=True)

    args = parser.parse_args()
    args.params_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    print(f"model_path:{args.model_name_or_path}, tp:{args.tensor_model_parallel_size}, output_dir:{args.output_dir}")
    model = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=args.params_dtype, low_cpu_mem_usage=True, device_map="cpu", trust_remote_code=True)
    new_state_dicts = [{"model": dict()} for _ in range(args.tensor_model_parallel_size)]
    convert_qwen2(model.language_model, new_state_dicts, args)
    convert_vision_model(model.vision_model,args.tensor_model_parallel_size, new_state_dicts, args.use_te)
    convert_vision_proj_model(model.mlp1, args.tensor_model_parallel_size, new_state_dicts)
    
    for i in range(args.tensor_model_parallel_size):
        output_dir_tp = os.path.join(args.output_dir, f"iter_0000001/mp_rank_0{i}")
        os.makedirs(output_dir_tp, exist_ok=True)
        output_path_tp = os.path.join(output_dir_tp, "model_optim_rng.pt")
        torch.save(new_state_dicts[i], output_path_tp)
        print("saved file", output_path_tp)
    ckpt_tracker_file_path = os.path.join(args.output_dir, "latest_checkpointed_iteration.txt")
    with open(ckpt_tracker_file_path, "w") as file:
        file.write("1")
    print("checkpoint tracker file created.")