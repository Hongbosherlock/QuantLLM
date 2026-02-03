"""
Qwen3-Next-80B-A3B FP8 Blockwise 权重量化脚本

量化方法：
    - 格式：FP8 E4M3
    - 块大小：128x128
    - 激活方案：动态量化
    - 为每个权重 tensor 生成对应的 weight_scale_inv 参数

适用模型：
    - Qwen3-Next-80B-A3B-Instruct (BF16)

使用示例：
    基础用法：
        python3 fp8_quant_qwen.py \\
            models/Qwen3-Next-80B-A3B-Instruct \\
            quantization_log.txt

    指定输出目录：
        python3 fp8_quant_qwen.py \\
            models/Qwen3-Next-80B-A3B-Instruct \\
            quantization_log.txt \\
            --output-dir models/My_FP8_Model

输出内容：
    - 量化后的 safetensors 文件（包含 weight 和 weight_scale_inv）
    - 完整的 config.json（含 quantization_config）
    - tokenizer 等其他必要文件
    - 量化日志文件

量化配置已经自动更新（写入 config.json）：
    {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [...]
    }

作者：xuhongbo02
版本：1.0
"""
from typing import Tuple
import torch
import triton
import triton.language as tl
import argparse
import os
import json
from safetensors.torch import load_file, save_file


@triton.jit
def fp8_blockwise_quant_act_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.
    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.
    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def fp8_blockwise_act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.
    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    fp8_blockwise_quant_act_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s

@triton.jit
def fp8_blockwise_quant_weight_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, s)


def fp8_blockwise_weight_quant(x: torch.Tensor, block_size: int = 128):
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.dim() == 2, 'Input tensor must have 2 dimensions'
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s_rows = triton.cdiv(M, block_size)
    s_cols = triton.cdiv(N, block_size)
    s = x.new_empty(s_rows, s_cols, dtype=torch.float32)
    grid = lambda meta: (s_rows, s_cols)
    fp8_blockwise_quant_weight_kernel[grid](x, y, s, M, N, BLOCK_SIZE=block_size)
    return y, s

@triton.jit
def fp8_blockwise_dequant_weight_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.
    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.
    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def fp8_blockwise_weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.
    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.
    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.
    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    fp8_blockwise_dequant_weight_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# 全局量化配置 - 基于 Qwen3-Next-80B-A3B-Instruct-FP8 官方模型
DEFAULT_QUANT_CONFIG = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
    "modules_to_not_convert": [
        "lm_head",
        "model.embed_tokens",
        # Layer 0-47 的排除模块（48层）
        *[f"model.layers.{i}.input_layernorm" for i in range(48)],
        *[f"model.layers.{i}.post_attention_layernorm" for i in range(48)],
        *[f"model.layers.{i}.mlp.gate" for i in range(48)],
        *[f"model.layers.{i}.mlp.shared_expert_gate" for i in range(48)],
        # Linear attention 层（层 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44,45,46）
        *[f"model.layers.{i}.linear_attn.A_log" for i in range(48) if i % 4 != 3],
        *[f"model.layers.{i}.linear_attn.conv1d" for i in range(48) if i % 4 != 3],
        *[f"model.layers.{i}.linear_attn.dt_bias" for i in range(48) if i % 4 != 3],
        *[f"model.layers.{i}.linear_attn.in_proj_ba" for i in range(48) if i % 4 != 3],
        *[f"model.layers.{i}.linear_attn.norm" for i in range(48) if i % 4 != 3],
        # Self attention 层（层 3,7,11,15,19,23,27,31,35,39,43,47）
        *[f"model.layers.{i}.self_attn.k_norm" for i in range(3, 48, 4)],
        *[f"model.layers.{i}.self_attn.q_norm" for i in range(3, 48, 4)],
        # MTP (Multi-Token Prediction) 模块
        "mtp.fc",
        "mtp.norm",
        "mtp.pre_fc_norm_embedding",
        "mtp.pre_fc_norm_hidden",
        "mtp.layers.0.input_layernorm",
        "mtp.layers.0.mlp.gate",
        "mtp.layers.0.mlp.shared_expert_gate",
        "mtp.layers.0.post_attention_layernorm",
        "mtp.layers.0.self_attn.k_norm",
        "mtp.layers.0.self_attn.q_norm",
    ]
}

def process_models(model_dir, output_dir, info_path):
    """
    处理模型目录中的所有safetensors文件，对符合条件的权重进行FP8量化
    
    Args:
        model_dir: 包含safetensors文件的目录路径
        output_dir: 量化后模型保存目录
        info_path: 修改记录保存路径
    """
    os.makedirs(output_dir, exist_ok=True)
    modified_log = []
    
    # 使用全局配置
    print("ℹ️  使用内置的默认量化配置")
    excluded_modules = set(DEFAULT_QUANT_CONFIG["modules_to_not_convert"])
    print(f"✓ 加载了 {len(excluded_modules)} 个排除模块")

    for filename in os.listdir(model_dir):
        if not filename.endswith(".safetensors"):
            continue

        filepath = os.path.join(model_dir, filename)
        output_filepath = os.path.join(output_dir, filename)
        print(f"🔍 正在处理文件: {filename}")

        try:
            tensors = load_file(filepath)
            modified = False
            # print(list(tensors.keys()))  # 调试用，已注释
            for key in list(tensors.keys()):
                tensor = tensors[key].to("cuda")
                
                # 检查是否需要量化（使用精确匹配）
                should_quantize = False
                if 'weight' in key:
                    param_name = key.replace('.weight', '')
                    should_quantize = param_name not in excluded_modules
                
                if should_quantize:
                    # 跳过非2D张量
                    if tensor.dim() != 2:
                        print(f"   ⚠️ 跳过非2D张量: {key}")
                        continue
                    
                    # 执行量化
                    try:
                        quantized, scale = fp8_blockwise_weight_quant(tensor)
                        
                        # 更新张量字典
                        tensors[key] = quantized.cpu()
                        scale_key = f"{key}_scale_inv"
                        tensors[scale_key] = scale.cpu()
                        
                        # 记录修改
                        log_entry = f"{filename} | {key} | {tensor.shape}→量化 | scale_shape: {scale.shape}"
                        modified_log.append(log_entry)
                        modified = True
                        print(f"   → 量化成功: {key} | scale形状: {scale.shape}")
                    
                    except Exception as e:
                        print(f"   ❌ 量化失败 {key}: {str(e)}")
                        continue

            save_file(tensors, output_filepath)
            print(f"💾 已保存到: {output_filepath}\n")

        except Exception as e:
            print(f"❌ 处理 {filename} 时出错: {str(e)}\n")
            continue
    
    # 保存修改记录
    with open(info_path, "w") as f:
        f.write("文件 | Tensor名称 | 修改记录\n")
        f.write("\n".join(modified_log))

    print(f"✅ 处理完成！共修改 {len(modified_log)} 个tensor")
    print(f"📝 修改记录已保存至: {info_path}")
    
    # 复制非 safetensors 文件到输出目录
    print("\n🔧 复制其他必要文件到输出目录...")
    copied_files = []
    for filename in os.listdir(model_dir):
        if not filename.endswith(".safetensors"):
            src = os.path.join(model_dir, filename)
            dst = os.path.join(output_dir, filename)
            try:
                if os.path.isdir(src):
                    continue  # 跳过目录
                import shutil
                shutil.copy2(src, dst)
                copied_files.append(filename)
            except Exception as e:
                print(f"    ❌ 复制 {filename} 失败: {e}")
    
    print(f"✓ 已复制 {len(copied_files)} 个文件")
    
    # 将量化配置写入输出目录的config.json
    print("\n📝 写入量化配置...")
    output_config_path = os.path.join(output_dir, "config.json")
    try:
        with open(output_config_path, 'r') as f:
            model_config = json.load(f)
        model_config["quantization_config"] = DEFAULT_QUANT_CONFIG
        with open(output_config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        print(f"✓ 已将量化配置写入 {output_config_path}")
    except Exception as e:
        print(f"⚠️ 写入量化配置失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='权重FP8量化工具')
    parser.add_argument('model_dir', help='模型文件所在目录')
    parser.add_argument('info_path', help='量化记录保存路径')
    parser.add_argument('--output-dir', help='量化后模型保存目录，默认为原目录加_fp8_quant', default=None)
    args = parser.parse_args()
    
    # 处理输出目录逻辑
    if args.output_dir is None:
        args.output_dir = f"{args.model_dir.rstrip('/')}_fp8_quant"
    
    process_models(args.model_dir, args.output_dir, args.info_path)
