import torch
import time
import argparse

def test_gpu_fp16_tflops(matrix_size=4096, num_iterations=100, warmup=10):
    """
    测试 GPU 的 FP16 算力 (TFLOPS)
    
    通过执行多个大矩阵乘法，计算达到的 TFLOPS 性能。
    
    参数:
        matrix_size (int): 矩阵的大小 (N x N)
        num_iterations (int): 测试循环次数
        warmup (int): 预热循环次数
    """
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("CUDA 不可用，请检查你的环境配置。")
        return None
    
    device = torch.device("cuda")
    
    # 创建两个 FP16 随机矩阵
    matrix_a = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
    matrix_b = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
    
    # 预热操作，让 GPU 达到稳定状态
    print(f"预热中... ({warmup} 次)")
    for _ in range(warmup):
        _ = torch.matmul(matrix_a, matrix_b)
    
    # 同步 CUDA 设备，确保预热完成
    torch.cuda.synchronize()
    
    # 开始计时
    print(f"开始测试 ({num_iterations} 次迭代，矩阵大小 {matrix_size}x{matrix_size})...")
    start_time = time.time()
    
    for _ in range(num_iterations):
        _ = torch.matmul(matrix_a, matrix_b)
    
    # 同步确保所有计算完成
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 计算性能指标
    elapsed_time = end_time - start_time
    # 每次矩阵乘法的浮点运算次数约为 2 * N^3 (乘加)
    flops_per_iteration = 2 * matrix_size ** 3
    total_flops = flops_per_iteration * num_iterations
    tflops = total_flops / (elapsed_time * 1e12)
    
    print(f"\n=== 测试结果 ===")
    print(f"GPU 设备: {torch.cuda.get_device_name(device)}")
    print(f"总耗时: {elapsed_time:.4f} 秒")
    print(f"FP16 混合算力: {tflops:.2f} TFLOPS")
    
    return tflops

def main():
    parser = argparse.ArgumentParser(description="测试 GPU FP16 算力")
    parser.add_argument("--size", type=int, default=4096, help="矩阵大小 (N x N)")
    parser.add_argument("--iter", type=int, default=100, help="测试迭代次数")
    parser.add_argument("--warmup", type=int, default=10, help="预热迭代次数")
    args = parser.parse_args()
    
    test_gpu_fp16_tflops(args.size, args.iter, args.warmup)

if __name__ == "__main__":
    main()