import torch

# 1. 检查 CUDA 是否可用
print("CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))

# 2. 简单的张量运算
a = torch.rand(3, 3)
b = torch.rand(3, 3)
c = torch.matmul(a, b)

print("矩阵 A:\n", a)
print("矩阵 B:\n", b)
print("结果 C = A x B:\n", c)

# 3. 如果有 GPU，把张量放到 GPU 上再算一次
if torch.cuda.is_available():
    a_gpu = a.cuda()
    b_gpu = b.cuda()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    print("在 GPU 上完成矩阵乘法，结果尺寸:", c_gpu.size())
