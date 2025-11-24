import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Create a simple tensor
x = torch.tensor([1.0, 2.0, 3.0])
print(f"Tensor: {x}")
