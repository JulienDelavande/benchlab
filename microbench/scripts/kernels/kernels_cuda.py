# /// script
# dependencies = [
#  "numpy",
#  "torch",
#  "kernels",
# ]
# ///

import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from kernels import get_kernel

DEVICE = "cuda"

# Make reproducible
torch.manual_seed(42)

# Download optimized activation kernels from the Hub
activation_kernels = get_kernel("kernels-community/activation")
liger_kernels = get_kernel("kernels-community/liger_kernels")

# Create a random tensor on the GPU
x = torch.randn((4, 4), dtype=torch.float16, device=DEVICE)

# Prepare an output tensor
y = torch.empty_like(x)

# list all available functions in the activation kernel module 
print("Available functions in 'kernels-community/activation':")
print(dir(activation_kernels))

# Run RMSNorm kernel
X = torch.randn(1, 1, 4096, dtype=torch.float16, device="cuda")
W = torch.ones(4096, dtype=torch.float16, device="cuda")  # Poids appris
eps = 1e-6  # Même epsilon que LLaMA

# Warmup the kernel
nvtx.range_push(f"Warmup_RMSNorm_kernel")
for _ in range(10):
    liger_kernels.rms_norm.LigerRMSNormFunction.apply(X, W, eps)
torch.cuda.synchronize()  # Ensure all operations are complete before timing
nvtx.range_pop()

torch.cuda.synchronize()  # Ensure all operations are complete before timing
nvtx.range_push(f"RMSNorm_kernel")
rmsnorm_output = liger_kernels.rms_norm.LigerRMSNormFunction.apply(X, W, eps)
torch.cuda.synchronize()  # Ensure all operations are complete before timing
nvtx.range_pop()

# Check if the RMSNorm output is close to PyTorch's built-in RMSNorm
normalized_shape = (X.shape[-1],)
# Warmup the PyTorch RMSNorm
nvtx.range_push(f"Warmup_PyTorch_RMSNorm")
for _ in range(10):
    F.rms_norm(X, normalized_shape, W, eps)
torch.cuda.synchronize()  # Ensure all operations are complete before timing
nvtx.range_pop()

torch.cuda.synchronize()  # Ensure all operations are complete before timing
nvtx.range_push(f"PyTorch_RMSNorm")
expected_rmsnorm = F.rms_norm(X, normalized_shape, W, eps)
torch.cuda.synchronize()  # Ensure all operations are complete before timing
nvtx.range_pop()
torch.testing.assert_close(rmsnorm_output, expected_rmsnorm, rtol=1e-2, atol=1e-2)
print("✅ RMSNorm kernel output matches PyTorch RMSNorm!")

# Run the fast GELU kernel
nvtx.range_push(f"Warmup_Fast_GELU")
for _ in range(10):
    activation_kernels.gelu_fast(y, x)
torch.cuda.synchronize()  # Ensure all operations are complete before timing
nvtx.range_pop()

torch.cuda.synchronize()  # Ensure all operations are complete before timing
nvtx.range_push(f"Fast_GELU")
activation_kernels.gelu_fast(y, x)
torch.cuda.synchronize()  # Ensure all operations are complete before timing
nvtx.range_pop()

# Get expected output using PyTorch's built-in GELU
nvtx.range_push(f"Warmup_PyTorch_GELU")
for _ in range(10):
    F.gelu(x)
torch.cuda.synchronize()  # Ensure all operations are complete before timing
nvtx.range_pop()

torch.cuda.synchronize()  # Ensure all operations are complete before timing
nvtx.range_push(f"PyTorch_GELU")
expected = F.gelu(x)
nvtx.range_pop()

# Compare the kernel output with PyTorch's result
torch.testing.assert_close(y, expected, rtol=1e-2, atol=1e-2)

print("✅ Kernel output matches PyTorch GELU!")

# Optional: print both tensors for inspection
print("\nInput tensor:")
print(x)
print("\nFast GELU kernel output:")
print(y)
print("\nPyTorch GELU output:")
print(expected)

# List available functions in the loaded kernel module
print("\nAvailable functions in 'kernels-community/activation':")
print(dir(activation_kernels))
