#!/usr/bin/python3
"""
GPU Information for PyTorch

Author: Jason A. Cox
4 June 2023
https://github.com/jasonacox/ProtosAI/

This is a simple GPU test for PyTorch workloads.

"""
# import
print("GPU Information for PyTorch")
print("   Version of torch: ",end="")
import torch
import os

# AMD ROCm requires environmental override to prevent segfault
sim = ""
if "rocm" in torch.__version__:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    sim = "(ROCm)"

print(f" {torch.__version__}")
print()
print("GPU Details")

if torch.cuda.is_available():
    # Get number of GPUs available
    gpus = torch.cuda.device_count()

    num_gpus = torch.cuda.device_count()
    print(f"   Number of GPUs available: {num_gpus}")
    print(f"   Type: cuda {sim}")
    
    # List each GPU's name, free memory, and total memory
    for i in range(num_gpus):
        mem_free, mem_total = 0,0
        gpu_name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        cuda_device_name = f"cuda:{i}"
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mem_free_mb = mem_free / 1024**3
        mem_total_mb = mem_total / 1024**3
        pct = (mem_free / mem_total) * 100
        print()
        print(f"   GPU {i}: {gpu_name} - {cuda_device_name}")
        print(f"      Compute Capability: {capability[0]}.{capability[1]}")
        print(f"      Memory: {mem_total_mb:.2f} GB")
        print(f"         Used: {mem_total_mb-mem_free_mb:.2f} GB ({100-int(pct)}%)")
        print(f"         Free: {mem_free_mb:.2f} GB ({int(pct)}%)")
        cuda_device_name = f"cuda:{i}"
    torch_device = torch.device(cuda_device_name)

# Check PyTorch has access to Apple MPS (Metal Performance Shader)
if torch.backends.mps.is_available():
    print("   Device: Apple Silicon Found")
    print(f"   MPS (Metal Performance Shader) built: {torch.backends.mps.is_built()}")
    print(f"   MPS available: {torch.backends.mps.is_available()}")
    torch_device = torch.device("mps")

# No GPUs Available
if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
    print("   ** No GPU support found **")
    print("   Device: CPU")
    torch_device = torch.device("cpu")

# Run a simple PyTorch test
print()
print(f"PyTorch Test with {torch_device} - Random 4x4 Array\n")
random_array = torch.randint(low=0, high=10000, size=(4, 4), device=torch_device)
print(random_array)
