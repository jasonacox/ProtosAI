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

    # Get index of currently selected device
    deviceno = torch.cuda.current_device() 
    name = torch.cuda.get_device_name(deviceno)

    # Returns the global free and total GPU memory occupied for a given device using cudaMemGetInfo.
    (mem_free,gpu_mem) = torch.cuda.mem_get_info()

    mem_free = mem_free / 1024**3
    gpu_mem = gpu_mem / 1024**3

    print(f"   Device #{deviceno}: {name}")
    print(f"   Type: cuda {sim}")
    print(f"   GPUs: {gpus}")
    print()
    print("Memory")
    print(f"   Global Free Memory: {mem_free} GB")
    print(f"   GPU Memory: {gpu_mem} GB") 
    print('   Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('   Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    torch_device = torch.device("cuda")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
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
