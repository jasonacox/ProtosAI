# Jupyter Lab

The following docker run witll set up a Jupyter Lab notebook server on your own local GPU server:

```bash
docker run -d  \
        --shm-size=10.24gb \
        --gpus all \
        -p 8888:8888 \
        -e JUPYTER_ENABLE_LAB=yes \
        -v "${PWD}":/home/jovyan/work \
        --name jupyter \
        quay.io/jupyter/datascience-notebook:2024-01-15 start-notebook.sh --NotebookApp.token='' --notebook-dir=/home/jovyan/work
```

This will:
* Use all GPUs available in the Nvidia container.
* Listen on http://localhost:8888/lab
* Store and persist all notebooks in the local directory.

Example notebook run:

```ipynb
!pip install torch
```

```python
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

    # Returns the global free and total GPU memory
    (mem_free,gpu_mem) = torch.cuda.mem_get_info()

    mem_free = mem_free / 1024**3
    gpu_mem = gpu_mem / 1024**3
    pct = (mem_free / gpu_mem) * 100

    print(f"   Device #{deviceno}: {name}")
    print(f"   Type: cuda {sim}")
    print(f"   GPUs: {gpus}")
    print()
    print("Memory")
    print(f"   GPU Total Memory: {gpu_mem} GB")
    print(f"   GPU Free Memory: {mem_free} GB ({int(pct)}%)")
    print('   Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('   Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    torch_device = torch.device("cuda")

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
```

```ipynb
GPU Information for PyTorch
   Version of torch:  2.2.0+cu121

GPU Details
   Device #0: NVIDIA GeForce RTX 3090
   Type: cuda 
   GPUs: 1

Memory
   GPU Total Memory: 23.69110107421875 GB
   GPU Free Memory: 1.71685791015625 GB (7%)
   Allocated: 0.0 GB
   Cached:    0.0 GB

PyTorch Test with cuda - Random 4x4 Array

tensor([[5820, 7863, 2464, 2813],
        [4027, 1586, 5967, 2303],
        [2072, 5165, 3202, 7322],
        [6688, 7199, 1020, 5595]], device='cuda:0')
```