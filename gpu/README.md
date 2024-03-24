# GPU support for PyTorch

Setting up GPU support for AI workloads can be non-trivial. If you happen to have an Nvidia GPU, chances are you have GPU support already installed via CUDA. If you have an AMD GPU, you are going to need to do some gymanastic to get to work. 

I'm going to log some of my adventure here. Hopefully this is helpful for anyone else going on this journey.

## GPU Information

The include `gpy.py` script will attempt to pull GPU information via PyTorch on your system:

```
GPU Information for PyTorch
   Version of torch:  2.0.1

GPU Details
   ** No GPU support found **
```

## Getting Started

Test the `gpu.py` script to see if your PyTorch installation already supports your GPU. If not, the steps below may help.

## PyTorch Support for GPUs

See the PyTorch friendly setup matrix here: https://pytorch.org/get-started/locally/

* Nvidia GPUs use the CUDA (Compute Unified Device Architecture) library
* AMD GPUs use the ROCm (Radeon Open Compute) library
* Apple Silicon M1/M2 GPUs use the Metal framework and PyTorch uses Apple’s Metal Performance Shaders (MPS) as a backend

### Nvidia - CUDA

To install support for CUDA, follow these steps:

1. Use the Nvidia download tool to get the right setup for your GPU: https://developer.nvidia.com/cuda-downloads
2. Reboot
3. The post-installation steps are important, specifically setting up `PATH`: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions
4. Test

```bash
# Ensure you have these working
nvidia-smi
nvcc -V

# Install python libraries 
pip install torch torchvision torchaudio

# Test
python3 gpu.py
```

### AMD - ROCm

If you have an AMD GPU, you will need to install and configure PyTorch to use the ROCm API for these AMD GPUs.

1. AMD ROCm Installation: https://rocmdocs.amd.com/en/latest/deploy/linux/quick_start.html

Example for Ubuntu 22.04
```bash
# Set up repository
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
sudo tee /etc/apt/sources.list.d/amdgpu.list <<'EOF'
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/latest/ubuntu jammy main
EOF
sudo tee /etc/apt/sources.list.d/rocm.list <<'EOF'
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/debian jammy main
EOF
sudo echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | sudo tee /etc/apt/preferences.d/rocm-pin-600

# Drivers
sudo apt update
sudo apt install amdgpu-dkms
sudo apt install rocm-hip-libraries
sudo reboot
```

2. Using PyTorch instruction ([here](https://pytorch.org/get-started/locally/)) - Install torch, torchvision and torchaudio

Example for Ubuntu 22.04
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

3. Test - run the `gpu.py` script. Note that some torch commands seem to segfault on the ROCm version.  You can set an environmental variable to fix that. The `gpu.py` script set this in the python code.

```bash
# Use this export if you get a segfault running Torch functions
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

```
GPU Information for PyTorch
   Version of torch:  2.0.1+rocm5.4.2

GPU Details
   Device #0: AMD Radeon RX 5700 XT
   Type: cuda (ROCm)
   GPUs: 1

Memory
   Global Free Memory: 7.984375 GB
   GPU Memory: 7.984375 GB
   Allocated: 0.0 GB
   Cached:    0.0 GB

PyTorch Test with cuda - Random 4x4 Array

tensor([[1561, 2237, 6565, 4058],
        [5793, 8552, 6102, 1498],
        [ 167, 7648,  333, 4254],
        [4603, 9456, 6009, 6043]], device='cuda:0')
```

### Apple MPS (Metal Performance Shader)

If you have an Apple Silicon processor (e.g. M1), PyTorch can use the built in MPS (Metal Performance Shader, Apple's GPU architecture).  PyTorch supports this with the latest builds.

1. Install PyTorch
```bash
pip3 install torch torchvision torchaudio
```

2. Test - run the `gpu.py` script.

Example
```
GPU Information for PyTorch
   Version of torch:  2.0.1

GPU Details
   Device: Apple Silicon Found
   MPS (Metal Performance Shader) built: True
   MPS available: True

PyTorch Test with mps - Random 4x4 Array

tensor([[1647,  740, 6492, 2094],
        [1884, 4286, 2299, 2086],
        [5135, 5838, 1956, 6912],
        [8681, 5118, 6668,  925]], device='mps:0')
```
