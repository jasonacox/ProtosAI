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

### PyTorch Support for GPUs

See the PyTorch friendly setup matrix here: https://pytorch.org/get-started/locally/

* Nvidia GPUs use the CUDA (Compute Unified Device Architecture) library
* AMD GPUs use the ROCm (Radeon Open Compute) library
* Apple Silicon M1/M2 GPUs use the Metal framework and PyTorch uses Apple’s Metal Performance Shaders (MPS) as a backend

### PyTorch Support for AMD GPUs

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

3. Test - run the `gpu.py` script.

```
GPU Information for PyTorch
   Version of torch:  2.0.1+rocm5.4.2

GPU Details
   Device #0: AMD Radeon RX 5700 XT
   Type: cuda
   GPUs: 1

Memory
   Global Free Memory: 7.984375 GB
   GPU Memory: 7.984375 GB
   Allocated: 0.0 GB
   Cached:    0.0 GB
```
