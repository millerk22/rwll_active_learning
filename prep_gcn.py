import torch
import subprocess
import sys


def format_pytorch_version(version):
  return version.split('+')[0]

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-f", f"https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html"])

def install_basic(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("torch-scatter")
install("torch-sparse")
install("torch-cluster")
install("torch-spline-conv")
install_basic("torch-geometric")
