import platform
import sys
import torch
import torchvision


def main():
    print("Python:", sys.version)
    print("Executable:", sys.executable)
    print("Platform:", platform.platform())
    print("PyTorch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("PyTorch CUDA runtime:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        x = torch.randn(16, 16, device="cuda", requires_grad=True)
        x.square().mean().backward()
        print("GPU forward/backward: OK")
    else:
        print("CPU 可做流程检查；正式实验建议切换到 NVIDIA GPU 环境。")


if __name__ == "__main__":
    main()
