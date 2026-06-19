import sys
import platform
import importlib


# Terminal color codes
class Colors:
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_status(name, version, is_ok, msg=""):
    status = f"{Colors.OKGREEN}[OK]{Colors.ENDC}" if is_ok else f"{Colors.FAIL}[FAIL]{Colors.ENDC}"
    ver_str = f"v{version}" if version else "N/A"
    print(f"{status} {name.ljust(15)} | {ver_str.ljust(10)} | {msg}")


def check_pkg(pkg_name, display_name=None):
    # Use display_name if provided, otherwise default to pkg_name
    display_name = display_name or pkg_name
    try:
        module = importlib.import_module(pkg_name)
        version = getattr(module, '__version__', 'Unknown')
        print_status(display_name, version, True)
    except ImportError:
        print_status(display_name, None, False, "Not installed")


print(f"{Colors.BOLD}=== SNN Photonics Environment Check ==={Colors.ENDC}\n")

# 1. System Info
print(f"{Colors.BOLD}[1. System Info]{Colors.ENDC}")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version.split()[0]}\n")

# 2. Dependencies
print(f"{Colors.BOLD}[2. Dependencies]{Colors.ENDC}")
packages = [
    'torch', 'torchvision', 'torchaudio', 'snntorch', 'tonic',
    'numpy', 'scipy', 'pandas', 'h5py', 'sklearn', 'skimage',
    'matplotlib', 'seaborn', 'tqdm', 'psutil', 'ptflops', 'thop'
]

# Check standard packages
for pkg in packages:
    check_pkg(pkg)

# Special handling for nvidia-ml-py (installed as nvidia-ml-py, but imported as pynvml)
check_pkg('pynvml', display_name='nvidia-ml-py')
print()

# 3. Hardware & CUDA Check
print(f"{Colors.BOLD}[3. Hardware & CUDA Check]{Colors.ENDC}")
try:
    import torch

    cuda_avail = torch.cuda.is_available()
    print_status("CUDA Available", torch.version.cuda if cuda_avail else None, cuda_avail)

    if cuda_avail:
        print_status("GPU Found", str(torch.cuda.device_count()), True, f"Using: {torch.cuda.get_device_name(0)}")
        try:
            _ = torch.zeros(1).cuda()
            print_status("VRAM Test", None, True, "Allocation successful")
        except Exception as e:
            print_status("VRAM Test", None, False, f"Failed: {e}")
except ImportError:
    print_status("CUDA Check", None, False, "PyTorch required")

# NVML Check (For power metrics)
try:
    import pynvml

    pynvml.nvmlInit()
    print_status("NVML Interface", pynvml.nvmlSystemGetDriverVersion(), True, "Ready for power profiling")
    pynvml.nvmlShutdown()
except Exception as e:
    print_status("NVML Interface", None, False, f"Warning: {e}")

print(f"\n{Colors.BOLD}=== Check Complete ==={Colors.ENDC}")