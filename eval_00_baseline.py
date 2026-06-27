"""
eval_00_baseline.py – CPU / GPU Baseline Performance Measurement
Records CPU/GPU model names along with latency, throughput, and energy.
Results saved to results/baseline_results.json
"""
import os
import time
import json
import platform
import subprocess
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from utils.Logger import logger

# ==================== 配置 ====================
ROOT_DIR = "./datasets"
FRAME_DIR = os.path.join(ROOT_DIR, "CIFAR10DVS", "frames_number_10_split_by_number")
TIME_STEPS = 10
NUM_CLASSES = 10
WEIGHT_PATH = "results/snn_vgg_best.pth"
BATCH_SIZE = 1
NUM_SAMPLES = 100  # 可调整

# ==================== 数据集 ====================
class CIFAR10DVSDirect(Dataset):
    CLASSES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

    def __init__(self, frame_dir):
        self.frame_dir = frame_dir
        self.class_to_idx = {c:i for i,c in enumerate(self.CLASSES)}
        self.file_paths = []
        self.labels = []
        for cls in self.CLASSES:
            cls_dir = os.path.join(frame_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            files = sorted(glob.glob(os.path.join(cls_dir, "*.npz")))
            for f in files:
                self.file_paths.append(f)
                self.labels.append(self.class_to_idx[cls])
        logger.info(f"Loaded {len(self.file_paths)} frame samples from {frame_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        d = np.load(self.file_paths[idx], allow_pickle=True)
        frame = torch.from_numpy(d['frames']).float()
        return frame, self.labels[idx]

# ==================== 模型 ====================
from models.snn_vgg import SpikingVGG

def load_model(device):
    model = SpikingVGG(num_classes=NUM_CLASSES).to(device)
    if os.path.exists(WEIGHT_PATH):
        state_dict = torch.load(WEIGHT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {WEIGHT_PATH}")
    else:
        logger.warning("No trained weights found, using random initialization.")
    model.eval()
    return model

# ==================== 获取硬件型号 ====================
def get_cpu_model():
    """尝试获取更详细的 CPU 型号，失败则返回 platform.processor()"""
    try:
        # Linux
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    return line.split(':')[1].strip()
    except:
        pass
    try:
        # Windows
        result = subprocess.run(
            ['wmic', 'cpu', 'get', 'name'],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            return lines[1].strip()
    except:
        pass
    # Fallback
    return platform.processor()

def get_gpu_model():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return None

# ==================== 功耗辅助（同前） ====================
def get_gpu_power_w():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')
        if lines and lines[0]:
            return float(lines[0])
    except:
        pass
    return None

def measure_power(device):
    if device.type == 'cuda':
        idle = get_gpu_power_w()
        if idle is None:
            idle = 30.0
            active = 130.0
        else:
            active = idle * 1.5  # 粗略估计
        return idle, active
    else:
        # CPU 典型值（可根据实际调整）
        return 15.0, 65.0

# ==================== 主函数 ====================
def main():
    logger.info("=== Baseline: CPU / GPU Performance Measurement ===")

    # 记录硬件型号
    cpu_model = get_cpu_model()
    gpu_model = get_gpu_model()
    logger.info(f"CPU Model: {cpu_model}")
    if gpu_model:
        logger.info(f"GPU Model: {gpu_model}")
    else:
        logger.info("GPU not available")

    # 数据集
    dataset = CIFAR10DVSDirect(FRAME_DIR)
    indices = list(range(min(NUM_SAMPLES, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    results = {
        "cpu_model": cpu_model,
        "gpu_model": gpu_model,
        "num_samples": len(indices)
    }

    # ---------- CPU ----------
    device_cpu = torch.device('cpu')
    model_cpu = load_model(device_cpu)

    # 预热
    for _ in range(5):
        sample_data, _ = dataset[0]
        sample_data = sample_data.unsqueeze(0).transpose(0, 1).to(device_cpu)
        _ = model_cpu(sample_data)

    torch.set_num_threads(1)  # 单线程模拟典型 CPU 推理
    start = time.perf_counter()
    with torch.no_grad():
        for data, _ in loader:
            data = data.transpose(0, 1).to(device_cpu)
            _ = model_cpu(data)
    elapsed_cpu = time.perf_counter() - start
    latency_cpu_us = (elapsed_cpu / len(indices)) * 1e6
    throughput_cpu = len(indices) / elapsed_cpu
    idle_cpu, active_cpu = measure_power(device_cpu)
    energy_cpu_j = active_cpu * elapsed_cpu / len(indices)

    results['cpu'] = {
        'latency_us': latency_cpu_us,
        'throughput_fps': throughput_cpu,
        'energy_per_image_uj': energy_cpu_j * 1e6,
        'estimated_active_power_w': active_cpu
    }
    logger.info(f"CPU: Latency={latency_cpu_us:.1f} us, Throughput={throughput_cpu:.0f} fps, Energy={energy_cpu_j*1e6:.1f} uJ")

    # ---------- GPU ----------
    if torch.cuda.is_available():
        device_gpu = torch.device('cuda')
        model_gpu = load_model(device_gpu)

        for _ in range(5):
            sample_data, _ = dataset[0]
            sample_data = sample_data.unsqueeze(0).transpose(0, 1).to(device_gpu)
            _ = model_gpu(sample_data)

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        with torch.no_grad():
            for data, _ in loader:
                data = data.transpose(0, 1).to(device_gpu)
                _ = model_gpu(data)
        ender.record()
        torch.cuda.synchronize()
        elapsed_gpu_ms = starter.elapsed_time(ender)
        latency_gpu_us = (elapsed_gpu_ms * 1000) / len(indices)
        throughput_gpu = len(indices) / (elapsed_gpu_ms / 1000)
        idle_gpu, active_gpu = measure_power(device_gpu)
        energy_gpu_j = active_gpu * (elapsed_gpu_ms / 1000) / len(indices)

        results['gpu'] = {
            'latency_us': latency_gpu_us,
            'throughput_fps': throughput_gpu,
            'energy_per_image_uj': energy_gpu_j * 1e6,
            'estimated_active_power_w': active_gpu
        }
        logger.info(f"GPU: Latency={latency_gpu_us:.1f} us, Throughput={throughput_gpu:.0f} fps, Energy={energy_gpu_j*1e6:.1f} uJ")
    else:
        results['gpu'] = None
        logger.warning("CUDA not available, skipping GPU baseline.")

    # 保存
    os.makedirs("results", exist_ok=True)
    output_path = "results/baseline_cifar10dvs.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Baseline results saved to {output_path}")

if __name__ == "__main__":
    main()