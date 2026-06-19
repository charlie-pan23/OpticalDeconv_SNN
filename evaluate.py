import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tonic.transforms as transforms
import numpy as np

from utils.Logger import logger
from utils.config import config
from src.dataset import NpyGestureDataset
from src.model import OpticalDeconvSNN
from train import pad_collate_fn  # 复用我们在 train.py 中写好的时间步对齐函数


def inject_noise_to_weights(model, noise_std):
    """
    模拟光子器件物理噪声：向核心解卷积层的权重注入高斯噪声
    """
    with torch.no_grad():
        # 获取光子解卷积核心的权重
        weights = model.deconv_core.weight
        # 生成高斯噪声 N(0, noise_std)
        noise = torch.randn_like(weights) * noise_std
        # 将噪声叠加到权重上
        model.deconv_core.weight.add_(noise)


def calculate_theoretical_energy():
    """
    利用 config.yaml 中的硬件参数，推算光子芯片的理论能耗
    """
    logger.info("--- Phase 4: Optical Hardware Energy Estimation ---")

    hw_config = config['hardware']
    macs = hw_config['macs']
    num_arrays = hw_config['num_arrays']
    array_size = hw_config['array_size']
    num_adcs = hw_config['num_adcs']

    # 业界光子乘加运算(MAC)的典型能耗通常在 0.1 ~ 1 pJ (皮焦耳) 左右
    # 这里我们假设一个典型的光子计算能效：0.5 pJ / MAC
    energy_per_mac_pj = 0.5 #==========================================

    # 计算理论总能耗
    total_energy_pj = macs * energy_per_mac_pj
    total_energy_uj = total_energy_pj / 1000000  # 转换为微焦耳 (uJ)

    logger.info(f"Hardware Configuration: {num_arrays} Arrays ({array_size}x{array_size}), {num_adcs} ADCs")
    logger.info(f"Total MACs for Deconv Layer: {macs:,}")
    logger.info(f"Assumed Energy per MAC: {energy_per_mac_pj} pJ")
    logger.info(f"Theoretical Energy Consumption per inference: {total_energy_uj:.4f} μJ")
    logger.info("*" * 60)


def evaluate():
    logger.info("=== Starting SNN Evaluation Pipeline ===")

    # ---------------------------------------------------------
    # 1. 配置与加载 (Setup & Loading)
    # ---------------------------------------------------------
    # 全部从 config.yaml 动态读取
    batch_size = config['training']['batch_size']
    time_window_us = config['training']['time_window_us']
    sensor_size = tuple(config['training']['sensor_size'])
    num_classes = config['training']['num_classes']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "optical_deconv_snn.pth")
    # 注意：评估时使用测试集
    data_dir = os.path.join(current_dir, "datasets", "DVSGesture", "ibmGestureTest")

    logger.info("Setting up test dataloader...")
    frame_transform = transforms.ToFrame(sensor_size=sensor_size, time_window=time_window_us)
    test_dataset = NpyGestureDataset(root_dir=data_dir, transform=frame_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        collate_fn=pad_collate_fn
    )

    # 加载模型权重
    logger.info(f"Loading trained weights from {model_path}...")
    model = OpticalDeconvSNN(num_classes=num_classes).to(device)
    if not os.path.exists(model_path):
        logger.error("Model weights not found! Please run train.py first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ---------------------------------------------------------
    # 2. 纯净环境准确率与 GPU 真实延迟 (Clean Baseline & GPU Latency)
    # ---------------------------------------------------------
    logger.info("--- Phase 1 & 2: Clean Baseline & GPU Latency ---")
    correct = 0
    total = 0
    gpu_times = []

    with torch.no_grad():
        # GPU 预热 (避免第一步初始化带来的时间误差)
        logger.info("Warming up GPU...")
        for data, _ in test_loader:
            model(data.to(device))
            break

        logger.info("Running baseline inference...")
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)

            # 使用 CUDA Event 进行高精度计时
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            spk_out, mem_out = model(data)
            end_event.record()

            # 等待 GPU 同步并记录时间
            torch.cuda.synchronize()
            gpu_times.append(start_event.elapsed_time(end_event))

            # 统计准确率 (基于膜电位累加)
            logits = mem_out.sum(dim=0)
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    clean_acc = 100. * correct / total
    avg_gpu_latency = np.mean(gpu_times)
    logger.info("*" * 60)
    logger.info(f"Clean Baseline Accuracy: {clean_acc:.2f}%")
    logger.info(f"Average GPU Latency per Batch ({batch_size} samples): {avg_gpu_latency:.2f} ms")
    logger.info("*" * 60)

    # ---------------------------------------------------------
    # 3. 光子器件物理噪声注入测试 (Noise Robustness)
    # ---------------------------------------------------------
    logger.info("--- Phase 3: Hardware Noise Injection Testing ---")

    # 在内存中保存一份干净的权重字典，用于每次加噪前重置
    clean_state_dict = torch.load(model_path, map_location=device)

    # 测试不同标准差的高斯噪声 (模拟论文图表中的 alpha/sigma 变化)
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5] #==================================

    for std in noise_levels:
        # 每次都将模型重置为纯净状态，防止噪声叠加累积
        model.load_state_dict(clean_state_dict)
        inject_noise_to_weights(model, std)

        noise_correct = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                spk_out, mem_out = model(data)

                logits = mem_out.sum(dim=0)
                _, predicted = logits.max(1)
                noise_correct += predicted.eq(targets).sum().item()

        noise_acc = 100. * noise_correct / total
        logger.info(f"Noise Level (std={std:4.2f}) -> Accuracy dropped to {noise_acc:.2f}%")
    logger.info("*" * 60)

    # ---------------------------------------------------------
    # 4. 光子加速器理论功耗评估 (Theoretical Energy)
    # ---------------------------------------------------------
    calculate_theoretical_energy()

    logger.info("=== Evaluation Pipeline Completed Successfully ===")


if __name__ == "__main__":
    evaluate()