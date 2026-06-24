import os
import copy
import torch
import yaml
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
from tqdm import tqdm

from utils.Logger import logger
from models.snn_vgg import SpikingVGG
from models.snn_cnn import SpikingCNN


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def inject_photonic_noise(model, noise_std=0.1):
    """
    模拟光子器件物理噪声 (WDM Crosstalk & Analog Partial Sum Noise)
    方法：在映射到光学 MVM 阵列的核心卷积层权重上注入高斯噪声
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            # 针对 VGG 的 conv3/conv4 或者 CNN 的 deconv_core 注入噪声
            if 'conv3' in name or 'conv4' in name or 'deconv_core' in name:
                weights = module.weight
                noise = torch.randn_like(weights) * noise_std
                module.weight.add_(noise)
                logger.debug(f"Injected Gaussian noise (σ={noise_std}) to {name}")


def main():
    logger.info("=== Phase 3: Hardware Robustness & Sensitivity Sweep ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = "configs/config_cifar10dvs.yaml"
    config = load_config(config_path)

    # ========================================================
    # 1. 加载测试集 (严格保证与训练时切片维度一致)
    # ========================================================
    sensor_size = tuple(config['dataset']['sensor_size'])
    time_steps = config['dataset']['time_steps']
    dataset_name = config['dataset']['name']
    root_dir = config['dataset']['root_dir']

    transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_steps)
    logger.info(f"Loading {dataset_name} test set for robustness evaluation...")

    if dataset_name == "cifar10dvs":
        # CIFAR10-DVS 官方未分 train/test，实操中可以用同一数据集抽样跑鲁棒性
        dataset = tonic.datasets.CIFAR10DVS(save_to=root_dir, transform=transform)
    elif dataset_name == "dvsgesture":
        dataset = tonic.datasets.DVSGesture(save_to=root_dir, train=False, transform=transform)
    else:
        raise ValueError("Unknown dataset.")

    # 鲁棒性评估时可以适当调大 batch_size 加速推理
    test_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        drop_last=True,
        collate_fn=tonic.collation.PadTensors(batch_first=False)
    )

    # ========================================================
    # 2. 实例化模型并加载最佳权重
    # ========================================================
    num_classes = config['dataset']['num_classes']
    network_type = config['network']['type']

    if network_type == "snn_vgg":
        model = SpikingVGG(num_classes=num_classes).to(device)
    else:
        model = SpikingCNN(num_classes=num_classes).to(device)

    weight_path = f"results/{network_type}_best.pth"
    if not os.path.exists(weight_path):
        logger.error(f"Cannot find trained weights at {weight_path}! Please run train.py first.")
        return

    # 加载纯净的 baseline 权重
    clean_state_dict = torch.load(weight_path, map_location=device)
    logger.info(f"Successfully loaded clean weights from {weight_path}")

    # ========================================================
    # 3. 开始噪声消融扫描 (Noise Sensitivity Sweep)
    # ========================================================
    # 噪声标准差 levels，对应论文 Figure 5 中的横坐标
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

    for std in noise_levels:
        # 每次测算前，必须将模型重置为纯净状态，防止噪声污染累加
        model.load_state_dict(copy.deepcopy(clean_state_dict))
        model.eval()

        if std > 0.0:
            inject_photonic_noise(model, noise_std=std)

        correct = 0
        total = 0

        # 评估循环
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc=f"Testing Noise σ={std:.2f}"):
                data, targets = data.to(device), targets.to(device)

                spk_out, mem_out = model(data)
                logits = mem_out.sum(dim=0)  # 使用全时间步膜电位总和进行分类判断
                _, predicted = logits.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        logger.info(f"[*] Photonic Noise Level (σ = {std:.2f}) -> Sustained Accuracy: {acc:.2f}%")

    logger.info("=== Robustness Sweep Completed ===")


if __name__ == "__main__":
    main()