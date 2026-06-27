import os
import yaml
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.Logger import logger

import numpy as np
import glob
from torch.utils.data import Dataset

# 你的模型（SpikingJelly 版本）
from models.snn_vgg import SpikingVGG


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class ActivityTracker:
    """
    挂载到 LIFNode 层，统计所有时间步的脉冲总数和总元素数。
    """
    def __init__(self, model):
        self.model = model
        self.total_spikes = 0
        self.total_elements = 0
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # SpikingJelly 的 LIFNode 类名是 'LIFNode'
        for name, module in self.model.named_modules():
            if 'LIFNode' in type(module).__name__:
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)
                logger.debug(f"Registered activity hook on: {name}")

    def _hook_fn(self, module, input, output):
        # SpikingJelly 的 LIFNode 输出是脉冲张量 (spike)
        # 注意：output 就是脉冲，不是元组
        spk = output
        self.total_spikes += torch.count_nonzero(spk).item()
        self.total_elements += spk.numel()

    def get_active_ratio(self):
        if self.total_elements == 0:
            return 0.0
        return self.total_spikes / self.total_elements

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def main():
    logger.info("=== Phase 1: SNN Activity & Sparsity Extraction (Direct Frame Loading) ===")

    # 1. 加载配置
    config_path = "configs/config_cifar10dvs.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ========================================================
    # 2. 准备数据集（直接读取本地帧文件，不依赖 SpikingJelly）
    # ========================================================
    time_steps = config['dataset']['time_steps']
    root_dir = config['dataset']['root_dir']          # e.g., "./datasets"
    dataset_name = config['dataset']['name']          # "cifar10dvs"

    # 读取 evaluation 参数
    eval_cfg = config.get('evaluation', {})
    num_samples = eval_cfg.get('num_samples', 1000)
    adc_activity = eval_cfg.get('adc_trigger_duty_cycle', 0.38)

    logger.info(f"Evaluation config: num_samples={num_samples}, adc_activity={adc_activity}")

    # 帧文件目录：./datasets/CIFAR10DVS/frames_number_10_split_by_number/
    frame_dir = os.path.join(root_dir, "CIFAR10DVS", f"frames_number_{time_steps}_split_by_number")
    if not os.path.isdir(frame_dir):
        logger.error(f"Frame directory not found: {frame_dir}")
        logger.error("Please run training script first to generate frames.")
        return

    # 自定义数据集类（直接读取 .npz 文件）
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
            print("Available keys:", d.files)  # Debug
            frame = torch.from_numpy(d['frames']).float()  # (T, 2, 128, 128)
            return frame, self.labels[idx]

    # 构建数据集和 DataLoader
    dataset = CIFAR10DVSDirect(frame_dir)
    total_available = len(dataset)
    logger.info(f"Total available samples: {total_available}")

    # 根据 num_samples 确定实际要评估的样本数
    if num_samples <= 0 or num_samples > total_available:
        num_batches_to_test = total_available
        logger.info(f"Using all {total_available} samples for evaluation.")
    else:
        num_batches_to_test = num_samples
        logger.info(f"Using {num_samples} samples for evaluation (out of {total_available}).")

    # 注意：DataLoader 的 batch_size 设为 1，shuffle 随机抽样
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
    )

    # ========================================================
    # 3. 实例化模型并加载权重
    # ========================================================
    num_classes = config['dataset']['num_classes']
    network_type = config['network']['type']   # "snn_vgg"

    if network_type == "snn_vgg":
        model = SpikingVGG(num_classes=num_classes).to(device)
    else:
        logger.error(f"Unsupported network type: {network_type}")
        return

    weight_path = f"results/{network_type}_best.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        logger.info(f"Loaded trained weights from {weight_path}")
    else:
        logger.warning(f"No trained weights found at {weight_path}! Using random init.")
        logger.warning("WARNING: Random initialized model will yield inaccurate Active Ratios!")

    model.eval()

    # 4. 挂载活性追踪器
    tracker = ActivityTracker(model)

    # 5. 运行推理并统计稀疏度
    logger.info("Running inference to extract active ratio...")
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(dataloader, total=num_batches_to_test)):
            if i >= num_batches_to_test:
                break
            # data shape: (1, T, 2, 128, 128) 需要转置为 (T, 1, 2, 128, 128)
            data = data.transpose(0, 1).to(device)   # (T, B, C, H, W)
            _ = model(data)   # 前向传播，钩子自动记录

    active_ratio = tracker.get_active_ratio()
    logger.info("*" * 50)
    logger.info(f"Total Evaluated Elements: {tracker.total_elements:,}")
    logger.info(f"Total Active Spikes (1s): {tracker.total_spikes:,}")
    logger.info(f"Extracted Average Active Ratio: {active_ratio * 100:.2f}%")
    logger.info("*" * 50)

    # 6. 保存结果
    os.makedirs("results", exist_ok=True)
    stats = {
        "dataset": dataset_name,
        "network": network_type,
        "time_steps": time_steps,
        "active_ratio": active_ratio,
        "adc_activity": adc_activity,
        "num_samples_used": num_batches_to_test
    }
    with open("results/activity_cifar10dvs.json", "w") as f:
        json.dump(stats, f, indent=4)
    logger.info("Activity statistics saved to results/activity_cifar10dvs.json")

if __name__ == "__main__":
    main()