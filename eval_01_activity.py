import os
import yaml
import json
import torch
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


class ActivityTracker:
    def __init__(self, model):
        self.model = model
        self.total_spikes = 0
        self.total_elements = 0
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # 遍历模型中的所有模块，找到 LIF 神经元层挂载监听器
        for name, module in self.model.named_modules():
            if 'Leaky' in str(type(module)):
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)
                logger.debug(f"Registered activity hook on: {name}")

    def _hook_fn(self, module, input, output):
        # snn.Leaky 的输出通常是 (spk, mem) 的元组
        spk = output[0]
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
    logger.info("=== Phase 1: SNN Activity & Sparsity Extraction ===")

    # 1. 加载配置
    config_path = "configs/config_cifar10dvs.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ========================================================
    # 2. 准备数据集 (严格切片与动态加载)
    # ========================================================
    sensor_size = tuple(config['dataset']['sensor_size'])
    time_steps = config['dataset']['time_steps']
    dataset_name = config['dataset']['name']
    root_dir = config['dataset']['root_dir']

    # 【关键修改】使用 n_time_bins 强制将事件流切分为严格的 T 帧
    transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_steps)
    logger.info(f"Loading dataset: {dataset_name} from {root_dir}...")

    try:
        if dataset_name == "cifar10dvs":
            dataset = tonic.datasets.CIFAR10DVS(save_to=root_dir, transform=transform)
        elif dataset_name == "dvsgesture":
            # 评估时严格使用测试集
            dataset = tonic.datasets.DVSGesture(save_to=root_dir, train=False, transform=transform)
        else:
            raise ValueError(f"Unknown dataset defined in config: {dataset_name}")

        # 硬件评估与活性采集严格要求 batch_size = 1，并打乱数据集进行随机抽样
        dataloader = DataLoader(
            dataset,
            batch_size=config['dataset']['batch_size'],
            shuffle=True,
            drop_last=True,
            collate_fn=tonic.collation.PadTensors(batch_first=False)
        )
        logger.info(f"Dataset successfully loaded! Total samples available: {len(dataset)}")

    except Exception as e:
        logger.error(f"Dataset load failed! Please check if files exist in {root_dir}. Error: {e}")
        return
        # ========================================================

    # 3. 实例化模型
    num_classes = config['dataset']['num_classes']
    network_type = config['network']['type']
    if network_type == "snn_vgg":
        model = SpikingVGG(num_classes=num_classes).to(device)
    else:
        model = SpikingCNN(num_classes=num_classes).to(device)

    # --- 适配修改 1：加载训练好的网络权重 ---
    weight_path = f"results/{network_type}_best.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        logger.info(f"Loaded trained weights from {weight_path}")
    else:
        logger.warning(f"No trained weights found at {weight_path}! Evaluating with RANDOM initialization.")
        logger.warning("WARNING: Random initialized model will yield inaccurate Active Ratios!")

    model.eval()

    # 4. 挂载活性追踪器
    tracker = ActivityTracker(model)

    # 5. 运行推理并统计稀疏度
    logger.info("Running inference to extract active ratio...")
    num_batches_to_test = 50  # 抽样 50 个 batch 计算统计学平均即可，无需全跑

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(dataloader, total=num_batches_to_test)):
            if i >= num_batches_to_test: break
            data = data.to(device)
            _ = model(data)

    active_ratio = tracker.get_active_ratio()
    logger.info("*" * 50)
    logger.info(f"Total Evaluated Elements: {tracker.total_elements:,}")
    logger.info(f"Total Active Spikes (1s): {tracker.total_spikes:,}")
    logger.info(f"Extracted Average Active Ratio: {active_ratio * 100:.2f}%")
    logger.info("*" * 50)

    # 6. 保存结果供 eval_02 功耗推导使用
    os.makedirs("results", exist_ok=True)
    stats = {
        "dataset": dataset_name,
        "network": network_type,
        "time_steps": time_steps,
        "active_ratio": active_ratio,
        # 这里为了主线闭环，暂用论文推导的 ADC 触发率 38%
        "adc_activity": 0.38
    }
    with open("results/activity_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    logger.info("Activity statistics saved to results/activity_stats.json")


if __name__ == "__main__":
    main()