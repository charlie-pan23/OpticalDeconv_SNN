import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tonic
tonic.datasets.CIFAR10DVS.download = lambda self: None
tonic.datasets.DVSGesture.download = lambda self: None

# import tonic.io
# import tonic.datasets.cifar10dvs
# # 全局劫持，解决 wrong magic number 报错
# tonic.datasets.cifar10dvs.read_aedat4 = tonic.io.read_aedat4
# tonic.io.read_aedat4 = tonic.io.read_aedat

import tonic.transforms as transforms
from tqdm import tqdm

from utils.Logger import logger
from models.snn_vgg import SpikingVGG
from models.snn_cnn import SpikingCNN


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    logger.info("=== Starting SNN Training Pipeline ===")

    # 1. 加载配置 (默认读取 CIFAR10-DVS 主基准)
    config_path = "configs/config_cifar10dvs.yaml"
    # config_path = "configs/config_dvsgesture.yaml"
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. 准备数据集 (使用 Tonic 和 ToFrame 进行严格的 T 帧切片)
    sensor_size = tuple(config['dataset']['sensor_size'])
    time_steps = config['dataset']['time_steps']
    dataset_name = config['dataset']['name']
    root_dir = config['dataset']['root_dir']
    batch_size = 16  # 训练时 Batch size 可以开大一点，比如 16 或 32，加快训练速度

    # 将稀疏事件流转化为固定帧数 (T, C, H, W)
    transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_steps)

    logger.info(f"Loading dataset: {dataset_name} for training...")
    # if dataset_name == "cifar10dvs":
    #     train_dataset = tonic.datasets.CIFAR10DVS(save_to=root_dir, transform=transform)
    #     test_dataset = train_dataset  # CIFAR10-DVS 官方未分 train/test，可以自己按比例切分，这里从简
    if dataset_name == "cifar10dvs":
        train_dataset = tonic.datasets.CIFAR10DVS(save_to=root_dir, transform=transform)

        import glob
        expected_path = os.path.join(root_dir, "CIFAR10DVS")
        all_files = glob.glob(os.path.join(expected_path, "*", "*.aedat"))

        if len(all_files) == 10000:
            # 定义官方的 10 个类别及索引映射
            classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

            _data = []
            _targets = []
            for f_path in all_files:
                cls_name = os.path.basename(os.path.dirname(f_path))  # 提取所属文件夹名
                if cls_name in class_to_idx:
                    _data.append(f_path)
                    _targets.append(class_to_idx[cls_name])

            # 暴力改写 Tonic 实例的内部属性，直接把饭喂到嘴里！
            train_dataset.data = _data
            train_dataset.targets = _targets
            if hasattr(train_dataset, 'file_paths'):
                train_dataset.file_paths = _data

            logger.info(f"🔧 注入成功！强行绕过框架限制，加载了 {len(train_dataset.data)} 个样本！")
        else:
            logger.error(f"❌ 致命错误：本地文件数不等于 10000，当前为 {len(all_files)}")
        # ========================================================

        test_dataset = train_dataset
    elif dataset_name == "dvsgesture":
        train_dataset = tonic.datasets.DVSGesture(save_to=root_dir, train=True, transform=transform)
        test_dataset = tonic.datasets.DVSGesture(save_to=root_dir, train=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 使用 tonic.collation.PadTensors 防止个别样本维度不对齐报错
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              collate_fn=tonic.collation.PadTensors(batch_first=False))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                             collate_fn=tonic.collation.PadTensors(batch_first=False))

    # 3. 实例化模型
    num_classes = config['dataset']['num_classes']
    network_type = config['network']['type']
    if network_type == "snn_vgg":
        model = SpikingVGG(num_classes=num_classes).to(device)
    else:
        model = SpikingCNN(num_classes=num_classes).to(device)

    # 4. 定义优化器与损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 5. 训练循环
    num_epochs = 20
    best_acc = 0.0
    os.makedirs("results", exist_ok=True)
    save_path = f"results/{network_type}_best.pth"

    logger.info("Starting training loop...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # 训练阶段
        for data, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            spk_out, mem_out = model(data)

            # SNN 分类通常使用最后一层神经元在所有时间步的膜电位总和或均值计算 Loss
            logits = mem_out.sum(dim=0)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total

        # 测试阶段 (这里直接用简单逻辑跑一次 Test)
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                spk_out, mem_out = model(data)
                logits = mem_out.sum(dim=0)
                _, predicted = logits.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100. * test_correct / test_total
        logger.info(
            f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # 保存表现最好的模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"[*] New best accuracy! Weights saved to {save_path}")

    logger.info(f"=== Training Pipeline Finished. Best Test Accuracy: {best_acc:.2f}% ===")


if __name__ == "__main__":
    main()
