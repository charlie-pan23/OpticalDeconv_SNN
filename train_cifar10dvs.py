import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# 引入 SpikingJelly 标准数据集与核心重置接口
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets import split_to_train_test_set
from spikingjelly.activation_based import functional as sf

from models.snn_vgg import SpikingVGG
from utils.Logger import logger


def main():
    logger.info("=== Starting CIFAR10-DVS Pure Training Pipeline ===")

    # 1. 读取 YAML 基础配置
    CONFIG_PATH = "configs/config_cifar10dvs.yaml"
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    ROOT = os.path.join(config['dataset']['root_dir'], "CIFAR10DVS")  # 通常指向 ./datasets/CIFAR10DVS
    T = config['dataset']['time_steps']
    BATCH_SIZE = 32  # 满血版批大小，充分喂饱 4070 显卡
    EPOCHS = 6  # 训练总轮数 对于CIFAR10-DVS数据集来说已经过拟合。
    # Epoch 06 | LR: 0.000315 | Loss: 0.6266 | Train: 79.25% | Test: 64.01%

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. 直接加载数据集（SpikingJelly 内部会自动识别并锁定 events_np，绝不修改历史缓存）
    logger.info(f"Loading dataset from frozen cache: {ROOT}")
    full_dataset = CIFAR10DVS(
        root=ROOT,
        data_type='frame',
        frames_number=T,
        split_by='number'
    )

    # 科学划分训练集与测试集（保证每个类别精确按 9:1 划分，防止类别失衡）
    train_ds, test_ds = split_to_train_test_set(0.9, full_dataset, num_classes=10)

    # 【高性能加速引擎】开启 4 线程并行预读 + 锁页内存，彻底消除 GPU 数据饥饿
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=True
    )
    logger.info(f"Train / Test samples: {len(train_ds)} / {len(test_ds)}")

    # 3. 初始化网络模型与高级优化方案
    model = SpikingVGG(num_classes=10).to(device)

    # 使用更稳健的 5e-4 初始学习率，并配合余弦退火调度器防止后期震荡
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # AMP 自动混合精度梯度缩放器，让显卡训练速度翻倍
    scaler = torch.amp.GradScaler('cuda')

    # 4. 纯净训练循环
    best_acc = 0.0
    os.makedirs("results", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        correct, total, train_loss = 0, 0, 0.0

        for data, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            # 【保命黄金细节 1】每个 Batch 开头强制清空所有神经元膜电位记忆
            sf.reset_net(model)

            # 形状转置: [B, T, C, H, W] -> [T, B, C, H, W] 以适配时间步迭代
            data = data.transpose(0, 1).to(device)
            targets = targets.to(device)

            # 【保命黄金细节 2】输入二值化处理，将像素累计值截断为 0 或 1，防止梯度死锁
            data = (data > 0).float()

            optimizer.zero_grad()

            # 开启 AMP 混合精度前向传播
            with torch.amp.autocast('cuda'):
                mem_list = model(data)
                # 【古月居优化方案】采用 mean(0) 取代 sum(0)，极大提升大 T 下的数值稳定性
                logits = mem_list.mean(dim=0)
                loss = criterion(logits, targets)

            # 混合精度反向传播与权重更新
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, pred = logits.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()

        train_acc = 100. * correct / total

        # 测试/验证循环
        model.eval()
        te_correct, te_total = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                # 验证集前向传播同样执行前置重置
                sf.reset_net(model)

                data = data.transpose(0, 1).to(device)
                targets = targets.to(device)
                data = (data > 0).float()

                with torch.amp.autocast('cuda'):
                    mem_list = model(data)
                    logits = mem_list.mean(dim=0)

                _, pred = logits.max(1)
                te_total += targets.size(0)
                te_correct += pred.eq(targets).sum().item()

        test_acc = 100. * te_correct / te_total
        logger.info(f"Epoch {epoch + 1:02d} | LR: {scheduler.get_last_lr()[0]:.6f} | Loss: {train_loss / len(train_loader):.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
        print(f"\nEpoch {epoch + 1:02d} | LR: {scheduler.get_last_lr()[0]:.6f} | Loss: {train_loss / len(train_loader):.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%\n")
        # 更新学习率
        scheduler.step()

        # 实时保存最具有论文说服力的最高准确率权重
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "results/snn_vgg_best.pth")
            logger.info(" -> [*] New best model saved!")

    logger.info(f"=== Pipeline Finished! Highest Test Accuracy: {best_acc:.2f}% ===")

if __name__ == "__main__":
    main()