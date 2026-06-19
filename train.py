import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tonic.transforms as transforms
from snntorch import spikegen
from snntorch import utils as snn_utils

from utils.Logger import logger
from utils.config import config
from src.dataset import NpyGestureDataset
from src.model import OpticalDeconvSNN

def pad_collate_fn(batch):
    data, targets = zip(*batch)
    # 找出当前 batch 里最长的时间步
    max_time = max([d.shape[0] for d in data])
    padded_data = []
    for d in data:
        # 补齐：在时间维度 (第 0 维) 的末尾填充 0
        pad_len = max_time - d.shape[0]
        if pad_len > 0:
            pad_tensor = torch.zeros((pad_len, *d.shape[1:]), dtype=d.dtype)
            d = torch.cat([d, pad_tensor], dim=0)
        padded_data.append(d)

    # 堆叠成 [Time_steps, Batch, Channels, Height, Width]
    return torch.stack(padded_data, dim=1), torch.tensor(targets)

def train():
    logger.info("=== Starting SNN Training Pipeline ===")

    # ---------------------------------------------------------
    # 1. 超参数与环境配置 (Hyperparameters & Setup)
    # ---------------------------------------------------------
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    learning_rate = float(config['training']['learning_rate'])
    time_window_us = config['training']['time_window_us']
    sensor_size = tuple(config['training']['sensor_size'])  # 将 yaml 的 list 转为 tuple
    num_classes = config['training']['num_classes']

    # 自动检测 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} | Batch Size: {batch_size} | Epochs: {epochs}")

    # ---------------------------------------------------------
    # 2. 数据准备 (Data Preparation)
    # ---------------------------------------------------------
    logger.info("Setting up data loaders...")
    # 配置 Tonic 转换器
    frame_transform = transforms.ToFrame(sensor_size=sensor_size, time_window=time_window_us)

    # 数据集路径 (目前用整个 ibmGestureTest 跑流程)
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "DVSGesture", "ibmGestureTest")

    train_dataset = NpyGestureDataset(root_dir=data_dir, transform=frame_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,  # 雇佣 4 个 CPU 子进程同时加载和处理数据
        pin_memory=False, # Windows 下避免 CUDA 映射错误
        collate_fn=pad_collate_fn)
    logger.info(f"Dataloader ready. Total batches per epoch: {len(train_loader)}")

    # ---------------------------------------------------------
    # 3. 模型构建与优化器 (Model & Optimizer)
    # ---------------------------------------------------------
    logger.info("Initializing OpticalDeconvSNN model...")
    model = OpticalDeconvSNN(num_classes=num_classes).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # SNN 损失函数
    loss_fn = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # 4. 训练循环 (Training Loop)
    # ---------------------------------------------------------
    logger.info("Starting training loop...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            # 将数据推至 GPU
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # 前向传播
            spk_rec, mem_rec = model(data)

            # 损失计算
            logits = mem_rec.sum(dim=0)
            loss = loss_fn(logits, targets)

            # 反向传播与权重更新
            loss.backward()
            optimizer.step()

            # 统计指标
            total_loss += loss.item()

            # 预测
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 5 == 0:
                logger.debug(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Epoch 结束汇总
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        logger.info(f"==> Epoch [{epoch + 1}/{epochs}] Summary | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # ---------------------------------------------------------
    # 5. 保存模型 (Save Model)
    # ---------------------------------------------------------
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optical_deconv_snn.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Training completed. Model saved to {save_path}")

if __name__ == "__main__":
    train()