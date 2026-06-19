import os
import sys
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.Logger import logger


class OpticalDeconvSNN(nn.Module):
    """
    模拟 OpticalDeconv 光子解卷积加速器的脉冲神经网络。
    针对 DVS128 Gesture 数据集设计 (输入: [Time, Batch, 2, 128, 128])
    """

    def __init__(self, num_classes=11, beta=0.9):
        super(OpticalDeconvSNN, self).__init__()

        logger.info(f"Initializing OpticalDeconvSNN with {num_classes} classes and beta={beta}")

        # 替代梯度函数：解决脉冲神经元“不可导”的问题
        # fast_sigmoid 是 SNN 训练中最常用、效果最稳定的替代梯度之一
        spike_grad = surrogate.fast_sigmoid()

        # ---------------------------------------------------------
        # 1. 降采样特征提取层 (128x128 -> 32x32 -> 8x8)
        # ---------------------------------------------------------
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # ---------------------------------------------------------
        # 2. 光子解卷积核心层 (OpticalDeconv Core)
        # ---------------------------------------------------------
        # 这一层代表了你们论文中的硬件加速器！将 8x8 的特征图上采样回 16x16
        # 后续的加噪测试 (Noise Injection) 主要针对这一层的权重或输出进行
        self.deconv_core = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # ---------------------------------------------------------
        # 3. 分类输出层
        # ---------------------------------------------------------
        self.flatten = nn.Flatten()
        # 经过上面的卷积和解卷积后，特征图大小为 16通道 * 16宽 * 16高 = 4096
        self.fc1 = nn.Linear(16 * 16 * 16, 128)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(128, num_classes)
        # 最后一层神经元，我们不再强制它发射脉冲，而是记录它的膜电位 (Membrane Potential)
        # 用于后续计算交叉熵 Loss
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    def forward(self, x):
        """
        前向传播函数
        x 的维度预期为: [Time_steps, Batch_size, Channels, Height, Width]
        """
        # 初始化所有 LIF 神经元的膜电位 (隐藏状态)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        # 记录最后一层的脉冲和膜电位序列，用于计算 Loss 和准确率
        spk5_rec = []
        mem5_rec = []

        # 获取时间步长 (Tonic 切片后的帧数)
        num_steps = x.size(0)

        # SNN 的灵魂：时间步循环 (Time-step Loop)
        for step in range(num_steps):
            # 取出当前时刻的一帧特征图: [Batch, Channels, Height, Width]
            cur_x = x[step]

            # 1. 降采样
            cur_x = self.pool1(self.conv1(cur_x))
            spk1, mem1 = self.lif1(cur_x, mem1)

            cur_x = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur_x, mem2)

            # 2. 光子解卷积 (核心硬件模拟)
            cur_x = self.deconv_core(spk2)
            spk3, mem3 = self.lif3(cur_x, mem3)

            # 3. 分类器
            cur_x = self.flatten(spk3)
            spk4, mem4 = self.lif4(self.fc1(cur_x), mem4)

            spk5, mem5 = self.lif5(self.fc2(spk4), mem5)

            # 记录输出层的状态
            spk5_rec.append(spk5)
            mem5_rec.append(mem5)

        # 将列表沿时间维度堆叠起来
        # 输出维度: [Time_steps, Batch_size, num_classes]
        return torch.stack(spk5_rec, dim=0), torch.stack(mem5_rec, dim=0)


if __name__ == "__main__":
    # --- 模型维度的单元测试 ---
    logger.info("Starting SNN Model Architecture Test...")

    # 模拟 Tonic 输出的一批数据
    # 假设：30个时间步, Batch为4, 2个通道, 128x128 像素
    mock_input = torch.randn(30, 4, 2, 128, 128)
    logger.info(f"Mock Input Shape: {mock_input.shape}")

    # 实例化模型
    model = OpticalDeconvSNN(num_classes=11)

    # 前向传播
    spk_out, mem_out = model(mock_input)

    logger.info("--- Forward Pass Successful ---")
    logger.info(f"Output Spikes Shape: {spk_out.shape} (Expected: [30, 4, 11])")
    logger.info(f"Output Membrane Potential Shape: {mem_out.shape} (Expected: [30, 4, 11])")