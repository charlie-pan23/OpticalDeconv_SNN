import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SpikingVGG(nn.Module):
    """
    HIPSA 架构基准测试网络 II: Spiking VGG-like (深层密集卷积)
    数据集: CIFAR10-DVS
    作为高事件密度与高 SOP 计算压力测试基准，验证 4个 64x64 PDPU Tile 的并行吞吐极值。
    """

    def __init__(self, num_classes=10, beta=0.9):
        super(SpikingVGG, self).__init__()

        spike_grad = surrogate.fast_sigmoid()

 
        # VGG Block 1 - Input: 128x128 -> Output: 64x64
 
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool1 = nn.MaxPool2d(2)

 
        # VGG Block 2 - Input: 64x64 -> Output: 32x32
 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool2 = nn.MaxPool2d(2)

 
        # VGG Block 3: Photonic MVM Mapping Blocks - Input: 32x32 -> Output: 16x16
        # 硬件映射: 连续的深度卷积层，将产生极大的 MAC 需求。
        # 它们是映射到 64x64 MRR 阵列并利用 HAPR 模拟流求和的核心压力层。
 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool3 = nn.MaxPool2d(2)

 
        # VGG Block 4 - Input: 16x16 -> Output: 8x8
 
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.lif6 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool4 = nn.MaxPool2d(2)

 
        # 分类输出层 (Flatten size: 256 * 8 * 8 = 16384)
 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16384, 512)
        self.lif7 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(512, num_classes)
        self.lif8 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    def forward(self, x):
        """
        x 形状: [Time_steps, Batch_size, Channels, Height, Width]
        """
        # 初始化所有层的膜电位
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()
        mem7 = self.lif7.init_leaky()
        mem8 = self.lif8.init_leaky()

        spk8_rec = []
        mem8_rec = []

        num_steps = x.size(0)

        for step in range(num_steps):
            cur_x = x[step]

            # Block 1
            cur_x = self.conv1(cur_x)
            spk1, mem1 = self.lif1(cur_x, mem1)
            cur_x = self.pool1(spk1)

            # Block 2
            cur_x = self.conv2(cur_x)
            spk2, mem2 = self.lif2(cur_x, mem2)
            cur_x = self.pool2(spk2)

            # Block 3
            cur_x = self.conv3(cur_x)
            spk3, mem3 = self.lif3(cur_x, mem3)
            cur_x = self.conv4(spk3)
            spk4, mem4 = self.lif4(cur_x, mem4)
            cur_x = self.pool3(spk4)

            # Block 4
            cur_x = self.conv5(cur_x)
            spk5, mem5 = self.lif5(cur_x, mem5)
            cur_x = self.conv6(spk5)
            spk6, mem6 = self.lif6(cur_x, mem6)
            cur_x = self.pool4(spk6)

            # Classifier
            cur_x = self.flatten(cur_x)
            cur_x = self.fc1(cur_x)
            spk7, mem7 = self.lif7(cur_x, mem7)

            cur_x = self.fc2(spk7)
            spk8, mem8 = self.lif8(cur_x, mem8)

            spk8_rec.append(spk8)
            mem8_rec.append(mem8)

        return torch.stack(spk8_rec, dim=0), torch.stack(mem8_rec, dim=0)