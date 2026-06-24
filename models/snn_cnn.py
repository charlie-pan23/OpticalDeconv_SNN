import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SpikingCNN(nn.Module):
    """
    HIPSA 架构基准测试网络 I: 浅层 Spiking CNN (带光学解卷积算子)
    数据集: DVS128 Gesture
    作为极致稀疏度测试基准，展现 TS-EGI 和 Shared ADC Pool 在极低输入活性下的极佳能效。
    """

    def __init__(self, num_classes=11, beta=0.9):
        super(SpikingCNN, self).__init__()

        # 使用 fast_sigmoid 作为平滑替代梯度，收敛快且稳定
        spike_grad = surrogate.fast_sigmoid()

        # 预处理特征提取层 (电学域/数字预处理域)
        # 输入: [Batch, 2, 128, 128]
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        # 此时特征图缩小至: 32通道 * 8 * 8

 
        # 光子算子核心模拟层 (Photonic PDPU Core)
        # 硬件映射: 这一层被映射到 4 个 64x64 的光子 MVM 宏块中。
        # 后续提取 Activity Ratio 时，我们将重点监控这里的稀疏度。
 
        self.deconv_core = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        # 特征图恢复至: 16通道 * 16 * 16

 
        # 分类输出层 (数字后端 LIF 与累加)
 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 16 * 16, 128)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(128, num_classes)
        # 最后一层神经元开启 output=True，直接返回膜电位用于计算 Loss
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    def forward(self, x):
        """
        x 形状: [Time_steps, Batch_size, Channels, Height, Width]
        """
        # 隐状态初始化 (SRAM 膜电位缓存清零)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        spk5_rec = []
        mem5_rec = []

        num_steps = x.size(0)

        # 时间步循环
        for step in range(num_steps):
            cur_x = x[step]

            # 下采样
            cur_x = self.pool1(self.conv1(cur_x))
            spk1, mem1 = self.lif1(cur_x, mem1)

            cur_x = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur_x, mem2)

            # 光子核心解卷积算子
            cur_x = self.deconv_core(spk2)
            spk3, mem3 = self.lif3(cur_x, mem3)

            # 数字分类
            cur_x = self.flatten(spk3)
            spk4, mem4 = self.lif4(self.fc1(cur_x), mem4)
            spk5, mem5 = self.lif5(self.fc2(spk4), mem5)

            spk5_rec.append(spk5)
            mem5_rec.append(mem5)

        return torch.stack(spk5_rec, dim=0), torch.stack(mem5_rec, dim=0)