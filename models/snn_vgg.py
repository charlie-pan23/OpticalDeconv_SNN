import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate


class SpikingVGG(nn.Module):
    """
    Spiking VGG for CIFAR10-DVS (SpikingJelly implementation)
    Input: (T, B, 2, 128, 128)  ->  Output: (T, B, 10)
    """

    def __init__(self, num_classes=10, tau=2.0, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        # 使用平滑度更好的 Sigmoid 替代梯度 (alpha=4.0)
        sg = surrogate.Sigmoid(alpha=4.0)

        self.conv1 = nn.Conv2d(2, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=sg)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=sg)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=sg)

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.lif4 = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=sg)
        self.pool3 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.lif5 = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=sg)

        self.conv6 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.lif6 = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=sg)
        self.pool4 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        # 【新增】Dropout 防过拟合
        self.dropout = nn.Dropout(0.2)

        # 128x128 经过 4 次池化 (缩小 2^4=16 倍) -> 8x8。 256 * 8 * 8 = 16384
        self.fc1 = nn.Linear(16384, 512, bias=False)
        self.lif7 = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=sg)

        self.fc2 = nn.Linear(512, num_classes, bias=False)

    def forward(self, x):
        """
        :param x: [T, B, C, H, W]
        :return: mem_list: [T, B, num_classes] (Logits 输出)
        """
        T = x.shape[0]
        mem_list = []

        # 手动时间步循环，保持最细粒度的底层控制
        for t in range(T):
            cur = x[t]

            cur = self.pool1(self.lif1(self.bn1(self.conv1(cur))))
            cur = self.pool2(self.lif2(self.bn2(self.conv2(cur))))

            cur = self.lif3(self.bn3(self.conv3(cur)))
            cur = self.pool3(self.lif4(self.bn4(self.conv4(cur))))

            cur = self.lif5(self.bn5(self.conv5(cur)))
            cur = self.pool4(self.lif6(self.bn6(self.conv6(cur))))

            cur = self.flatten(cur)
            cur = self.dropout(cur)
            cur = self.lif7(self.fc1(cur))

            out = self.fc2(cur)
            mem_list.append(out)

        return torch.stack(mem_list, dim=0)