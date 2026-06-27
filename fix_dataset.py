import numpy as np

# =====================================================================
# 🚨 终极环境补丁 (Monkey Patch)
# 强行把 Python 原生的 bool 赋值给 np.bool，解决 NumPy 1.24+ 兼容性报错
# =====================================================================
np.bool = bool
np.object = object  # 顺手把 object 也补上，防患于未然

import os
import glob
from tqdm import tqdm
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS


def fix_cifar10dvs_events():
    print("=== 🚀 开启真正上帝模式：CIFAR10-DVS 强制单线程转换 ===")

    root = os.path.join("datasets", "CIFAR10DVS")
    extract_dir = os.path.join(root, "extract")
    events_np_dir = os.path.join(root, "events_np")

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if not os.path.exists(extract_dir):
        print(f"❌ 找不到解压目录: {extract_dir} (请确保程序已经走完 extract 阶段)")
        return

    total_files = 0
    success_files = 0

    for cls in classes:
        cls_extract = os.path.join(extract_dir, cls)
        cls_events = os.path.join(events_np_dir, cls)
        os.makedirs(cls_events, exist_ok=True)

        aedat_files = glob.glob(os.path.join(cls_extract, "*.aedat"))
        if len(aedat_files) == 0:
            print(f"⚠️ 警告: 在 {cls_extract} 中没有找到 .aedat 文件")
            continue

        print(f"\n📂 正在处理类别: {cls} ({len(aedat_files)} 个文件)")

        for f in tqdm(aedat_files, desc=cls):
            total_files += 1
            file_name = os.path.basename(f)
            npz_path = os.path.join(cls_events, file_name.replace(".aedat", ".npz"))

            # 如果已经存在且大小正常，跳过以节省时间
            if os.path.exists(npz_path) and os.path.getsize(npz_path) > 1024:
                success_files += 1
                continue

            try:
                # 调用官方自带的静态方法
                CIFAR10DVS.read_aedat_save_to_np(f, npz_path)
                success_files += 1
            except Exception as e:
                print(f"\n❌ 致命报错！文件 {file_name} 转换失败: {str(e)}")

    print(f"\n✅ 强制转换完成！成功率: {success_files}/{total_files}")
    if success_files == 10000:
        print("🎉 你的底层事件流缓存已完美建立！现在去运行你的 train_cifar10dvs.py 吧！")


if __name__ == "__main__":
    fix_cifar10dvs_events()