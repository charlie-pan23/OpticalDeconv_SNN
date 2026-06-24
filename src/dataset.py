import os
import tonic


def download_datasets():
    # 1. 确定目标路径：获取当前脚本所在目录的上一级，并拼接 'Datasets'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath(os.path.join(current_dir, "..", "Datasets"))

    # 确保目标文件夹存在
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"准备就绪，所有数据集将自动下载并解压至:\n   {dataset_dir}\n")

    # ---------------------------------------------------------
    # 数据集 1: DVS128 Gesture (~1.2 GB)
    # ---------------------------------------------------------
    print("[1/3] 开始下载 DVS128 Gesture...")
    try:
        # 实例化 train 和 test 集，tonic 如果检测到文件不存在会自动触发下载和解压
        tonic.datasets.DVSGesture(save_to=dataset_dir, train=True)
        tonic.datasets.DVSGesture(save_to=dataset_dir, train=False)
        print("DVS128 Gesture 下载并解压完成！\n")
    except Exception as e:
        print(f"DVS128 Gesture 下载失败: {e}\n")

    # ---------------------------------------------------------
    # 数据集 2: CIFAR10-DVS (~1.2 GB)
    # ---------------------------------------------------------
    print("[2/3] 开始下载 CIFAR10-DVS...")
    try:
        # CIFAR10-DVS 官方原版没有严格的 train/test 划分，直接实例化即可下载全部 10000 个样本
        tonic.datasets.CIFAR10DVS(save_to=dataset_dir)
        print("CIFAR10-DVS 下载并解压完成！\n")
    except Exception as e:
        print(f"CIFAR10-DVS 下载失败: {e}\n")

    # # ---------------------------------------------------------
    # # 数据集 3: DDD17 (DAVIS Driving Dataset)
    # # ---------------------------------------------------------
    # print("[3/3] 开始下载 DDD17 (DAVIS Driving Dataset)...")
    # print(" 提示：DDD17 包含了真实的驾驶录像，体积非常大（原始记录文件几十GB）。")
    # print(" 下载和提取时间可能较长，请确保磁盘空间充足（建议预留至少 40GB）。")
    # try:
    #     # tonic 会自动处理大量 .hdf5 或原始文件的下载
    #     tonic.datasets.DDD17(save_to=dataset_dir, train=True)
    #     tonic.datasets.DDD17(save_to=dataset_dir, train=False)
    #     print("DDD17 下载并配置完成！\n")
    # except Exception as e:
    #     print(f"DDD17 下载失败: {e}\n")


if __name__ == "__main__":
    download_datasets()
    print("所有数据集下载流水线执行完毕！")