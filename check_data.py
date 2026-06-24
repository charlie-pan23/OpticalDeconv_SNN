import os
import glob


def diagnose_cifar10dvs():
    root = "./datasets/CIFAR10DVS"

    if not os.path.exists(root):
        print(f"❌ 错误: 找不到目录 {root}")
        return

    # 检查一级子文件夹
    folders = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    print(f"✅ 找到 {len(folders)} 个子文件夹.")
    if len(folders) < 10:
        print(f"❌ 错误: 文件夹数量不对！应该有 10 个类的文件夹，现在只有 {len(folders)} 个。")
        print(f"   (请确认 automobile.zip, bird.zip 等其他 9 个压缩包是否也解压了！)")

    # 检查总文件数
    aedat_files = glob.glob(os.path.join(root, "*", "*.aedat"))
    print(f"✅ 找到 {len(aedat_files)} 个 .aedat 脉冲文件.")

    if len(aedat_files) != 10000:
        print(f"❌ 致命错误: Tonic 要求精确的 10000 个文件，你当前只有 {len(aedat_files)} 个！")
        print("   -> Tonic 因此触发了强制重新下载并导致 403 报错。")
        print("   -> 请重新把缺失的类解压补齐！")
    else:
        print("🎉 恭喜！你的数据集完美无缺！Tonic 没有任何理由拒绝。")


if __name__ == "__main__":
    diagnose_cifar10dvs()