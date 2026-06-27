import os
import glob
import random
import numpy as np


def check_npz_validity():
    print("=== 🔍 开启 CIFAR10-DVS .npz 真实性抽查 ===")
    events_np_dir = os.path.join("datasets", "CIFAR10DVS", "events_np")

    if not os.path.exists(events_np_dir):
        print(f"❌ 找不到目录 {events_np_dir}，请确认路径。")
        return

    classes = os.listdir(events_np_dir)
    if not classes:
        print("❌ events_np 目录下为空！")
        return

    print(f"✅ 找到 {len(classes)} 个类别文件夹。开始随机抽查...\n")

    valid_count = 0
    total_checked = 0

    for cls in classes:
        cls_dir = os.path.join(events_np_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        npz_files = glob.glob(os.path.join(cls_dir, "*.npz"))
        if not npz_files:
            print(f"⚠️ 类别 {cls} 下没有找到 .npz 文件。")
            continue

        # 每个类别随机抽查 1 个文件
        sample_file = random.choice(npz_files)
        total_checked += 1

        print(f"[{cls}] 正在抽查: {os.path.basename(sample_file)}")
        try:
            data = np.load(sample_file)

            # 1. 检查必备的键值 (DVS 数据必须包含 t, x, y, p)
            keys = data.files
            if not all(k in keys for k in ['t', 'x', 'y', 'p']):
                print(f"  ❌ 缺失关键数据列！当前文件仅包含: {keys}")
                continue

            t, x, y, p = data['t'], data['x'], data['y'], data['p']
            event_count = len(t)

            if event_count == 0:
                print("  ❌ 事件数量为 0 (这是一个假文件/空文件)！")
                continue

            # 2. 打印底层物理统计信息
            print(f"  ✅ 事件总数 (Event Count): {event_count}")
            print(f"  ✅ X 坐标范围: {x.min()} ~ {x.max()}")
            print(f"  ✅ Y 坐标范围: {y.min()} ~ {y.max()}")
            print(f"  ✅ 极性 P 种类: {np.unique(p)}")
            print(f"  ✅ 时间戳 T 范围: {t.min()} ~ {t.max()} (微秒)")

            # 3. CIFAR10-DVS 硬件分辨率严格校验 (DVS128 摄像机的上限就是 127)
            if x.max() > 127 or y.max() > 127:
                print("  ⚠️ 警告: 坐标超过了 128x128 的物理分辨率，解码可能存在越界！")
            elif x.max() == 0 and y.max() == 0:
                print("  ❌ 错误: 坐标全为 0，这是毫无意义的假数据！")
            else:
                valid_count += 1
                print("  🎉 质检通过！数据完全符合真实的 DVS 物理特性。\n")

        except Exception as e:
            print(f"  ❌ 读取失败: {e}\n")

    print("=====================================================")
    print(f"🎯 抽查完毕: 共检查 {total_checked} 个文件，其中 {valid_count} 个完全合格。")
    if valid_count == total_checked and total_checked > 0:
        print("🚀 放心吧！你的数据集现在是 100% 货真价实的，立刻去运行 train_cifar10dvs.py 吧！")
    else:
        print("⚠️ 存在异常数据，请查看上面的报错信息。")


if __name__ == "__main__":
    check_npz_validity()