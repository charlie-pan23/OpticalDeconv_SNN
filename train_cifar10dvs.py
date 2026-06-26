"""
train_cifar10dvs.py – 手写 CIFAR10-DVS 全流程 (解压 + aedat -> npz + npz -> frame)
脱离 Tonic/SJ 黑盒； aedat -> npznpz -> frame 按类 10 线程并发（每类一个 worker，类间零共享）
训练循环保留 transpose + mem_out.sum 风格，config 不动
"""
import os
import zipfile
import glob
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from tonic.io import read_davis_346
from utils.Logger import logger
from models.snn_vgg import SpikingVGG

# ==============================================================
# 工具常量
# ==============================================================
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
DAVIS_HW = (260, 346)
CROP_HW = (128, 128)
CROP_TOP = (DAVIS_HW[0] - CROP_HW[0]) // 2    # 66
CROP_LEFT = (DAVIS_HW[1] - CROP_HW[1]) // 2   # 109
T_FRAMES = 10
MAX_WORKERS = 10  # 10 类  ->  10 线程，每类一个


# ==============================================================
#  解压（串行，zip 单资源）
# ==============================================================
def ensure_extract(root_dir):
    download_dir = os.path.join(root_dir, "download")
    extract_dir = os.path.join(root_dir, "extract")

    zips = glob.glob(os.path.join(download_dir, "*.zip"))
    if len(zips) == 0:
        zips = glob.glob(os.path.join(root_dir, "*.zip"))
    if len(zips) == 0:
        raise FileNotFoundError(
            f"No zip found in {download_dir} or {root_dir}. "
            "Put the official CIFAR10-DVS zip under download/ ."
        )

    need = False
    for cls in CLASSES:
        d = os.path.join(extract_dir, cls)
        if len(glob.glob(os.path.join(d, "*.aedat"))) < 1000:
            need = True
            break

    if not need and os.path.isdir(extract_dir):
        logger.info("[Extract] extract/ completed, skip.")
        return

    logger.info(f"[Extract] Unzipping {len(zips)} zip(s) ...")
    for z in zips:
        with zipfile.ZipFile(z, 'r') as zf:
            zf.extractall(extract_dir)
    logger.info("[Extract] Done.")


# ==============================================================
# purge：清坏 npz（写到一半中断留下的 truncated）
# ==============================================================
def purge_corrupted_npz(npz_dir):
    bad = 0
    for cls in CLASSES:
        d = os.path.join(npz_dir, cls)
        if not os.path.isdir(d):
            continue
        for fp in glob.glob(os.path.join(d, "*.npz")):
            try:
                with np.load(fp, allow_pickle=True) as f:
                    _ = f['events']
            except Exception as e:
                logger.warning(f"  [purge] corrupted: {fp} ({e})")
                os.remove(fp)
                bad += 1
    logger.info(f"[purge] removed {bad} corrupted npz")


# ==============================================================
# 单类 aedat -> npz（给线程池用）
# ==============================================================
def _convert_cls_aedat_to_npz(cls, src_root, dst_root):
    """
    处理单个类：src_root/{cls}/*.aedat  ->  dst_root/{cls}/*.npz
    返回 (cls, converted_count)
    """
    src = os.path.join(src_root, cls)
    dst = os.path.join(dst_root, cls)
    os.makedirs(dst, exist_ok=True)

    aedats = sorted(glob.glob(os.path.join(src, "*.aedat")))
    cnt = 0
    for fp in tqdm(aedats, desc=f"  {cls}", unit="file", position=1, leave=False):
        base = os.path.splitext(os.path.basename(fp))[0]
        out = os.path.join(dst, base + ".npz")
        tmp = out + ".tmp"
        if os.path.exists(out):
            continue
        try:
            ret = read_davis_346(fp)
            events = ret[-1]   # (N,) structured
            np.savez_compressed(tmp, events=events)
            os.replace(tmp, out)   # 原子替换，防中断留半截
            cnt += 1
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            logger.warning(f"  [aedat -> npz] {fp} failed: {e}")
    return cls, cnt


def ensure_events_np(root_dir):
    extract_dir = os.path.join(root_dir, "extract")
    npz_dir = os.path.join(root_dir, "events_np")

    # 先清坏 npz
    purge_corrupted_npz(npz_dir)

    # 再检查是否全满
    logger.info("[aedat -> npz] Checking ...")
    all_ok = True
    for cls in CLASSES:
        d = os.path.join(npz_dir, cls)
        os.makedirs(d, exist_ok=True)
        if len(glob.glob(os.path.join(d, "*.npz"))) < 1000:
            all_ok = False
            break
    if all_ok:
        logger.info("[aedat -> npz] events_np/ completed, skip.")
        return

    logger.info(f"[aedat -> npz] Converting (10 threads, per-class) ...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(_convert_cls_aedat_to_npz, cls,
                        extract_dir, npz_dir): cls
            for cls in CLASSES
        }
        for fut in tqdm(as_completed(futs), total=10, desc="   aedat -> npz classes"):
            cls, cnt = fut.result()
            if cnt:
                logger.info(f"    {cls}: +{cnt} npz")

    # 再校验一遍
    short = [c for c in CLASSES
             if len(glob.glob(os.path.join(npz_dir, c, "*.npz"))) < 1000]
    if short:
        raise RuntimeError(f"[aedat -> npz] classes incomplete: {short}")
    logger.info("[aedat -> npz] Done.")


# ==============================================================
# 单类 npz -> frame（给线程池用）
# ==============================================================
def _convert_cls_npz_to_frame(cls, npz_root, frame_root, T):
    src = os.path.join(npz_root, cls)
    dst = os.path.join(frame_root, cls)
    os.makedirs(dst, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src, "*.npz")))
    cnt = 0
    for fp in tqdm(files, desc=f"  {cls}", unit="file", position=1, leave=False):
        base = os.path.splitext(os.path.basename(fp))[0]
        out = os.path.join(dst, base + ".npz")
        tmp = out + ".tmp"
        if os.path.exists(out):
            continue
        try:
            ev = np.load(fp, allow_pickle=True)['events']

            t = ev['t'].astype(np.float64)
            t_min, t_max = t.min(), t.max()
            if t_max - t_min < 1e-9:
                frame = np.zeros((T, 2, *CROP_HW), dtype=np.float32)
                np.savez_compressed(tmp, frame=frame)
                os.replace(tmp, out)
                cnt += 1
                continue

            edges = np.linspace(t_min, t_max, T + 1)
            frame = np.zeros((T, 2, *DAVIS_HW), dtype=np.float32)

            for ti in range(T):
                mask = (t >= edges[ti]) & (t < edges[ti + 1])
                sub = ev[mask]
                if len(sub) == 0:
                    continue
                x = sub['x'].astype(np.int64)
                y = sub['y'].astype(np.int64)
                p = sub['p'].astype(np.int64)
                if p.min() < 0:
                    p = ((p + 1) // 2).astype(np.int64)
                frame[ti, 0, y[p == 0], x[p == 0]] += 1
                frame[ti, 1, y[p == 1], x[p == 1]] += 1

            frame = frame[:, :,
                          CROP_TOP:CROP_TOP + CROP_HW[0],
                          CROP_LEFT:CROP_LEFT + CROP_HW[1]]
            np.savez_compressed(tmp, frame=frame)
            os.replace(tmp, out)
            cnt += 1
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            logger.warning(f"  [npz -> frame] {fp} failed: {e}")
    return cls, cnt


def ensure_frames(root_dir, T=T_FRAMES):
    npz_dir = os.path.join(root_dir, "events_np")
    frame_dir = os.path.join(root_dir, f"frames_number_{T}_split_by_number")

    logger.info("[npz -> frame] Checking ...")
    all_ok = True
    for cls in CLASSES:
        d = os.path.join(frame_dir, cls)
        if not os.path.isdir(d):
            all_ok = False
            break
        if len(glob.glob(os.path.join(d, "*.npz"))) < 1000:
            all_ok = False
            break
    if all_ok:
        logger.info(f"[npz -> frame] frames_number_{T}_split_by_number/ completed, skip.")
        return

    logger.info(f"[npz -> frame] Converting (10 threads, per-class) ...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(_convert_cls_npz_to_frame, cls,
                        npz_dir, frame_dir, T): cls
            for cls in CLASSES
        }
        for fut in tqdm(as_completed(futs), total=10, desc="classes"):
            cls, cnt = fut.result()
            if cnt:
                logger.info(f"    {cls}: +{cnt} frame-npz")

    short = [c for c in CLASSES
             if len(glob.glob(os.path.join(frame_dir, c, "*.npz"))) < 1000]
    if short:
        raise RuntimeError(f"[npz -> frame] classes incomplete: {short}")
    logger.info("[npz -> frame] Done.")


# ==============================================================
# Dataset（不动）
# ==============================================================
class CIFAR10DVSFrames(Dataset):
    def __init__(self, root_dir, T=T_FRAMES, train=True, split_ratio=0.9):
        self.root = os.path.join(root_dir, f"frames_number_{T}_split_by_number")
        self.T = T
        self.classes = CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        self.file_paths = []
        self.labels = []
        for cls in CLASSES:
            cls_dir = os.path.join(self.root, cls)
            files = sorted(glob.glob(os.path.join(cls_dir, "*.npz")))
            n = len(files)
            k = int(n * split_ratio)
            if train:
                files = files[:k]
            else:
                files = files[k:]
            for f in files:
                self.file_paths.append(f)
                self.labels.append(self.class_to_idx[cls])
        mode = "train" if train else "test"
        logger.info(f"  CIFAR10DVSFrames({mode}): {len(self.file_paths)} "
                    f"(split_ratio={split_ratio})")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        d = np.load(self.file_paths[idx], allow_pickle=True)
        return torch.from_numpy(d['frame']).float(), self.labels[idx]


# ==============================================================
# main
# ==============================================================
def main():
    logger.info("=== CIFAR10-DVS train ===")

    with open("configs/config_cifar10dvs.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    ds = config['dataset']
    root_dir = os.path.join(ds['root_dir'], "CIFAR10DVS")
    T = ds['time_steps']

    ensure_extract(root_dir)
    ensure_events_np(root_dir)     # 内含 purge + 10 线程
    ensure_frames(root_dir, T=T)   # 10 线程

    logger.info("Building CIFAR10DVSFrames (split_ratio=0.9) ...")
    train_dataset = CIFAR10DVSFrames(root_dir, T=T, train=True, split_ratio=0.9)
    test_dataset = CIFAR10DVSFrames(root_dir, T=T, train=False, split_ratio=0.9)

    loader_kw = dict(
        batch_size=ds['batch_size_train'],
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kw)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kw)

    model = SpikingVGG(num_classes=ds['num_classes']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs("results", exist_ok=True)
    save_path = "results/snn_vgg_best.pth"

    for epoch in range(20):
        model.train()
        correct, total, train_loss = 0, 0, 0.0
        for data, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/20"):
            data = data.transpose(0, 1).to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            _, mem_out = model(data)
            logits = mem_out.sum(dim=0)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = logits.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()

        train_acc = 100. * correct / total

        model.eval()
        te_correct, te_total = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.transpose(0, 1).to(device)
                targets = targets.to(device)
                _, mem_out = model(data)
                logits = mem_out.sum(dim=0)
                _, pred = logits.max(1)
                te_total += targets.size(0)
                te_correct += pred.eq(targets).sum().item()

        test_acc = 100. * te_correct / te_total
        logger.info(
            f"Epoch {epoch + 1} | "
            f"Loss: {train_loss / len(train_loader):.4f} | "
            f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"  -> [*] New best {best_acc:.2f}%, saved.")

    logger.info(f"=== Done. Best Test Acc: {best_acc:.2f}% ===")


if __name__ == "__main__":
    main()