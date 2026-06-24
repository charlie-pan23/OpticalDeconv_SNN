"""
Unzip all CIFAR10-DVS zip files and count .aedat samples.
"""

import os
import zipfile

# 自动定位 datasets/CIFAR10DVS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(SCRIPT_DIR, "..", "datasets", "CIFAR10DVS")


def unzip_all(zip_root):
    if not os.path.exists(zip_root):
        print(f"Directory not found: {zip_root}")
        return False

    zip_files = [f for f in os.listdir(zip_root) if f.lower().endswith(".zip")]

    if not zip_files:
        print(f"No zip files found in {zip_root}")
        return False

    print(f"Found {len(zip_files)} zip file(s).")

    for zip_name in zip_files:
        zip_path = os.path.join(zip_root, zip_name)
        extract_to = zip_root

        print(f"Unzipping: {zip_name}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Extracted: {zip_name}")
        except Exception as e:
            print(f"Failed to unzip {zip_name}: {e}")
            return False

    print("\n🎉 Unzip completed successfully.")
    return True


def count_aedat_files(dataset_root):
    total = 0
    class_counts = {}

    for root, dirs, files in os.walk(dataset_root):
        aedat_files = [f for f in files if f.lower().endswith(".aedat")]
        if aedat_files:
            class_name = os.path.basename(root)
            class_counts[class_name] = len(aedat_files)
            total += len(aedat_files)

    print("\nAEDAT Sample Count:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls:15s}: {cnt}")

    print(f"\nTotal .aedat files: {total}")

    if total == 10000:
        print("Perfect! You have exactly 10,000 samples (CIFAR10-DVS full dataset).")
    else:
        print(f"Warning: Expected 10,000 samples, but found {total}.")

    return total


if __name__ == "__main__":
    success = unzip_all(DATASET_ROOT)
    if success:
        count_aedat_files(DATASET_ROOT)