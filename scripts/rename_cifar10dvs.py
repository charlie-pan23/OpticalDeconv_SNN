import os

# 自动定位 datasets/CIFAR10DVS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(SCRIPT_DIR, "..", "datasets", "CIFAR10DVS")

PREFIX_TO_REMOVE = "cifar10_"


def rename_files(root):
    renamed_count = 0

    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.startswith(PREFIX_TO_REMOVE) and filename.endswith(".aedat"):
                old_path = os.path.join(dirpath, filename)
                new_filename = filename.replace(PREFIX_TO_REMOVE, "", 1)
                new_path = os.path.join(dirpath, new_filename)

                print(f"Renaming:\n  {filename}\n-> {new_filename}\n")
                os.rename(old_path, new_path)
                renamed_count += 1

    print(f"Done. Total files renamed: {renamed_count}")


if __name__ == "__main__":
    if not os.path.exists(DATASET_ROOT):
        print(f"Dataset root not found: {DATASET_ROOT}")
    else:
        rename_files(DATASET_ROOT)