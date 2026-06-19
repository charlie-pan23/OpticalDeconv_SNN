import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NpyGestureDataset(Dataset):

    def __init__(self, root_dir):
        self.filepaths = []
        self.labels = []

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.npy'):
                    self.filepaths.append(os.path.join(root, file))

                    label = int(file.split('.')[0])
                    self.labels.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]

        data = np.load(path)

        tensor_data = torch.from_numpy(data).float()

        label = self.labels[idx]

        return tensor_data, label


if __name__ == "__main__":

    dataset_dir = r"D:\ProgramProject_Hub\OpticalDeconv_SNN\datasets\DVSGesture\ibmGestureTest"

    test_dataset = NpyGestureDataset(root_dir=dataset_dir)
    print(f"Loaded {len(test_dataset)} samples")

    # 抽取第一个样本，看看它的维度
    sample_data, sample_label = test_dataset[0]
    print(f"\n--- Sample Information ---")
    print(f"Label: {sample_label}")
    print(f"Shape: {sample_data.shape}")