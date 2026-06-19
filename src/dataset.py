import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import tonic.transforms as transforms

# Ensure the project root is in the system path so we can import 'utils'
# regardless of where this script is executed from.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.Logger import logger


class NpyGestureDataset(Dataset):
    """
    A custom PyTorch Dataset for loading pre-processed DVS Gesture data stored as .npy files.
    It reads raw [N, 4] event arrays and prepares them for Tonic transformations
    """

    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        Args:
            root_dir (str): Directory containing the user folders with .npy files.
            transform (callable, optional): Optional transform to be applied on a sample (e.g., tonic transforms).
            target_transform (callable, optional): Optional transform to be applied on the label.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        self.filepaths = []
        self.labels = []

        logger.info(f"Initializing NpyGestureDataset from directory: {self.root_dir}")

        if not os.path.exists(self.root_dir):
            logger.error(f"Dataset directory does not exist: {self.root_dir}")
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        # Walk through the directory to find all .npy files
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.npy'):
                    self.filepaths.append(os.path.join(root, file))
                    try:
                        # Extract label from filename (e.g., "0.npy" -> 0)
                        label = int(file.split('.')[0])
                        self.labels.append(label)
                    except ValueError:
                        logger.warning(f"Could not parse label from filename: {file}. Skipping this file.")
                        self.filepaths.pop()  # Remove the appended filepath if label parsing fails

        logger.info(f"Successfully loaded {len(self.filepaths)} samples into the dataset index.")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]

        try:
            # 1. Load the raw [N, 4] numpy array
            events = np.load(path)

            # Edge case: Handle empty event arrays
            if len(events) == 0:
                logger.warning(f"Empty event array found at {path}. Returning empty tensor.")
                events = np.zeros((1, 4))

            # 2. Convert to Tonic-compatible structured array
            # Tonic expects a 1D structured array with specific field names: 'x', 'y', 't', 'p'
            structured_events = np.zeros(len(events), dtype=[('x', np.int16),
                                                             ('y', np.int16),
                                                             ('t', np.int64),
                                                             ('p', np.bool_)])
            structured_events['x'] = events[:, 0]
            structured_events['y'] = events[:, 1]

            # Autodetect which column is the timestamp (t) and which is polarity (p)
            # Timestamps are usually strictly increasing and have a much larger maximum value than polarity (0 or 1)
            if np.max(events[:, 2]) > np.max(events[:, 3]):
                structured_events['t'] = events[:, 2] * 1000
                structured_events['p'] = events[:, 3]
            else:
                structured_events['t'] = events[:, 3] * 1000
                structured_events['p'] = events[:, 2]

            # 3. Apply Tonic transforms (e.g., framing, denoising)
            if self.transform is not None:
                # Tonic ToFrame outputs a numpy array of shape [Time_steps, Channels, Height, Width]
                processed_events = self.transform(structured_events)
                # Convert to PyTorch Tensor
                tensor_data = torch.from_numpy(processed_events).float()
            else:
                # If no transform is provided, just return the raw tensor
                tensor_data = torch.from_numpy(events).float()

            # 4. Apply target transforms if any
            if self.target_transform is not None:
                label = self.target_transform(label)

            return tensor_data, label

        except Exception as e:
            logger.error(f"Failed to load sample at {path}. Error: {e}")
            raise


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Unit Testing Block
    # Run this file directly to test the dataset loading logic
    # ---------------------------------------------------------
    logger.info("Starting Dataset Unit Test...")

    # Define the sensor configuration for DVS128 Gesture (128x128 resolution, 2 polarities)
    sensor_size = (128, 128, 2)

    # Define a transform: Group events into frames every 30,000 microseconds (30 ms)
    # This converts the sparse event stream into dense tensors suitable for CNNs/SNNs
    time_window_us = 30000
    frame_transform = transforms.ToFrame(sensor_size=sensor_size, time_window=time_window_us)

    # Define the dataset path (Update this if your folder structure changes)
    test_data_dir = os.path.join(project_root, "datasets", "DVSGesture", "ibmGestureTest")

    try:
        # Initialize the dataset
        test_dataset = NpyGestureDataset(root_dir=test_data_dir, transform=frame_transform)

        # Test fetching a specific sample
        sample_idx = 0
        data, target = test_dataset[sample_idx]

        logger.info("--- Sample Loading Test Successful ---")
        logger.info(f"Sample Index : {sample_idx}")
        logger.info(f"Target Label : {target}")
        logger.info(f"Data Shape   : {data.shape}")
        logger.info(f"Data Type    : {data.dtype}")

        # Explain the output shape
        # Format is usually [T, C, H, W] for Tonic frames
        logger.info("Shape Explanation: [Time_steps, Channels (Polarity), Height, Width]")

    except Exception as e:
        logger.error(f"Dataset Test Failed: {e}")
