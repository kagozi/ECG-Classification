import torch
import random
import numpy as np

import torch
import random
import numpy as np

DATA_PATH = '/path/to/ptb-xl-dataset/'  # Update this path as needed
OUTPUT_PATH = '/path/to/output/'  # Update this path as needed
# --- Constants for Preprocessing ---
TARGET_FS = 100  # Target sampling rate (Hz)
SIGNAL_LEN_SECONDS = 10
LOWCUT = 0.5
HIGHCUT = 40.0
THRESHOLD = 0.5  # for binarizing probabilities
IMAGE_SIZE = 224  # for ResNet50 input resize
NUM_WORKERS = 4
SEED = 42  # for reproducibility
def setup_device_and_seed(seed: int = 42):
    """
    Sets up the computation device (CUDA or CPU) and seeds all
    relevant random number generators for reproducibility.

    Args:
        seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        device (torch.device): The device to be used for computation.
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set all seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"✅ Using device: {device}")
    print(f"🔒 Random seed set to: {seed}")

    return device

