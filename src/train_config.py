import os
from datetime import datetime
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torchvision import transforms

DATE = datetime.now().strftime("%d%m%Y")

# No need for dataset_switch anymore
# Use environment variable if available, otherwise use default path
DATA_DIR = os.environ.get("DATA_DIR")
if DATA_DIR is None:
    # Set default data directory to parent data folder
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Updated to use caxton_dataset_final.csv
DATASET_NAME = "dataset_mini"
DATA_CSV = os.path.join(
    DATA_DIR,
    "caxton_dataset/caxton_dataset_mini.csv",
)
DATASET_MEAN = [0.2915257, 0.27048784, 0.14393276]  # Keep same unless you recompute
DATASET_STD = [0.066747, 0.06885352, 0.07679665]


INITIAL_LR = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 50

NUM_NODES = 1
NUM_GPUS = 1
ACCELERATOR = "ddp"

def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def make_dirs(path):
    try:
        os.makedirs(path)
    except:
        pass

# Preprocessing (same normalization used above)
preprocess = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            DATASET_MEAN,
            DATASET_STD,
        )
    ],
)