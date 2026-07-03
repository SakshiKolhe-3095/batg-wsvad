import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility.
    Call this at the top of train.py and eval.py before anything else.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU safe, harmless if single GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42)
    print("Seed set to 42 — all random sources fixed.")
    print("numpy random:", np.random.rand(3))
    print("torch random:", torch.rand(3))