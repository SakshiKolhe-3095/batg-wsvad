import os
import torch


def save_checkpoint(state: dict, path: str) -> None:
    """Save training checkpoint.
    
    Call periodically during training (every N epochs or every N minutes)
    to survive AI Kosh 4hr session cap without losing progress.
    
    Args:
        state: dict containing at minimum:
            - epoch: int
            - model_state_dict: model.state_dict()
            - optimizer_state_dict: optimizer.state_dict()
            - loss: current loss value
            - auc: current best AUC (optional but useful)
            - budget: budget level B being trained (25/50/75/100)
        path: full path to save .pt file
               e.g. "checkpoints/batg_B50_epoch010.pt"
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[checkpoint] Saved → {path}")


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    device: str = "cpu") -> dict:
    """Load checkpoint and restore model (and optionally optimizer) state.
    
    Args:
        path: path to .pt checkpoint file
        model: model instance to load weights into
        optimizer: optimizer to restore state (pass None to skip)
        device: 'cuda' or 'cpu'
    
    Returns:
        state dict (so caller can read epoch, auc, budget etc.)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    
    epoch = state.get("epoch", "?")
    auc   = state.get("auc", "?")
    budget = state.get("budget", "?")
    print(f"[checkpoint] Loaded ← {path}  (epoch={epoch}, auc={auc}, budget={budget})")
    return state


def latest_checkpoint(checkpoint_dir: str, prefix: str = "batg") -> str | None:
    """Find the most recent checkpoint file in a directory.
    
    Useful at start of training: if a checkpoint exists, resume from it.
    
    Args:
        checkpoint_dir: folder to search
        prefix: filename prefix to filter by (default 'batg')
    
    Returns:
        path to latest .pt file, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith(prefix) and f.endswith(".pt")
    ]
    
    if not files:
        return None
    
    files.sort()  # lexicographic sort works if filenames include epoch number zero-padded
    latest = os.path.join(checkpoint_dir, files[-1])
    print(f"[checkpoint] Found latest: {latest}")
    return latest


if __name__ == "__main__":
    # quick smoke test — save a dummy checkpoint, reload it
    import torch.nn as nn

    dummy_model = nn.Linear(1024, 1)
    dummy_opt   = torch.optim.Adam(dummy_model.parameters())

    save_checkpoint({
        "epoch": 1,
        "model_state_dict": dummy_model.state_dict(),
        "optimizer_state_dict": dummy_opt.state_dict(),
        "loss": 0.42,
        "auc": 0.55,
        "budget": 50,
    }, path="checkpoints/test_checkpoint.pt")

    state = load_checkpoint(
        "checkpoints/test_checkpoint.pt",
        dummy_model, dummy_opt, device="cpu"
    )
    print("Smoke test passed. epoch =", state["epoch"], "auc =", state["auc"])

    # cleanup
    os.remove("checkpoints/test_checkpoint.pt")
    print("Test checkpoint removed.")