import os
import csv
import datetime


class CSVLogger:
    """Logs experiment metrics to a timestamped CSV file.
    
    Usage in train.py:
        logger = CSVLogger("results/tables", prefix="batg_B50")
        logger.log(epoch=1, auc=0.55, loss=0.42, flops_pct=50.0, fps=24.3)
    
    Creates: results/tables/batg_B50_2026-07-03_14-22.csv
    """

    def __init__(self, log_dir: str, prefix: str = "batg") -> None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.path = os.path.join(log_dir, f"{prefix}_{timestamp}.csv")
        self._header_written = False
        print(f"[logger] Logging to → {self.path}")

    def log(self, **kwargs) -> None:
        """Log one row. Pass any keyword args as columns.
        
        First call defines the column headers.
        Subsequent calls must use the same keys.
        """
        write_header = not self._header_written

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(kwargs.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(kwargs)


def print_log(msg: str, also_print: bool = True) -> str:
    """Return a timestamped log string, optionally print it."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line)
    return line


if __name__ == "__main__":
    logger = CSVLogger("results/tables", prefix="test_run")
    for ep in range(3):
        logger.log(epoch=ep, auc=0.5 + ep * 0.1, loss=1.0 - ep * 0.2, budget=50)
    print_log("Logger smoke test passed.")
    # cleanup
    os.remove(logger.path)
    print("Test CSV removed.")