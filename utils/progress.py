# utils/progress.py

from contextlib import contextmanager
from tqdm import tqdm
from tensorflow.keras.callbacks import Callback


def tqdm_bar(iterable, desc="", unit="it", total=None, leave=True, disable=False):
    """
    Wrapper for tqdm to enforce consistent progress bar formatting.

    Parameters:
    - iterable: iterable to wrap
    - desc: progress bar description
    - unit: label for iteration units (e.g., 'pkt', 'file', 'row')
    - total: total expected iterations (optional)
    - leave: whether to keep the bar after completion
    - disable: disable the progress bar entirely (e.g., for logging-only mode)

    Returns:
    - tqdm-wrapped iterable
    """
    return tqdm(iterable, desc=desc, unit=unit, total=total, leave=leave, disable=disable)


class TqdmKerasCallback(Callback):
    """
    Keras Callback that shows tqdm-based progress per epoch.
    """

    def __init__(self, total_epochs, desc="Training (Epochs)", unit="epoch"):
        super().__init__()
        self.total_epochs = total_epochs
        self.pbar = tqdm(total=total_epochs, desc=desc, unit=unit, leave=True)

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()
        

@contextmanager
def single_bar(desc="", total=1, unit="it", leave=True, disable=False):
    """
    Context manager for progress bars where only one update is needed.

    Example:
        with single_bar("Training model") as update:
            model.fit(X, y)
            update()  # fill bar
    """
    bar = tqdm(total=total, desc=desc, unit=unit, leave=leave, disable=disable)
    yield lambda: bar.update(1)
    bar.close()
