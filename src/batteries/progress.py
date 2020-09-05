import sys
from functools import partial
from tqdm import tqdm


tqdm = partial(
    tqdm,
    file=sys.stdout,
    bar_format="{desc}: {percentage:3.0f}%|{bar:20}{r_bar}",
    leave=True,
)
