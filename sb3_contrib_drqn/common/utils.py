from typing import Union

from collections import deque
import numpy as np

def safe_n_mean(arr: Union[np.ndarray, list, deque], n: int) -> float:
    if len(arr) == 0:
        return np.nan 
    elif len(arr) < n:
        return float(np.mean(arr))
    else:
        if isinstance(arr, deque):
            arr = list(arr)  # deque를 list로 변환하여 슬라이싱 지원
        return float(np.mean(arr[-n:]))