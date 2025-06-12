import numpy as np
from numpy.typing import NDArray

def caret_to_exponent(expr_str: str) -> str:
    # first check if the string even contains a caret
    if "^" not in expr_str:
        return expr_str
    else:
        return expr_str.replace("^", "**")
    
def normalize(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    magnitude = np.linalg.norm(vector)
    return vector / magnitude