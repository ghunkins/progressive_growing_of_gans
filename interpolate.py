"""
Programmer: Gregory D. Hunkins

Institution: University of Rochester
"""
import numpy as np

def slerp(val, low, high):
    """
	Interpolates two Numpy arrays based on the ratio defined
	by val.
    Code from https://github.com/soumith/dcgan.torch/issues/14
    """
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high