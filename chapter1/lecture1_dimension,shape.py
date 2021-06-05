import tensorflow as tf
import torch as tc
import numpy as np

scalar = np.array(8)
print("----- 1. scalar -----")
print("shape: ", scalar.shape)
print("dimension: ", scalar.ndim)
print()

vector = np.array([1, 2])
print("----- 2. vector -----")
print("shape: ", vector.shape)
print("dimension: ", vector.ndim)
print()

matrix = np.array([[1, 2, 3],
                  [4, 5, 6]])
print("----- 3. matrix -----")
print("shape: ", matrix.shape)
print("dimension: ", matrix.ndim)
print()

tensor = np.array([[[1, 2, 3],
                   [4, 5, 6]],
                   [[1, 2, 3],
                   [4, 5, 6]],
                   [[1, 2, 3],
                   [4, 5, 6]]])
print("----- 4. tensor -----")
print("shape: ", tensor.shape)
print("dimension: ", tensor.ndim)
print()