import numpy as np

# Define tensors I and D
I = np.array([[[[ 1,  2],
                [ 3,  4]],

               [[ 5,  6],
                [ 7,  8]]],

              [[[ 9, 10],
                [11, 12]],

               [[13, 14],
                [15, 16]]]])

D = np.array([[1, 2],
              [3, 4]])

# Perform einsum calculation
J = np.einsum('pqrs,rs->pq', I, D)

# Print the result
print("Result of einsum calculation:")
print(J)
