import numpy as np
import time

# Define tensors I and D
size = 100
I = np.zeros((size, size, size, size), dtype=np.int32)
D = np.zeros((size, size), dtype=np.int32)

# Initialize tensor I with the same formula: I(p,q,r,s) = p + q + r + s
for p in range(size):
    for q in range(size):
        for r in range(size):
            for s in range(size):
                I[p, q, r, s] = p + q + r + s

# Initialize tensor D with the same formula: D(r,s) = r + s
for r in range(size):
    for s in range(size):
        D[r, s] = r + s

# Perform einsum calculation
start_einsum = time.time()
J = np.einsum('pqrs,rs->pq', I, D)
end_einsum = time.time()

# Print the result (printing such a large matrix might not be practical, so we just print a part of it)
print("Result of einsum calculation (first 10x10 block):")
print(J[:10, :10])

# Calculate time taken for einsum calculation
duration_einsum = (end_einsum - start_einsum) * 1000  # in milliseconds

print("\nEinsum calculation time: {:.2f} ms".format(duration_einsum))