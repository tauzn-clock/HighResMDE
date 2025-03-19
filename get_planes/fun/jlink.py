import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

SIGMA = 0.00
PT_PER_LINE = 50
MODELS = 8000
THRESHOLD = 0.001

pts = []

for i in range(5):
    for j in range(PT_PER_LINE):
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.randn() * SIGMA + 0.16
        # Rotate by i * 2 * pi / 5
        x, y = x * np.cos(i * 2 * np.pi / 5) - y * np.sin(i * 2 * np.pi / 5), x * np.sin(i * 2 * np.pi / 5) + y * np.cos(i * 2 * np.pi / 5)
        pts.append((x, y))

for _ in range(0):
    pts.append((np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)))

# Plot the points
pts = np.array(pts)
plt.scatter(pts[:, 0], pts[:, 1])

# Save the img
plt.savefig('jlink.png')

mss = np.zeros((len(pts),MODELS), dtype=np.bool)

for i in range(MODELS):
    # Randomly select 2 points
    p1, p2 = np.random.choice(len(pts), 2, replace=False)
    # Calculate the line
    grad = (pts[p2, 1] - pts[p1, 1]) / (pts[p2, 0] - pts[p1, 0])
    c = pts[p1, 1] - grad * pts[p1, 0]

    grad_normal = -1 / (grad + 1e-10)
    c_normal = pts[:, 1] - grad_normal * pts[:, 0]
    x_int = (c_normal - c) / (grad - grad_normal)
    y_int = grad * x_int + c
    dist = np.sqrt((pts[:, 0] - x_int)**2 + (pts[:, 1] - y_int)**2)
    mss[:, i] = dist < THRESHOLD

print(mss.shape)

plt.imsave('jlink_mss.png', mss, cmap='gray')

def get_jaccard(a, b):
    union = np.logical_or(a, b).sum()
    intersection = np.logical_and(a, b).sum()
    return intersection / union

parent = np.linspace(0, len(pts)-1, len(pts), dtype=np.int32)
index = np.linspace(0, len(pts)-1, len(pts), dtype=bool)

for _ in range(len(pts)):
    best_jaccard = 0
    best_a = -1
    best_b = -1

    for idx_i, i in enumerate(index):
        if not i:
            continue
        for idx_j in range(idx_i+1, len(index)):
            j = index[idx_j]
            if not j:
                continue
            jaccard = get_jaccard(mss[parent[idx_i],:], mss[parent[idx_j],:])
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_a = idx_i
                best_b = idx_j

    
    print(f'Best jaccard: {best_jaccard}')
    if best_jaccard < 0.5:
        break
    
    parent[best_b] = parent[best_a]
    mss[parent[best_a], :] = np.logical_and(mss[parent[best_a], :], mss[parent[best_b], :])
    index[best_b] = False
    
for i in range(parent.shape[0]):
    cur = parent[i]
    while parent[cur] != cur:
        cur = parent[cur]
    parent[i] = cur

# Visualise best fit lines
features, counts = np.unique(parent, return_counts=True)
features = features[np.argsort(counts)[::-1]]
counts = counts[np.argsort(counts)[::-1]]

print(features)
print(counts)

for i in range(len(features)):
    if counts[i] < 10:
        break
    # Find best fit line for each mask
    x = pts[parent == features[i], 0]
    y = pts[parent == features[i], 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print(f'Line {i}: y = {m}x + {c}')
    plt.plot(x, m*x + c)

plt.savefig('jlink_fit.png')
plt.imsave("mss.png", mss, cmap='gray')