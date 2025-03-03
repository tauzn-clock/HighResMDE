from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

IMG_PATH = "/scratchdata/test.jpg"

def full_svd_compress(img, k):
    start = time.time()
    U, S, V = np.linalg.svd(img)
    U_k = U[:, :k]
    S_k = S[:k]
    V_k = V[:k, :]
    print(f"Time taken for k={k}: {time.time()-start}")
    return U_k, S_k, V_k

def randomised_sv_compress(img, k):
    H, W = img.shape

    start = time.time()
    # Generate a random matrix P for projection
    P = np.random.randn(W, k)

    # Project the image onto the subspace spanned by the columns of P
    Z = img @ P
    Q, _ = np.linalg.qr(Z)

    # Compute the SVD of the projected image
    U, S, V = np.linalg.svd(Q.T @ img)
    V = V[:k, :]
    U = Q @ U
    print(f"Time taken for k={k}: {time.time()-start}")

    return U, S, V

img = Image.open(IMG_PATH, 'r')
img = img.convert('L')
img = np.array(img)/255.0
H, W = img.shape

k = [1, 5, 25, 125, 625]

for i in k:
    U, S, V = full_svd_compress(img, i)
    plt.imsave(f"{i}.png", U@np.diag(S)@V, cmap='gray')
    U_c, S_c, V_c = randomised_sv_compress(img, i)    
    plt.imsave(f"{i}_randomised.png", U_c@np.diag(S_c)@V_c, cmap='gray')

    #Plot the singular values
    fig, ax = plt.subplots()
    ax.plot(S)
    ax.plot(S_c)
    ax.legend(["Full SVD", "Randomised SVD"])
    plt.savefig(f"{i}_singular_values.png")