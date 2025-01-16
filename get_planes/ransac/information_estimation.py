import numpy as np

def default_ransac(POINTS, R, EPSILON, SIGMA, CONFIDENCE=0.99, INLIER_THRESHOLD=0.01, MAX_PLANE=1):
    assert(POINTS.shape[1] == 3)

    information = [0 for _ in range(MAX_PLANE+1)]
    N = POINTS.shape[0]
    
    # O
    information[0] = N * 3 * np.log(R/EPSILON)

    # 1P + 0

    information[1] += N * np.log(2) # Mask that classify points
    information[1] += 1 * 3 * np.log(R/EPSILON) # Plane

    ITERATION = int(np.log(1 - CONFIDENCE) / np.log(1 - (INLIER_THRESHOLD)**3))

    BEST_INLIERS = 0
    BEST_PLANE = None
    TOLERANCE = (np.log(R/EPSILON) - np.log(SIGMA/EPSILON) - 0.5 * np.log(2*np.pi)) / (0.5 / SIGMA**2)
    assert TOLERANCE > 0, "TOLERANCE must be positive, reduce the value of SIGMA"
    TOLERANCE = np.sqrt(TOLERANCE)
    print("TOLERANCE", TOLERANCE)

    for _ in range(ITERATION):
        # Get 3 random points
        idx = np.random.randint(0, N, 3)

        # Get the normal vector and distance
        A = POINTS[idx[0]]
        B = POINTS[idx[1]]
        C = POINTS[idx[2]]

        AB = B - A
        AC = C - A
        normal = np.cross(AB, AC)
        normal = normal / np.linalg.norm(normal)
        distance = np.dot(normal, A)

        # Count the number of inliers
        inliers = 0
        test = (np.dot(POINTS, normal.T)-distance)
        cnt = np.abs(np.dot(POINTS, normal.T)-distance) < TOLERANCE
        cnt = np.sum(cnt)

        if cnt > BEST_INLIERS:
            BEST_INLIERS = cnt
            BEST_PLANE = (normal, distance)

    print("BEST_INLIERS", BEST_INLIERS)
    print("BEST_PLANE", BEST_PLANE)
    
    print(information)
    return 