import numpy as np
from tqdm import tqdm

def default_ransac(POINTS, R, EPSILON, SIGMA, CONFIDENCE=0.99, INLIER_THRESHOLD=0.01, MAX_PLANE=1, valid_mask=None):
    assert(POINTS.shape[1] == 3)
    assert(MAX_PLANE > 0), "MAX_PLANE must be greater than 0"
    N = POINTS.shape[0]

    TOLERANCE = (np.log(R/EPSILON) - np.log(SIGMA/EPSILON) - 0.5 * np.log(2*np.pi)) / (0.5 / SIGMA**2)
    assert TOLERANCE > 0, "TOLERANCE must be positive, reduce the value of SIGMA"
    TOLERANCE = np.sqrt(TOLERANCE)
    print("TOLERANCE", TOLERANCE)

    ITERATION = int(np.log(1 - CONFIDENCE) / np.log(1 - (INLIER_THRESHOLD)**3))
    print("ITERATION", ITERATION)

    information = np.zeros(MAX_PLANE+1, dtype=float)
    mask = np.zeros((MAX_PLANE+1, N), dtype=int)
    plane = np.zeros((MAX_PLANE+1, 4), dtype=float)
    availability_mask = np.ones(N, dtype=bool)
    if valid_mask is not None:
        availability_mask = valid_mask
    
    # O
    information[0] = N * 3 * np.log(R/EPSILON)

    # nP + 0
    for plane_cnt in range(1, MAX_PLANE+1):
        BEST_INLIERS_CNT = 0
        BEST_INLIERS_MASK = np.zeros(N, dtype=bool)
        BEST_ERROR = np.zeros(N, dtype=float)
        BEST_PLANE = np.zeros(4, dtype=float)

        available_index = np.linspace(0, N-1, N, dtype=int)
        available_index = np.where(availability_mask)[0]
        if (availability_mask).sum() < 3:
            break

        for _ in tqdm(range(ITERATION)):
            # Get 3 random points
            idx = np.random.choice(available_index, 3, replace=False)

            # Get the normal vector and distance
            A = POINTS[idx[0]]
            B = POINTS[idx[1]]
            C = POINTS[idx[2]]

            AB = B - A
            AC = C - A
            normal = np.cross(AB, AC)
            normal = normal / (np.linalg.norm(normal) + 1e-6)
            distance = -np.dot(normal, A)

            # Count the number of inliers
            inliers = 0
            error = np.abs(np.dot(POINTS, normal.T)+distance)
            trial_mask = error < TOLERANCE
            trial_mask = trial_mask & availability_mask
            trial_cnt = np.sum(trial_mask)

            if trial_cnt > BEST_INLIERS_CNT:
                BEST_INLIERS_CNT = trial_cnt
                BEST_INLIERS_MASK = trial_mask
                BEST_PLANE = np.concatenate((normal, [distance]))
                BEST_ERROR = error
        
        information[plane_cnt] =  information[plane_cnt-1]
        information[plane_cnt] -= N * np.log(plane_cnt) # Remove previous mask 
        information[plane_cnt] += N * np.log(plane_cnt+1) # New Mask that classify points
        information[plane_cnt] += 3 * np.log(R/EPSILON) # New Plane
        total_error = np.sum(BEST_ERROR[BEST_INLIERS_MASK]**2) * 0.5 / SIGMA**2
        information[plane_cnt] += total_error + 0.5 * np.log(2*np.pi) * BEST_INLIERS_CNT + np.log(SIGMA/EPSILON) * BEST_INLIERS_CNT
        information[plane_cnt] -= BEST_INLIERS_CNT * np.log(R/EPSILON) # Information saving from inliers

        tmp_mask = mask[plane_cnt-1].copy()
        tmp_mask[BEST_INLIERS_MASK] = plane_cnt
        mask[plane_cnt] = tmp_mask
        plane[plane_cnt] = BEST_PLANE

        availability_mask[BEST_INLIERS_MASK] = 0
    
    return information, mask, plane