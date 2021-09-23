import numpy as np
import cv2
from sklearn.cluster import KMeans

if __name__ == '__main__':
    result = cv2.imread('test_images/image_2.png')
    image = cv2.imread('test_images/image_2.png', 0)

    dim = (int(2480/4), int(3508/4))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    result = cv2.resize(result, dim, interpolation=cv2.INTER_AREA)

    image[image > 133] = 255
    image = cv2.bitwise_not(image)
    image = cv2.dilate(image, np.ones((3, 3)), iterations=1)
    image = cv2.adaptiveThreshold(image, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 20)

    # ------------Counters------------
    counters = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counters = counters[0] if len(counters) == 2 else counters[1]
    data = np.empty((0, 2), int)
    for counter in counters:
        area = cv2.contourArea(counter)
        if area > 50:
            M = cv2.moments(counter)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            data = np.append(data, [[cX, cY]], axis=0)

    # ------------Kmeans clustering------------
    k = 17
    n = data.shape[0]
    c = data.shape[1]
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    centers_new = np.random.randn(k, c) * std + mean

    clusters = np.zeros(n)
    centers_old = np.zeros(centers_new.shape)
    distances = np.zeros((n, k))
    error = np.linalg.norm(centers_new - centers_old)

    while error >= 0.05:
        for i in range(k):
            distances[:, i] = np.linalg.norm(data - centers_new[i], axis=1)
        clusters = np.argmin(distances, axis=1)
        centers_old = centers_new.copy()
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == i], axis=0)
        error = np.linalg.norm(centers_new - centers_old)

    # ------------Colors for rectangles------------
    colors = np.zeros((1, 3))
    for i in range(k-1):
        buf_color = []
        for j in range(3):
            buf_color = np.append(buf_color, np.random.randint(100, 255, 1, dtype=int), axis=0)
        colors = np.append(colors, [buf_color], axis=0)

    # ------------Rectangles------------
    subt = 0
    for i, counter in enumerate(counters):
        area = cv2.contourArea(counter)
        if area > 50:
            x, y, w, h = cv2.boundingRect(counter)
            cv2.rectangle(result, (x, y), (x + w, y + h), colors[clusters[i-subt]], 1)
        else:
            subt += 1

    # ------------Centers of clusters------------
    for i, center in enumerate(centers_new):
        cv2.circle(result, tuple(map(int, center)), 3, (0, 255, 0), -1)

    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
