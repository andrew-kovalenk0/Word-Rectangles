import numpy as np
import cv2
from sklearn.cluster import KMeans

if __name__ == '__main__':
    result = cv2.imread('test_images/image.png')
    image = cv2.imread('test_images/image.png', 0)

    dim = (int(2480/4), int(3508/4))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    result = cv2.resize(result, dim, interpolation=cv2.INTER_AREA)

    image[image > 140] = 255
    image = cv2.bitwise_not(image)
    image = cv2.dilate(image, np.ones((3, 3)), iterations=1)
    image = cv2.adaptiveThreshold(image, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 20)

    counters = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counters = counters[0] if len(counters) == 2 else counters[1]
    centers = []
    for counter in counters:
        M = cv2.moments(counter)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers.append([cX, cY])
        x, y, w, h = cv2.boundingRect(counter)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 1)

    model = KMeans(n_clusters=7)
    model.fit(centers)
    for i in model.cluster_centers_:
        cv2.circle(result, (int(i[0]), int(i[1])), 2, (255, 0, 0), -1)
    for counter in counters:
        M = cv2.moments(counter)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers.append([cX, cY])
        cv2.circle(result, (cX, cY), 2, (0, 255, 0), -1)

    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
