import numpy as np
import cv2

if __name__ == '__main__':
    result = cv2.imread('image_2.png')
    image = cv2.imread('image_2.png', 0)
    image = cv2.bitwise_not(image)
    dilate_kernel = np.ones((3, 3))
    image = cv2.dilate(image, dilate_kernel, iterations=4)
    # image[image < 100] = 0
    counters = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counters = counters[0] if len(counters) == 2 else counters[1]
    for counter in counters:
        x, y, w, h = cv2.boundingRect(counter)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imshow('result', result)
    cv2.waitKey()
