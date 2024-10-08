import cv2
import numpy as np

if __name__ == '__main__':
    image = cv2.imread('566-shadow.png')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV值范围1（红色在HSV色系中不连续）
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])

    # 定义红色的HSV值范围2
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # 找出在该HSV范围内的像素点
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # 计算红色区域的面积（即像素点数）
    area_red = cv2.countNonZero(mask_red)

    # 定义白色的BGR值
    white = [255, 255, 255]

    # 计算白色区域的面积（即像素点数）
    area_white = np.sum(np.all(image == white, axis=2))

    # 计算红色与白色的面积比
    ratio = area_red / area_white if area_white != 0 else 'inf'
    ratio2 = area_red / (area_white + area_red) if area_white != 0 else 'inf'

    print(f"The ratio of the areas of red to white is: {ratio}")

    print(area_red)
    print(area_white)
    print(ratio)
    print(ratio2)
