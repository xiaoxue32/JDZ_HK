import cv2

if __name__ == '__main__':
    img = cv2.imread("../P1.jpg")
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('../P1.jpg', img2gray)