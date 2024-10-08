import cv2

if __name__ == '__main__':
    # for i in range(1, 2936):
    #     img = cv2.imread('../mask/{}.jpg'.format(i))
    #     img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     ret, mask = cv2.threshold(img2gray, 127, 255, cv2.THRESH_BINARY)
    #     cv2.imwrite('../output/{}.jpg'.format(i), mask)
    # print("处理完毕")
    img = cv2.imread("67.png")
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 2, 255, cv2.THRESH_BINARY)
    cv2.imwrite('O1.png', mask)