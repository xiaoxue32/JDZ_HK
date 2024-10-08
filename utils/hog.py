import cv2
from skimage.feature import hog
from skimage import io


def get_hog_f(img, dim):
    if dim == 3780:
        fd, hog_image = hog(img, orientations=6, pixels_per_cell=(6, 6), cells_per_block=(2, 2), visualize=True)

    elif dim == 2520:
        fd, hog_image = hog(img, orientations=6, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

    elif dim == 1680:
        fd, hog_image = hog(img, orientations=4, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

    elif dim == 756:
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)

    elif dim == 504:
        fd, hog_image = hog(img, orientations=6, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)

    else:
        print('HOG Dimension not defined.')

    return fd, hog_image


if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread('data/images/img.png'), cv2.COLOR_BGR2GRAY)
    fd, hog_image = get_hog_f(img, 3780)
    # cv2.imshow("hog_image", hog_image)
    io.imshow(hog_image)
    io.show()
    cv2.waitKey(0)
    cv2.imwrite("data/hog.png", hog_image)
