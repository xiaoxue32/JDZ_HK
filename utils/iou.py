import cv2


# 计算交并比
def compute_IOU(A, B):
    intersection = (A * B).sum()
    union = A.sum() + B.sum() - intersection
    IOU = intersection / union

    return IOU


def main():
    GT = cv2.imread("D:/U2-Net/dataset/DUTS-TE/DUTS-TE-Mask/385.png") / 255
    P1 = cv2.imread("D:/U2-Net/results/385.png") / 255
    print(compute_IOU(P1, GT))


if __name__ == '__main__':
    main()
