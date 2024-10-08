import os
import cv2
import numpy as np
import csv

files_path = "data/sorce"
assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)
files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
files_num = len(files_name)
with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["number", "images", "width", "height", "scale","length"])
    for index, file_name in enumerate(files_name):
        img = cv2.imread('data/sorce/{}.png'.format(file_name), 0)
        y_indices, x_indices = np.where(img == 255)
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)
        length = len(x_indices)
        width = x_max - x_min
        height = y_max - y_min
        scale = height / width
        if 1.5 < scale < 1.9:
            writer.writerow([index, file_name, width, height, scale, length])

print("done")
