import os
import time
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

from src import u2net_full, u2net_lite


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    files_path = "images"
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)
    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])

    weights_path = "inference/model_499.pth"
    threshold = 0.5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(320, antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    model = u2net_full()
    weights = torch.load(weights_path, map_location='cpu')
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)
    model.to(device)
    print("start eval")
    model.eval()
    for index, file_name in enumerate(files_name):
        origin_img = cv2.cvtColor(cv2.imread(files_path + '/{}.png'.format(file_name), flags=cv2.IMREAD_COLOR),
                                  cv2.COLOR_BGR2RGB)

        h, w = origin_img.shape[:2]
        img = data_transform(origin_img)
        img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]

        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)
            t_start = time_synchronized()
            pred = model(img)
            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start))
            pred = torch.squeeze(pred).to("cpu").numpy()
            pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            pred_mask = np.where(pred > threshold, 1, 0)
            mask_img = np.array(torch.ones((h, w, 1)) * 255, dtype=np.uint8)
            seg_img = mask_img * pred_mask[..., None]
            cv2.imwrite("results/{}.png".format(file_name),
                        cv2.cvtColor(seg_img.astype(np.uint8), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
