import os
import time
import datetime
from typing import Union, List

import torch
from torch.utils import data
from tensorboardX import SummaryWriter
from src import u2net_full, u2net_lite
from train_utils import train_one_epoch, evaluate, get_params_groups, create_lr_scheduler
from my_dataset import DUTSDataset
import transforms as T


class SODPresetTrain:
    def __init__(self, base_size: Union[int, List[int]], crop_size: int,
                 hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            # T.Resize(base_size, resize_mask=True),
            # T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(hflip_prob),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class SODPresetEval:
    def __init__(self, base_size: Union[int, List[int]], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            # T.Resize(base_size, resize_mask=False),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = args.batch_size
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # 用来保存数据的文件
    writer = SummaryWriter()

    train_dataset = DUTSDataset(args.data_path, train=True, transforms=SODPresetTrain([320, 320], crop_size=288))
    val_dataset = DUTSDataset(args.data_path, train=False, transforms=SODPresetEval([320, 320]))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        pin_memory=True,
                                        collate_fn=train_dataset.collate_fn)

    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      collate_fn=val_dataset.collate_fn)
    model = u2net_full()
    # model = u2net_lite()
    model.to(device)

    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=2)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    current_mae, current_f1 = 1.0, 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch, args.epochs,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        writer.add_scalar(tag="train_loss", scalar_value=mean_loss, global_step=epoch)
        writer.add_scalar(tag="lr", scalar_value=lr, global_step=epoch)
        if args.save:
            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
        else:
            save_file = {"model": model.state_dict()}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            mae_metric, f1_metric = evaluate(model, val_data_loader, device=device, epoch=epoch, epochs=args.epochs)
            mae_info, f1_info = mae_metric.compute(), f1_metric.compute()
            writer.add_scalar(tag="val_MAE", scalar_value=mae_info, global_step=epoch)
            writer.add_scalar(tag="val_maxF1", scalar_value=f1_info, global_step=epoch)

            # save_best
            if current_mae >= mae_info and current_f1 <= f1_info:
                torch.save(save_file, "inference/model_defect.pth")

        if os.path.exists(f"save_weights/model_{epoch - 10}.pth"):
            os.remove(f"save_weights/model_{epoch - 10}.pth")

        torch.save(save_file, f"save_weights/model_{epoch}.pth")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch u2net training")
    parser.add_argument("--data-path", default="./", help="DUTS root")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--save", default=False, help="save all weights")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=500, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--eval-interval", default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--print-freq', default=1000, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument("--amp", action='store_true',
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
