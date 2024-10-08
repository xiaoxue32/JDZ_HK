import math
import time
import datetime
import torch
from tqdm import tqdm
from torch.nn import functional as F
import train_utils.distributed_utils as utils


def criterion(inputs, target):
    losses = [F.binary_cross_entropy_with_logits(inputs[i], target) for i in range(len(inputs))]
    total_loss = sum(losses)

    return total_loss


def muti_ce_loss(d0, d1, d2, d3, d4, d5, d6, labels_v):
    crossentropy_loss = torch.nn.CrossEntropyLoss()
    loss0 = crossentropy_loss(d0, labels_v.type(dtype=torch.long))
    loss1 = crossentropy_loss(d1, labels_v.type(dtype=torch.long))
    loss2 = crossentropy_loss(d2, labels_v.type(dtype=torch.long))
    loss3 = crossentropy_loss(d3, labels_v.type(dtype=torch.long))
    loss4 = crossentropy_loss(d4, labels_v.type(dtype=torch.long))
    loss5 = crossentropy_loss(d5, labels_v.type(dtype=torch.long))
    loss6 = crossentropy_loss(d6, labels_v.type(dtype=torch.long))
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    bce_loss = torch.nn.BCELoss(size_average=True)
    loss0 = bce_loss(d0, labels_v.type(dtype=torch.long))
    loss1 = bce_loss(d1, labels_v.type(dtype=torch.long))
    loss2 = bce_loss(d2, labels_v.type(dtype=torch.long))
    loss3 = bce_loss(d3, labels_v.type(dtype=torch.long))
    loss4 = bce_loss(d4, labels_v.type(dtype=torch.long))
    loss5 = bce_loss(d5, labels_v.type(dtype=torch.long))
    loss6 = bce_loss(d6, labels_v.type(dtype=torch.long))
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss


def evaluate(model, data_loader, device, epoch, epochs):
    model.eval()
    mae_metric = utils.MeanAbsoluteError()
    f1_metric = utils.F1Score()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc=f'Eval Epoch {epoch}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
            for images, targets in metric_logger.log_every(data_loader, 100, header):
                images, targets = images.to(device), targets.to(device)
                output = model(images)

                # post norm
                # ma = torch.max(output)
                # mi = torch.min(output)
                # output = (output - mi) / (ma - mi)

                mae_metric.update(output, targets)
                f1_metric.update(output, targets)
                pbar.set_postfix(**{'mae_metric': str(mae_metric).replace("MAE: ", ""),
                                    'f1_metric': str(f1_metric).replace("maxF1: ", "")})
                pbar.update(1)

            mae_metric.gather_from_all_processes()
            f1_metric.reduce_from_all_processes()
    return mae_metric, f1_metric


def train_one_epoch(model, optimizer, data_loader, device, epoch, epochs, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    with tqdm(total=len(data_loader), desc=f'Train Epoch {epoch}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image, target = image.to(device), target.to(device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output = model(image)
                loss = criterion(output, target)
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            lr_scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss.item(), lr=lr)
            pbar.set_postfix(**{'train_loss': metric_logger.meters["loss"].global_avg,
                                'lr': lr})
            pbar.update(1)
    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
    params_group = [{"params": [], "weight_decay": 0.},  # no decay
                    {"params": [], "weight_decay": weight_decay}]  # with decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
            params_group[0]["params"].append(param)  # no decay
        else:
            params_group[1]["params"].append(param)  # with decay

    return params_group
