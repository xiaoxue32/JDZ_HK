from thop import profile
import torch
import math
from src import u2net_full,u2net_lite

model = u2net_lite()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
input = torch.zeros((1, 3, 224, 224)).to(device)
flops, params = profile(model.to(device), inputs=(input,))

print("参数量：", params)
print("FLOPS：", flops/(pow(10,9)))