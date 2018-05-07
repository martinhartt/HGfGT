import torch


def apply_cuda(obj):
    return obj.cuda() if torch.cuda.is_available() else obj
