import torch
import math
import numpy as np
import torch.nn.functional as F


def pseudolabel(target, epoch, args):
    ratio = ((epoch + 1) / args.epochs) * args.factor

    prob = F.softmax(target, dim=1)
    entp = torch.mul(prob, torch.log2(prob + 1e-30))
    entp = torch.div(torch.sum(entp, dim=1), - math.log2(args.numclass))

    num = math.floor(entp.numel() * ratio)
    idx = np.argpartition(entp.detach().cpu().numpy().reshape(-1), num)[num:]

    trlb = torch.max(target, dim=1)[1].reshape(-1)
    trlb[idx] = 0
    trlb = trlb.reshape(entp.size())

    return trlb

