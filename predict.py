import argparse
import os
from PIL import Image
from unet import UNet
from dataloaders.utils import *
import torch
import torch.nn as nn
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None


parser = argparse.ArgumentParser(description="Prediction")
parser.add_argument('--inputpath', default='None', type=str,
                    help='path of input images')
parser.add_argument('--outputpath', default='None', type=str,
                    help='path of output results')
parser.add_argument('--modelname', default='None', type=str,
                    help='name of model')


def main():
    args = parser.parse_args()

    numclass = 24 + 1
    pch = 512
    srd = int(pch / 2)
    gap = int((pch - srd) / 2)
    model = UNet(n_channels=4, n_classes=numclass, bilinear=True)

    checkpoint = torch.load(args.modelname, map_location=lambda storge, loc: storge.cuda(0))
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()

    if args.modelname[5:-8] == 'wuhan':
        normalize = transforms.Normalize(mean=[0.537, 0.375, 0.374, 0.370],
                                         std=[0.239, 0.237, 0.230, 0.229])  # wuhan
    elif args.modelname[5:-8] == 'chengdu':
        normalize = transforms.Normalize(mean=[0.582, 0.359, 0.411, 0.372],
                                         std=[0.233, 0.244, 0.229, 0.229])  # chengdu
    elif args.modelname[5:-8] == 'shanghai':
        normalize = transforms.Normalize(mean=[0.527, 0.430, 0.432, 0.406],
                                         std=[0.284, 0.269, 0.263, 0.260])  # shanghai
    elif args.modelname[5:-8] == 'beijing':
        normalize = transforms.Normalize(mean=[0.520, 0.391, 0.416, 0.385],
                                         std=[0.247, 0.255, 0.241, 0.236])  # beijing
    elif args.modelname[5:-8] == 'guangzhou':
        normalize = transforms.Normalize(mean=[0.590, 0.286, 0.323, 0.308],
                                         std=[0.238, 0.238, 0.226, 0.225])  # guangzhou

    for name in os.listdir(args.inputpath):
        image = Image.open(args.inputpath + name).convert('CMYK')
        if args.modelname[5:-8] == 'chengdu' or args.modelname[5:-8] == 'shanghai':
            W, H = image.size
            image = image.resize((int(W*3/4), int(H*3/4)), resample=Image.NEAREST)
        w, h = image.size
        pad2 = int(((w - srd) // srd + 1) * srd + pch - w - (pch - srd) / 2)
        pad4 = int(((h - srd) // srd + 1) * srd + pch - h - (pch - srd) / 2)
        tra = transforms.ToTensor()
        pad = nn.ZeroPad2d(padding=(gap, pad2, gap, pad4))
        image = pad(tra(image))
        hNum, wNum = int((image.size()[1] - pch) / srd + 1), int((image.size()[2] - pch) / srd + 1)
        classMap = np.zeros((image.size()[1], image.size()[2]))
        for i in range(hNum):
            for j in range(wNum):
                patch = image[:, i * srd:i * srd + pch, j * srd:j * srd + pch]
                patch = normalize(patch).unsqueeze_(dim=0)
                with torch.no_grad():
                    output = model(patch.cuda())
                    output = torch.max(output[:, 1:, :, :], 1)[1].detach().cpu().numpy().squeeze(0)
                    output = output[gap:gap + srd, gap:gap + srd]

                classMap[gap + i * srd:gap + srd + i * srd, gap + j * srd:gap + srd + j * srd] = output
        classMap = decode_seg_map_sequence(classMap[gap:gap + h, gap:gap + w])
        classMap = Image.fromarray(classMap.astype('uint8'), 'RGB')
        if args.modelname[5:-8] == 'chengdu' or args.modelname[5:-8] == 'shanghai':
            classMap = classMap.resize((W, H), resample=Image.NEAREST)
        classMap.save(args.outputpath + '{}_result.tif'.format(name[0:-4]))
        print('Image:{}, done.'.format(name))


if __name__ == '__main__':
    main()

