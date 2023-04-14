import argparse
import numpy as np
from tqdm import tqdm

from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.pseudo_label import pseudolabel
from dataloaders.datasets import gid24
from dataloaders.datasets import target

import torch
import torch.optim
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from unet import UNet


parser = argparse.ArgumentParser(description="DPA with U-Net Training")
# data path
parser.add_argument('--source_dir', default='None', type=str,
                    help='path of source data')
parser.add_argument('--target_dir', default='None', type=str,
                    help='path of target data')
# training hyper params
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train (default: 120)')
parser.add_argument('--start-epoch', type=int, default=0,
                    metavar='N', help='start epochs (default:0)')
parser.add_argument('--batch-size', type=int, default=16,
                    metavar='N', help='batch size for each branch')
# optimizer params
parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr-scheduler', type=str, default='poly',
                    choices=['poly', 'step', 'cos'],
                    help='lr scheduler mode: (default: poly)')
parser.add_argument('--loss-type', type=str, default='ce',
                    choices=['ce', 'focal'],
                    help='loss func type (default: ce)')
parser.add_argument('--momentum', type=float, default=0.9,
                    metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-5,
                    metavar='M', help='w-decay (default: 1e-5)')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='whether use nesterov')
parser.add_argument('--amp', action='store_true', default=True,
                    help='Use mixed precision')
# distributed training
parser.add_argument('--workers', type=int, default=8,
                    metavar='N', help='dataloader threads')
parser.add_argument('--gpu', type=int, default='0',
                    help='use which gpu to train, must be a \
                    comma-separated list of integers only (default=0)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://0.0.0.0', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# target
parser.add_argument('--target', default='None', type=str,
                    choices=['beijing', 'chengdu', 'guangzhou', 'shanghai', 'wuhan'],
                    help='name of target city')
parser.add_argument('--factor', default=0.5, type=float,
                    help='ratio of pseudo-label')


def main():
    args = parser.parse_args()
    args.numclass = 24+1

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    print(args)

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training.".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        print("Start distributed training. Use GPU: {}.".format(args.gpu))

    # Define network
    # n_channels=3 for RGB images
    model = UNet(n_channels=4, n_classes=args.numclass, bilinear=True)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            print("Move to GPU: {}".format(args.gpu))
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        print("Move to GPU: {}".format(args.gpu))

    # Define Criterion
    weight = np.array([0, 22.7467, 24.6431,  3.2611, 12.7603, 37.6410, 15.1353, 27.0354, 49.7190,
                       38.1156, 45.3688, 19.8079, 15.6410, 14.6416, 35.8430, 34.6233, 49.8047,
                       45.1986, 17.1660, 50.2263, 50.1459, 22.2144, 46.8094, 48.8383, 48.9092])
    weight = torch.from_numpy(weight.astype(np.float32))

    criterion = SegmentationLosses(weight=weight, ignore_index=0, gpu=args.gpu).build_loss(mode=args.loss_type)

    # Define Optimizer
    if args.distributed:
        optimizer = torch.optim.SGD(model.module.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

    # Define Dataloader
    target_set = target.TRGData(args)

    if args.distributed:
        target_sampler = torch.utils.data.distributed.DistributedSampler(target_set)
    else:
        target_sampler = None

    target_loader = torch.utils.data.DataLoader(
        target_set, batch_size=args.batch_size, shuffle=(target_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=target_sampler)

    # Define lr Scheduler
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(target_loader))

    # Automatic Mixed Precision
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    cudnn.benchmark = True
    best_pred = 0.0

    trainer = Trainer(args, model, criterion, optimizer, scheduler, target_loader, grad_scaler, best_pred)

    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):

        train_set = gid24.GIDData(args, split='train')
        val_set = gid24.GIDData(args, split='val')

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        trainer.training(epoch, train_loader)
        trainer.validation(epoch, val_loader)

    trainer.writer.close()


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, scheduler, target_loader, grad_scaler, best_pred):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.target_loader = target_loader
        self.grad_scaler = grad_scaler

        # Define Evaluator
        self.evaluator = Evaluator(self.args.numclass)
        self.best_pred = best_pred

    def training(self, epoch, train_loader):
        train_loss = 0.0
        self.model.train()
        trbar = tqdm(self.target_loader)
        num_img_tr = len(self.target_loader)

        for i, data in enumerate(zip(train_loader, trbar)):
            source_set, target_set = data
            srimg, srlbl = source_set['image'], source_set['label']
            trimg = target_set['image']

            if self.args.gpu is not None:
                trimg = trimg.cuda(self.args.gpu, non_blocking=True)
                srimg = srimg.cuda(self.args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                srlbl = srlbl.cuda(self.args.gpu, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.args.amp):
                output = self.model(torch.cat([srimg, trimg], dim=0))
                srout, trout = output.chunk(2, dim=0)
                class_loss_sr = self.criterion(srout, srlbl)

                trlbl = pseudolabel(trout.detach(), epoch, self.args)
                class_loss_tr = self.criterion(trout, trlbl)

                loss = class_loss_sr + class_loss_tr

            self.optimizer.zero_grad()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)

            train_loss += loss.item()
            trbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + trimg.data.shape[0]))
        print('Loss: %.3f' % train_loss)

    def validation(self, epoch, val_loader):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(val_loader)
        test_loss = 0.0

        with torch.no_grad():
            for i, sample in enumerate(tbar):
                image, label = sample['image'], sample['label']
                if self.args.gpu is not None:
                    image = image.cuda(self.args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    label = label.cuda(self.args.gpu, non_blocking=True)

                output = self.model(image)
                loss = self.criterion(output, label)

                test_loss += loss.item()
                tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
                pred = output.data.cpu().numpy()
                label = label.cpu().numpy()
                pred = np.argmax(pred[:, 1:, :, :], axis=1)
                # Add batch sample into evaluator
                self.evaluator.add_batch(label, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        is_best = False
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred

        if self.args.distributed:
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
        else:
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


if __name__ == '__main__':
    main()
