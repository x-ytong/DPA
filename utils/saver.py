import os
import shutil
import torch
from collections import OrderedDict
import glob


class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run')
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='unet_checkpoint'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, (filename + '_' + self.args.target + '.pth.tar'))
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))

            shutil.copyfile(filename, os.path.join(self.directory,
                                                       ('unet_model_best' + '_' + self.args.target + '.pth.tar')))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()

