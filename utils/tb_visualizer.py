import numpy as np
import os
import time
from . import util
from tensorboardX import SummaryWriter


class TBVisualizer:
    def __init__(self, opt):
        self.opt = opt
        self.save_path = os.path.join(opt.checkpoints_dir, opt.name)

        self.log_path = os.path.join(self.save_path, 'loss_log2.txt')
        self.tb_path = os.path.join(self.save_path, 'summary.json')
        self.writer = SummaryWriter(self.save_path)

        with open(self.log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def __del__(self):
        self.writer.close()

    def display_current_results(self, visuals, it, is_train, save_visuals=False):
        for label, image_numpy in visuals.items():
            sum_name = '{}/{}'.format('Train' if is_train else 'Test', label)
            if type(image_numpy) is list:
                if len(image_numpy) == 0 or len(image_numpy[0]) == 0:
                    continue
                images = (np.stack(image_numpy).transpose(0, 3, 1, 2))/255.0
                self.writer.add_images(sum_name, images, it)
            else:
                self.writer.add_image(sum_name, image_numpy.transpose((2, 0, 1)), it)

        self.writer.export_scalars_to_json(self.tb_path)

    def plot_scalars(self, scalars, it, is_train):
        for label, scalar in scalars.items():
            sum_name = '{}/{}'.format('Train' if is_train else 'Test', label)
            self.writer.add_scalar(sum_name, scalar, it)

    def print_current_train_errors(self, epoch, i, iters_per_epoch, errors, t, visuals_were_stored):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        visuals_info = "v" if visuals_were_stored else ""
        message = '%s (T%s, epoch: %d, it: %d/%d, t/smpl: %.3fs) ' % (log_time, visuals_info, epoch, i, iters_per_epoch, t)
        for k, v in errors.items():
            message += '%s:%.3f ' % (k, v)

        print(message)
        with open(self.log_path, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_validate_errors(self, epoch, errors, t):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = '%s (V, epoch: %d, time_to_val: %ds) ' % (log_time, epoch, t)
        for k, v in errors.items():
            message += '%s:%.3f ' % (k, v)

        print(message)
        with open(self.log_path, "a") as log_file:
            log_file.write('%s\n' % message)

    def save_images(self, visuals):
        for label, image_numpy in visuals.items():
            image_name = '%s.png' % label
            save_path = os.path.join(self.save_path, "samples", image_name)
            util.save_image(image_numpy, save_path)
