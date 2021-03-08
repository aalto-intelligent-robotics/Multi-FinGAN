import time
from options.train_options import TrainOptions
from datasets.custom_dataset_data_loader import CustomDatasetDataLoader
from models.models import ModelsFactory
from utils.tb_visualizer import TBVisualizer
from utils import util
import torch
import os
import subprocess


class Train:
    def __init__(self):
        self.opt = TrainOptions().parse()
        self.model = ModelsFactory.get_by_name(self.opt)
        self.tb_visualizer = TBVisualizer(self.opt)
        if self.get_training_data():
            self.setup_train_test_sets()
            self.train()

    def get_training_data(self):
        if not os.path.isdir("data/train_data/") or len(os.listdir("data/train_data/")) == 0 or not os.path.isdir("data/meshes/") or len(os.listdir("data/meshes/")) == 1:
            while True:
                download_data = input("Training data does not exist. Want to download it (y/n)? ")
                if download_data == "y":
                    dir = os.path.dirname(os.path.realpath(__file__)) + "/data/"
                    print("Downloading training data. This will take some time so just sit back and relax.")
                    subprocess.Popen([dir+'download_train_data.sh %s' % dir], shell=True).wait()
                    print("Done downloading training data. Continuing with training.")
                    return True
                elif download_data == "n":
                    print("You chose to not download the data. Terminating training")
                    return False
        else:
            print("Training data exists. Proceeding with training.")
            return True

    def setup_train_test_sets(self):
        data_loader_train = CustomDatasetDataLoader(self.opt, mode='train')
        self.dataset_train = data_loader_train.load_data()
        self.dataset_train_size = len(data_loader_train)
        print('#train images = %d' % self.dataset_train_size)
        data_loader_val = CustomDatasetDataLoader(self.opt, mode='test')
        self.dataset_val = data_loader_val.load_data()
        self.dataset_val_size = len(data_loader_val)
        print('#val images = %d' % self.dataset_val_size)

    def train(self):
        # Here we set the start epoch. It is nonzero only if we continue train or test as the epoch saved for the network
        # we load is used
        start_epoch = self.model.get_epoch()
        self.total_steps = start_epoch * self.dataset_train_size
        self.iters_per_epoch = self.dataset_train_size / self.opt.batch_size
        self.last_display_time = None
        self.last_display_time_val = None
        self.last_save_latest_time = None
        self.last_print_time = time.time()
        self.visuals_per_batch = self.iters_per_epoch//2
        for i_epoch in range(start_epoch, self.opt.nepochs_no_decay + self.opt.nepochs_decay + 1):
            epoch_start_time = time.time()

            # train epoch
            self.train_epoch(i_epoch)
            # print epoch info
            self.print_epoch_info(time.time() - epoch_start_time, i_epoch)
            # update learning rate
            self.update_learning_rate(i_epoch)
            # save model
            self.model.save("latest", i_epoch+1)
            self.display_visualizer_train(i_epoch)
            if (i_epoch) % 5 == 0:  # Only test the network every fifth epoch
                self.test(i_epoch, self.total_steps)

    def print_epoch_info(self, time_epoch, epoch_num):
        print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
              (epoch_num, self.opt.nepochs_no_decay + self.opt.nepochs_decay, time_epoch,
               time_epoch / 60, time_epoch / 3600))

    def update_learning_rate(self, epoch_num):
        if epoch_num > self.opt.nepochs_no_decay:
            self.model.update_learning_rate()

    def train_epoch(self, i_epoch):
        self.model.set_train()
        self.epoch_losses_G = []
        self.epoch_losses_D = []
        self.epoch_scalars = []
        self.epoch_visuals = []
        for i_train_batch, train_batch in enumerate(self.dataset_train):
            iter_start_time = time.time()

            self.model.set_input(train_batch)
            train_generator = self.train_generator(i_train_batch)

            self.model.optimize_parameters(train_generator=train_generator)

            self.total_steps += self.opt.batch_size

            self.bookkeep_epoch_data(train_generator)

            if ((i_train_batch+1) % self.visuals_per_batch == 0):
                self.bookkeep_epoch_visualizations()
                self.display_terminal(
                    iter_start_time, i_epoch, i_train_batch, True)

    def train_generator(self, batch_num):
        return ((batch_num+1) % self.opt.train_G_every_n_iterations) == 0

    def bookkeep_epoch_visualizations(self):
        self.epoch_visuals.append(
            self.model.get_current_visuals())

    def bookkeep_epoch_data(self, train_generator):
        if train_generator:
            self.epoch_losses_G.append(self.model.get_current_errors_G())
            self.epoch_scalars.append(self.model.get_current_scalars())
        self.epoch_losses_D.append(self.model.get_current_errors_D())

    def display_terminal(self, iter_start_time, i_epoch, i_train_batch, visuals_flag):
        errors = self.model.get_current_errors()
        t = (time.time() - iter_start_time) / self.opt.batch_size
        self.tb_visualizer.print_current_train_errors(
            i_epoch, i_train_batch, self.iters_per_epoch, errors, t, visuals_flag)

    def display_visualizer_train(self, total_steps):
        self.tb_visualizer.display_current_results(
            util.concatenate_dictionary(self.epoch_visuals), total_steps, is_train=True, save_visuals=True)
        self.tb_visualizer.plot_scalars(
            util.average_dictionary(self.epoch_losses_G), total_steps, is_train=True)
        self.tb_visualizer.plot_scalars(
            util.average_dictionary(self.epoch_losses_D), total_steps, is_train=True)
        self.tb_visualizer.plot_scalars(
            util.average_dictionary(self.epoch_scalars), total_steps, is_train=True)

    def display_visualizer_test(self, test_epoch_visuals, epoch_num, average_test_results, test_time, total_steps):
        self.tb_visualizer.print_current_validate_errors(
            epoch_num, average_test_results, test_time)
        self.tb_visualizer.plot_scalars(
            average_test_results, epoch_num, is_train=False)
        self.tb_visualizer.display_current_results(
            util.concatenate_dictionary(test_epoch_visuals), total_steps, is_train=False, save_visuals=True)

    def test(self, i_epoch, total_steps):
        val_start_time = time.time()

        self.model.set_eval()
        test_epoch_visuals = []

        iters_per_epoch_val = self.dataset_val_size / self.opt.batch_size
        visuals_per_val_epoch = max(1, round(iters_per_epoch_val//2))
        errors = []
        with torch.no_grad():
            for i_val_batch, val_batch in enumerate(self.dataset_val):
                self.model.set_input(val_batch)
                self.model.forward_G(train=True)
                errors.append(self.model.get_current_errors_G())
                if (i_val_batch+1) % visuals_per_val_epoch == 0:
                    test_epoch_visuals.append(
                        self.model.get_current_visuals())

        average_test_results = util.average_dictionary(errors)
        test_time = (time.time() - val_start_time)
        self.display_visualizer_test(test_epoch_visuals, i_epoch, average_test_results, test_time, total_steps)
        self.model.set_train()


if __name__ == "__main__":
    Train()
