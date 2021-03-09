import argparse
import os

from utils import util
import torch
import yaml
import sys
from easydict import EasyDict


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--object_mesh_dir', type=str,
                                 default='./data/meshes/ycb_meshes/', help='path to dataset')
        self.parser.add_argument('--grasp_dir', type=str,
                                 default='./data/train_data/graspit_training_grasps/', help='path to dataset')
        self.parser.add_argument(
            '--batch_size', type=int, default=100, help='input batch size')
        self.parser.add_argument(
            '--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('-n', '--name', type=str, default='',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument(
            '--n_threads_test', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument(
            '--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--precomputed_rotations', action='store_true',
                                 help='Use precomputed rotations')
        self.parser.add_argument('-ofct', '--object_finger_contact_threshold', type=float, default=0.01,
                                 help='The threshold for object-finger contact in the hand refinement part')
        self.parser.add_argument('--random_rot_std', type=float, default=0.1)
        self.parser.add_argument('--collision_loss_threshold',
                                 type=float, default=0.01)
        self.parser.add_argument(
            '--manual_seed', type=int, help='manual seed')
        self.parser.add_argument(
            '--extra_name', type=str, default='', help='string appended to end of folder which we save data to')

        self.parser.add_argument(
            '--pregenerate_data', action='store_true', help='If we want to pregenerate viewpoints')
        self.parser.add_argument(
            '--debug',
            action='store_true',
            help='If true, we will only train on the first object'
        )
        self.parser.add_argument(
            '--num_viewpoints_per_object',
            type=int,
            default=100,
        )
        self.parser.add_argument(
            '--constrain_method', choices=['soft', 'hard', 'cvx'], default='soft')
        self.parser.add_argument('--optimize_fingertip', action='store_true',
                                 help='If we also want to optimize the finger tip')

        self.initialized = True

    def parse(self):
        pass

    def set_folder_name(self):
        folder_name = "views_"+str(self.opt.num_viewpoints_per_object)
        if not self.opt.no_classification_loss:
            folder_name += "_classification_" + \
                str(int(self.opt.lambda_G_classification))
        if not self.opt.no_contact_loss:
            folder_name += "_contact_"+str(int(self.opt.lambda_G_contactloss))
        if not self.opt.no_intersection_loss:
            folder_name += "_intersection_" + \
                str(int(self.opt.lambda_G_intersections))
        if not self.opt.no_orientation_loss:
            folder_name += "_orientation_" + \
                str(int(self.opt.lambda_G_orientation))
        if not self.opt.no_discriminator:
            folder_name += "_discriminator"
            folder_name += "_adv_"+str(int(self.opt.lambda_D_prob))
            folder_name += "_gp_"+str(int(self.opt.lambda_D_gp))
        folder_name += "_coll_threshold_" + \
            str(self.opt.collision_loss_threshold).split(".")[-1]
        folder_name += "_obj_finger_contact_threshold_" + \
            str(self.opt.object_finger_contact_threshold).split(".")[-1]
        folder_name += "_finger_constrainer_"+self.opt.constrain_method
        folder_name += "_optimizer_"+self.opt.optimizer
        folder_name += "_number_of_epochs_" + \
            str(self.opt.nepochs_no_decay+self.opt.nepochs_decay)
        folder_name += self.opt.extra_name
        return folder_name

    def load_opt_file(self, file):
        with open(file, "r") as f:
            opt = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
            opt.load_epoch = self.opt.load_epoch
            opt.checkpoint_dir_load = self.opt.checkpoint_dir_load
            opt.batch_size = self.opt.batch_size
            opt.gpu_ids = self.opt.gpu_ids
            #opt.gpu_ids = ','.join([str(elem) for elem in opt.gpu_ids])
            self.opt.update(opt)

    def set_epoch(self):
        folder_name = self.set_folder_name()
        self.opt.name = folder_name+self.opt.name
        models_dir = os.path.join(
            self.opt.checkpoints_dir, self.opt.name)
        if os.path.exists(models_dir):
            print("Terminating. The folder " + models_dir +
                  " exists and you chose to train from start. Either remove the folder or choose to continue train")
            sys.exit()

    def load_epoch(self):
        if os.path.exists(self.opt.checkpoint_dir_load):
            self.load_opt_file(self.opt.checkpoint_dir_load+"opt_train.yaml")
            found = False
            for file in os.listdir(self.opt.checkpoint_dir_load):
                if file.startswith("net_epoch_"+self.opt.load_epoch):
                    found = True
                    break
            if not found:
                print("Terminating. Epoch "+self.opt.load_epoch+" not found ")
                sys.exit()
        else:
            print("Terminating. You want to continue train but the folder " + self.opt.checkpoint_dir_load +
                  " does not exist. First you need to train a model.")
            sys.exit()

    def get_set_gpus(self):
        # get gpu ids
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
            elif id == -1:
                self.opt.gpu_ids.append(id)
                return
        if torch.cuda.is_available():
            while self.opt.gpu_ids[0] >= torch.cuda.device_count():
                self.opt.gpu_ids[0] -= 1
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

    def print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def save(self, args):
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        print(expr_dir)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' %
                                 ('train' if self.is_train else 'test'))
        file_name_yaml = os.path.join(expr_dir, 'opt_%s.yaml' %
                                      ('train' if self.is_train else 'test'))
        with open(file_name_yaml, 'w') as opt_file:
            yaml.dump(args, opt_file)
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        file_name = os.path.join(expr_dir, 'command_line.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(" ".join(sys.argv))
