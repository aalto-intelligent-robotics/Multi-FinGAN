from easydict import EasyDict
import numpy as np
import torch
import random
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--n_threads_train', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--num_iters_validate', default=10,
                                 type=int, help='# batches to use when validating')
        self.parser.add_argument('--print_freq_s', type=int, default=60,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--display_freq_s', type=int, default=300,
                                 help='frequency [s] of showing training results on screen')
        self.parser.add_argument('--display_freq_s_val', type=int, default=300,
                                 help='frequency [s] of showing training results on screen')
        self.parser.add_argument('--save_latest_freq_s', type=int, default=3600,
                                 help='frequency of saving the latest results')

        self.parser.add_argument('--nepochs_no_decay', type=int, default=400,
                                 help='# of epochs at starting learning rate')
        self.parser.add_argument('--nepochs_decay', type=int, default=400,
                                 help='# of epochs to linearly decay learning rate to zero')

        self.parser.add_argument('--train_G_every_n_iterations', type=int,
                                 default=5, help='train G every n interations')
        self.parser.add_argument(
            '--optimizer', default="Adam", choices=["Adam", "SGD"], type=str)

        self.parser.add_argument(
            '--poses_g_sigma', type=float, default=0.06, help='initial learning rate for adam')
        self.parser.add_argument(
            '--lr_G', type=float, default=0.0001, help='initial learning rate for G adam')
        self.parser.add_argument(
            '--G_adam_b1', type=float, default=0.5, help='beta1 for G adam')
        self.parser.add_argument(
            '--G_adam_b2', type=float, default=0.999, help='beta2 for G adam')
        self.parser.add_argument(
            '--lr_D', type=float, default=0.0001, help='initial learning rate for D adam')
        self.parser.add_argument(
            '--D_adam_b1', type=float, default=0.5, help='beta1 for D adam')
        self.parser.add_argument(
            '--D_adam_b2', type=float, default=0.999, help='beta2 for D adam')
        self.parser.add_argument('--lambda_D_prob', type=float, default=1,
                                 help='lambda for real/fake discriminator loss')
        self.parser.add_argument(
            '--lambda_D_gp', type=float, default=10, help='lambda gradient penalty loss')

        self.parser.add_argument(
            '--lambda_G_classification', type=float, default=1.0, help='')
        self.parser.add_argument(
            '--lambda_G_contactloss', type=float, default=100.0, help='')
        self.parser.add_argument(
            '--lambda_G_intersections', type=float, default=100.0, help='')
        self.parser.add_argument('--no_discriminator', action='store_true',
                                 help='if true, do not train the discriminator')
        self.parser.add_argument('--no_classification_loss', action='store_true',
                                 help='if true, do not train with intersection loss')
        self.parser.add_argument('--no_intersection_loss', action='store_true',
                                 help='if true, do not train with intersection loss')
        self.parser.add_argument('--no_contact_loss', action='store_true',
                                 help='if true, do not train with intersection loss')
        self.parser.add_argument('--no_orientation_loss', action='store_true',
                                 help='if true, do not train with orientation loss')
        self.parser.add_argument(
            '--lambda_G_orientation', type=float, default=1.0, help='')
        self.parser.add_argument(
            '--continue_train', action='store_true', help='if true then we continue to train')
        self.parser.add_argument('--ablation_study', action='store_true',
                                 help='if true then we train for ablation study')
        opts, _ = self.parser.parse_known_args()
        if opts.continue_train:
            self.parser.add_argument(
                '--checkpoint_dir_load', type=str, help='path to checkpoint we want to continue train from', required="true")
            self.parser.add_argument('--load_epoch', type=str, default='latest',
                                     help='which epoch to load? ')
        self.is_train = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = EasyDict(vars(self.parser.parse_args()))
        if not self.opt.pregenerate_data:
            self.opt.n_threads_train = 0
            self.opt.n_threads_test = 0

        # set is train or set
        self.opt.is_train = True
        if self.opt.manual_seed is None:
            self.opt.manual_seed = random.randint(1, 10000)

        # set and check load_epoch
        if self.opt.continue_train:
            self.load_epoch()
            self.opt.load_network = True
        else:
            self.set_epoch()
            self.opt.load_network = False
        torch.manual_seed(self.opt.manual_seed)
        np.random.seed(self.opt.manual_seed)
        # get and set gpus
        if self.opt.ablation_study:
            self.opt.checkpoints_dir = self.opt.checkpoints_dir+"/ablation_study/"

        self.opt.dataset_name = "ycb"
        self.get_set_gpus()

        args = vars(self.opt)

        # print in terminal args
        self.print(args)

        # save args to file
        self.save(args)

        return self.opt
