from .base_options import BaseOptions
import torch
import numpy as np
import random
from easydict import EasyDict


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--graspit', action='store_true')
        self.parser.add_argument('--threshold_intersections', type=float, default=50)
        self.parser.add_argument('--threshold_contact', type=float, default=0.009,
                                 help="threshold in meters to determine if the distance between two vertices make contact")
        self.parser.add_argument('--display', action='store_true')
        self.parser.add_argument('--save_folder', type=str, default='./results/simulation_results/')
        self.parser.add_argument('--test_set', default="ycb", choices=["ycb", "egad"], type=str)
        opts, _ = self.parser.parse_known_args()
        if not opts.graspit:
            self.parser.add_argument('--num_viewpoints_per_object_test', default=5, type=int)
            self.parser.add_argument('--num_grasps_to_sample', type=int, choices=range(2, 200), default=30)
            self.parser.add_argument('--checkpoint_dir_load', type=str, help='path to checkpoint we want to load', required="true")
            self.parser.add_argument('--load_epoch', type=str, default='latest',
                                     help='which epoch to load? ')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = EasyDict(vars(self.parser.parse_args()))

        # set and check load_epoch
        if self.opt.graspit:
            self.opt.save_folder += "graspit/"
            if self.opt.test_set == "ycb":
                self.opt.graspit_grasp_dir = './data/test_data/ycb_graspit_grasps/'
                self.opt.object_mesh_dir = './data/meshes/ycb_meshes/0*/google_16k/textured_simplified.obj'
            elif self.opt.test_set == "egad":
                self.opt.graspit_grasp_dir = './data/test_data/egad_graspit_grasps/'
                self.opt.object_mesh_dir = './data/meshes/egad_val_set_meshes/*_simplified.ply'
        else:
            self.opt.save_folder += "multifin_gan/"
            if not self.opt.pregenerate_data:
                self.opt.n_threads_train = 0
                self.opt.n_threads_test = 0

            self.load_epoch()
            # In val mode we do not want to run the inference on any seed that was used while training or testing
            seed_list = list(range(1, 10000))
            seed_list.remove(self.opt.manual_seed)
            self.opt.manual_seed = random.choice(seed_list)
            self.opt.no_discriminator = True
            self.opt.precomputed_rotations = True
            torch.manual_seed(self.opt.manual_seed)
            np.random.seed(self.opt.manual_seed)
            self.get_set_gpus()
            if self.opt.test_set == "ycb":
                self.opt.object_mesh_dir = './data/meshes/ycb_meshes/'
                self.opt.dataset_mode = "ycb_synthetic_one_object"
            elif self.opt.test_set == "egad":
                self.opt.object_mesh_dir = './data/meshes/egad_val_set_meshes/'
                self.opt.dataset_mode = "egad_synthetic_one_object"

        self.opt.is_train = False
        self.opt.load_network = True
        args = vars(self.opt)

        # print in terminal args
        self.print(args)

        # save args to file
        # self.save(args)

        return self.opt
