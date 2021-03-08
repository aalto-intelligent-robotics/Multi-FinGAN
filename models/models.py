import os
import torch
from torch.optim import lr_scheduler


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(*args, **kwargs):
        model = None

        from .multi_fingan import Model
        model = Model(*args, **kwargs)

        print("Model %s was created" % model.name)
        return model


class BaseModel(object):
    def __init__(self, opt):
        self.name = 'BaseModel'

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train

        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def get_name(self):
        return self.name

    def is_train(self):
        return self.is_train

    def set_input(self, input):
        assert False, "set_input not implemented"

    def set_train(self):
        assert False, "set_train not implemented"

    def set_eval(self):
        assert False, "set_eval not implemented"

    def forward(self, keep_data_for_visuals=False):
        assert False, "forward not implemented"

    # used in test time, no backprop
    def test(self):
        assert False, "test not implemented"

    def get_image_paths(self):
        return {}

    def optimize_parameters(self):
        assert False, "optimize_parameters not implemented"

    def get_current_visuals(self):
        return {}

    def get_current_errors(self):
        return {}

    def get_current_scalars(self):
        return {}

    def save(self, label):
        assert False, "save not implemented"

    def load(self):
        assert False, "load not implemented"

    def save_optimizer(self, optimizer, optimizer_label, label):
        save_filename = 'opt_epoch_%s_id_%s.pth' % (label,
                                                    optimizer_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(
            {
                'optimizer_state_dict': optimizer.state_dict(),
                'learning_rate_G': self.current_lr_image_encoder_and_grasp_predictor,
            }, save_path)

    def load_optimizer(self, optimizer, optimizer_label, label, device):
        load_filename = 'opt_epoch_%s_id_%s.pth' % (label,
                                                    optimizer_label)
        load_path = os.path.join(self.save_dir, load_filename)
        assert os.path.exists(
            load_path
        ), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path

        checkpoint = torch.load(load_path, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_lr_image_encoder_and_grasp_predictor = checkpoint['learning_rate_G']
        print('loaded optimizer: %s' % load_path)

    def save_network(self, network, network_label, label, epoch):
        save_filename = 'net_epoch_%s_id_%s.pth' % (label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
            }, save_path)

        #print('saved net: %s' % save_path)

    def load_network(self, network, network_label, label, device):
        load_filename = 'net_epoch_%s_id_%s.pth' % (label, network_label)
        load_path = os.path.join(self.save_dir, load_filename)
        assert os.path.exists(
            load_path
        ), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path
        checkpoint = torch.load(load_path, map_location=device)
        network.load_state_dict(checkpoint['model_state_dict'])
        self.set_epoch(checkpoint["epoch"])
        print('loaded net: %s' % load_path)

    def update_learning_rate(self):
        pass

    def print_network(self, network):
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()
        print(network)
        print('Total number of parameters: %d' % num_params)

    def get_scheduler(self, optimizer, opt):
        if opt.lr_policy == 'lambda':

            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count -
                                 opt.niter) / float(opt.niter_decay + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer,
                                            step_size=opt.lr_decay_iters,
                                            gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.2,
                                                       threshold=0.01,
                                                       patience=5)
        else:
            return NotImplementedError(
                'learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler
