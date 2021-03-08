import utils.plots as plot_utils
from utils import forward_kinematics_barrett as fk
from barrett_kinematics.barrett_layer.barrett_layer import BarrettLayer
import numpy as np
from networks.networks import NetworksFactory
from .models import BaseModel
from torch.autograd import Variable
from collections import OrderedDict
import torch
from utils import contactutils, util, constrain_hand


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__(opt)
        self.name = 'Multi_FinGan'

        self.setup_touching_vertices(opt)

        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids[0] != -1 and torch.cuda.is_available() else torch.device('cpu')

        # create networks
        self.create_and_init_networks()

        self.constrain_hand = constrain_hand.ConstrainHand(
            self.opt.constrain_method, self.opt.batch_size, self.device)
        # init train variables
        if self.is_train:
            self.create_and_init_optimizer()

        self.i_epoch = 0
        if self.opt.load_network:
            self.load()

        self.init_losses()
        self.taxonomy_poses = np.load(
            './data/average_hand_joints_per_taxonomy.npy')
        self.taxonomy_tensor = torch.FloatTensor(
            self.taxonomy_poses).to(self.device)
        self.gradient_accumulation_every = 2
        self.gradient_accumulation_current_step = 0

        if self.opt.precomputed_rotations:
            all_approach_orientation = np.load(
                "files/uniform_rotations.npy", allow_pickle=True)
            self.approach_orientation = torch.FloatTensor(
                all_approach_orientation[self.opt.num_grasps_to_sample-2])
        self.setup_bookkeeping_variables()

    def setup_bookkeeping_variables(self):
        self.delta_T = torch.zeros(1).to(self.device)
        self.delta_R = torch.zeros(1).to(self.device)
        self.delta_HR = torch.zeros(1).to(self.device)
        self.invalid_hand_conf = torch.zeros(1).to(self.device)
        self.true_positive = torch.zeros(1).to(self.device)
        self.true_negative = torch.zeros(1).to(self.device)
        self.false_positive = torch.zeros(1).to(self.device)
        self.false_negative = torch.zeros(1).to(self.device)
        self.display_hand_gt_pose = None
        self.display_hand_gt_rep = None
        self.display_mesh_vertices = None
        self.display_mesh_faces = None

    def setup_touching_vertices(self, opt):
        self.touching_hand_vertices = [769, 802, 809, 815, 912, 915,
                                       923, 929, 934, 937, 1026, 1029, 1030, 1037, 1043, 1045, 1048]
        self.touching_hand_vertices += [18, 129, 205]

    def create_and_init_networks(self):
        self.image_encoder_and_grasp_predictor = self.create_image_encoder_and_grasp_predictor()
        self.image_encoder_and_grasp_predictor.init_weights()
        self.grasp_generator = self.create_grasp_generator()
        self.grasp_generator.init_weights()

        if len(self.gpu_ids) > 1:
            self.image_encoder_and_grasp_predictor = torch.nn.DataParallel(self.image_encoder_and_grasp_predictor, device_ids=self.gpu_ids)
            self.grasp_generator = torch.nn.DataParallel(self.grasp_generator,
                                                         device_ids=self.gpu_ids)
        self.image_encoder_and_grasp_predictor.to(self.device)
        self.grasp_generator.to(self.device)

        # Initialize Barrett layer
        self.barrett_layer = BarrettLayer(
            device=self.device).to(self.device)
        if not self.opt.no_discriminator:
            # Discriminator network
            self.discriminator = self.create_discriminator()
            self.discriminator.init_weights()
            if len(self.gpu_ids) > 1:
                self.discriminator = torch.nn.DataParallel(self.discriminator, device_ids=self.gpu_ids)
            self.discriminator.to(self.device)

    def create_image_encoder_and_grasp_predictor(self):
        return NetworksFactory.get_by_name(
            'img_encoder_and_grasp_predictor')  # The output is 6 or 7 as we consider 6 or 7 grasp taxonomy classes for the Barrett hand

    def create_grasp_generator(self):
        return NetworksFactory.get_by_name(
            'grasp_generator', input_dim=3 + 3 + 7)  # 3 for rotation 3 for tranlsation and 7 for hand joints

    def create_discriminator(self):
        return NetworksFactory.get_by_name('discriminator',
                                           input_dim=3 + 3 + 1)  # 3 for rotation 3 for tranlsation and 1 for finger spread

    def create_and_init_optimizer(self):
        self.current_lr_image_encoder_and_grasp_predictor = self.opt.lr_G
        # initialize optimizers
        if self.opt.optimizer == "Adam":
            optimizer = torch.optim.Adam
            self.optimizer_image_enc_and_grasp_predictor = optimizer(
                self.image_encoder_and_grasp_predictor.parameters(),
                lr=self.current_lr_image_encoder_and_grasp_predictor,
                betas=[self.opt.G_adam_b1, self.opt.G_adam_b2])

            self.optimizer_grasp_generator = optimizer(self.grasp_generator.parameters(),
                                                       lr=self.current_lr_image_encoder_and_grasp_predictor)
            if not self.opt.no_discriminator:
                self.current_lr_discriminator = self.opt.lr_D
                self.optimizer_discriminator = torch.optim.Adam(
                    self.discriminator.parameters(),
                    lr=self.current_lr_discriminator,
                    betas=[self.opt.D_adam_b1, self.opt.D_adam_b2])
        elif self.opt.optimizer == "SGD":
            optimizer = torch.optim.SGD
            self.optimizer_image_enc_and_grasp_predictor = optimizer(
                self.image_encoder_and_grasp_predictor.parameters(),
                lr=self.current_lr_image_encoder_and_grasp_predictor,
                momentum=0.9)

            self.optimizer_grasp_generator = optimizer(self.grasp_generator.parameters(),
                                                       lr=self.current_lr_image_encoder_and_grasp_predictor, momentum=0.9)
            if not self.opt.no_discriminator:
                self.current_lr_discriminator = self.opt.lr_D
                self.optimizer_discriminator = optimizer(
                    self.discriminator.parameters(),
                    lr=self.current_lr_discriminator,
                    momentum=0.9
                )
        else:
            raise ValueError("Optimizer ", self.opt.optimizer,
                             " not availabel.")

    def set_epoch(self, epoch):
        self.i_epoch = epoch

    def get_epoch(self):
        return self.i_epoch

    def init_losses(self):
        self.loss_g_contactloss = Variable(self.Tensor([0]))
        self.loss_g_interpenetration = Variable(self.Tensor([0]))
        self.loss_g_CE = Variable(self.Tensor([0]))
        self.acc_g = Variable(self.Tensor([0]))
        self.criterion_CE = torch.nn.BCEWithLogitsLoss().to(self.device)
        # if self.opt.rot_loss:
        self.canonical_approach_vec = torch.FloatTensor(
            [[0, 0, 1]]).to(self.device)
        self.loss_g_orientation = Variable(self.Tensor([0]))
        if not self.opt.no_discriminator:
            self.loss_g_fake = Variable(self.Tensor([0]))
            self.loss_d_real = Variable(self.Tensor([0]))
            self.loss_d_fake = Variable(self.Tensor([0]))
            self.loss_d_fakeminusreal = Variable(self.Tensor([0]))
            self.loss_d_gp = Variable(self.Tensor([0]))

    def set_input(self, data_input):
        self.input_rgb_img = data_input['rgb_img'].float().permute(
            0, 3, 1, 2).contiguous()
        self.input_object_id = data_input['object_id']
        if self.opt.dataset_mode == 'ycb_synthetic_one_object':
            self.input_taxonomy = data_input['taxonomy'].float()
        else:
            self.input_taxonomy = data_input['taxonomy']
        self.input_obj_verts = data_input['3d_points_object']
        self.input_obj_faces = data_input['3d_faces_object']
        self.input_obj_resampled_verts = data_input['object_points_resampled']
        self.input_hand_gt_rep = data_input['hand_gt_representation'].float()
        self.input_hand_gt_pose = data_input['hand_gt_pose'].float()

        if torch.cuda.is_available():
            self.input_rgb_img = self.input_rgb_img.to(self.device)
            self.input_taxonomy = self.input_taxonomy.to(self.device)
            self.input_hand_gt_rep = self.input_hand_gt_rep.to(self.device)
            self.input_hand_gt_pose = self.input_hand_gt_pose.to(self.device)

        self.batch_size = self.input_rgb_img.size(0)
        self.calculate_center_of_objects()

    def calculate_center_of_objects(self):
        center_objects = []
        for i in range(self.batch_size):
            center_objects.append(self.input_obj_resampled_verts[i][
                0].mean(0))

        self.input_center_objects = torch.FloatTensor(
            center_objects).to(self.device)

    def set_train(self):
        self.image_encoder_and_grasp_predictor.train()
        self.grasp_generator.train()
        if not self.opt.no_discriminator:
            self.discriminator.train()
        self.zero_losses()
        self.zero_bookkeeping()
        self.is_train = True

    def zero_bookkeeping(self):
        self.true_positive *= 0
        self.true_negative *= 0
        self.false_positive *= 0
        self.false_negative *= 0
        self.delta_HR *= 0
        self.delta_T *= 0
        self.delta_R *= 0
        self.invalid_hand_conf *= 0

    def zero_losses(self):
        self.loss_g_CE *= 0
        self.acc_g *= 0
        # if self.opt.intersection_loss:
        self.loss_g_interpenetration *= 0
        # if self.opt.contact_loss:
        self.loss_g_contactloss *= 0
        # if self.opt.rot_loss:
        self.loss_g_orientation *= 0
        if not self.opt.no_discriminator:
            self.loss_g_fake *= 0
            self.loss_d_real *= 0
            self.loss_d_fake *= 0
            self.loss_d_fake *= 0
            self.loss_d_real *= 0
            self.loss_d_gp *= 0

    def set_eval(self):
        self.image_encoder_and_grasp_predictor.eval()
        self.grasp_generator.eval()
        if not self.opt.no_discriminator:
            self.discriminator.eval()
        # Zero all losses
        self.zero_losses()
        self.is_train = False

    def create_pose(self, T, R):
        pose = torch.eye(4).view(1, 4, 4).to(self.device)
        pose = pose.repeat(T.shape[0], 1, 1)
        pose[:, :3, :3] = R
        pose[:, :3, 3] = T
        return pose

    def calculate_interpenetration(self, hand_vertices, hand_vertice_areas, batch_size):
        interpenetration = torch.FloatTensor([0]).to(self.device)
        # INTERSECTION LOSS ON OPTIMIZED HAND!
        for i in range(batch_size):
            numobjects = len(self.input_obj_verts[i])
            all_triangles = []
            # all_verts = []_
            for j in range(numobjects):
                obj_triangles = self.input_obj_verts[i][j][
                    self.input_obj_faces[i][j]]
                obj_triangles = torch.FloatTensor(
                    obj_triangles).to(self.device)
                all_triangles.append(obj_triangles)

            all_triangles = torch.cat(all_triangles)

            exterior = contactutils.batch_mesh_contains_points(
                hand_vertices[i].unsqueeze(0), all_triangles.unsqueeze(0), device=self.device)
            penetr_mask = ~exterior
            if penetr_mask.sum() == 0:
                continue

            allpoints_resampled = torch.FloatTensor(
                self.input_obj_resampled_verts[i]).to(self.device).reshape(
                    -1, 3).unsqueeze(0)
            dists = util.batch_pairwise_dist(
                hand_vertices[i, penetr_mask[0]].unsqueeze(0), allpoints_resampled)
            mins21, _ = torch.min(dists, 2)
            mins21[mins21 < 1e-4] = 0
            mins21 = mins21*hand_vertice_areas[i, penetr_mask[0]]

            interpenetration = interpenetration + mins21.mean()

        return interpenetration

    def calculate_contact(self, points_3d):
        relevantobjs_resampled = torch.FloatTensor([
            self.input_obj_resampled_verts[i][0]
            for i in range(self.batch_size)
        ]).to(self.device)

        distance_touching_vertices_fake = self.get_touching_distances(
            points_3d, relevantobjs_resampled)
        distance_touching_vertices_fake[distance_touching_vertices_fake <
                                        self.opt.collision_loss_threshold] = 0

        return distance_touching_vertices_fake

    def get_touching_distances(self, hand_points, object_points):

        relevant_vertices = hand_points[:, self.touching_hand_vertices]
        n1 = len(self.touching_hand_vertices)
        n2 = len(object_points[0])

        matrix1 = relevant_vertices.unsqueeze(1).repeat(1, n2, 1, 1)
        matrix2 = object_points.unsqueeze(2).repeat(1, 1, n1, 1)

        dists = torch.norm(matrix1 - matrix2, dim=-1)
        dists = dists.min(1)[0]
        return dists

    def get_distances_single_example(self, relevant_vertices, object_points):
        n1 = len(relevant_vertices)
        # TODO: HAVE TO DO IT IN A LOOP SINCE OBJECTS ALL HAVE DIFFERENT AMOUNT OF VERTICES
        n2 = len(object_points)

        matrix1 = relevant_vertices.unsqueeze(0).repeat(n2, 1, 1)
        if torch.cuda.is_available():
            matrix2 = torch.FloatTensor(object_points).to(self.device).unsqueeze(
                1).repeat(1, n1, 1)
        else:
            matrix2 = torch.FloatTensor(object_points).unsqueeze(1).repeat(
                1, n1, 1)
        dists = torch.sqrt(((matrix1 - matrix2)**2).sum(-1))
        return dists.min(0)[0]

    def optimize_parameters(self,
                            train_generator=True,
                            ):
        if self.is_train:
            # convert tensor to variables
            self.batch_size = self.input_rgb_img.size(0)

            # train discriminator_
            if not self.opt.no_discriminator:
                fake_input_D, real_input_D, loss_D = self.forward_D()
                loss_D_gp = self.gradient_penalty_D(fake_input_D, real_input_D)

                self.optimizer_discriminator.zero_grad()
                loss = loss_D + loss_D_gp
                loss.backward(retain_graph=True)
                self.optimizer_discriminator.step()

            # train generator
            if train_generator:
                self.forward_G(True)
                loss_G = self.combine_generator_losses()
                loss_G.backward()
                self.gradient_accumulation_current_step += 1

                if self.gradient_accumulation_current_step % self.gradient_accumulation_every == 0:
                    self.optimizer_image_enc_and_grasp_predictor.step()
                    self.optimizer_grasp_generator.step()
                    self.optimizer_image_enc_and_grasp_predictor.zero_grad()
                    self.optimizer_grasp_generator.zero_grad()
                    self.gradient_accumulation_current_step = 0

    def evaluate_prediction(self, prediction):
        self.loss_g_CE = self.criterion_CE(prediction, self.input_taxonomy)*self.opt.lambda_G_classification
        thresholded_predictions = (torch.sigmoid(prediction) > 0.5).long()
        self.acc_g = (thresholded_predictions.cpu().data.numpy(
        ) == self.input_taxonomy.cpu().data.numpy()).mean()
        tp, tn, fp, fn = util.calculate_classification_statistics(
            self.input_taxonomy, thresholded_predictions)
        self.true_positive += tp
        self.true_negative += tn
        self.false_positive += fp
        self.false_negative += fn

    def finger_refinement(self, HR, rot_matrix, T, batched=False):
        if batched:
            fk.optimize_fingers_batched(
                HR, rot_matrix,
                T, self.input_obj_resampled_verts[0][
                    0], self.barrett_layer, self.opt.object_finger_contact_threshold, optimize_finger_tip=self.opt.optimize_fingertip, device=self.device)
        else:
            for i in range(self.batch_size):
                _ = fk.optimize_fingers(
                    HR[i].view(1, -1), rot_matrix[i],
                    T[i], self.input_obj_resampled_verts[i][
                        0], self.barrett_layer, self.opt.object_finger_contact_threshold, optimize_finger_tip=self.opt.optimize_fingertip, device=self.device)

    def forward_G(self, train=True):
        input_img = self.input_rgb_img
        prediction, img_representations = self.image_encoder_and_grasp_predictor.forward(
            input_img)
        if train:
            self.evaluate_prediction(prediction)
        hand_representations = self.hand_representation_from_prediction(
            prediction)

        if self.opt.precomputed_rotations:
            # If we use precomputed rotations it means that we are doing the simulation experiments
            # and for that we generate self.opt.num_grasps_to_sample grasps but only on one object at a time.
            # Therefore, we need to repeat the translation_in, img_representation, and hand_representation
            # accordingly
            random_rots = torch.FloatTensor(self.approach_orientation).to(
                self.device)
            translation_in = self.input_center_objects.repeat(self.opt.num_grasps_to_sample, 1)
            img_representations = img_representations.repeat(self.opt.num_grasps_to_sample, 1)
            hand_representations = hand_representations.repeat(self.opt.num_grasps_to_sample, 1)
        else:
            axis_angles = util.rotm_to_axis_angles(
                self.input_hand_gt_pose[:, :3, :3], self.device)
            random_rots = axis_angles + torch.normal(
                mean=torch.zeros(self.batch_size, 3)).to(self.device) / 5
            translation_in = self.input_center_objects

        hand_configuration, R, T = self.grasp_generator.forward(img_representations, hand_representations,
                                                                random_rots, translation_in)
        rot_matrix = util.axis_angles_to_rotation_matrix(R)

        hand_configuration, hand_configuration_unconstrained = self.constrain_hand(hand_configuration)
        # Batched hand refinement is only possible when not training, i.e. when we do the simulation experiments in https://arxiv.org/pdf/2012.09696.pdf.
        # The reason for this is because in training mode we operate with much higher batch sizes and the batched hand-refinement layer cannot handle large
        # batch sizes
        self.finger_refinement(hand_configuration, rot_matrix, T, batched=not train)

        self.display_hand_gt_pose = self.input_hand_gt_pose
        self.display_hand_gt_rep = self.input_hand_gt_rep
        self.display_mesh_vertices = self.input_obj_verts
        self.display_mesh_faces = self.input_obj_faces
        pose = self.create_pose(T, rot_matrix)

        * points_3d, _ = self.barrett_layer(pose, hand_configuration)
        if train:
            self.delta_T = util.euclidean_distance(
                translation_in, T).mean()

            self.invalid_hand_conf = util.valid_hand_conf(util.rad2deg(
                hand_configuration), self.device).squeeze()
            self.delta_R = util.axis_angle_distance(random_rots, R).mean()
            self.delta_HR = (hand_representations-hand_configuration).abs().mean()

            points_3d, points_3d_area = util.concatenate_barret_vertices_and_vertice_areas(
                points_3d, self.barrett_layer.vertice_face_areas, use_torch=True)
            self.refined_handpose = points_3d.cpu().data.numpy()
            points_3d_area = points_3d_area.to(self.device)
            self.calculate_generator_losses(points_3d, points_3d_area, R, T, translation_in, hand_configuration_unconstrained)
        else:
            points_3d_concatenated, faces_concatenated = util.concatenate_barret_vertices_and_faces(
                points_3d, self.barrett_layer.gripper_faces, use_torch=True)

            self.refined_handpose_concatenated = points_3d_concatenated
            self.refined_handpose = points_3d
            self.refined_handpose_faces = faces_concatenated
            self.refined_HR = hand_configuration
            self.refined_R = R
            self.refined_T = T

    def calculate_generator_losses(self, points_3d, points_3d_area, R, T, translation_in, hand_configuration_unconstrained):
        interpenetration = torch.FloatTensor([0]).to(self.device)
        # INTERSECTION LOSS ON OPTIMIZED HAND
        interpenetration = self.calculate_interpenetration(
            points_3d, points_3d_area, self.batch_size)
        self.loss_g_interpenetration = interpenetration / \
            self.batch_size * self.opt.lambda_G_intersections
        # CONTACT LOSS ON OPTIMIZED HAND
        distance_touching_vertices_fake = self.calculate_contact(points_3d)
        self.loss_g_contactloss = distance_touching_vertices_fake.mean(
        ) * self.opt.lambda_G_contactloss
        # ORIENTATION LOSS
        self.loss_g_orientation = self.opt.lambda_G_orientation * \
            self.calculate_rot_loss(
                R, T, self.input_center_objects).mean()
        if not self.opt.no_discriminator:
            fake_input_D = torch.cat(
                (
                    R,
                    hand_configuration_unconstrained[:, 0].unsqueeze(1),
                    T - translation_in),
                1)
            d_fake_prob = self.discriminator(fake_input_D)
            self.loss_g_fake = self.compute_loss_D(
                d_fake_prob, True) * self.opt.lambda_D_prob

    def combine_generator_losses(self):
        combined_losses = 0
        if not self.opt.no_classification_loss:
            combined_losses += self.loss_g_CE
        if not self.opt.no_intersection_loss:
            combined_losses += self.loss_g_interpenetration[0]
        if not self.opt.no_contact_loss:
            combined_losses += self.loss_g_contactloss
        if not self.opt.no_orientation_loss:
            combined_losses += self.loss_g_orientation
        if not self.opt.no_discriminator:
            combined_losses += self.loss_g_fake
        return combined_losses

    def calculate_rot_loss(self, rot, T, object_center):
        hand_to_obj_vec = object_center-T
        hand_to_obj_vec_normed = hand_to_obj_vec / \
            torch.norm(hand_to_obj_vec, dim=-1, keepdim=True)
        hand_approach_vector = util.axis_angle_rot(
            rot, self.canonical_approach_vec)

        rot_loss = 1-(hand_to_obj_vec_normed*hand_approach_vector).sum(-1)
        return rot_loss

    def is_nan(self, tensor, type):
        if torch.any(torch.isnan(tensor)):
            print(type + " is nan")

    def forward_D(self):
        input_img = self.input_rgb_img
        axis_angles = util.rotm_to_axis_angles(
            self.input_hand_gt_pose[:, :3, :3], self.device)
        input_hand_gt_rot = axis_angles
        random_rots = axis_angles + torch.normal(
            mean=torch.zeros(self.batch_size, 3)).to(self.device) / 5

        prediction, img_representations = self.image_encoder_and_grasp_predictor.forward(
            input_img)
        self.evaluate_prediction(prediction)
        hand_representations = self.hand_representation_from_prediction(
            prediction)

        translation_in = self.input_center_objects

        HR, R, T = self.grasp_generator.forward(img_representations, hand_representations,
                                                random_rots, translation_in)

        fake_input_D = torch.cat(
            (R, HR[:, 0].unsqueeze(1), T - translation_in), 1).detach()
        real_input_D = torch.cat(
            (input_hand_gt_rot, self.input_hand_gt_rep[:, 0].unsqueeze(1),
             self.input_hand_gt_pose[:, :3, 3] - translation_in), 1).detach()

        d_fake_prob = self.discriminator(fake_input_D)
        d_real_prob = self.discriminator(real_input_D)

        self.loss_d_real = self.compute_loss_D(
            d_real_prob, True) * self.opt.lambda_D_prob
        self.loss_d_fake = self.compute_loss_D(
            d_fake_prob, False) * self.opt.lambda_D_prob

        return fake_input_D, real_input_D, self.loss_d_real + self.loss_d_fake

    def hand_representation_from_prediction(self, prediction):
        prediction_probs = torch.sigmoid(
            prediction) > 0.5
        idx = prediction_probs.nonzero()
        hand_representations = torch.zeros((self.batch_size, 7))
        rand_nonzero_cols = torch.randint(
            self.taxonomy_tensor.shape[0], (self.batch_size,))
        for batch_idx in torch.unique(idx[:, 0]):
            nonzero_columns = prediction_probs[batch_idx].nonzero()
            num_nonzero_cols = len(nonzero_columns)
            rand_nonzero_cols[batch_idx] = nonzero_columns[torch.randint(
                num_nonzero_cols, (1,))][0]
        hand_representations = self.taxonomy_tensor[rand_nonzero_cols]
        return hand_representations

    def gradient_penalty_D(self, fake_input_D, real_input_D):
        # interpolate sample
        alpha = torch.rand(self.batch_size, fake_input_D.shape[1]).to(self.device)
        alpha.requires_grad = True
        interpolated = alpha * real_input_D + (1 - alpha) * fake_input_D
        interpolated_prob = self.discriminator(interpolated)

        # compute gradients
        grad = torch.autograd.grad(outputs=interpolated_prob,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(
                                       interpolated_prob.size()).to(self.device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        # penalize gradients
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad**2, dim=1))
        self.loss_d_gp = torch.mean(
            (grad_l2norm - 1)**2) * self.opt.lambda_D_gp

        return self.loss_d_gp

    def compute_loss_D(self, estim, is_real):
        return -torch.mean(estim) if is_real else torch.mean(estim)

    def get_current_errors(self):
        losses = []
        losses.append(
            ('g CE', self.loss_g_CE.cpu().data.numpy()))
        losses.append(
            ('g acc', self.acc_g))

        losses.append(
            ('g contact loss', self.loss_g_contactloss.cpu().data.numpy()))
        losses.append(
            ('g intersection loss',  self.loss_g_interpenetration.cpu().data.numpy()))
        losses.append(('g orientation loss', self.loss_g_orientation .cpu().data.numpy()))
        if not self.opt.no_discriminator:
            losses.append(('g fake', self.loss_g_fake.cpu().data.numpy()))
            losses.append(('d real', self.loss_d_real.cpu().data.numpy()))
            losses.append(('d fake', self.loss_d_fake.cpu().data.numpy()))
            losses.append(('d fakeminusreal', self.loss_d_fake.cpu().data.numpy() -
                           self.loss_d_real.cpu().data.numpy()))
            losses.append(('d gp', self.loss_d_gp.cpu().data.numpy()))
        return OrderedDict(losses)

    def get_current_errors_G(self):
        losses = []
        losses.append(
            ('g CE', self.loss_g_CE.cpu().data.numpy()))
        losses.append(
            ('g acc', self.acc_g))
        losses.append(
            ('g contact loss', self.loss_g_contactloss.cpu().data.numpy()))
        losses.append(
            ('g intersection loss',  self.loss_g_interpenetration.cpu().data.numpy()))
        losses.append(('g orientation loss', self.loss_g_orientation .cpu().data.numpy()))
        if not self.opt.no_discriminator:
            losses.append(('g fake', self.loss_g_fake.cpu().data.numpy()))
        return OrderedDict(losses)

    def get_current_errors_D(self):
        losses = []
        if not self.opt.no_discriminator:
            losses.append(('d real', self.loss_d_real.cpu().data.numpy()))
            losses.append(('d fake', self.loss_d_fake.cpu().data.numpy()))
            losses.append(('d fakeminusreal', self.loss_d_fake.cpu().data.numpy() -
                           self.loss_d_real.cpu().data.numpy()))
            losses.append(('d gp', self.loss_d_gp.cpu().data.numpy()))
        return OrderedDict(losses)

    def get_current_scalars(self):
        scalars = [
            ('mean euclidean dist', self.delta_T.cpu().data.numpy()),
            ('mean rotation dist', self.delta_R.cpu().data.numpy()),
            ('mean finger spread angle', self.delta_HR.cpu().data.numpy()),
            ('lr G', self.current_lr_image_encoder_and_grasp_predictor)
        ]
        if not self.opt.no_discriminator:
            scalars.append(('lr D', self.current_lr_discriminator))
        scalars.append(
            ('f1 score', self.f1_score()))
        scalars.append(
            ('number of invalid configurations', self.invalid_hand_conf.cpu().data.numpy().astype('float32')))

        return OrderedDict(scalars)

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = {}
        groundtruths = []
        predictions = []
        if self.display_hand_gt_rep is not None:
            rand_grasp_idx = np.random.randint(
                len(self.display_hand_gt_rep))
        else:
            rand_grasp_idx = 0

        if self.display_hand_gt_rep is not None:
            * hand_verts, _ = self.barrett_layer(
                self.display_hand_gt_pose[rand_grasp_idx].view(1, 4, 4), self.display_hand_gt_rep[rand_grasp_idx].view(1, 7))
            gt_verts, gt_faces = util.concatenate_barret_vertices_and_faces(
                hand_verts, self.barrett_layer.gripper_faces)
            groundtruths = plot_utils.plot_scene_w_grasps(
                self.display_mesh_vertices[rand_grasp_idx], self.display_mesh_faces[rand_grasp_idx], gt_verts,
                gt_faces)
        try:
            predictions = plot_utils.plot_scene_w_grasps(
                self.display_mesh_vertices[rand_grasp_idx], self.display_mesh_faces[rand_grasp_idx],
                [self.refined_handpose[rand_grasp_idx]], gt_faces)
        except Exception as e:
            pass

        visuals['1_groundtruth'] = groundtruths
        visuals['2_prediction'] = predictions
        return OrderedDict(visuals)

    def save(self, label, epoch):
        # save networks and optimizers
        self.save_network(self.image_encoder_and_grasp_predictor, 'image_encoder_and_grasp_predictor', label, epoch)
        self.save_optimizer(self.optimizer_image_enc_and_grasp_predictor, 'image_encoder_and_grasp_predictor', label)
        self.save_network(self.grasp_generator, 'grasp_generator', label, epoch)
        self.save_optimizer(self.optimizer_grasp_generator, 'grasp_generator', label)
        if not self.opt.no_discriminator:
            self.save_network(self.discriminator, 'discriminator', label, epoch)
            self.save_optimizer(self.optimizer_discriminator, 'discriminator', label)

    def load(self):
        load_epoch = self.opt.load_epoch

        # load image_encoder
        self.load_network(self.image_encoder_and_grasp_predictor, 'image_encoder_and_grasp_predictor', load_epoch, self.device)
        self.load_network(self.grasp_generator, 'grasp_generator', load_epoch, self.device)
        if not self.opt.no_discriminator:
            self.load_network(self.discriminator, 'discriminator', load_epoch, self.device)

        if self.is_train:
            # load optimizers
            self.load_optimizer(self.optimizer_image_enc_and_grasp_predictor, 'image_encoder_and_grasp_predictor',
                                load_epoch, self.device)
            self.load_optimizer(self.optimizer_grasp_generator, 'grasp_generator',
                                load_epoch, self.device)
            if not self.opt.no_discriminator:
                self.load_optimizer(self.optimizer_discriminator, 'discriminator',
                                    load_epoch, self.device)
            self.set_learning_rate()

    def update_learning_rate(self):
        # updated learning rateimage_encoder
        lr_decay_G = self.opt.lr_G / self.opt.nepochs_decay
        self.current_lr_image_encoder_and_grasp_predictor -= lr_decay_G
        if not self.opt.no_discriminator:
            lr_decay_D = self.opt.lr_D / self.opt.nepochs_decay
            self.current_lr_discriminator -= lr_decay_D
        self.set_learning_rate(plot=False)
        print('update image_encoder_and_grasp_predictor and grasp_generator learning rate: %f -> %f' %
              (self.current_lr_image_encoder_and_grasp_predictor + lr_decay_G, self.current_lr_image_encoder_and_grasp_predictor))
        if not self.opt.no_discriminator:
            print('update discriminator learning rate: %f -> %f' %
                  (self.current_lr_discriminator + lr_decay_D, self.current_lr_discriminator))

    def set_learning_rate(self, plot=True):
        for param_group in self.optimizer_image_enc_and_grasp_predictor.param_groups:
            param_group['lr'] = self.current_lr_image_encoder_and_grasp_predictor
        for param_group in self.optimizer_grasp_generator.param_groups:
            param_group['lr'] = self.current_lr_image_encoder_and_grasp_predictor
        if not self.opt.no_discriminator:
            for param_group in self.optimizer_discriminator.param_groups:
                param_group['lr'] = self.current_lr_discriminator
        if plot:
            print('set image_encoder_and_grasp_predictor and grasp_generator learning rate to: %f' %
                  (self.current_lr_image_encoder_and_grasp_predictor))
            if not self.opt.no_discriminator:
                print('set discriminator learning rate to: %f' %
                      (self.current_lr_discriminator))

    def f1_score(self):
        f1_score = util.f1_score(self.true_positive, self.true_negative, self.false_positive, self.false_negative)
        return f1_score.cpu().numpy()
