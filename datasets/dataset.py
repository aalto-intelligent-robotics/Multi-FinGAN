import torch.utils.data as data
import kaolin
from kaolin.graphics.nmr.util import get_points_from_angles
import torch
import numpy as np
import pyquaternion


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, opt, mode):
        if dataset_name == 'ycb':
            from datasets.ycb_synthetic_oneobject import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'egad':
            from datasets.egad_synthetic_oneobject import Dataset
            dataset = Dataset(opt, mode)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)

        print('Dataset {} was created'.format(dataset.name))
        return dataset


class DatasetBase(data.Dataset):
    def __init__(self, opt, mode):
        super(DatasetBase, self).__init__()
        self.name = 'BaseDataset'
        self.opt = opt
        self.mode = mode
        self.setup_camera()

        self.all_object_vertices = []
        self.all_object_faces = []
        self.all_object_textures = []
        self.all_object_vertices_simplified = []
        self.resampled_objects_800verts = []
        self.all_object_faces_simplified = []
        self.all_grasp_translations = []
        self.all_grasp_rotations = []
        self.all_grasp_hand_configurations = []
        self.all_grasp_taxonomies = []

        # Resnet normalization values
        self.means_rgb = [0.485, 0.456, 0.406]
        self.std_rgb = [0.229, 0.224, 0.225]

        self.IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    def get_name(self):
        return self.name

    def setup_camera(self):
        self.renderer = kaolin.graphics.NeuralMeshRenderer(
            camera_mode='look_at')
        self.renderer.light_intensity_directional = 0.0
        self.renderer.light_intensity_ambient = 1.0
        camera_distance = 0.4
        elevation = 0.0
        azimuth = 0.0
        self.renderer.eye = get_points_from_angles(
            camera_distance, elevation, azimuth)

    def generate_data(self, id_obj):
        random_rot_np = pyquaternion.Quaternion().random().rotation_matrix
        random_rot = torch.FloatTensor(random_rot_np)
        rotated_verts = torch.matmul(
            random_rot, torch.FloatTensor(self.all_object_vertices[id_obj]).T).T
        rotated_verts = rotated_verts.unsqueeze(0).cuda()
        faces = torch.LongTensor(
            self.all_object_faces[id_obj]).unsqueeze(0).cuda()
        textures = torch.FloatTensor(self.all_object_textures[id_obj]).cuda()

        # Reshape as needed for renderer:
        textures = textures.reshape(
            1, len(textures), 1, 1, 1, 3).repeat(1, 1, 2, 2, 2, 1)

        # Rendering with CUDA:
        color, _, _ = self.renderer(rotated_verts, faces, textures=textures)
        all_object_vertices = [
            np.matmul(random_rot_np, self.all_object_vertices_simplified[id_obj].T).T]
        all_obj_verts_resampled800 = [
            np.matmul(random_rot_np, self.resampled_objects_800verts[id_obj].T).T]
        all_obj_faces = [self.all_object_faces_simplified[id_obj]]

        return color.cpu().data.numpy(), random_rot_np, all_object_vertices, all_obj_verts_resampled800, all_obj_faces

    def pregenerate_data(self, num_viewpoints_per_object):
        self.pregenerated_data_per_object = []
        for id_obj in range(self.training_models):
            current_object_data = []
            for _ in range(num_viewpoints_per_object):
                color, random_rot, all_object_vertices, all_obj_verts_resampled800, all_obj_faces = self.generate_data(
                    id_obj)
                current_object_data.append(
                    (color, random_rot, all_object_vertices, all_obj_verts_resampled800, all_obj_faces))
            self.pregenerated_data_per_object.append(current_object_data)

    def collate_fn(self, args):
        length = len(args)
        keys = list(args[0].keys())
        data = {}

        for _, key in enumerate(keys):
            data_type = []

            if key == 'rgb_img' or key == 'mask_img' or key == 'noise_img' or key == 'plane_eq' or key == 'hand_gt_representation' or key == 'hand_gt_pose':
                for j in range(length):
                    data_type.append(torch.FloatTensor(args[j][key]))
                data_type = torch.stack(data_type)
            elif key == 'label' or key == 'taxonomy':
                labels = []
                for j in range(length):
                    labels.append(args[j][key])
                data_type = torch.LongTensor(labels)
            else:
                for j in range(length):
                    data_type.append(args[j][key])
            data[key] = data_type
        return data
