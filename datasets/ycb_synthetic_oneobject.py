import glob

import numpy as np
import pickle

from datasets.dataset import DatasetBase
from utils import util
import pyquaternion
import trimesh
import math


class Dataset(DatasetBase):
    def __init__(self, opt, mode):
        super(Dataset, self).__init__(opt, mode)
        self.name = 'Dataset_ycb_synthetic_one_object'
        self.debug = opt.debug
        self.object_mesh_dir = opt.object_mesh_dir
        self.grasp_dir = opt.grasp_dir
        self.should_pregenerate_data = opt.pregenerate_data
        # All models in the ycb dataset we train on are:
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 17, 18, 20, 21, 30, 31, 32, 34, 40, 41, 49]
        # Test set is now the same as the val set
        self.test_models = np.asarray([3, 7, 40])
        self.num_viewpoints_per_object = opt.num_viewpoints_per_object
        # read dataset
        self.setup_data()

        # Either we pre-generate data once and reuse these all the time which significantly speeds
        # up data loading or we generate new data for each batch
        if self.should_pregenerate_data:
            self.pregenerate_data(self.num_viewpoints_per_object)

    def split_index(self, index):
        obj_id = math.floor(index/self.num_viewpoints_per_object)
        viewpoint_idx = index-self.num_viewpoints_per_object*obj_id
        return obj_id, viewpoint_idx

    def __getitem__(self, index):
        # Get object at random:

        id_obj, viewpoint_idx = self.split_index(index)
        id_grasp = np.random.randint(0, len(self.all_grasp_rotations[id_obj]))

        if self.should_pregenerate_data:
            color, random_rot, all_object_vertices, all_obj_verts_resampled800, all_obj_faces = self.pregenerated_data_per_object[
                id_obj][viewpoint_idx]
        else:
            color, random_rot, all_object_vertices, all_obj_verts_resampled800, all_obj_faces = self.generate_data(
                id_obj)

        # Get random grasp:
        grasp_rot = self.all_grasp_rotations[id_obj][id_grasp]
        grasp_trans = self.all_grasp_translations[id_obj][id_grasp]
        grasp_dof = self.all_grasp_hand_configurations[id_obj][id_grasp]

        # Normalize:
        img = color[0].transpose(1, 2, 0)/256
        img = img - self.means_rgb
        img = img / self.std_rgb

        # Taxonomy ground truth:
        all_grasp_taxonomies = self.all_grasp_taxonomies[id_obj]

        grasp_rot = pyquaternion.Quaternion(
            np.array(grasp_rot)[[3, 0, 1, 2]]).rotation_matrix
        grasp_pose = np.eye(4)
        grasp_pose[:3, :3] = grasp_rot
        grasp_pose[:3, 3] = grasp_trans

        grasp_to_object_transformation = np.eye(4)
        grasp_to_object_transformation[:3, :3] = random_rot
        grasp_pose = np.matmul(grasp_to_object_transformation, grasp_pose)

        grasp_repr = util.joints_to_grasp_representation(grasp_dof)

        # pack data
        sample = {'rgb_img': img,
                  'object_id': id_obj,
                  'taxonomy': all_grasp_taxonomies,
                  'hand_gt_representation': grasp_repr,
                  'hand_gt_pose': grasp_pose,
                  '3d_points_object': all_object_vertices,
                  '3d_faces_object': all_obj_faces,
                  'object_points_resampled': all_obj_verts_resampled800,
                  }

        return sample

    def __len__(self):
        return self.num_viewpoints_per_object*self.training_models

    def setup_data(self):

        models_original = glob.glob(
            self.object_mesh_dir + '/0*/google_16k/textured.obj')
        models_simplified = glob.glob(
            self.object_mesh_dir + '/0*/google_16k/textured_simplified.obj')

        models_original.sort()
        models_simplified.sort()
        objects_in_YCB = np.load('./data/objects_in_YCB.npy')
        if self.mode == "train":
            objects_in_YCB = np.setdiff1d(objects_in_YCB, self.test_models)
        elif self.mode == "test":
            objects_in_YCB = self.test_models
        elif self.mode == "val":
            # Val models are both train and test models but from new viewpoints
            objects_in_YCB = objects_in_YCB

        self.models_original = np.array(models_original)[objects_in_YCB]
        self.models_simplified = []
        for i in self.models_original:
            for j in models_simplified:
                if i.split("/")[-3] in j:
                    self.models_simplified.append(j)
                    break

        self.models_simplified = np.asarray(self.models_simplified)
        self.training_models = self.models_original.shape[0]
        for i in range(self.training_models):
            # Kaolin doesn't load textures from obj
            # so using Trimesh
            obj_orig = trimesh.load(self.models_original[i])
            obj_simp = trimesh.load(self.models_simplified[i])
            object_center = obj_orig.centroid
            obj_orig.vertices -= object_center
            obj_simp.vertices -= object_center
            resampled = trimesh.sample.sample_surface_even(obj_orig, 800)[0]
            if resampled.shape[0] < 800:
                resampled = trimesh.sample.sample_surface(obj_orig, 800)[0]
            self.resampled_objects_800verts.append(resampled)

            self.all_object_faces.append(obj_orig.faces)
            self.all_object_vertices.append(obj_orig.vertices)
            # Get texture (it has to be face colors):
            visual = obj_orig.visual.to_color()
            colors = visual.vertex_colors[:, :3]
            triangles = colors[obj_orig.faces]
            self.all_object_textures.append(np.uint8(triangles.mean(1)))
            self.all_object_vertices_simplified.append(obj_simp.vertices)
            self.all_object_faces_simplified.append(obj_simp.faces)

            # Get all grasps for this object:
            object_grasp_translations = []
            object_grasp_rotations = []
            object_grasp_hand_configurations = []
            grasp_files = glob.glob(
                self.grasp_dir + "obj_%d_*" % objects_in_YCB[i])
            grasp_files.sort()
            available_taxonomy = np.zeros(7)
            for file in grasp_files:
                data = pickle.load(open(file, 'rb'), encoding='latin')
                object_grasp_translations.append(data['pose'][:3] - object_center)
                object_grasp_rotations.append(data['pose'][3:])
                object_grasp_hand_configurations.append(data['joints'])
                available_taxonomy[data['taxonomy'] - 1] += 1

            self.all_grasp_translations.append(object_grasp_translations)
            self.all_grasp_rotations.append(object_grasp_rotations)
            self.all_grasp_hand_configurations.append(object_grasp_hand_configurations)
            self.all_grasp_taxonomies.append((available_taxonomy > 0)*1)

        self.resampled_objects_800verts = np.asarray(
            self.resampled_objects_800verts)
