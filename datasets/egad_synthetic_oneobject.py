import glob

import numpy as np

from datasets.dataset import DatasetBase
import trimesh
import math


class Dataset(DatasetBase):
    def __init__(self, opt, mode):
        super(Dataset, self).__init__(opt, mode)
        self.name = 'Dataset_egad_synthetic_one_object'
        self.debug = opt.debug
        self.object_mesh_dir = opt.object_mesh_dir
        self.grasp_dir = opt.grasp_dir
        self.should_pregenerate_data = opt.pregenerate_data
        self.num_viewpoints_per_object = opt.num_viewpoints_per_object
        self.setup_data()
        if self.should_pregenerate_data:
            self.pregenerate_data(self.num_viewpoints_per_object)

    def split_index(self, index):
        obj_id = math.floor(index/self.num_viewpoints_per_object)
        viewpoint_idx = index-self.num_viewpoints_per_object*obj_id
        return obj_id, viewpoint_idx

    def __getitem__(self, index):
        # Get object at random:

        id_obj, viewpoint_idx = self.split_index(index)

        if self.should_pregenerate_data:
            color, _, all_obj_verts, all_obj_verts_resampled800, all_obj_faces = self.pregenerated_data_per_object[
                id_obj][viewpoint_idx]
        else:
            color, _, all_obj_verts, all_obj_verts_resampled800, all_obj_faces = self.generate_data(
                id_obj)

        # Normalize:
        img = color[0].transpose(1, 2, 0)/256
        img = img - self.means_rgb
        img = img / self.std_rgb
        # pack data
        sample = {'rgb_img': img,
                  'object_id': id_obj,
                  '3d_points_object': all_obj_verts,
                  '3d_faces_object': all_obj_faces,
                  'object_points_resampled': all_obj_verts_resampled800,
                  'taxonomy': 0,
                  'hand_gt_representation': 0,
                  'hand_gt_pose': 0,
                  }

        return sample

    def __len__(self):
        return self.num_viewpoints_per_object*self.training_models

    def setup_data(self):

        models_original = glob.glob(
            self.object_mesh_dir + '*[0-9].ply')
        models_simplified = glob.glob(
            self.object_mesh_dir + '*_simplified.ply')
        models_original.sort()
        models_simplified.sort()
        self.models_original = np.array(models_original)
        self.models_simplified = []
        for i in self.models_original:
            for j in models_simplified:
                if i.split("/")[-1].split(".")[0] in j:
                    self.models_simplified.append(j)
                    break

        self.models_simplified = np.asarray(self.models_simplified)
        self.training_models = self.models_original.shape[0]
        for i in range(self.training_models):
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
            visual = 255*np.ones((obj_orig.vertices.shape[0], 3))
            colors = visual
            triangles = colors[obj_orig.faces]
            self.all_object_textures.append(np.uint8(triangles.mean(1)))
            self.all_object_vertices_simplified.append(obj_simp.vertices)
            self.all_object_faces_simplified.append(obj_simp.faces)

        self.resampled_objects_800verts = np.asarray(
            self.resampled_objects_800verts)
