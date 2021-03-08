from barrett_kinematics.barrett_layer.barrett_layer import BarrettLayer
import pickle
import pyquaternion
import trimesh
import os
import glob
from utils import util
import utils.plots as plot_utils
import torch
import numpy as np
from options.test_options import TestOptions

import time
from datasets.custom_dataset_data_loader import CustomDatasetDataLoader
from models.models import ModelsFactory

from joblib import Parallel, delayed

import pandas as pd
import subprocess


class Test:
    def __init__(self):
        self.opt = TestOptions().parse()
        if self.get_test_data():
            self.display = self.opt.display
            self.threshold_intersections = self.opt.threshold_intersections
            self.threshold_contact = self.opt.threshold_contact
            self.vertices_and_parts_on_barrett_hand_that_can_touch_object()
            try:
                os.makedirs(self.opt.save_folder)
            except:
                pass
            if self.opt.graspit:
                self.barrett_layer = BarrettLayer()

                self.save_path = self.opt.save_folder+self.opt.test_set
                self.simulation_experiment_graspit()
            else:
                self.setup_multi_fingan()
                with torch.no_grad():
                    self.simulation_experiment_multi_fingan()

    def get_test_data(self):
        if not os.path.isdir("data/test_data/") or len(os.listdir("data/test_data/")) < 2 or not os.path.isdir("data/meshes/") or len(os.listdir("data/meshes/")) < 2:
            while True:
                download_data = input("Test data does not exist. Want to download it (y/n)? ")
                if download_data == "y":
                    dir = os.path.dirname(os.path.realpath(__file__)) + "/data/"
                    print("Downloading test data. This will take some time so just sit back and relax.")
                    subprocess.Popen([dir+'download_test_data.sh %s' % dir], shell=True).wait()
                    print("Done downloading test data. Continuing with testing.")
                    return True
                elif download_data == "n":
                    print("You chose to not download test data. Terminating testing")
                    return False
        else:
            print("Testing data exists. Proceeding with Testing.")
            return True

    def vertices_and_parts_on_barrett_hand_that_can_touch_object(self):
        self.finger_vertices = [7, 37, 21, 51, 71, 34, 61, 86, 46, 22, 53, 40, 69, 52, 38, 13, 29, 19, 8, 81, 60, 72, 97, 36,
                                28, 45, 49, 66, 3, 70, 65, 68, 25, 95, 48, 57, 98, 27, 79, 15, 80, 89, 62, 100, 91, 5, 44, 63]
        self.finger_tip_vertices = [35, 98, 44, 90, 95, 77, 84, 73, 76, 25, 108, 64, 22, 24, 96, 23, 85, 79, 83, 30, 45, 47, 68, 54, 42, 69, 92, 86,
                                    19, 7, 94, 37, 99, 91, 11, 107, 0, 89, 57, 59, 109, 4, 65, 31, 2, 1, 10, 101, 52, 97, 87, 50, 72, 15, 106, 82, 12, 56, 78, 32, 46, 8]
        self.palm_vertices = [18, 205, 20, 90, 129, 166, 107, 70, 27, 161, 34, 211, 80, 114]
        self.parts_that_can_touch_vertices = [self.palm_vertices, self.finger_vertices, self.finger_vertices,
                                              self.finger_vertices, self.finger_tip_vertices, self.finger_tip_vertices, self.finger_tip_vertices]
        # Only the palm (0), three proximal links (3,4,5) and three distal links (6,7,8) can touch an object
        # We exclute the knuckle links on finger 1 and 2 in https://support.barrett.com/wiki/Hand/280/KinematicsJointRangesConversionFactors
        self.parts_that_can_touch = [0, 3, 4, 5, 6, 7, 8]

    def setup_multi_fingan(self):
        self.device = torch.device('cuda:{}'.format(
            self.opt.gpu_ids[0])) if self.opt.gpu_ids[0] != -1 and torch.cuda.is_available() else torch.device('cpu')
        self.save_path = self.opt.save_folder+self.opt.test_set
        self.num_grasps_to_sample = self.opt.num_grasps_to_sample
        # Let's set batch size to 1 as we generate grasps from one viewpoint only
        self.opt.batch_size = 1

        self.opt.object_finger_contact_threshold = 0.004
        self.opt.optimize_finger_tip = True
        self.opt.num_viewpoints_per_object = self.opt.num_viewpoints_per_object_test

        self.opt.n_threads_train = self.opt.n_threads_test

        self.model = ModelsFactory.get_by_name(self.opt)

        data_loader_test = CustomDatasetDataLoader(self.opt, mode='val')
        self.dataset_test = data_loader_test.load_data()
        self.dataset_test_size = len(data_loader_test)
        print('#test images = %d' % self.dataset_test_size)
        self.model.set_eval()

        self.hand_in_parts = self.calculate_number_of_hand_vertices_per_part(self.model.barrett_layer)

    def calculate_number_of_hand_vertices_per_part(self, hand):
        hand_in_parts = [0]  # One palm link
        hand_in_parts.append(hand.num_vertices_per_part[0])
        for _ in range(2):  # Two knuckles
            hand_in_parts.append(hand_in_parts[-1]+hand.num_vertices_per_part[1])
        for _ in range(3):  # three proximal links
            hand_in_parts.append(hand_in_parts[-1]+hand.num_vertices_per_part[2])
        for _ in range(3):  # Three distal links
            hand_in_parts.append(hand_in_parts[-1]+hand.num_vertices_per_part[3])
        return hand_in_parts

    def import_meshes(self):
        models = glob.glob(self.opt.object_mesh_dir)
        models.sort()
        models = np.array(models)
        all_meshes = {}
        for i in range(models.shape[0]):
            curr_mesh = {}
            if self.opt.test_set == "egad":
                name = os.path.splitext(models[i].split("/")[-1])[0]
            elif self.opt.test_set == "ycb":
                name = models[i].split("/")[-3]
            obj = trimesh.load(models[i])
            resampled_objects_800verts = trimesh.sample.sample_surface_even(obj, 800)[
                0]
            curr_mesh["vertices"] = np.expand_dims(obj.vertices, 0)
            curr_mesh["faces"] = np.expand_dims(obj.faces, 0)
            curr_mesh["resampled_vertices"] = resampled_objects_800verts
            all_meshes[name] = curr_mesh
        return all_meshes

    def simulation_experiment_graspit(self):
        grasp_files = glob.glob(self.opt.graspit_grasp_dir+"*pkl")
        time_graspit = np.loadtxt(self.opt.graspit_grasp_dir+"time.txt")

        grasp_files.sort()
        all_meshes = self.import_meshes()
        df = pd.DataFrame(columns=["Number of contacts", "Intersection", "Epsilon quality",
                                   "Epsilon quality from GraspIt!", "Volume quality from GraspIt!", "Objects name", "Time for graspit"])

        for i, grasp_file in enumerate(grasp_files):
            grasps = pickle.load(open(grasp_file, "rb"))
            mesh_name = os.path.splitext(os.path.basename(grasp_file))[0]
            all_grasp_stabilities = []
            try:
                mesh = all_meshes[mesh_name]
            except:
                continue
            if type(grasps) is list:
                for grasp_idx, grasp in enumerate(grasps):
                    data = self.evaluate_graspit_grasp(grasp, mesh, grasp_idx)
                    all_grasp_stabilities.append(data)
            else:
                all_grasp_stabilities.append(self.evaluate_graspit_grasp(
                    grasps, mesh, i))
            #np.save(self.save_path+"_"+str(i)+".npy", np.asarray(all_grasp_stabilities))
            all_grasp_stabilities = np.asarray(all_grasp_stabilities)
            sorted_according_to_quality = (-1*all_grasp_stabilities[:, 2]).argsort()
            all_grasp_stabilities = all_grasp_stabilities[sorted_according_to_quality]
            df_per_object = pd.DataFrame(data=np.asarray(all_grasp_stabilities), columns=["Number of contacts", "Intersection", "Epsilon quality",
                                                                                          "Epsilon quality from GraspIt!", "Volume quality from GraspIt!"])
            df_per_object["Objects name"] = mesh_name
            df_per_object["Time for graspit"] = time_graspit[i]
            df = df.append(df_per_object)
        df.to_csv(self.save_path+"_results.csv")

    def evaluate_graspit_grasp(self, grasp, mesh, i):
        grasp_pose = grasp["pose"]
        grasp_pose_torch = torch.eye(4).unsqueeze(0)
        grasp_pose_torch[:, :3, :3] = torch.from_numpy(pyquaternion.Quaternion(
            np.array(grasp_pose[3:])[[3, 0, 1, 2]]).rotation_matrix)
        grasp_pose_torch[:, :3, 3] = torch.FloatTensor(grasp_pose[:3])
        grasp_joints_torch = torch.FloatTensor(
            util.joints_to_grasp_representation(grasp["joints"])).unsqueeze(0)

        * verts, _ = self.barrett_layer(grasp_pose_torch, grasp_joints_torch)
        verts_concatenated, faces_concatenated = util.concatenate_barret_vertices_and_faces(
            verts, self.barrett_layer.gripper_faces)

        forces_post, torques_post, normals_post, finger_is_touching, vertices_that_touch = self.get_contact_points(
            verts, self.barrett_layer, mesh["resampled_vertices"])
        obj_mesh = util.create_mesh(mesh['vertices'][0], mesh['faces'][0])
        hand_mesh = util.create_mesh(
            verts_concatenated[0], faces_concatenated[0])
        intersections = util.intersect_vox(hand_mesh, obj_mesh, 0.005)
        if intersections > self.threshold_intersections or len(forces_post) < 3:
            data = [
                finger_is_touching.sum(-1), intersections, 0.0, grasp["epsilon"], grasp["volume"]]
        else:
            G = util.grasp_matrix(np.array(forces_post).transpose(), np.array(
                torques_post).transpose(), np.array(normals_post).transpose())
            grasp_metric = util.min_norm_vector_in_facet(G)[0]
            data = [finger_is_touching.sum(-1), intersections, grasp_metric, grasp["epsilon"],
                    grasp["volume"]]
        if self.display:
            object_vertices = [np.asarray(mesh['vertices']).squeeze()]
            object_faces = [np.asarray(mesh['faces']).squeeze()]
            hand_vertices = [verts_concatenated[0]]
            hand_faces = [faces_concatenated[0]]
            self.save_grasp_image(hand_vertices, hand_faces, object_vertices, object_faces, "/tmp/grasp_graspit"+str(i)+".png")
        return data

    def save_grasp_image(self, hand_vertices, hand_faces, object_vertices, object_faces, save_dir):
        image_numpy = plot_utils.plot_scene_w_grasps(object_vertices, object_faces, hand_vertices, hand_faces)
        util.save_image(image_numpy, save_dir)

    def simulation_experiment_multi_fingan(self):
        all_data_to_save = np.zeros(
            (self.dataset_test_size*self.num_grasps_to_sample, 9))

        print("SAMPLING %d POSSIBLE GRASPS" % (self.num_grasps_to_sample))
        for i_test_batch, test_batch in enumerate(self.dataset_test):
            print("PROCESSING TEST SAMPLE %d" % (i_test_batch))
            time_batch = time.time()
            self.model.set_input(test_batch)

            resampled_obj_verts = torch.FloatTensor(
                self.model.input_obj_resampled_verts[0][0]).to(self.device)
            #self.model.input_object_id = [0]
            # self.model.input_center_objects = torch.FloatTensor(
            #    [self.model.input_obj_resampled_verts[0][0].mean(0)]).cuda()
            time_s = time.time()
            self.model.forward_G(False)
            time_generating = time.time()-time_s
            print("Time for generating grasps " + str(time_generating))
            verts = self.model.refined_handpose_concatenated
            time_s = time.time()
            forces_post, torques_post, normals_post, parts_are_touching, vertices_that_touch = self.get_contact_points_batch(
                verts, self.model.barrett_layer, resampled_obj_verts, self.num_grasps_to_sample)
            time_contact = time.time()-time_s
            enough_contacts = parts_are_touching.sum(-1) >= 3
            time_s = time.time()
            intersections = np.zeros(self.num_grasps_to_sample)
            obj_mesh = util.create_mesh(
                test_batch['3d_points_object'][0][0], test_batch['3d_faces_object'][0][0])

            intersections = np.asarray(Parallel(n_jobs=24)(
                delayed(util.get_intersection)(verts[i].cpu().numpy(), self.model.refined_handpose_faces[i].cpu().numpy(), obj_mesh)
                for i in range(self.num_grasps_to_sample))
            )
            time_intersection = time.time()-time_s
            print("Time to calculate all intersections " + str(time_intersection))
            no_intersection = intersections < self.threshold_intersections
            grasp_to_check_quality = no_intersection & enough_contacts.cpu().numpy()

            time_s = time.time()
            qualities = np.zeros(self.num_grasps_to_sample)
            for dim in grasp_to_check_quality.nonzero()[0]:
                try:
                    G = util.grasp_matrix(torch.stack(forces_post[dim]).T.cpu().data.numpy(), torch.stack(torques_post[dim]).T.cpu().data.numpy(), torch.stack(
                        normals_post[dim]).T.cpu().data.numpy())
                except:
                    print("Problem with batch "+str(i_test_batch))
                    continue
                grasp_metric = util.min_norm_vector_in_facet(G)[0]
                qualities[dim] = grasp_metric

            if self.display:
                best_grasps = (-1*qualities).argsort()
                object_vertices = [self.model.input_obj_verts[0][0]]
                object_faces = [self.model.input_obj_faces[0][0]]
                hand_vertices = [self.model.refined_handpose_concatenated[best_grasps][0].cpu()]
                hand_faces = [self.model.refined_handpose_faces[0].cpu()]
                self.save_grasp_image(hand_vertices, hand_faces, object_vertices, object_faces,
                                      "/tmp/multi_fingan_best_grasp_batch_"+str(i_test_batch)+".png")

            time_for_batch = time.time()-time_batch
            time_quality = time.time()-time_s
            print("Time to calculate qualities " + str(time_quality))
            print("Total time for batch " + str(time_for_batch))
            batch_data_to_save = self.populate_batch_data(
                parts_are_touching.sum(-1).data, intersections, qualities, time_generating, time_contact, time_intersection, time_quality, time_for_batch)
            all_data_to_save[i_test_batch*self.num_grasps_to_sample:(i_test_batch+1)*self.num_grasps_to_sample] = batch_data_to_save
        df = pd.DataFrame(data=all_data_to_save, columns=["Object id", "Num touches", "Intersection", "Qualities", "Average time per grasp generating",
                                                          "Average time per grasp contact", "Average time per grasp intersection", "Average time per grasp qualities", "Average time total"])

        df.to_csv(self.save_path+"_results.csv")

    def populate_batch_data(self, num_contacts, intersections, qualities, time_to_generate, time_for_contact, time_for_intersections, time_for_quality, total_time):
        array_for_data = np.zeros(
            (self.num_grasps_to_sample, 9))
        array_for_data[:, 0] = int(self.model.input_object_id[0])
        array_for_data[:, 1] = num_contacts
        array_for_data[:, 2] = intersections
        array_for_data[:, 3] = qualities
        array_for_data[:, 4] = time_to_generate/self.num_grasps_to_sample
        array_for_data[:, 5] = time_for_contact/self.num_grasps_to_sample
        array_for_data[:, 6] = time_for_intersections / self.num_grasps_to_sample
        array_for_data[:, 7] = time_for_quality/self.num_grasps_to_sample
        array_for_data[:, 8] = total_time/self.num_grasps_to_sample
        return array_for_data

    def concatenate_faces(self, hand, use_torch=False):
        if use_torch:
            faces = [hand.gripper_faces[0]]
            for _ in range(2):
                faces.append(hand.gripper_faces[1])
            for _ in range(3):
                faces.append(hand.gripper_faces[2])
            for _ in range(3):
                faces.append(hand.gripper_faces[3])
        else:
            faces = [hand.gripper_faces[0].cpu().data.numpy()]
            for i in range(3):
                faces.append(hand.gripper_faces[2].cpu().data.numpy())
            for i in range(3):
                faces.append(hand.gripper_faces[3].cpu().data.numpy())
        return faces

    def get_contact_points_batch(self, all_hand_vertices, hand, resampled_obj_verts, batch_size):
        faces = self.concatenate_faces(hand, True)
        all_forces = [[] for _ in range(batch_size)]
        all_torques = [[] for _ in range(batch_size)]
        all_normals = [[] for _ in range(batch_size)]
        vertices_that_touches = [[] for _ in range(batch_size)]
        # We consider seven parts: the palm, the three finger tips, and the three finger bases
        # parts_touching = torch.zeros(palm_vertices.shape[0], 7)
        # distance_touching_vertices = self.calculate_contact_batch(points_3d)
        resampled_obj_verts = resampled_obj_verts.unsqueeze(
            0).repeat(batch_size, 1, 1)
        dists = util.get_distance_vertices_batched(
            resampled_obj_verts, all_hand_vertices)

        parts_are_touching = torch.zeros(batch_size, len(self.parts_that_can_touch))
        for j, i in enumerate(self.parts_that_can_touch):
            val, dims = dists[:, self.hand_in_parts[i]:self.hand_in_parts[i+1]][:, self.parts_that_can_touch_vertices[j]].min(dim=1)
            parts_are_touching[:, j] = val < self.threshold_contact
            batches_to_eval = torch.where(val < self.threshold_contact)[0]
            for batch in batches_to_eval:
                temp = torch.where(dims[batch] == faces[i])[0]
                normal = util.get_normal_face_batched(
                    all_hand_vertices[batch], faces[i][temp]+self.hand_in_parts[i]).mean(0) * 1e5
                normal = normal/normal.norm()
                all_torques[batch].append(
                    torch.FloatTensor([0, 0, 0]).to(self.device))
                all_normals[batch].append(normal)
                all_forces[batch].append(normal)
                vertices_that_touches[batch].append(
                    all_hand_vertices[batch][self.hand_in_parts[i]+dims[batch]])
        return all_forces, all_torques, all_normals, parts_are_touching, vertices_that_touches

    def get_contact_points(self, all_hand_vertices, hand, resampled_obj_verts, debug=False):
        palm_vertices, _, all_finger_vertices, all_finger_tip_vertices = all_hand_vertices
        hand_vertices = [palm_vertices[0].cpu().data.numpy()]
        for base_finger in all_finger_vertices[0]:
            hand_vertices.append(base_finger.cpu().data.numpy())
        for finger_tip in all_finger_tip_vertices[0]:
            hand_vertices.append(finger_tip.cpu().data.numpy())
        hand_faces = self.concatenate_faces(hand)
        forces = []
        torques = []
        normals = []
        # We consider seven parts: the palm, the three finger tips, and the three finger bases
        part_is_touching = torch.zeros(len(self.parts_that_can_touch))

        vertices_that_touches = []

        for i in range(len(self.parts_that_can_touch)):
            # Get the distance between all vertices on hand and all sampled points on the object
            dists = util.get_distance_vertices(
                resampled_obj_verts, hand_vertices[i])[self.parts_that_can_touch_vertices[i]]
            if np.min(dists) < self.threshold_contact:
                part_is_touching[i] = 1
                vertices_that_touches.append(
                    hand_vertices[i][np.argmin(dists)])
                # Get all incident faces of the vertice on the hand that is in contact with the object
                faces = np.where(np.argmin(dists) == hand_faces[i])[0]
                normal = []
                # Calculate the normal of all incident faces
                for j in range(len(faces)):
                    normal.append(util.get_normal_face(hand_vertices[i][hand_faces[i][faces[j], 0]], hand_vertices[i]
                                                       [hand_faces[i][faces[j], 1]], hand_vertices[i][hand_faces[i][faces[j], 2]]))
                # The contact normal is the average of all face norlas that are incident to the vertice in contact
                normal = np.mean(normal, 0) * 1e5
                normal = normal/np.sqrt((np.array(normal)**2).sum())
                torques.append([0, 0, 0])
                normals.append(normal)
                forces.append(normal)

        return forces, torques, normals, part_is_touching, vertices_that_touches


if __name__ == '__main__':
    Test()
