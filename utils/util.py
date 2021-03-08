from __future__ import print_function
from collections import OrderedDict
import trimesh
import quaternion
import torch
import math
import os
import numpy as np
from PIL import Image
import copy
import cvxopt as cvx
cvx.solvers.options['show_progress'] = False


def batch_pairwise_dist(x, y, use_cuda=True):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    if use_cuda:
        dtype = torch.cuda.LongTensor
    else:
        dtype = torch.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)
    rx = (xx[:, diag_ind_x,
             diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1)))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_image(image_numpy, image_path):
    mkdir(os.path.dirname(image_path))
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def convert_qt_to_T_matrix(qt):
    q = qt[3:]
    t = qt[:3]
    q = np.quaternion(q[-1], q[0], q[1], q[2])
    R = quaternion.as_rotation_matrix(q)
    T = np.eye(4)
    T[:3, 3] = t
    T[:3, :3] = R
    return T


def concatenate_barret_vertices_and_faces(vertices, faces, use_torch=False):
    all_vertices_per_batch = []
    all_faces_per_batch = []

    for i in range(vertices[0].shape[0]):
        flattened_vertices = []
        flattened_faces = []
        num_vertices = 0
        for v, f in zip(vertices, faces):
            curr_v = v[i]
            curr_f = f+num_vertices
            if len(v[i].shape) > 2:
                curr_f = curr_f.unsqueeze(0).repeat(v[i].shape[0], 1, 1)
                curr_f[1:] += curr_v.shape[1]
                if curr_f.shape[0] > 2:
                    curr_f[2] += curr_v.shape[1]
                curr_f = curr_f.reshape(
                    curr_f.shape[0]*curr_f.shape[1], curr_f.shape[2])
                num_vertices += curr_v.shape[1]*curr_v.shape[0]
                curr_v = v[i].reshape(
                    v[i].shape[0] * v[i].shape[1], v[i].shape[2])
            else:
                num_vertices += curr_v.shape[0]
            if use_torch:
                flattened_vertices.append(curr_v)
                flattened_faces.append(curr_f)
            else:
                flattened_vertices.append(curr_v.cpu().data.numpy())
                flattened_faces.append(curr_f.cpu().data.numpy())
        if use_torch:
            all_vertices_per_batch.append(torch.cat(flattened_vertices))
            all_faces_per_batch.append(torch.cat(flattened_faces))
        else:
            all_vertices_per_batch.append(np.concatenate(flattened_vertices))
            all_faces_per_batch.append(np.concatenate(flattened_faces))
    if use_torch:
        return torch.stack(all_vertices_per_batch), all_faces_per_batch
    else:
        # np.concatenate(flattened_vertices)
        return all_vertices_per_batch, all_faces_per_batch


def concatenate_barret_vertices(vertices, use_torch=False):
    all_vertices_per_batch = []
    for i in range(vertices[0].shape[0]):
        flattened_vertices = []
        for v in vertices:
            curr_v = v[i]
            if len(v[i].shape) > 2:
                curr_v = v[i].reshape(
                    v[i].shape[0] * v[i].shape[1], v[i].shape[2])
            if use_torch:
                flattened_vertices.append(curr_v)
            else:
                flattened_vertices.append(curr_v.cpu().data.numpy())
        if use_torch:
            all_vertices_per_batch.append(torch.cat(flattened_vertices))
        else:
            all_vertices_per_batch.append(np.concatenate(flattened_vertices))
    if use_torch:
        return torch.stack(all_vertices_per_batch)
    else:
        return all_vertices_per_batch


def concatenate_barret_vertices_and_vertice_areas(vertices, vertice_areas, use_torch=False):
    all_vertices_per_batch = []
    all_vertice_areas_per_batch = []
    for i in range(vertices[0].shape[0]):
        flattened_vertices = []
        flattened_v_areas = []
        for v, v_a in zip(vertices, vertice_areas):
            curr_v = v[i]
            curr_v_a = v_a
            if len(v[i].shape) > 2:
                curr_v = v[i].reshape(
                    v[i].shape[0] * v[i].shape[1], v[i].shape[2])
                curr_v_a = curr_v_a.repeat(v[i].shape[0])
            if use_torch:
                flattened_vertices.append(curr_v)
                flattened_v_areas.append(curr_v_a)
            else:
                flattened_vertices.append(curr_v.cpu().data.numpy())
                flattened_v_areas.append(curr_v_a.cpu().data.numpy())
        if use_torch:
            all_vertices_per_batch.append(torch.cat(flattened_vertices))
            all_vertice_areas_per_batch.append(torch.cat(flattened_v_areas))
        else:
            all_vertices_per_batch.append(np.concatenate(flattened_vertices))
            all_vertice_areas_per_batch.append(
                np.concatenate(flattened_v_areas))
    if use_torch:
        return torch.stack(all_vertices_per_batch), torch.stack(all_vertice_areas_per_batch)
    else:
        return all_vertices_per_batch, all_vertice_areas_per_batch


def joints_to_grasp_representation(joints):
    dofs = np.zeros(7)
    dofs[0] = joints[0]
    dofs[1] = joints[4]
    dofs[2] = joints[1]
    dofs[3] = joints[6]
    dofs[4] = joints[5]
    dofs[5] = joints[2]
    dofs[6] = joints[7]
    return dofs


def load_mesh(mesh_file):
    mesh = trimesh.load(mesh_file)
    object_info = {}
    object_info["verts"] = mesh.vertices
    object_info["faces"] = mesh.faces
    object_info["verts_resampled"] = trimesh.sample.sample_surface_even(mesh, 800)[
        0]

    return object_info


def calculate_classification_statistics(y_true, y_pred):
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    return tp, tn, fp, fn


def f1_score(tp, tn, fp, fn):
    '''Calculate F1 score. Can work with gpu tensors

    The original implementation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision*recall) / (precision + recall + epsilon)
    return f1


def rotm_to_axis_angles(R, device="cpu"):
    trace = torch.einsum('bii->b', R)
    theta = torch.acos((trace-1)/2)
    eps1 = 0.01
    eps2 = 0.1
    axis_angles = torch.zeros((R.shape[0], 3)).to(device)
    axis_angles[:, 0] = R[:, 2, 1]-R[:, 1, 2]
    axis_angles[:, 1] = R[:, 0, 2]-R[:, 2, 0]
    axis_angles[:, 2] = R[:, 1, 0]-R[:, 0, 1]
    temp = 1/(2*torch.sin(theta)).unsqueeze(-1)
    axis_angles *= temp
    singularities = torch.where(((R[:, 0, 1]-R[:, 1, 0]).abs() < eps1) & ((R[:, 0, 2]-R[:, 2, 0]).abs()
                                                                          < eps1) & ((R[:, 1, 2]-R[:, 2, 1]).abs() < eps1))[0]
    if singularities.nelement() > 0:
        theta[singularities] = math.pi
        for i in singularities:
            trace_sing = R[i].trace()
            if (((R[i, 0, 1]+R[i, 1, 0]).abs() < eps2) & ((R[i, 0, 2] + R[i, 2, 0]).abs() < eps2) & ((R[i, 1, 2]+R[i, 2, 1]).abs() < eps1) & ((trace_sing-3).abs() < eps2)):
                axis_angles[i] = torch.zeros(3)
                axis_angles[i, 0] = 1
                theta[i] = 0
            else:
                theta[i] = math.pi
                xx = 0.5*(R[i, 0, 0]+1)
                yy = 0.5*(R[i, 1, 1]+1)
                zz = 0.5*(R[i, 2, 2]+1)
                xy = (R[i, 0, 1]+R[i, 1, 0])/4
                xz = (R[i, 0, 2]+R[i, 2, 0])/4
                yz = (R[i, 1, 2]+R[i, 2, 1])/4
                if (xx > yy) & (xx > zz):
                    if (xx < eps1):
                        x = 0
                        y = 0.7071
                        z = 0.7071
                    else:
                        x = torch.sqrt(xx)
                        y = xy/x
                        z = xz/x
                elif (yy > xx):
                    if (yy < eps1):
                        x = 0.7071
                        y = 0
                        z = 0.7071
                    else:
                        y = torch.sqrt(yy)
                        x = xy/y
                        z = yz/y
                else:
                    if (zz < eps1):
                        x = 0.7071
                        y = 0.7071
                        z = 0
                    else:
                        z = torch.sqrt(zz)
                        x = xz/z
                        y = yz/z
                axis_angles[i, 0] = x
                axis_angles[i, 1] = y
                axis_angles[i, 2] = z
    return axis_angles*theta.view(theta.shape[0], 1)


def axis_angles_to_rotation_matrix(angle_axis, eps=1e-6, device="cpu"):
    theta = angle_axis.norm(dim=-1)
    angle_axis = angle_axis/theta.view(theta.shape[0], 1)
    k_one = 1.0
    wxyz = angle_axis  # / (theta + eps)
    wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
    cos_theta = torch.cos(theta).unsqueeze(1)
    sin_theta = torch.sin(theta).unsqueeze(1)
    r00 = cos_theta + wx * wx * (k_one - cos_theta)
    r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
    r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
    r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
    r11 = cos_theta + wy * wy * (k_one - cos_theta)
    r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
    r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
    r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
    r22 = cos_theta + wz * wz * (k_one - cos_theta)
    rotation_matrix = torch.cat(
        [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1).to(device)
    return rotation_matrix.view(-1, 3, 3)


def rad2deg(rads):
    return 180. * rads / math.pi


def deg2rad(degs):
    return math.pi * degs / 180.


def valid_hand_conf(degs, device="cpu", only_check_spread=False):
    deg_max = torch.FloatTensor([180, 140, 140, 140, 48, 48, 48]).to(device)
    deg_min = torch.zeros(7).to(device)
    if only_check_spread:
        return torch.sum(degs[:, 0]-1e-2 > deg_max[0])+torch.sum(degs[:, 0]+1e-2 < deg_min[0])
    else:
        return torch.sum(degs-1e-2 > deg_max)+torch.sum(degs+1e-2 < deg_min)


def euclidean_distance(d1, d2):
    return torch.norm(d1-d2, dim=-1)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def axis_angle_rot(angle_axis, vec):
    theta = angle_axis.norm(dim=-1)
    vec = vec.repeat(theta.shape[0], 1)
    angle_axis = angle_axis/theta.view(theta.shape[0], 1)
    angle_cos = torch.cos(theta)
    angle_sin = torch.sin(theta)
    vec_rot = vec*angle_cos.view(theta.shape[0], 1)+torch.cross(angle_axis, vec)*angle_sin.view(theta.shape[0], 1) + \
        angle_axis*(angle_axis*vec).sum(-1, keepdim=True) * \
        (1-angle_cos.view(theta.shape[0], 1))
    return vec_rot


def axis_angle_distance(random_rots, R):
    random_rots_normed = random_rots/random_rots.norm(dim=-1, keepdim=True)
    R_normed = R/R.norm(dim=-1, keepdim=True)
    return 1-(random_rots_normed*R_normed).sum(-1)


def average_dictionary(list_dict):
    averaged_dictionary = OrderedDict()
    for dict_losses in list_dict:
        for key in dict_losses.keys():
            if key in averaged_dictionary:
                averaged_dictionary[key] += dict_losses[key]
            else:
                averaged_dictionary[key] = copy.deepcopy(dict_losses[key])
    for key in averaged_dictionary.keys():
        averaged_dictionary[key] /= len(list_dict)
    return averaged_dictionary


def concatenate_dictionary(list_dict):
    concatenated_dictionary = {}
    for curr_dict in list_dict:
        for key in curr_dict.keys():
            if key in concatenated_dictionary:
                concatenated_dictionary[key].append(curr_dict[key])
            else:
                concatenated_dictionary[key] = [curr_dict[key]]
    return concatenated_dictionary


def intersect_vox(obj_mesh, hand_mesh, pitch=0.01):
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume


def create_mesh(input_vertices, input_faces):
    return trimesh.Trimesh(input_vertices, input_faces)


def get_intersection(hand_vertices, hand_faces, obj_mesh):
    hand_mesh = create_mesh(hand_vertices, hand_faces)
    intersections = intersect_vox(hand_mesh, obj_mesh, 0.005)
    return intersections


def min_norm_vector_in_facet(facet, wrench_regularizer=1e-10):
    """ Finds the minimum norm point in the convex hull of a given facet (aka simplex) by solving a QP.

    Parameters
    ----------
    facet : 6xN :obj:`numpy.ndarray`
        vectors forming the facet
    wrench_regularizer : float
        small float to make quadratic program positive semidefinite

    Returns
    -------
    float
        minimum norm of any point in the convex hull of the facet
    Nx1 :obj:`numpy.ndarray`
        vector of coefficients that achieves the minimum
    """
    dim = facet.shape[1]  # num vertices in facet

    # create alpha weights for vertices of facet
    G = facet.T.dot(facet)
    grasp_matrix = G + wrench_regularizer * np.eye(G.shape[0])

    # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h, Ax = b
    P = cvx.matrix(2 * grasp_matrix)   # quadratic cost for Euclidean dist
    q = cvx.matrix(np.zeros((dim, 1)))
    G = cvx.matrix(-np.eye(dim))       # greater than zero constraint
    h = cvx.matrix(np.zeros((dim, 1)))
    A = cvx.matrix(np.ones((1, dim)))  # sum constraint to enforce convex
    b = cvx.matrix(np.ones(1))         # combinations of vertices
    sol = cvx.solvers.qp(P, q, G, h, A, b)
    v = np.array(sol['x'])
    min_norm = np.sqrt(sol['primal objective'])

    return abs(min_norm), v


def grasp_matrix(forces, torques, normals, soft_fingers=False,
                 finger_radius=0.005, params=None):
    if params is not None and 'finger_radius' in params.keys():
        finger_radius = params.finger_radius
    num_forces = forces.shape[1]
    num_torques = torques.shape[1]
    if num_forces != num_torques:
        raise ValueError('Need same number of forces and torques')

    num_cols = num_forces
    if soft_fingers:
        num_normals = 2
        if normals.ndim > 1:
            num_normals = 2*normals.shape[1]
        num_cols = num_cols + num_normals

    torque_scaling = 1
    G = np.zeros([6, num_cols])
    for i in range(num_forces):
        G[:3, i] = forces[:, i]
        # G[3:,i] = forces[:,i] # ZEROS
        G[3:, i] = torque_scaling * torques[:, i]

    if soft_fingers:
        torsion = np.pi * finger_radius**2 * \
            params.friction_coef * normals * params.torque_scaling
        pos_normal_i = -num_normals
        neg_normal_i = -num_normals + num_normals / 2
        G[3:, pos_normal_i:neg_normal_i] = torsion
        G[3:, neg_normal_i:] = -torsion

    return G


def get_normal_face(p1, p2, p3):
    U = p2 - p1
    V = p3 - p1
    Nx = U[1]*V[2] - U[2]*V[1]
    Ny = U[2]*V[0] - U[0]*V[2]
    Nz = U[0]*V[1] - U[1]*V[0]
    return [-1*Nx, -1*Ny, -1*Nz]


def get_normal_face_batched(vertices, faces):
    p1 = vertices[faces[:, 0]]
    p2 = vertices[faces[:, 1]]
    p3 = vertices[faces[:, 2]]
    U = p2 - p1
    V = p3 - p1
    Nx = U[:, 1]*V[:, 2] - U[:, 2]*V[:, 1]
    Ny = U[:, 2]*V[:, 0] - U[:, 0]*V[:, 2]
    Nz = U[:, 0]*V[:, 1] - U[:, 1]*V[:, 0]
    return -1*torch.stack((Nx, Ny, Nz)).T


def get_distance_vertices(obj, hand):
    n1 = len(hand)
    n2 = len(obj)

    matrix1 = hand[np.newaxis].repeat(n2, 0)
    matrix2 = obj[:, np.newaxis].repeat(n1, 1)
    dists = np.sqrt(((matrix1-matrix2)**2).sum(-1))

    return dists.min(0)


def get_distance_vertices_batched(obj, hand):
    n1 = len(hand[1])
    n2 = len(obj[0])

    matrix1 = hand.unsqueeze(1).repeat(1, n2, 1, 1)
    matrix2 = obj.unsqueeze(2).repeat(1, 1, n1, 1)
    dists = torch.norm(matrix1 - matrix2, dim=-1)
    dists = dists.min(1)[0]

    return dists
