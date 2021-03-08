import torch
from utils import util


def get_relevant_vertices():
    finger_vertices = [7, 37, 21, 51, 71, 34, 61, 86, 46, 22, 53, 40, 69, 52, 38, 13, 29, 19, 8, 81, 60, 72, 97, 36,
                       28, 45, 49, 66, 3, 70, 65, 68, 25, 95, 48, 57, 98, 27, 79, 15, 80, 89, 62, 100, 91, 5, 44, 63]
    finger_tip_vertices = [35, 98, 44, 90, 95, 77, 84, 73, 76, 25, 108, 64, 22, 24, 96, 23, 85, 79, 83, 30, 45, 47, 68, 54, 42, 69, 92, 86,
                           19, 7, 94, 37, 99, 91, 11, 107, 0, 89, 57, 59, 109, 4, 65, 31, 2, 1, 10, 101, 52, 97, 87, 50, 72, 15, 106, 82, 12, 56, 78, 32, 46, 8]
    return finger_vertices, finger_tip_vertices


def optimize_fingers(handfullpose, R, T, obj_verts, barrett_layer, object_finger_threshold, optimize_finger_tip=False, step=25, device="cpu"):
    relevant_finger_vertices, relevant_finger_tip_vertices = get_relevant_vertices()
    handfullpose_converged = handfullpose.clone().to(device)
    touching_indexes = 0

    num_samples = 1000//step + 1
    pose = torch.zeros((1, 4, 4)).to(device)
    pose[0, :3, :3] = R
    pose[0, :3, 3] = T
    pose = pose.repeat(num_samples, 1, 1)

    inds = torch.linspace(0, util.deg2rad(140), num_samples).to(device)

    def solve_for_rotation(idx):
        _, _, all_finger_vertices, all_finger_tip_vertices, _ = barrett_layer(pose, handfullpose_repeated)
        # all_relevant_finger_vertices = all_finger_vertices  # [:, :, vertices["finger"][0]]
        # all_relevant_finger_tip_vertices = all_finger_tip_vertices  # [:, :, vertices["finger_tip"][0]]
        all_relevant_finger_vertices = all_finger_vertices[:, :, relevant_finger_vertices]
        all_relevant_finger_tip_vertices = all_finger_tip_vertices[:, :, relevant_finger_tip_vertices]
        # We have three fingers on the barrett hand
        touching_indexes = 0
        for i in range(3):
            current_finger_vertices = all_relevant_finger_vertices[:, i]
            current_finger_tip_vertices = all_relevant_finger_tip_vertices[:, i]
            current_concat_finger_vertices = torch.cat((current_finger_vertices, current_finger_tip_vertices), dim=1).squeeze()
            vertex_solution, converged = get_optimization_angle(
                current_concat_finger_vertices, obj_verts, object_finger_threshold, device)
            if converged:
                touching_indexes += 1
                delta_angle = inds[vertex_solution]-handfullpose[0, i+idx]  # - inds[vertex_solution]
                handfullpose[0, i+idx] = handfullpose[0, i+idx] + delta_angle
        return touching_indexes
    handfullpose_repeated = handfullpose_converged.repeat(num_samples, 1)
    handfullpose_repeated[:, 1] = inds
    handfullpose_repeated[:, 2] = inds
    handfullpose_repeated[:, 3] = inds
    touching_indexes = solve_for_rotation(1)
    if optimize_finger_tip:
        inds = torch.linspace(0, util.deg2rad(40), num_samples).to(device)
        handfullpose_repeated = handfullpose.clone().to(device).repeat(num_samples, 1)
        handfullpose_repeated[:, 4] = inds
        handfullpose_repeated[:, 5] = inds
        handfullpose_repeated[:, 6] = inds
        touching_indexes += solve_for_rotation(4)
    return touching_indexes


def optimize_fingers_batched(handfullpose, R, T, obj_verts, barrett_layer, object_finger_threshold, optimize_finger_tip=False, step=25, device="cpu"):
    if type(obj_verts) is not torch.Tensor:
        obj_verts = torch.FloatTensor(obj_verts).to(device)

    relevant_finger_vertices, relevant_finger_tip_vertices = get_relevant_vertices()

    handfullpose_converged = handfullpose.clone().to(device)
    touching_indexes = 0
    batch_size = handfullpose.shape[0]

    num_samples = 1000//step + 1
    inds = torch.linspace(0, util.deg2rad(140), num_samples).to(device).repeat(batch_size)

    # Here we need to create, for each barrett hand in the batch, a new barrett hand with joints
    # set from fully open to fully closed
    pose = torch.zeros((inds.shape[0], 4, 4)).to(device)
    for i in range(batch_size):
        pose[i*num_samples:(i+1)*num_samples, :3, :3] = R[i]
        pose[i*num_samples:(i+1)*num_samples, :3, 3] = T[i]

    def solve_for_rotation(idx):
        _, _, all_finger_vertices, all_finger_tip_vertices, _ = barrett_layer(pose, handfullpose_repeated)
        all_relevant_finger_vertices = all_finger_vertices[:, :, relevant_finger_vertices]
        all_relevant_finger_tip_vertices = all_finger_tip_vertices[:, :, relevant_finger_tip_vertices]
        # We have three fingers on the barrett hand
        for i in range(3):
            current_finger_vertices = all_relevant_finger_vertices[:, i]
            current_finger_tip_vertices = all_relevant_finger_tip_vertices[:, i]
            current_concat_finger_vertices = torch.cat((current_finger_vertices, current_finger_tip_vertices), dim=1)

            current_concat_finger_vertices = current_concat_finger_vertices.reshape(
                batch_size, current_concat_finger_vertices.shape[0]//batch_size, current_concat_finger_vertices.shape[-2], 3)
            vertex_solution, _ = get_optimization_angle_batched(
                current_concat_finger_vertices, obj_verts, object_finger_threshold, device)
            handfullpose[:, i+idx] = handfullpose[:, i+idx] + inds[vertex_solution]-handfullpose[:, i+idx]  # delta_angle

    handfullpose_repeated = handfullpose_converged.repeat_interleave(num_samples, dim=0)
    handfullpose_repeated[:, 1] = inds
    handfullpose_repeated[:, 2] = inds
    handfullpose_repeated[:, 3] = inds
    solve_for_rotation(1)
    if optimize_finger_tip:
        # TODO: Here the GPUs are synchronizing and the execution time is way slower than not optimizing the fingers
        inds = torch.linspace(0, util.deg2rad(40), num_samples, device=device).repeat(batch_size)  # .to(device)
        handfullpose_repeated = handfullpose.clone().repeat(1, num_samples).view(-1, 7)  # repeat_interleave(num_samples, dim=0)
        handfullpose_repeated[:, 4] = inds
        handfullpose_repeated[:, 5] = inds
        handfullpose_repeated[:, 6] = inds

    return touching_indexes


def get_optimization_angle_batched(arc_points, obj_verts, object_finger_threshold, device):
    arc_points = arc_points.unsqueeze(1)
    obj_verts = obj_verts.unsqueeze(1).unsqueeze(1)
    dists = obj_verts - arc_points
    eu_dists = (dists**2).sum(-1)
    eu_dists = torch.sqrt(eu_dists.min(1)[0])
    solutions = eu_dists < object_finger_threshold
    earliest_in_arc = solutions.max(1)[1]
    zero = (earliest_in_arc == 0)
    vertex_solution = (999*zero+earliest_in_arc).min(-1)[0]
    vertex_solution = vertex_solution*(vertex_solution != 999)
    converged = solutions.view(solutions.shape[0], -1).max(-1)[0] == 1
    return vertex_solution, converged


def get_optimization_angle(arc_points, obj_verts, object_finger_threshold, device):
    if type(obj_verts) is not torch.Tensor:
        obj_verts = torch.FloatTensor(obj_verts).to(device)

    obj_verts = obj_verts.unsqueeze(1).unsqueeze(1)
    arc_points = arc_points.unsqueeze(0)
    dists = obj_verts - arc_points
    eu_dists = torch.sqrt((dists**2).sum(-1))
    threshold = torch.FloatTensor([object_finger_threshold]).to(device)

    eu_dists = eu_dists.min(0)[0]
    solutions = eu_dists < threshold

    earliest_in_arc = solutions.cpu().data.numpy().argmax(0)
    earliest_in_arc[earliest_in_arc == 0] = 999

    vertex_solution = earliest_in_arc.min()
    if vertex_solution == 999:
        vertex_solution = 0

    converged = solutions.max().cpu().data.numpy() == 1
    return vertex_solution, converged
