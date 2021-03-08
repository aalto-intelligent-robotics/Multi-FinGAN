from __future__ import print_function
from mayavi import mlab
mlab.options.offscreen = True


def plot_scene_w_grasps(list_obj_verts, list_obj_faces, list_obj_handverts, list_obj_handfaces):
    figure = mlab.figure(1, bgcolor=(1, 1, 1),
                         fgcolor=(0, 0, 0), size=(640, 480))
    mlab.clf()
    for i in range(len(list_obj_verts)):
        vertices = list_obj_verts[i]
        mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                             list_obj_faces[i], color=(1, 0, 0), opacity=0.5)
    for i in range(len(list_obj_handverts)):
        vertices = list_obj_handverts[i]
        mlab.triangular_mesh(vertices[:, 0], vertices[:, 1],
                             vertices[:, 2], list_obj_handfaces[i], color=(0, 0, 1))
    mlab.view(azimuth=-90, distance=1.5)
    data = mlab.screenshot(figure)
    return data
