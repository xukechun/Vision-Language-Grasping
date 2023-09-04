import math
import numpy as np
import pybullet as p
import cv2
import open3d as o3d
import open3d_plus as o3dp
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from constants import WORKSPACE_LIMITS, PIXEL_SIZE


reconstruction_config = {
    'nb_neighbors': 50,
    'std_ratio': 2.0,
    'voxel_size': 0.0015,
    'icp_max_try': 5,
    'icp_max_iter': 2000,
    'translation_thresh': 3.95,
    'rotation_thresh': 0.02,
    'max_correspondence_distance': 0.02
}

graspnet_config = {
    'graspnet_checkpoint_path': 'models/graspnet/logs/log_rs/checkpoint.tar',
    'refine_approach_dist': 0.01,
    'dist_thresh': 0.05,
    'angle_thresh': 15,
    'mask_thresh': 0.5
}

def get_heightmap(points, colors, bounds, pixel_size):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

    Args:
        points: HxWx3 float array of 3D points in world coordinates.
        colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
        bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
            region in 3D space to generate heightmap in world coordinates.
        pixel_size: float defining size of each pixel in meters.
    Returns:
        heightmap: HxW float array of height (from lower z-bound) in meters.
        colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
    """
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
    iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
    iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[px, py] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
        colormap[px, py, c] = colors[:, c]
    return heightmap, colormap


def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.
    Args:
        depth: HxW float array of perspective depth in meters.
        intrinsics: 3x3 float array of camera intrinsics matrix.
    Returns:
        points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.
    Args:
        points: HxWx3 float array of 3D points in camera coordinates.
        transform: 4x4 float array representing a rigid transformation matrix.
    Returns:
        points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding, "constant", constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points


def reconstruct_heightmaps(color, depth, configs, bounds, pixel_size):
    """Reconstruct top-down heightmap views from multiple 3D pointclouds."""
    heightmaps, colormaps = [], []
    for color, depth, config in zip(color, depth, configs):
        intrinsics = config["intrinsics"]
        xyz = get_pointcloud(depth, intrinsics)
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        xyz = transform_pointcloud(xyz, transform)
        heightmap, colormap = get_heightmap(xyz, color, bounds, pixel_size)
        heightmaps.append(heightmap)
        colormaps.append(colormap)

    return heightmaps, colormaps


def get_fuse_heightmaps(obs, configs, bounds, pixel_size):
    """Reconstruct orthographic heightmaps with segmentation masks."""
    heightmaps, colormaps = reconstruct_heightmaps(
        obs["color"], obs["depth"], configs, bounds, pixel_size
    )
    colormaps = np.float32(colormaps)
    heightmaps = np.float32(heightmaps)

    # Fuse maps from different views.
    valid = np.sum(colormaps, axis=3) > 0
    repeat = np.sum(valid, axis=0)
    repeat[repeat == 0] = 1
    cmap = np.sum(colormaps, axis=0) / repeat[Ellipsis, None]
    cmap = np.uint8(np.round(cmap))
    hmap = np.max(heightmaps, axis=0)  # Max to handle occlusions.

    return cmap, hmap


def get_true_heightmap(env):
    """Get RGB-D orthographic heightmaps and segmentation masks in simulation."""

    # Capture near-orthographic RGB-D images and segmentation masks.
    color, depth, segm = env.render_camera(env.oracle_cams[0])

    # Combine color with masks for faster processing.
    color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

    # Reconstruct real orthographic projection from point clouds.
    hmaps, cmaps = reconstruct_heightmaps(
        [color], [depth], env.oracle_cams, env.bounds, env.pixel_size
    )

    # Split color back into color and masks.
    cmap = np.uint8(cmaps)[0, Ellipsis, :3]
    hmap = np.float32(hmaps)[0, Ellipsis]
    mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()

    return cmap, hmap, mask


def get_heightmap_from_real_image(color, depth, segm, env):
    # Combine color with masks for faster processing.
    color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

    # Reconstruct real orthographic projection from point clouds.
    hmaps, cmaps = reconstruct_heightmaps(
        [color], [depth], env.camera.configs, env.bounds, env.pixel_size
    )

    # Split color back into color and masks.
    cmap = np.uint8(cmaps)[0, Ellipsis, :3]
    hmap = np.float32(hmaps)[0, Ellipsis]
    mask = np.uint8(cmaps)[0, Ellipsis, 3:].squeeze()

    return cmap, hmap, mask


def process_pcds(pcds, reconstruction_config):
    trans = dict()
    pcd = pcds[0]
    pcd.estimate_normals()
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors = reconstruction_config['nb_neighbors'],
        std_ratio = reconstruction_config['std_ratio']
    )
    for i in range(1, len(pcds)):
        voxel_size = reconstruction_config['voxel_size']
        income_pcd, _ = pcds[i].remove_statistical_outlier(
            nb_neighbors = reconstruction_config['nb_neighbors'],
            std_ratio = reconstruction_config['std_ratio']
        )
        income_pcd.estimate_normals()
        income_pcd = income_pcd.voxel_down_sample(voxel_size)
        transok_flag = False
        for _ in range(reconstruction_config['icp_max_try']): # try 5 times max
            reg_p2p = o3d.pipelines.registration.registration_icp(
                income_pcd,
                pcd,
                reconstruction_config['max_correspondence_distance'],
                np.eye(4, dtype = np.float),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(reconstruction_config['icp_max_iter'])
            )
            if (np.trace(reg_p2p.transformation) > reconstruction_config['translation_thresh']) \
                and (np.linalg.norm(reg_p2p.transformation[:3, 3]) < reconstruction_config['rotation_thresh']):
                # trace for transformation matrix should be larger than 3.5
                # translation should less than 0.05
                transok_flag = True
                break
        if not transok_flag:
            reg_p2p.transformation = np.eye(4, dtype = np.float32)
        income_pcd = income_pcd.transform(reg_p2p.transformation)
        trans[i] = reg_p2p.transformation
        pcd = o3dp.merge_pcds([pcd, income_pcd])
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals()
    return trans, pcd


def get_fuse_pointcloud(env):
    pcds = []
    configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
    # Capture near-orthographic RGB-D images and segmentation masks.
    for config in configs:
        color, depth, _ = env.render_camera(config)
        xyz = get_pointcloud(depth, config["intrinsics"])
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        points = transform_pointcloud(xyz, transform)
        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
        iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
        iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
        valid = ix & iy & iz
        points = points[valid]
        colors = color[valid]
        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        pcd.voxel_down_sample(reconstruction_config['voxel_size'])
        # # visualization
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])
        # the first pcd is the one for start fusion
        pcds.append(pcd)

    _, fuse_pcd = process_pcds(pcds, reconstruction_config)
    # visualization
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # o3d.visualization.draw_geometries([fuse_pcd, frame])

    return fuse_pcd


def get_true_bboxs(env, color_image, depth_image, mask_image):
    # get mask of all objects
    bbox_images = []
    bbox_positions = []
    for obj_id in env.obj_ids["rigid"]:
        mask = np.zeros(mask_image.shape).astype(np.uint8)
        mask[mask_image == obj_id] = 255
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        stats = stats[stats[:,4].argsort()]
        if stats[:-1].shape[0] > 0:
            bbox = stats[:-1][0]
            # for bbox
            # |(y0, x0)         |   
            # |                 |
            # |                 |
            # |         (y1, x1)|
            x0, y0 = bbox[0], bbox[1]
            x1 = bbox[0] + bbox[2]
            y1 = bbox[1] + bbox[3]

            # visualization
            start_point, end_point = (x0, y0), (x1, y1)
            color = (0, 0, 255) # Red color in BGR
            thickness = 1 # Line thickness of 1 px 
            mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_bboxs = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)
            cv2.imwrite('mask_bboxs.png', mask_bboxs)

            bbox_image = color_image[y0:y1, x0:x1]
            bbox_images.append(bbox_image)
            
            pixel_x = (x0 + x1) // 2
            pixel_y = (y0 + y1) // 2
            bbox_pos = [
                pixel_y * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                pixel_x * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
                depth_image[pixel_y][pixel_x] + WORKSPACE_LIMITS[2][0],
            ]
            bbox_positions.append(bbox_pos)

    return bbox_images, bbox_positions


def relabel_mask(env, mask_image):
    assert env.target_obj_id != -1
    num_obj = 50
    for i in np.unique(mask_image):
        if i == env.target_obj_id:
            mask_image[mask_image == i] = 255
        elif i in env.obj_ids["rigid"]:
            mask_image[mask_image == i] = num_obj
            num_obj += 10
        else:
            mask_image[mask_image == i] = 0
    mask_image = mask_image.astype(np.uint8)
    return mask_image


def relabel_mask_real(masks):
    """Assume the target object is labeled to 255"""
    mask_image = np.zeros_like(masks[0], dtype=np.uint8)
    num_obj = 50
    for idx, mask in enumerate(masks):
        if idx == 0:
            mask_image[mask == 255] = 255
        else:
            mask_image[mask == 255] = num_obj
            num_obj += 10
    mask_image = mask_image.astype(np.uint8)
    return mask_image


def get_real_heightmap(env):
    """Get RGB-D orthographic heightmaps in real world."""

    color, depth = env.get_camera_data()
    cv2.imwrite("temp.png", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

    # Reconstruct real orthographic projection from point clouds.
    hmaps, cmaps = reconstruct_heightmaps(
        [color], [depth], env.camera.configs, env.bounds, env.pixel_size
    )

    # Split color back into color and masks.
    cmap = np.uint8(cmaps)[0, Ellipsis]
    hmap = np.float32(hmaps)[0, Ellipsis]

    return cmap, hmap


def rotate(image, angle, is_mask=False):
    """Rotate an image using cv2, counterclockwise in degrees"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    if is_mask:
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST)
    else:
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

    return rotated


def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


# Preprocess of model input
def preprocess(bbox_images, bbox_positions, grasp_pose_set, n_px):
    transform = Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    remain_bboxes = []
    remain_bbox_positions = []
    for i in range(len(bbox_images)):
        if bbox_images[i].shape[0] >= 15 and bbox_images[i].shape[1] >= 15:
            remain_bboxes.append(bbox_images[i])  # shape = [n_obj, H, W, C]
            remain_bbox_positions.append(bbox_positions[i])
    print('Remaining bbox number', len(remain_bboxes))
    bboxes = None
    for remain_bbox in remain_bboxes:
        remain_bbox = Image.fromarray(remain_bbox)
        # padding
        w,h = remain_bbox.size
        if w >= h:
            remain_bbox_ = Image.new(mode='RGB', size=(w,w))
            remain_bbox_.paste(remain_bbox, box=(0, (w-h)//2))
        else:
            remain_bbox_ = Image.new(mode='RGB', size=(h,h))
            remain_bbox_.paste(remain_bbox, box=((h-w)//2, 0))
        remain_bbox_ = transform(remain_bbox_)

        remain_bbox_ = remain_bbox_.unsqueeze(0)
        if bboxes == None:
            bboxes = remain_bbox_
        else:
            bboxes = torch.cat((bboxes, remain_bbox_), dim=0) # shape = [n_obj, C, patch_size, patch_size]
    if bboxes != None:
        bboxes = bboxes.unsqueeze(0) # shape = [1, n_obj, C, patch_size, patch_size]
    
    pos_bboxes = None
    for bbox_pos in remain_bbox_positions:
        bbox_pos = torch.from_numpy(np.array(bbox_pos))
        bbox_pos = bbox_pos.unsqueeze(0)
        if pos_bboxes == None:
            pos_bboxes = bbox_pos
        else:
            pos_bboxes = torch.cat((pos_bboxes, bbox_pos), dim=0) # shape = [n_obj, pos_dim]
    if pos_bboxes != None:
        pos_bboxes = pos_bboxes.unsqueeze(0).to(dtype=torch.float32) # shape = [1, n_obj, pos_dim]
    

    grasps = None
    for grasp in grasp_pose_set:
        grasp = torch.from_numpy(grasp)
        grasp = grasp.unsqueeze(0)
        if grasps == None:
            grasps = grasp
        else:
            grasps = torch.cat((grasps, grasp), dim=0) # shape = [n_grasp, grasp_dim]
    grasps = grasps.unsqueeze(0).to(dtype=torch.float32) # shape = [1, n_grasp, grasp_dim]

    return remain_bboxes, bboxes, pos_bboxes, grasps


def plot_probs(text, bboxes, probs):
    plt.figure()
    plt.suptitle(text)
    for i in range(len(bboxes)):
        # bboxes[i] = cv2.cvtColor(bboxes[i], cv2.COLOR_RGB2BGR)
        ax = plt.subplot(1, len(bboxes), i+1)
        plt.imshow(bboxes[i])
        plt.xticks([])
        plt.yticks([])
        ax.set_title(str(probs[0][i]), fontsize=10)
    plt.show()
    
def plot_attnmap(attn_map):
    fig, ax = plt.subplots()
    ax.set_yticks([])
    # ax.set_yticks(range(attn_map.shape[0]))
    ax.set_xticks([])
    im = ax.imshow(attn_map, cmap="YlGnBu", interpolation='nearest')
    # R
    plt.colorbar(im)

    # for i in range(attn_map.shape[0]):
    #     for j in range(attn_map.shape[1]):
    #         # print('data[{},{}]:{}'.format(i, j, attn_map[i, j]))
    #         ax.text(j, i, round(attn_map[i, j]*100, 2),
    #                 ha="center", va="center", color="black")

    # plt.xlabel('cross feat')
    # plt.ylabel('grasp')
    # show
    fig.tight_layout()
    plt.show()


# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )
    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )
    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


# Checks if a matrix is a valid rotation matrix.
def isRotm(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
def rotm2euler(R):

    assert isRotm(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def angle2rotm(angle, axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis / np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:

        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def rotm2angle(R):
    # From: euclideanspace.com

    epsilon = 0.01  # Margin to allow for rounding errors
    epsilon2 = 0.1  # Margin to distinguish between 0 and 180 degrees

    assert isRotm(R)

    if (
        (abs(R[0][1] - R[1][0]) < epsilon)
        and (abs(R[0][2] - R[2][0]) < epsilon)
        and (abs(R[1][2] - R[2][1]) < epsilon)
    ):
        # Singularity found
        # First check for identity matrix which must have +1 for all terms in leading diagonaland zero in other terms
        if (
            (abs(R[0][1] + R[1][0]) < epsilon2)
            and (abs(R[0][2] + R[2][0]) < epsilon2)
            and (abs(R[1][2] + R[2][1]) < epsilon2)
            and (abs(R[0][0] + R[1][1] + R[2][2] - 3) < epsilon2)
        ):
            # this singularity is identity matrix so angle = 0
            return [0, 1, 0, 0]  # zero angle, arbitrary axis

        # Otherwise this singularity is angle = 180
        angle = np.pi
        xx = (R[0][0] + 1) / 2
        yy = (R[1][1] + 1) / 2
        zz = (R[2][2] + 1) / 2
        xy = (R[0][1] + R[1][0]) / 4
        xz = (R[0][2] + R[2][0]) / 4
        yz = (R[1][2] + R[2][1]) / 4
        if (xx > yy) and (xx > zz):  # R[0][0] is the largest diagonal term
            if xx < epsilon:
                x = 0
                y = 0.7071
                z = 0.7071
            else:
                x = np.sqrt(xx)
                y = xy / x
                z = xz / x
        elif yy > zz:  # R[1][1] is the largest diagonal term
            if yy < epsilon:
                x = 0.7071
                y = 0
                z = 0.7071
            else:
                y = np.sqrt(yy)
                x = xy / y
                z = yz / y
        else:  # R[2][2] is the largest diagonal term so base result on this
            if zz < epsilon:
                x = 0.7071
                y = 0.7071
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz / z
                y = yz / z
        return [angle, x, y, z]  # Return 180 deg rotation

    # As we have reached here there are no singularities so we can handle normally
    s = np.sqrt(
        (R[2][1] - R[1][2]) * (R[2][1] - R[1][2])
        + (R[0][2] - R[2][0]) * (R[0][2] - R[2][0])
        + (R[1][0] - R[0][1]) * (R[1][0] - R[0][1])
    )  # used to normalise
    if abs(s) < 0.001:
        s = 1

    # Prevent divide by zero, should not happen if matrix is orthogonal and should be
    # Caught by singularity test above, but I've left it in just in case
    angle = np.arccos((R[0][0] + R[1][1] + R[2][2] - 1) / 2)
    x = (R[2][1] - R[1][2]) / s
    y = (R[0][2] - R[2][0]) / s
    z = (R[1][0] - R[0][1]) / s
    return [angle, x, y, z]
