import copy
import sys
import os
sys.path.append(os.getcwd())
import torch
import numpy as np
import cv2
import math
import open3d as o3d
import argparse
import pyquaternion
import mmcv
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion

# from lib.utils.general_utils import matrix_to_quaternion
from nuscenes.nuscenes import NuScenes as NuScenesDatabase
from PIL import Image
import pyquaternion
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R




def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))



def rotation_translation_to_pose(r_quat, t_vec):
    """Convert quaternion rotation and translation vectors to 4x4 matrix"""

    pose = np.eye(4)

    # NB: Nuscenes recommends pyquaternion, which uses scalar-first format (w x y z)
    # https://github.com/nutonomy/nuscenes-devkit/issues/545#issuecomment-766509242
    # https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L299
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    pose[:3, :3] = pyquaternion.Quaternion(r_quat).rotation_matrix

    pose[:3, 3] = t_vec
    return pose

def draw_box_on_img(vertices, img, id):
    colour = (255, 128, 128)
    # Draw the edges of the 3D bounding box
    for k in [0, 1]:
        for l in [0, 1]:
            for idx1,idx2 in [((0,k,l),(1,k,l)), ((k,0,l),(k,1,l)), ((k,l,0),(k,l,1))]:
                cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=1)
    # Draw a cross on the front face to identify front & back.
    for idx1,idx2 in [((1,0,0),(1,1,1)), ((1,1,0),(1,0,1))]:
        cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)  # Blue color in BGR

    img = cv2.putText(img, str(id), vertices[0,0,0], font, fontScale, color)

def get_intrinsic(camera_calibration):
    intrinsic = camera_calibration.intrinsic
    # Camera model:
    # | fx  0 cx 0 |
    # |  0 fy cy 0 |
    # |  0  0  1 0 |
    camera_model = np.array([
        [intrinsic[0], 0, intrinsic[2], 0],
        [0, intrinsic[1], intrinsic[3], 0],
        [0, 0,                       1, 0]])

    return camera_model

def get_3d_box_projected_corners(vehicle_to_image, box, obj_pose):
    """Get the 2D coordinates of the 8 corners of a label's 3D bounding box.

    vehicle_to_image: Transformation matrix from the global frame to the image frame.
    label: The object label
    """

    sl, sw, sh = box[0], box[1], box[2]
    scale_matrix = np.eye(4)
    scale_matrix[0, 0], scale_matrix[1, 1], scale_matrix[2, 2] = sl, sw, sh
    box_to_vehicle = np.matmul(obj_pose, scale_matrix)
    # Calculate the projection from the box space to the image space.
    box_to_image = np.matmul(vehicle_to_image, box_to_vehicle)

    # Loop through the 8 corners constituting the 3D box
    # and project them onto the image
    vertices = np.empty([2,2,2,2])
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                # 3D point in the box space
                v = np.array([(k-0.5), (l-0.5), (m-0.5), 1.])

                # Project the point onto the image
                v = np.matmul(box_to_image, v)

                # If any of the corner is behind the camera, ignore this object.
                if v[2] < 0:
                    return None

                vertices[k,l,m,:] = [v[0]/v[2], v[1]/v[2]]

    vertices = vertices.astype(np.int32)

    return vertices



def _load_points(pts_filename):
    """Private function to load point clouds data.
    Args:
        pts_filename (str): Filename of point clouds data.
    Returns:
        np.ndarray: An array containing point clouds data.
    """
    file_client_args=dict(backend='disk')
    file_client = None
    if file_client is None:
        file_client = mmcv.fileio.FileClient(**file_client_args)
    try:
        pts_bytes = file_client.get(pts_filename)
        points = np.frombuffer(pts_bytes, dtype=np.float32)
    except ConnectionError:
        mmcv.check_file_exist(pts_filename)
        if pts_filename.endswith('.npy'):
            points = np.load(pts_filename)
        else:
            points = np.fromfile(pts_filename, dtype=np.float32)
    return points

def get_bbox_corners(center, wlh, rotation_matrix):
    """
    Get the 8 corners of a 3D bounding box.
    :param center: Center of the bbox (x, y, z).
    :param wlh: Width, length, height of the bbox.
    :param rotation_matrix: 3x3 rotation matrix.
    :return: 8x3 numpy array of corners.
    """
    w, l, h = wlh / 2.0  # Half dimensions

    # Relative coordinates of the corners before rotation
    corners = np.array([
        [ l,  w,  h], [ l, -w,  h], [-l, -w,  h], [-l,  w,  h],  # Top four corners
        [ l,  w, -h], [ l, -w, -h], [-l, -w, -h], [-l,  w, -h]   # Bottom four corners
    ])

    # Apply rotation
    rotated_corners = np.dot(corners, rotation_matrix[0].T)

    # Translate corners to world coordinates
    world_corners = rotated_corners + center

    return world_corners

def transform_to_box_frame(points, bbox_center, bbox_rotation_matrix):
    """
    Transform points to the bounding box's coordinate frame.
    :param points: Nx3 array of points.
    :param bbox_center: The center of the bounding box.
    :param bbox_rotation_matrix: The rotation matrix of the bounding box.
    :return: Nx3 array of transformed points.
    """
    # Translate points to the bbox center
    translated_points = points - bbox_center

    # Rotate points to align with the bbox axes
    return np.dot(translated_points, bbox_rotation_matrix)

def check_points_in_bbox(points, bbox_center, bbox_rotation_matrix, bbox_dimensions):
    """
    Check if points are inside the 3D bounding box.
    :param points: Nx3 array of points.
    :param bbox_center: Center of the bbox.
    :param bbox_rotation_matrix: Rotation matrix of the bbox.
    :param bbox_dimensions: Dimensions of the bbox (width, length, height).
    :return: Boolean array indicating whether each point is inside the bbox.
    """
    # Transform points to the bounding box frame
    transformed_points = transform_to_box_frame(points, bbox_center, bbox_rotation_matrix)

    # Half dimensions for width, length, and height
    half_dims = bbox_dimensions / 2.0

    # Check if points are inside the bbox
    inside_x = np.logical_and(transformed_points[:, 0] >= -half_dims[0], transformed_points[:, 0] <= half_dims[0])
    inside_y = np.logical_and(transformed_points[:, 1] >= -half_dims[1], transformed_points[:, 1] <= half_dims[1])
    inside_z = np.logical_and(transformed_points[:, 2] >= -half_dims[2], transformed_points[:, 2] <= half_dims[2])

    return np.logical_and(np.logical_and(inside_x, inside_y), inside_z)


def parse_seq_rawdata(args, root_dir, save_dir, seq_name='scene-400'):
    print(f'Processing sequence {seq_name}...')
    seq_save_dir = os.path.join(save_dir, seq_name, ID)
    os.makedirs(seq_save_dir, exist_ok=True)

    pld_save_dir = os.path.join(seq_save_dir, "sparse/0/")
    os.makedirs(pld_save_dir, exist_ok=True)

    nusc = NuScenesDatabase(version="v1.0-mini", dataroot=root_dir, verbose=False)
    cameras_name = ["CAM_" + camera for camera in cameras]
    n_cam = len(cameras_name)

    data_idx = 0 # set for split test/train dataset

    samples = [
            samp for samp in nusc.sample if nusc.get("scene", samp["scene_token"])["name"] == seq_name
        ]
    samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))


    print("Processing tracking data...")
    object_ids = dict()

    track_infos_path = os.path.join(seq_save_dir, "track_info.txt")
    track_infos_file = open(track_infos_path, 'w')
    row_info_title = "frame_id " + "track_id " + "object_class " + "alpha " + \
        "box_height " + "box_width " + "box_length " + "box_center_x " + "box_center_y " + "box_center_z " \
        + "box_rotation_x " + "box_rotation_y " + "box_rotation_z " + "box_rotation_w " \
        + "box2d_minx " + "box2d_miny " + "box2d_maxx " + "box2d_maxy " + "camera" + "\n"
    track_infos_file.write(row_info_title)
    track_vis_save_dir = os.path.join(seq_save_dir, 'images')
    os.makedirs(track_vis_save_dir, exist_ok=True)

    c2ws = []
    obj_poses_list = []
    ixts = []

    pts_all = []

    num=0
    for camera_id, camera in enumerate(cameras_name):
        for frame_id_cur_cam, sample in enumerate(samples):
            frame_id = len(samples) * camera_id + frame_id_cur_cam
            camera_data = nusc.get("sample_data", sample["data"][camera])
            calibrated_sensor_data = nusc.get("calibrated_sensor", camera_data["calibrated_sensor_token"])
            ego_pose_data = nusc.get("ego_pose", camera_data["ego_pose_token"])
            image_path, boxes, K = nusc.get_sample_data(sample["data"][camera])
            anno_tokens_cur_cam = [b.token for b in boxes]
            cur_frame_obj_cnt = 0
            # ego_pose == ego2golbal
            ego_pose = rotation_translation_to_pose(ego_pose_data["rotation"], ego_pose_data["translation"])

            # cam_pose == cam2ego
            cam_pose = rotation_translation_to_pose(
                calibrated_sensor_data["rotation"], calibrated_sensor_data["translation"]
            )
            img = np.array(Image.open(os.path.join(root_dir, camera_data["filename"])))
            h, w = img.shape[:2]
            #自车坐标系是[forward, left, up]
            opencv2camera = np.array([[0., 0., 1., 0.],
                                [-1., 0., 0., 0.],
                                [0., -1., 0., 0.],
                                [0., 0., 0., 1.]])

            # c2v = np.matmul(cam_pose, opencv2camera) # [right, down, forward]
            c2w = ego_pose @ cam_pose
            # c2w = np.matmul(c2w, opencv2camera)
            camera_intrinsic = np.array(calibrated_sensor_data['camera_intrinsic'])
            fx=camera_intrinsic[0, 0]
            fy=camera_intrinsic[1, 1]
            cx=camera_intrinsic[0, 2]
            cy=camera_intrinsic[1, 2]
            intrinsic_M = np.array([[fx, 0, cx, 0],[0, fy, cy, 0],[0, 0, 1, 0]])
            vehicle_to_image = np.matmul(intrinsic_M, np.linalg.inv(c2w))

            for anno_token in sample['anns']:


                anno_info = nusc.get('sample_annotation',anno_token)
                obj_id = anno_info['instance_token']
                main_category = anno_info['category_name'].split('.')[0]
                category = anno_info['category_name'].split('.')[1]

                if anno_token not in anno_tokens_cur_cam:
                    continue
                if category != "car":
                    continue
                if obj_id  != ID:
                    continue

                track_id = nusc.getind('instance', anno_info['instance_token'])
                dims = anno_info['size']  # w, l, h
                pos_world = anno_info['translation']  # x, y, z
                quat_world = anno_info['rotation']  # w, x, y, z
                dims[0], dims[1] = dims[1], dims[0] # l, w, h

                rot_world = pyquaternion.Quaternion(quat_world).rotation_matrix
                print(pyquaternion.Quaternion(quat_world).yaw_pitch_roll)
                obj_pose_global = np.eye(4, dtype=np.float32)
                obj_pose_global[:3, :3] = rot_world
                obj_pose_global[:3, 3] = pos_world


                image_width = img.shape[1]
                image_height= img.shape[0]

                vertices = get_3d_box_projected_corners(vehicle_to_image, dims, obj_pose_global)
                vertices_8 = vertices.reshape(-1,2)
                min = np.min(vertices_8,axis=0)
                max = np.max(vertices_8,axis=0)

                x_min = min[0]
                y_min = min[1]
                x_max = max[0]
                y_max = max[1]

                # if x_min <0 or y_min<0 or x_max<0 or y_max<0:
                #     continue
                # if x_max>image_width or y_max >image_height:
                #     continue
                # img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255),thickness=2)
                w = x_max - x_min
                h = y_max - y_min


                if w < 0.1 * image_width or h < 0.1 * image_height:  # limit box size
                    continue

                if vertices is None:
                    continue
                bbox_visible_width = np.logical_and((vertices[..., 0] > 0), (vertices[..., 0] < w)).any()
                bbox_visible_height = np.logical_and((vertices[..., 1] > 0), (vertices[..., 1] < h)).any()
                bbox_visible = np.logical_and(bbox_visible_width, bbox_visible_height)




                # draw_box_on_img(vertices, img, id=obj_id)
                alpha = -10
                obj_rotation = torch.from_numpy(obj_pose_global[:3, :3]).float().unsqueeze(0)
                obj_quaternion = matrix_to_quaternion(obj_rotation).squeeze(0).numpy() #obj_to_global

                # lines_info = f"{frame_id} {track_id} {main_category} {alpha} {dims[2]} {dims[1]} {dims[0]} {pos_world[0]} {pos_world[1]} {pos_world[2]} {quat_world[0]} {quat_world[1]} {quat_world[2]} {quat_world[3]} {x_min} {y_min} {x_max} {y_max}\n"
                lines_info = f"{num} {track_id} {main_category} {alpha} {dims[2]} {dims[1]} {dims[0]} {pos_world[0]} {pos_world[1]} {pos_world[2]} {quat_world[0]} {quat_world[1]} {quat_world[2]} {quat_world[3]} {x_min} {y_min} {x_max} {y_max} {camera}\n"

                track_infos_file.write(lines_info)



                cv2.imwrite(os.path.join(track_vis_save_dir, '%06d.png'%num), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                num+=1

                ## save ex and ins
                camera_intrinsic = np.array(calibrated_sensor_data['camera_intrinsic'])
                fx = camera_intrinsic[0, 0]
                fy = camera_intrinsic[1, 1]
                cx = camera_intrinsic[0, 2]
                cy = camera_intrinsic[1, 2]
                intrinsic_M = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                ixts.append(intrinsic_M)
                data_idx += 1


                ## transfer  c2w in obj coordinate
                # c2w_ori = copy.deepcopy(c2w)
                # c2w =  np.linalg.inv(obj_pose_global) @ c2w
                c2ws.append(c2w)
                obj_poses_list.append(obj_pose_global)


                ## save point cloud
                lidar_data = nusc.get("sample_data", sample["data"]['LIDAR_TOP'])
                lidar_ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
                lidar_path = os.path.join(root_dir, lidar_data["filename"])
                # points_pre = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
                points = _load_points(lidar_path).reshape(-1, 5).copy()

                calibrated_sensor_data = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
                rot_matrix_lidar = pyquaternion.Quaternion(calibrated_sensor_data['rotation']).rotation_matrix[None, ...]
                points[:, :3] = np.matmul(rot_matrix_lidar, points[:, :3, None]).squeeze()  # lidar frame -> ego frame
                for i in range(3):
                    points[:, i] += calibrated_sensor_data['translation'][i]
                rot_matrix_ego = pyquaternion.Quaternion(lidar_ego_pose['rotation']).rotation_matrix[None, ...]
                points[:, :3] = np.matmul(rot_matrix_ego, points[:, :3, None]).squeeze()  # ego frame -> world frame
                for i in range(3):
                    points[:, i] += lidar_ego_pose['translation'][i]



                inside_bbox_mask = check_points_in_bbox(points[:,:3], pos_world, rot_world, np.array(dims))
                instance_points = points[inside_bbox_mask, :3]

                # transfer world point clound to obj coordiante
                instance_points = np.matmul(np.linalg.inv(obj_pose_global)[:3, :3], instance_points[:, :3, None]).squeeze()
                for i in range(3):
                    instance_points[:, i] += np.linalg.inv(obj_pose_global)[:3, -1][i]


                pts_all.append(instance_points)

    pts_all = np.concatenate(pts_all, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all[...,:3])
    o3d.io.write_point_cloud(f'{pld_save_dir}/points3D.ply', pcd)
    print("Processing LiDAR data done...")


    c2ws = np.stack(c2ws, axis=0)
    obj_poses_list = np.stack(obj_poses_list, axis=0)
    ixts = np.stack(ixts, axis=0)
    np.save(f'{seq_save_dir}/extrinsics.npy', c2ws)
    np.save(f'{seq_save_dir}/obj_poses.npy', obj_poses_list)
    np.save(f'{seq_save_dir}/intrinsics.npy', ixts)

    track_infos_file.close()
    print("Processing tracking data done...")




def valid_c2w(save_dir, scene_name, id):
    c2w_path = os.path.join(save_dir, scene_name, id,"extrinsics.npy")
    c2w = np.load(c2w_path)
    w2c_0 = np.linalg.inv(c2w[0])
    C_30_0 = w2c_0 @ c2w[2]
    print(C_30_0[:3, 3])
    print(C_30_0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_image', action='store_true')
    parser.add_argument('--skip_lidar', action='store_true')
    parser.add_argument('--skip_tracking', action='store_true')
    args = parser.parse_args()
    root_dir = "/mnt/hdd/xiaobiaodu/data/v1.0-mini"
    save_dir = "./"
    for scene_name in scene_list:
        parse_seq_rawdata(args, root_dir, save_dir, scene_name)
        valid_c2w(save_dir, scene_name, ID)

if __name__ == '__main__':
    cameras=("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT", )



    scene_list = ["scene-0103"]
    # ID = "d0f5aace57684923b8a44b554b006fb2"
    # ID = "3620feb00d744241a94855f3a8913a78"

    scene_list = ["scene-0655"]
    ID = "94b33ce331b844dcb991a2020742cebf"
    ID = "32e7ed87deb6491685b5c621c6db9b66"
    main()
