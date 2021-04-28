from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes import NuScenes
import os,numpy as np

nusc = NuScenes(version='v1.0-trainval', dataroot='/home/dante0shy/dataset/nuScenes', verbose=True)

scene = nusc.get('scene', nusc.scene[0]['token'])
sample = nusc.get('sample', scene['first_sample_token'])
lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

poss = nusc.get('ego_pose', lidar['ego_pose_token'])


cs_record_lid = nusc.get('calibrated_sensor',lidar['calibrated_sensor_token'])
lid_to_ego = transform_matrix(cs_record_lid["translation"], Quaternion(cs_record_lid["rotation"]), inverse=False)
lid_ego_to_world = transform_matrix(poss["translation"], Quaternion(poss["rotation"]), inverse=False)
kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
lid_to_world = np.dot(lid_ego_to_world, np.dot(lid_to_ego,kitti_to_nu_lidar.transformation_matrix))