from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes import NuScenes
import os,numpy as np

nusc = NuScenes(version='v1.0-trainval', dataroot='/home/dante0shy/dataset/nuScenes', verbose=True)

scene = nusc.get('scene', nusc.scene[0]['token'])


sample = nusc.get('sample', scene['first_sample_token'])
lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
xyzf = np.fromfile(os.path.join(nusc.dataroot, lidar['filename']), dtype=np.float32, count=-1).reshape([-1, 5])

lidar_sd_token = sample['data']['LIDAR_TOP']
lidarseg_labels_filename = os.path.join(nusc.dataroot,
                                                nusc.get('lidarseg', lidar_sd_token)['filename'])
points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])