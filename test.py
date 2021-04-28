from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes import NuScenes
import os,numpy as np

nusc = NuScenes(version='v1.0-trainval', dataroot='/home/dante0shy/dataset/nuScenes', verbose=True)

scene = nusc.get('scene', nusc.scene[0]['token'])
# sample = nusc.get('sample', scene['first_sample_token'])
sample = {'next':  scene['first_sample_token']}

def return_pose(lidar):
    poss = nusc.get('ego_pose', lidar['ego_pose_token'])
    cs_record_lid = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
    lid_to_ego = transform_matrix(cs_record_lid["translation"], Quaternion(cs_record_lid["rotation"]), inverse=False)
    lid_ego_to_world = transform_matrix(poss["translation"], Quaternion(poss["rotation"]), inverse=False)
    # kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
    lid_to_world = np.dot(lid_ego_to_world, lid_to_ego)#np.dot(lid_to_ego, kitti_to_nu_lidar.transformation_matrix)
    return lid_to_world

paths = '/home/dante0shy/remote_worplace/nuscene_test/extras/lidar'
count = 0
while sample:
    sample = nusc.get('sample', sample['next'])
    lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    xyzf = np.fromfile(os.path.join(nusc.dataroot, lidar['filename']), dtype=np.float32, count=-1).reshape([-1, 5])

    lidar_sd_token = sample['data']['LIDAR_TOP']
    lidarseg_labels_filename = os.path.join(nusc.dataroot,
                                                    nusc.get('lidarseg', lidar_sd_token)['filename'])
    points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])

    pose = return_pose(lidar)
    count+=1

    np.save(open(os.path.join(paths,'{:03d}.npy'.format(count)),'wb'), xyzf)
    np.save(open(os.path.join(paths,'{:03d}_pose.npy'.format(count)),'wb'), pose)

    if sample['token'] ==  scene['last_sample_token']:
        break