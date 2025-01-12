import os
import argparse
import math
from os import path as osp

import numpy as np

import mmcv

from common_utils import get_scenario_map, get_filter_parameters
from data_utils import *
from tqdm import tqdm
import quaternion

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import CameraChannel

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import discover_log_dbs

### setup env variables required for nuplan
NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', r'/mnt/g/GIT/nuplan/dataset')
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', r'/mnt/g/GIT/nuplan/dataset/maps')
NUPLAN_DB_FILES = os.getenv('NUPLAN_DB_FILES', r'/mnt/g/GIT/nuplan/dataset/nuplan-v1.1/splits/mini')
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')
NUPLAN_EXP_ROOT = os.getenv('NUPLAN_EXP_ROOT', r"/mnt/g/GIT/nuplan/exp")
NUPLAN_SENSOR_ROOT = os.getenv('NUPLAN_SENSOR_ROOT', r"/mnt/g/GIT/nuplan/dataset/nuplan-v1.1/sensor_blobs/", )

PAST_TIME_HORIZON = 2 #secs
NUM_PAST_POSES = 20
NUM_AGENTS = 20
FUTURE_TIME_HORIZON = 6 #secs
NUM_FUTURE_POSES = 12
NUM_VAL_RATIO = 0.1 # percentage of dataset that should be used for validation

TEST_MODE = False
ENABLE_BIRDSVIEW_PLOT = False

NuplanNuscenesNameMapping = {
    "vehicle": "car",
    "pedestrian": "pedestrian",
    "bicycle": "bicycle",
    "traffic_cone": "traffic_cone",
    "barrier": "barrier",
    "czone_sign": "czone_sign",
    "generic_object": "generic_object",
}

def get_ego_agent_future(scenario):
    current_absolute_state = scenario.initial_ego_state

    trajectory_absolute_states = scenario.get_ego_future_trajectory(
        iteration=0, num_samples=6, time_horizon=3
    )

    # Get all future poses of the ego relative to the ego coordinate system
    trajectory_relative_poses = convert_absolute_to_relative_poses(
        current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
    )
    
    ego_future = trajectory_relative_poses[:, :2]
    
    # set mask
    ego_future_mask = np.ones(ego_future.shape[0])
    
    # Rotate the trajectories (ego is always with direction angle pi/2 from x axis in birdsview)
    x_coords, y_coords = zip(*ego_future)
    
    coordinates = torch.stack((torch.tensor(x_coords), torch.tensor(y_coords)), dim=1)
    cos_theta = torch.cos(torch.tensor(-math.pi/2.0, dtype=torch.float32))
    sin_theta = torch.sin(torch.tensor(-math.pi/2.0, dtype=torch.float32))
    
    rotation_matrix = torch.tensor([[cos_theta, -sin_theta],[sin_theta,  cos_theta]])
    rotated_coordinates = coordinates @ rotation_matrix
    
    x_coords_reconstruct = rotated_coordinates[:, 0].tolist()
    y_coords_reconstruct = rotated_coordinates[:, 1].tolist()
    
    # Combine along the last dimension
    recombined = np.concatenate([np.expand_dims(x_coords_reconstruct, axis=-1), np.expand_dims(y_coords_reconstruct, axis=-1)], axis=-1)

    if recombined[-1][0] >= 2:
        command = np.array([1, 0, 0])  # Turn Right
    elif recombined[-1][0] <= -2:
        command = np.array([0, 1, 0])  # Turn Left
    else:
        command = np.array([0, 0, 1])  # Go Straight

    # convert to increments
    recombined[1:] = recombined[1:] - recombined[:-1]

    return recombined, command.astype(np.float32), ego_future_mask.astype(np.float32)

def get_neighbor_agents_future(scenario, agent_index):
    current_ego_state = scenario.initial_ego_state
    present_tracked_objects = scenario.initial_tracked_objects.tracked_objects

    # Get all future poses of of other agents
    future_tracked_objects = [
        tracked_objects.tracked_objects
        for tracked_objects in scenario.get_future_tracked_objects(
            iteration=0, time_horizon=FUTURE_TIME_HORIZON, num_samples=NUM_FUTURE_POSES
        )
    ]

    sampled_future_observations = [present_tracked_objects] + future_tracked_objects
    future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_tensor_list(sampled_future_observations)
    agent_futures = agent_future_process(current_ego_state, future_tracked_objects_tensor_list, len(agent_index), agent_index)

    # process agent future masks
    final_agent_future = np.zeros((agent_futures.shape[0], agent_futures.shape[1] - 1, 2))
    agents_masks = np.zeros((agent_futures.shape[0], agent_futures.shape[1] -1))
    for i, agent_future in enumerate(agent_futures):
        valid_steps = np.where((agent_future[:, 0] == 0) & (agent_future[:, 1] == 0))[0]
        if valid_steps.size > 0 and valid_steps[0] > 1:
            agents_masks[i, :valid_steps[0]-1] = 1
            final_agent_future[i, 0:valid_steps[0]-1] = agent_future[1:valid_steps[0], :2]  - agent_future[0:valid_steps[0]-1, :2]
        elif valid_steps.size == 0:
            agents_masks[i, :] = 1
            final_agent_future[i, :] = agent_future[1:, :2]  - agent_future[:-1, :2]
        
    return final_agent_future, agents_masks.astype(np.float32)

def create_nuplan_infos(out_path,
                          info_prefix,
                          version='v1.1-mini'):
    """Create info file of nuplan dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        out_path (str): location where generated file should locate.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.1-mini'
    """

    print(version, NUPLAN_DATA_ROOT)
    
    # get scenarios
    map_version = "nuplan-maps-v1.0"    
    sensor_root = NUPLAN_SENSOR_ROOT
    db_files = discover_log_dbs(NUPLAN_DB_FILES)
    db_files = db_files[:7]
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(NUPLAN_DB_FILES, NUPLAN_MAPS_ROOT, sensor_root, db_files, map_version, include_cameras = True, scenario_mapping=scenario_mapping)
    scenario_filter = ScenarioFilter(*get_filter_parameters(1000, None, False))
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")

    if version == 'v1.0-mini':
        out_path = osp.join(out_path, 'mini')
    else:
        raise ValueError('unknown')
    os.makedirs(out_path, exist_ok=True)

    total_nuplan_infos = _fill_trainval_infos(scenarios)
    num_of_val = int(len(total_nuplan_infos) * NUM_VAL_RATIO)
    train_nusc_infos = total_nuplan_infos[num_of_val:]
    val_nusc_infos = train_nusc_infos[:num_of_val]

    metadata = dict(version=version)
    
    print('train sample: {}, val sample: {}'.format(
        len(train_nusc_infos), len(val_nusc_infos)))
    data = dict(infos=train_nusc_infos, metadata=metadata)
    info_path = osp.join(out_path,'{}_infos_train.pkl'.format(info_prefix))
    mmcv.dump(data, info_path)
    data['infos'] = val_nusc_infos
    info_val_path = osp.join(out_path,'{}_infos_val.pkl'.format(info_prefix))
    mmcv.dump(data, info_val_path)

def _fill_trainval_infos(sces,
                         fut_ts=12,
                         ego_fut_ts=6):
    """Generate the train/val infos from the raw data.
    Args:
    Returns:
    """
    train_nuplan_infos = []
    
    numberLoops = 0
    
    for scenario in tqdm(sces):
        numberLoops += 1
        if numberLoops > 100 and TEST_MODE:
            break

        map_location = scenario._map_name
        lidar2ego_transform = scenario.get_lidar_to_ego_transform()
        initial_ego_state = scenario.initial_ego_state
        # the ego2global_translation and rotation might be obtained from initial_ego_state.rear_axle.x, y, heading?
        
        lidar2ego_translation = lidar2ego_transform[:3,3]
        lidar2ego_rotation_matric = lidar2ego_transform[:3,:3]
        lidar2ego_rotation_q = quaternion.from_rotation_matrix(lidar2ego_rotation_matric)
        lidar2ego_rotation = [lidar2ego_rotation_q.w, lidar2ego_rotation_q.x, lidar2ego_rotation_q.y, lidar2ego_rotation_q.z]
        
        info = {
            'lidar_path': "",
            'token': scenario.token,
            'sweeps': [],
            'cams': dict(),
            'scene_token': scenario.token,
            'lidar2ego_translation': lidar2ego_translation,
            'lidar2ego_rotation': lidar2ego_rotation,
            'ego2global_translation': [initial_ego_state.rear_axle.x, initial_ego_state.rear_axle.y, 0.0],
            'ego2global_rotation': [math.cos(initial_ego_state.rear_axle.heading/2), 0, 0, math.sin(initial_ego_state.rear_axle.heading/2)],
            'timestamp': scenario._initial_lidar_timestamp,
            'map_location': map_location,
            'map_annos': [],
        }
        
        tracked_objects=scenario.get_tracked_objects_at_iteration(0)
        gt_boxes = []
        gt_names = []
        gt_velocity = []
        ego_status = []
        valid_flag = []
        instance_inds = []
        gt_agent_fut_masks = []
        
        ego_state = scenario.get_ego_state_at_iteration(0)
        ego_status.extend([ego_state.dynamic_car_state.rear_axle_acceleration_2d.x, ego_state.dynamic_car_state.rear_axle_acceleration_2d.y, 9.81]) # acceleration in ego vehicle frame, m/s/s. 9.81 is the g, similiar value as used in nuscenes, not sure how important it is, assume not
        ego_status.extend([0.0, 0.0, ego_state.dynamic_car_state.angular_velocity]) # angular velocity in ego vehicle frame, rad/s
        ego_status.extend([ego_state.dynamic_car_state.speed, 0.0, 0.0]) # velocity in ego vehicle frame, m/s
        ego_status.append(ego_state.tire_steering_angle) # steering angle, positive: left turn, negative: right turn
        
        for tracked_object in tracked_objects.tracked_objects:
            ### process gt_boxes (rotate the coordinate to keep same as in nuscenes_converter)
            x_ori = tracked_object.box.center.x - ego_state.center.x
            y_ori = tracked_object.box.center.y - ego_state.center.y
            heading_angle = math.pi / 2.0 - ego_state.center.heading # the heading angle definition in nuplan and nuscenes is different (based on x axis and y axis)
            
            x_new = x_ori * math.cos(heading_angle) - y_ori * math.sin(heading_angle)
            y_new = x_ori * math.sin(heading_angle) + y_ori * math.cos(heading_angle)
            # Add the rotation angle to the heading
            heading_new = tracked_object.box.center.heading + heading_angle
            gt_boxes.append([x_new, y_new, 0, tracked_object.box.dimensions.length, tracked_object.box.dimensions.width, tracked_object.box.dimensions.height, heading_new])
            
            ### process gt_velocity (rotate the coordinate to keep same as in nuscenes_converter)
            vx_ori = tracked_object.velocity.x
            yx_ori = tracked_object.velocity.y
            vx_new = vx_ori * math.cos(heading_angle) - yx_ori * math.sin(heading_angle)
            vy_new = vx_ori * math.sin(heading_angle) + yx_ori * math.cos(heading_angle)
            gt_velocity.append([vx_new, vy_new])
            
            ###  
            gt_names.append(NuplanNuscenesNameMapping[tracked_object.tracked_object_type.fullname])
            instance_inds.append(tracked_object.metadata.track_id)
            valid_flag.append(True)
            
        # get agent future tracks
        agents_future, gt_agent_fut_masks = get_neighbor_agents_future(scenario, instance_inds)
        
        # get ego future tracks
        ego_future, gt_ego_fut_cmd, ego_fut_masks = get_ego_agent_future(scenario)
        
        info['gt_boxes'] = np.array(gt_boxes)
        info['gt_names'] = np.array(gt_names)
        info['gt_velocity'] = np.array(gt_velocity)
        # not sure whehter num pts are really required. Give all value 1 for now. Also assume all datas are valid
        info['num_lidar_pts'] = np.ones(len(gt_boxes))
        info['num_radar_pts'] = np.ones(len(gt_boxes))
        info['valid_flag'] = np.array(valid_flag)
        info['instance_inds'] = np.array(instance_inds)
        info['gt_agent_fut_trajs'] = agents_future
        info['gt_agent_fut_masks'] = gt_agent_fut_masks
        info['gt_ego_fut_trajs'] = ego_future
        info['gt_ego_fut_masks'] = ego_fut_masks
        info['gt_ego_fut_cmd'] = gt_ego_fut_cmd
        info['ego_status'] = np.array(ego_status).astype(np.float32)
        
        sensors, image_pathes = scenario.get_sensors_at_iteration(0, [CameraChannel.CAM_F0, CameraChannel.CAM_B0, CameraChannel.CAM_L0, CameraChannel.CAM_L2, CameraChannel.CAM_R0, CameraChannel.CAM_R2])
        
        update_info_cams(info, image_pathes)
        
        if ENABLE_BIRDSVIEW_PLOT:
            plotBirdsview(info, ego_state.agent.box.half_length, ego_state.agent.box.half_width, sensors)
        
        train_nuplan_infos.append(info)
    
    return train_nuplan_infos

def update_info_cams(info, image_pathes):
    ### update info['cams']
    ### values that not available from nuplan, are taken over from nuscenes
    cam_front = dict()
    cam_front.update({'data_path': NUPLAN_SENSOR_ROOT + image_pathes['CAM_F0']})
    cam_front.update({'type': 'CAM_FRONT'})
    cam_front.update({'sample_data_token': 'not defined'})
    cam_front.update({'timestamp': 'not defined'})
    cam_front.update({'sensor2ego_translation': [1.72200568478, 0.00475453292289, 1.49491291905]})
    cam_front.update({'sensor2ego_rotation': [0.5077241387638071, -0.4973392230703816, 0.49837167536166627, -0.4964832014373754]})
    cam_front.update({'ego2global_translation': info['ego2global_translation']})
    cam_front.update({'ego2global_rotation': info['ego2global_rotation']})
    cam_front.update({'sensor2lidar_translation': np.array([-0.00627514,  0.44372303, -0.33161267])})
    cam_front.update({'sensor2lidar_rotation': np.array([[ 0.99988013, -0.01013819, -0.0117025 ],[ 0.01223258,  0.05390464,  0.99847116],[-0.00949188, -0.99849462,  0.05402219]])})
    cam_front.update({'cam_intrinsic': np.array([[1.25281310e+03, 0.00000000e+00, 8.26588115e+02],[0.00000000e+00, 1.25281310e+03, 4.69984663e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])})
    
    
    cam_front_right = dict()
    cam_front_right.update({'data_path': NUPLAN_SENSOR_ROOT + image_pathes['CAM_R0']})
    cam_front_right.update({'type': 'CAM_FRONT_RIGHT'})
    cam_front_right.update({'sample_data_token': 'not defined'})
    cam_front_right.update({'timestamp': 'not defined'})
    cam_front_right.update({'sensor2ego_translation': [1.58082565783, -0.499078711449, 1.51749368405]})
    cam_front_right.update({'sensor2ego_rotation': [0.20335173766558642, -0.19146333228946724, 0.6785710044972951, -0.6793609166212989]})
    cam_front_right.update({'ego2global_translation': info['ego2global_translation']})
    cam_front_right.update({'ego2global_rotation': info['ego2global_rotation']})
    cam_front_right.update({'sensor2lidar_translation': np.array([0.49830135,  0.37303191, -0.30971647])})
    cam_front_right.update({'sensor2lidar_rotation': np.array([[ 0.53727368, -0.00136775,  0.84340686],[-0.84173947,  0.06200031,  0.53631206],[-0.05302503, -0.99807519,  0.03215985]])})
    cam_front_right.update({'cam_intrinsic': np.array([[1.25674851e+03, 0.00000000e+00, 8.17788757e+02],[0.00000000e+00, 1.25674851e+03, 4.51954178e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])})
    
    cam_front_left = dict()
    cam_front_left.update({'data_path': NUPLAN_SENSOR_ROOT + image_pathes['CAM_L0']})
    cam_front_left.update({'type': 'CAM_FRONT_LEFT'})
    cam_front_left.update({'sample_data_token': 'not defined'})
    cam_front_left.update({'timestamp': 'not defined'})
    cam_front_left.update({'sensor2ego_translation': [1.5752559464, 0.500519383135, 1.50696032589]})
    cam_front_left.update({'sensor2ego_rotation': [0.6812088525125634, -0.6687507165046241, 0.2101702448905517, -0.21108161122114324]})
    cam_front_left.update({'ego2global_translation': info['ego2global_translation']})
    cam_front_left.update({'ego2global_rotation': info['ego2global_rotation']})
    cam_front_left.update({'sensor2lidar_translation': np.array([-0.5023761 ,  0.22914752, -0.33165801])})
    cam_front_left.update({'sensor2lidar_rotation': np.array([[ 0.56725815, -0.01433343, -0.82341529],[ 0.82281279,  0.05187402,  0.5659401 ],[ 0.034602  , -0.99855077,  0.04121969]])})
    cam_front_left.update({'cam_intrinsic': np.array([[1.25786253e+03, 0.00000000e+00, 8.27241063e+02],[0.00000000e+00, 1.25786253e+03, 4.50915498e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])})
    
    cam_back = dict()
    cam_back.update({'data_path': NUPLAN_SENSOR_ROOT + image_pathes['CAM_B0']})
    cam_back.update({'type': 'CAM_BACK'})
    cam_back.update({'sample_data_token': 'not defined'})
    cam_back.update({'timestamp': 'not defined'})
    cam_back.update({'sensor2ego_translation': [0.05524611077, 0.0107882366898, 1.56794286957]})
    cam_back.update({'sensor2ego_rotation': [0.5067997344989889, -0.4977567019405021, -0.4987849934090844, 0.496594225837321]})
    cam_back.update({'ego2global_translation': info['ego2global_translation']})
    cam_back.update({'ego2global_rotation': info['ego2global_rotation']})
    cam_back.update({'sensor2lidar_translation': np.array([-0.0095122 , -1.00464249, -0.3205656 ])})
    cam_back.update({'sensor2lidar_rotation': np.array([[-0.99992834, -0.00859485, -0.0083335 ],[ 0.00799071,  0.03917429, -0.99920044],[ 0.00891444, -0.99919543, -0.0391028 ]])})
    cam_back.update({'cam_intrinsic': np.array([[796.89106345, 0., 857.77743269],[0. , 796.89106345, 476.88489884],[0., 0., 1.]])})
    
    cam_back_left = dict()
    cam_back_left.update({'data_path': NUPLAN_SENSOR_ROOT + image_pathes['CAM_L2']})
    cam_back_left.update({'type': 'CAM_BACK_LEFT'})
    cam_back_left.update({'sample_data_token': 'not defined'})
    cam_back_left.update({'timestamp': 'not defined'})
    cam_back_left.update({'sensor2ego_translation': [1.04852047718, 0.483058131052, 1.56210154484]})
    cam_back_left.update({'sensor2ego_rotation': [0.7048620297871717, -0.6907306801461466, -0.11209091960167808, 0.11617345743327073]})
    cam_back_left.update({'ego2global_translation': info['ego2global_translation']})
    cam_back_left.update({'ego2global_rotation': info['ego2global_rotation']})
    cam_back_left.update({'sensor2lidar_translation': np.array([-0.48218189,  0.07357368, -0.27649454])})
    cam_back_left.update({'sensor2lidar_rotation': np.array([[-0.31910314, -0.01589122, -0.94758675],[ 0.94686077,  0.03722081, -0.31948287],[ 0.04034692, -0.9991807 ,  0.00316949]])})
    cam_back_left.update({'cam_intrinsic': np.array([[1.25498606e+03, 0.00000000e+00, 8.29576933e+02],[0.00000000e+00, 1.25498606e+03, 4.67168056e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])})
    
    cam_back_right = dict()
    cam_back_right.update({'data_path': NUPLAN_SENSOR_ROOT + image_pathes['CAM_R2']})
    cam_back_right.update({'type': 'CAM_BACK_RIGHT'})
    cam_back_right.update({'sample_data_token': 'not defined'})
    cam_back_right.update({'timestamp': 'not defined'})
    cam_back_right.update({'sensor2ego_translation': [1.05945173053, -0.46720294852, 1.55050857555]})
    cam_back_right.update({'sensor2ego_rotation': [0.13819187705364147, -0.13796718183628456, -0.6893329941542625, 0.697630335509333]})
    cam_back_right.update({'ego2global_translation': info['ego2global_translation']})
    cam_back_right.update({'ego2global_rotation': info['ego2global_rotation']})
    cam_back_right.update({'sensor2lidar_translation': np.array([ 0.46738986, -0.08280982, -0.29607485])})
    cam_back_right.update({'sensor2lidar_rotation': np.array([[-0.38201342,  0.01385406,  0.92405293],[-0.92305064,  0.04318667, -0.38224655],[-0.04520244, -0.99897096, -0.00370989]])})
    cam_back_right.update({'cam_intrinsic': np.array([[1.24996293e+03, 0.00000000e+00, 8.25376805e+02],[0.00000000e+00, 1.24996293e+03, 4.62548164e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])})
    
    info['cams'].update({'CAM_FRONT': cam_front})
    info['cams'].update({'CAM_FRONT_RIGHT': cam_front_right})
    info['cams'].update({'CAM_FRONT_LEFT': cam_front_left})
    info['cams'].update({'CAM_BACK': cam_back})
    info['cams'].update({'CAM_BACK_LEFT': cam_back_left})
    info['cams'].update({'CAM_BACK_RIGHT': cam_back_right})

def nuplan_data_prep(info_prefix,
                       version,
                       out_dir,
                       data_root):
    """Prepare data related to nuScenes dataset.
    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the database info.
        data_root (str): dataset directory, not used yet
    """
    create_nuplan_infos(out_dir, info_prefix, version=version)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='nuplan', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/nuplan',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for nuplan')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/nuplan',
    required='False',
    help='specify the root path of output files')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--data-root',
    type=str,
    default='/mnt/g/GIT/nuplan/dataset/',
    required='False',
    help='specify the root path of datasets')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'nuplan' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuplan_data_prep(
            info_prefix=args.dataset,
            version=train_version,
            out_dir=args.out_dir,
            data_root=args.data_root)
