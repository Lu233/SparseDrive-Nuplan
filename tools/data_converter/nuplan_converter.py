import os
import argparse
import math
from os import path as osp

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

import numpy as np

import mmcv

from common_utils import get_scenario_map, get_filter_parameters
from data_utils import *
from tqdm import tqdm
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

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
NUPLAN_SENSOR_ROOT = os.getenv('NUPLAN_SENSOR_ROOT', r"/mnt/g/GIT/nuplan/dataset/nuplan-v1.1/sensor_blobs", )

PAST_TIME_HORIZON = 2 #secs
NUM_PAST_POSES = 20
NUM_AGENTS = 20
FUTURE_TIME_HORIZON = 6 #secs
NUM_FUTURE_POSES = 12


NameMapping = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}


def get_ego_agent(scenario):
    anchor_ego_state = scenario.initial_ego_state
    
    past_ego_states = scenario.get_ego_past_trajectory(
        iteration=0, num_samples=NUM_PAST_POSES, time_horizon=PAST_TIME_HORIZON
    )

    sampled_past_ego_states = list(past_ego_states) + [anchor_ego_state]
    past_ego_states_tensor = sampled_past_ego_states_to_tensor(sampled_past_ego_states)

    past_time_stamps = list(
        scenario.get_past_timestamps(
            iteration=0, num_samples=NUM_PAST_POSES, time_horizon=PAST_TIME_HORIZON
        )
    ) + [scenario.start_time]

    past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

    return past_ego_states_tensor, past_time_stamps_tensor


def get_neighbor_agents(scenario):
    present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
    past_tracked_objects = [
        tracked_objects.tracked_objects
        for tracked_objects in scenario.get_past_tracked_objects(
            iteration=0, time_horizon=PAST_TIME_HORIZON, num_samples=NUM_PAST_POSES
        )
    ]

    sampled_past_observations = past_tracked_objects + [present_tracked_objects]
    past_tracked_objects_tensor_list, past_tracked_objects_types = \
            sampled_tracked_objects_to_tensor_list(sampled_past_observations)

    return past_tracked_objects_tensor_list, past_tracked_objects_types


def get_ego_agent_future(scenario):
    current_absolute_state = scenario.initial_ego_state

    trajectory_absolute_states = scenario.get_ego_future_trajectory(
        iteration=0, num_samples=6, time_horizon=3
    )

    # Get all future poses of the ego relative to the ego coordinate system
    trajectory_relative_poses = convert_absolute_to_relative_poses(
        current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
    )

    return trajectory_relative_poses

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
    agent_futures = agent_future_process(current_ego_state, future_tracked_objects_tensor_list, NUM_AGENTS, agent_index)

    return agent_futures

def create_nuplan_infos(out_path,
                          info_prefix,
                          version='v1.1-mini',
                          max_sweeps=10,
                          roi_size=(30, 60),):
    """Create info file of nuplan dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """

    print(version, NUPLAN_SENSOR_ROOT)
    
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

    train_nusc_infos = _fill_trainval_infos(scenarios, max_sweeps=max_sweeps)
    val_nusc_infos = train_nusc_infos

    metadata = dict(version=version)
    
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(out_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)


def _plotstr(info, ego_stat, cameras):
    fig, ax = plt.subplots(figsize=(8, 8))
    for box, instanceId, name in zip(info["gt_boxes"], info["instance_inds"], info["gt_names"]):
        x = box[0] - box[4] / 2   # 左下角的 x 坐标
        y = box[1] - box[3] / 2  # 左下角的 y 坐标
        rect = patches.Rectangle((x, y), box[4], box[3],
                                linewidth=1, edgecolor="blue", facecolor="cyan", alpha=0.5)
        box_heading_degree = math.degrees(box[6])
        t = Affine2D().rotate_deg_around(box[0], box[1], box_heading_degree) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        # 标注矩形中心
        ax.text(box[0], box[1], f"{instanceId}", ha="center", va="center")
        if (name == "vehicle"):
            # draw  driving direction arrow
            arrow_start = np.array([box[0], box[1]])  # Arrow starts at the center of the box
            direction = np.array([np.cos(np.radians(box_heading_degree)), np.sin(np.radians(box_heading_degree))])
            arrow_end = arrow_start + direction * 3

            # Draw the arrow
            ax.arrow(
                arrow_start[0], arrow_start[1],
                arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
                head_width=0.2, head_length=0.3, fc='black', ec='black'
            )
        
    # draw ego vehicle
    x = 0 - ego_stat.agent.box.half_length
    y = 0 - ego_stat.agent.box.half_width
    
    rect = patches.Rectangle((x, y), ego_stat.agent.box.half_length * 2, ego_stat.agent.box.half_width * 2,
                                linewidth=1, edgecolor="red", facecolor="red", alpha=0.5)
    
    ego_heading_degree = math.degrees(ego_stat.center.heading)
    t = Affine2D().rotate_deg_around(0, 0, ego_heading_degree) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    # 标注矩形中心
    ax.text(0, 0, "ego", ha="center", va="center")
    
    # draw ego driving direction arrow
    arrow_start = np.array([0, 0])  # Arrow starts at the center of the box
    direction = np.array([np.cos(np.radians(ego_heading_degree)), np.sin(np.radians(ego_heading_degree))])
    arrow_end = arrow_start + direction * 3

    # Draw the arrow
    ax.arrow(
        arrow_start[0], arrow_start[1],
        arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
        head_width=0.2, head_length=0.3, fc='black', ec='black'
    )
    
    # 设置坐标轴范围和网格
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    # 显示图像
    plt.title("Bird's View with Multiple Boxes")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    
    # 保存为 .png 文件
    output_filename = 'savefig/' + info["token"] + "_birdsview.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")  # dpi: 分辨率，bbox_inches: 修正边框
    plt.clf()  # Clears the current figure
    
    img_CAM_F0 = cameras.images[CameraChannel.CAM_F0]
    plt.imshow(img_CAM_F0.as_numpy, label = 'CAM_F0')
    filename = 'savefig/' + info["token"] + '_CAM_F0.png'
    plt.savefig(filename)
    
    img_CAM_B0 = cameras.images[CameraChannel.CAM_B0]
    plt.imshow(img_CAM_B0.as_numpy, label = 'CAM_B0')
    filename = 'savefig/' + info["token"] + '_CAM_B0.png'
    plt.savefig(filename)
    
    img_CAM_L0= cameras.images[CameraChannel.CAM_L0]
    plt.imshow(img_CAM_L0.as_numpy, label = 'CAM_L0')
    filename = 'savefig/' + info["token"] + '_CAM_L0.png'
    plt.savefig(filename)
    
    img_CAM_L2= cameras.images[CameraChannel.CAM_L2]
    plt.imshow(img_CAM_L2.as_numpy, label = 'CAM_L2')
    filename = 'savefig/' + info["token"] + '_CAM_L2.png'
    plt.savefig(filename)
    
    img_CAM_R0= cameras.images[CameraChannel.CAM_R0]
    plt.imshow(img_CAM_R0.as_numpy, label = 'CAM_R0')
    filename = 'savefig/' + info["token"] + '_CAM_R0.png'
    plt.savefig(filename)
    
    img_CAM_R2= cameras.images[CameraChannel.CAM_R2]
    plt.imshow(img_CAM_R2.as_numpy, label = 'CAM_R2')
    filename = 'savefig/' + info["token"] + '_CAM_R2.png'
    plt.savefig(filename)
    
    # 关闭图像以释放内存
    plt.close()
        

def _fill_trainval_infos(sces,
                         test=False,
                         max_sweeps=10,
                         fut_ts=12,
                         ego_fut_ts=6):
    """Generate the train/val infos from the raw data.
    Args:
    Returns:
    """
    train_nuplan_infos = []
    
    for scenario in tqdm(sces):
        map_location = scenario._map_name
        lidar2ego_transform = scenario.get_lidar_to_ego_transform()
        initial_ego_state = scenario.initial_ego_state
        # the ego2global_translation and rotation might be obtained from initial_ego_state.rear_axle.x, y, heading?
        
        lidar2ego_translation = lidar2ego_transform[:3,3]
        lidar2ego_rotation = lidar2ego_transform[:3,:3]
        
        info = {
            'lidar_path': "xxx",
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
        }
        
        tracked_objects=scenario.get_tracked_objects_at_iteration(0)
        gt_boxes = []
        gt_names = []
        gt_velocity = []
        ego_status = []
        num_lidar_pts = []
        num_radar_pts = []
        valid_flag = []
        instance_inds = []
        gt_agent_fut_trajs = []
        
        ego_state = scenario.get_ego_state_at_iteration(0)
        ego_status.extend([ego_state.dynamic_car_state.rear_axle_acceleration_2d.x, ego_state.dynamic_car_state.rear_axle_acceleration_2d.y, 0.0]) # acceleration in ego vehicle frame, m/s/s
        ego_status.extend([ego_state.dynamic_car_state.angular_velocity, 0, 0]) # angular velocity in ego vehicle frame, rad/s
        ego_status.extend([ego_state.dynamic_car_state.rear_axle_velocity_2d.x, ego_state.dynamic_car_state.rear_axle_velocity_2d.y, 0]) # velocity in ego vehicle frame, m/s
        ego_status.append(ego_state.tire_steering_angle) # steering angle, positive: left turn, negative: right turn
        
        for tracked_object in tracked_objects.tracked_objects:
            gt_boxes.append([tracked_object.box.center.x - ego_state.center.x, tracked_object.box.center.y - ego_state.center.y, 0, tracked_object.box.dimensions.width, tracked_object.box.dimensions.length, tracked_object.box.dimensions.height, tracked_object.box.center.heading])
            gt_names.append(tracked_object.tracked_object_type.fullname)
            gt_velocity.append([tracked_object.velocity.x, tracked_object.velocity.y])
            instance_inds.append(tracked_object._metadata.track_id)
            valid_flag.append(True)
            
        # get agent past tracks
        ego_agent_past, time_stamps_past = get_ego_agent(scenario)
        neighbor_agents_past, neighbor_agents_types = get_neighbor_agents(scenario)
        ego_agent_past, neighbor_agents_past, neighbor_indices = agent_past_process(ego_agent_past, time_stamps_past, neighbor_agents_past, neighbor_agents_types, NUM_AGENTS)
        agents_future = get_neighbor_agents_future(scenario, neighbor_indices)
        agents_future = agents_future[:, :, :2]
        ego_future = get_ego_agent_future(scenario)
        ego_future = ego_future[:, :2]
        
        # drive command according to final fut step offset
        if ego_future[-1][1] >= 2:
            command = np.array([1, 0, 0])  # Turn Right
        elif ego_future[-1][1] <= -2:
            command = np.array([0, 1, 0])  # Turn Left
        else:
            command = np.array([0, 0, 1])  # Go Straight
            
        gt_ego_fut_cmd = command.astype(np.float32)
        
        info['gt_boxes'] = np.array(gt_boxes)
        info['gt_names'] = np.array(gt_names)
        info['gt_velocity'] = np.array(gt_velocity)
        #info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in annotations])
        #info['num_radar_pts'] = np.array([a['num_radar_pts'] for a in annotations])
        info['valid_flag'] = np.array(valid_flag)
        info['instance_inds'] = np.array(instance_inds)
        info['gt_agent_fut_trajs'] = agents_future
        #info['gt_agent_fut_masks'] = gt_fut_masks.astype(np.float32)
        info['gt_ego_fut_trajs'] = ego_future
        #info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)
        info['gt_ego_fut_cmd'] = gt_ego_fut_cmd
        info['ego_status'] = np.array(ego_status).astype(np.float32)

        
        train_nuplan_infos.append(info)
        
        sensors = scenario.get_sensors_at_iteration(0, [CameraChannel.CAM_F0, CameraChannel.CAM_B0, CameraChannel.CAM_L0, CameraChannel.CAM_L2, CameraChannel.CAM_R0, CameraChannel.CAM_R2])
        
        _plotstr(info, ego_state, sensors)


    return train_nuplan_infos


def nuplan_data_prep(info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    create_nuplan_infos(out_dir, info_prefix, version=version, max_sweeps=max_sweeps)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'nuplan' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuplan_data_prep(
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuplanDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
