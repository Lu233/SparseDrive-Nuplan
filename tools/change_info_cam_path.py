import pickle
import pandas as pd

LOCAL_PATH_NUPLAN = "/mnt/g/GIT/nuplan/dataset/nuplan-v1.1/sensor_blobs/"
REMOTE_PATH_NUPLAN = "/root/autodl-tmp/nuplan-v1.1_mini_camera_0/"

LOCAL_PATH_NUSCENES = "/mnt/g/GIT/nuscene/"
REMOTE_PATH_NUSCENES = "/root/autodl-tmp/nuscenes/"

# with open('./data_nuplan/infos/mini/nuplan_infos_val.pkl', 'rb') as f:
#     data = pickle.load(f)
#     df = pd.DataFrame(data['infos'])
#     for cam in df.cams:
#         for cam_it in cam:
#             cam[cam_it]['data_path'] = cam[cam_it]['data_path'].replace(LOCAL_PATH_NUPLAN, REMOTE_PATH_NUPLAN)
#     with open("./data_nuplan/infos/mini/nuplan_infos_val_remote.pkl", "wb") as file:
#         pickle.dump(df, file)
    
#     df.to_csv(r'./data_nuplan/infos/mini/data_nuplan_infos_Val_remote.csv')

'/mnt/g/GIT/nuscene/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg'

with open('./data/infos/mini/nuscenes_infos_val.pkl', 'rb') as f:
    data = pickle.load(f)
    df = pd.DataFrame(data['infos'])
    for cam in df.cams:
        for cam_it in cam:
            cam[cam_it]['data_path'] = cam[cam_it]['data_path'].replace(LOCAL_PATH_NUSCENES, REMOTE_PATH_NUSCENES)
    with open("./data/infos/mini/nuscenes_infos_val_remote.pkl", "wb") as file:
        pickle.dump(df, file)
    
    df.to_csv(r'./data/infos/mini/data_nuscenes_infos_Val_remote.csv')

print("Process finish")

