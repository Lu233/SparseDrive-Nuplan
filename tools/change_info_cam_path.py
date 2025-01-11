import pickle
import pandas as pd

data_nuscenes_infos_train = None
data_nuscenes_infos_Val = None

ORIGIN_PATH = "/mnt/g/GIT/nuplan/dataset/nuplan-v1.1/sensor_blobs/"
TARGET_PATH = "/root/autodl-tmp/nuplan-v1.1_mini_camera_0/"

with open('./data_nuplan/infos/mini/nuplan_infos_val.pkl', 'rb') as f:
    data_nuscenes_infos_Val = pickle.load(f)
    df = pd.DataFrame(data_nuscenes_infos_Val['infos'])
    for cam in df.cams:
        for cam_it in cam:
            cam[cam_it]['data_path'] = cam[cam_it]['data_path'].replace(ORIGIN_PATH, TARGET_PATH)
    with open("./data_nuplan/infos/mini/nuplan_infos_val_remote.pkl", "wb") as file:
        pickle.dump(df, file)
    
    df.to_csv(r'./data_nuplan/infos/mini/data_nuplan_infos_Val_remote.csv')
    
# with open('./data/infos/mini/nuscenes_infos_val.pkl', 'rb') as f:
#     data_nuscenes_infos_Val = pickle.load(f)
#     df = pd.DataFrame(data_nuscenes_infos_Val['infos'])
#     for cam in df.cams:
#         for cam_it in cam:
#             cam[cam_it]['data_path'] = cam[cam_it]['data_path'].replace(ORIGIN_PATH, TARGET_PATH)
#     with open("./data/infos/mini/nuscenes_infos_val_remote.pkl", "wb") as file:
#         pickle.dump(df, file)
    
#     df.to_csv(r'./data/infos/mini/data_nuplan_infos_Val_remote.csv')

print("Process finish")

