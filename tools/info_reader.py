import pickle
import numpy as np
import pandas as pd

data_nuscenes_infos_train = None
data_nuscenes_infos_Val = None

SAVE_NPY = False
SAVE_PKL = True

# Define a custom function to combine elements
def combine_elements(pair):
    return str(pair[0]) + ', ' + str(pair[1])

def combine_elements_x(pair):
    return '(' + str(pair[0]) + ', ' + str(pair[1]) + ')'

def combine_elements_y(pair):
    result_str = ''
    for idx, x in enumerate(pair):
        result_str += (pair[idx] + ', ')
    return result_str[:-2]

# with open('./data/infos/mini/nuscenes_infos_train.pkl', 'rb') as f:
#     data_nuscenes_infos_train = pickle.load(f)
#     df = pd.DataFrame(data_nuscenes_infos_train['infos'])
#     df.to_csv(r'./data/infos/mini/nuscenes_infos_train.csv')
    
# with open('./data/infos/mini/nuscenes_infos_val.pkl', 'rb') as f:
#     data_nuscenes_infos_Val = pickle.load(f)
#     df = pd.DataFrame(data_nuscenes_infos_Val['infos'])
#     df.to_csv(r'./data/infos/mini/data_nuscenes_infos_Val.csv')
        
        
if SAVE_PKL:
    with open('./data_nuplan/infos/mini/nuplan_infos_val.pkl', 'rb') as f:
        data_nuscenes_infos_Val = pickle.load(f)
        df = pd.DataFrame(data_nuscenes_infos_Val['infos'])
        df.to_csv(r'./data_nuplan/infos/mini/data_nuplan_infos_Val.csv')
    with open('./data_nuplan/infos/mini/nuplan_infos_train.pkl', 'rb') as f:
        data_nuscenes_infos_Train = pickle.load(f)
        df = pd.DataFrame(data_nuscenes_infos_Train['infos'])
        df.to_csv(r'./data_nuplan/infos/mini/data_nuplan_infos_Train.csv')

if SAVE_NPY:
    # data_kmeans_det_900 = np.load('./data/kmeans/kmeans_det_900.npy')
    # df_data_kmeans_det_900 = pd.DataFrame(data_kmeans_det_900)
    # df.to_csv(r'./data/kmeans/data_kmeans_det_900.csv')
    
    data_kmeans_map_100 = np.load('./data/kmeans/kmeans_map_100.npy')
    tensor_combined = np.apply_along_axis(lambda x: combine_elements(x), -1, data_kmeans_map_100)
    df_data_kmeans_map_100 = pd.DataFrame(tensor_combined)
    df_data_kmeans_map_100.to_csv(r'./data/kmeans/data_kmeans_map_100.csv')

    data_kmeans_motion_6 = np.load('./data/kmeans/kmeans_motion_6.npy')
    tensor_combined_x = np.apply_along_axis(lambda x: combine_elements_x(x), -1, data_kmeans_motion_6)
    tensor_combined_y = np.apply_along_axis(lambda x: combine_elements_y(x), -1, tensor_combined_x)
    df_data_kmeans_motion_6 = pd.DataFrame(tensor_combined_y)
    df_data_kmeans_motion_6.to_csv(r'./data/kmeans/data_kmeans_motion_6.csv')

    data_kmeans_plan_6 = np.load('./data/kmeans/kmeans_plan_6.npy')
    tensor_combined_x = np.apply_along_axis(lambda x: combine_elements_x(x), -1, data_kmeans_plan_6)
    tensor_combined_y = np.apply_along_axis(lambda x: combine_elements_y(x), -1, tensor_combined_x)
    df_data_kmeans_plan_6 = pd.DataFrame(tensor_combined_y)
    df_data_kmeans_plan_6.to_csv(r'./data/kmeans/data_kmeans_plan_6.csv')

print("read data finish")

