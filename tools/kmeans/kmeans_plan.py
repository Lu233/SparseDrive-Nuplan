import os
import pickle
import copy
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

K = 6
Version = "mini"

fp = 'data/infos/mini/nuscenes_infos_train.pkl'
data = mmcv.load(fp)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
navi_trajs = [[], [], []]
for idx in tqdm(range(len(data_infos))):
    info = data_infos[idx]
    plan_traj = info['gt_ego_fut_trajs'].cumsum(axis=-2)
    plan_mask = info['gt_ego_fut_masks']
    cmd = info['gt_ego_fut_cmd'].astype(np.int32)
    cmd = cmd.argmax(axis=-1)
    if not plan_mask.sum() == 6:
        continue
    navi_trajs[cmd].append(plan_traj)
    
if Version == "mini":
    # the mini version data does not contain cmd = [1, 0, 0]
    navi_trajs[0] = copy.deepcopy(navi_trajs[1])
    for traj in navi_trajs[0]:
        for sublist in traj:
            sublist[0] = sublist[0]*(-1)

clusters = []
for trajs in navi_trajs:
    if len(trajs) == 0:
        continue
    trajs = np.concatenate(trajs, axis=0).reshape(-1, 12)
    cluster = KMeans(n_clusters=K).fit(trajs).cluster_centers_
    cluster = cluster.reshape(-1, 6, 2)
    clusters.append(cluster)
    for j in range(K):
        plt.scatter(cluster[j, :, 0], cluster[j, :,1])
plt.savefig(f'vis_test/kmeans/plan_{K}', bbox_inches='tight')
plt.close()

clusters = np.stack(clusters, axis=0)
np.save(f'data_test/kmeans/kmeans_plan_{K}.npy', clusters)