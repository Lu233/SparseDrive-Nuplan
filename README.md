# Migration of SparseDrive from nuscenes to nuplan

## Introduction
This is a fork of the official [SparseDrive](https://github.com/swc-17/SparseDrive) repository. The primary goal of this fork is to migrate the SparseDrive model to the NuPlan dataset. SparseDrive is a Sparse-Centric paradigm for end-to-end autonomous driving, originally trained and validated on the NuScenes dataset.
Since 2022, the NuPlan dataset has been available as the world's first large-scale planning benchmark for autonomous driving. It provides both open-loop and closed-loop resimulation capabilities, enabling comprehensive validation of planners. This project aims to explore how SparseDrive performs with the NuPlan dataset.


## Works have been done so far
- Developed a nuplan_converter to convert the NuPlan dataset into the same PKL file format used by the original [SparseDrive](https://github.com/swc-17/SparseDrive). A built-in bird's-eye-view visualization tool was implemented to verify the correctness of the converted data format.
- Both test and visualization are slightly adapted so that it works with the nuplan converted pkl.
- As my local machine lacks a powerful GPU compatible with FlashAttention, I utilized a cloud environment with an NVIDIA 4090 for training and inference. Data conversion and visualization tasks were kept on the local machine. To facilitate this, I developed the change_info_cam_path.py script to adapt image paths in PKL files between local and remote environments. Benefits: Data conversion (especially for the full dataset) doesn’t require a powerful GPU, saving costs.
- Created script files to streamline steps such as data preparation, testing, and visualization on both local and remote environments.
- Tested so far using the NuPlan mini dataset.

## Intermediate result
### inference nuplan mini dataset
Model: SparseDrive pre-trained model released by the official repository: [ckpt](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2.pth) 
![Nuplan dataset inference](docs/intermediate_result.gif)

### Bird's-Eye View Plot Comparison: NuScenes vs. NuPlan After Data Conversion
nuscenes example
![nuscenes example](docs/nuscenes_birdsview_example.png)

nuplan example
![nuplan example](docs/nuplan_birdsview_example.png)

## Conclusion
The inference results indicate that detection and prediction generally work; however, the performance remains suboptimal. This could be attributed to:
- Differences in the actual camera mounting positions and intrinsics between NuPlan and NuScenes vehicles. Unfortunately, the NuPlan dataset does not provide detailed camera specifications, so the data from NuScenes was used as a substitute.

## next step
- Fine-tune the pre-trained SparseDrive model using the NuPlan dataset to improve performance.
- Integrate NuPlan inference into NuPlan’s closed-loop resimulation.

## Quick Start
The quick start is slightly different compare to the orignal [Quick Start](docs/quick_start.md). But it is compatible with all the original SparseDrive functions.

### Set up a new virtual environment, 
python 3.9 is used in oder to be compatible with nuplan
```bash
conda create -n sparsedrive python=3.9 -y
conda activate sparsedrive
```

### Install dependency packpages
```bash
sparsedrive_path="path/to/sparsedrive"
cd ${sparsedrive_path}
pip install --upgrade pip
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirement.txt
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
```

### Compile the deformable_aggregation CUDA op
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
```
### Prepare the data
Download the [Nuplan dataset](https://www.nuscenes.org/nuplan#download), it should be put as below file structure<br>
-dataset <br>
--maps<br>
--nuplan-v1.1<br>
---sensor_blobs<br>
---splits<br>

```bash
cd ${sparsedrive_path}
mkdir data_nuplan
```
Open project with VSCode. Launch Nuplan data preparation Script from launch.json file (make sure you have adapted the pathes in launch.json accordingly). Results file will be written under data_nuplan as .pkl. <br>

If you need to use the .pkl file also on another machine, you could run change_info_cam_path.py (make sure you have adapted the LOCAL_PATH and REMOTE_PATH accordingly), this will generate a new .pkl that could be used on the target machine.

### Generate anchors by K-means
Gnerated anchors are saved to data/kmeans and can be visualized in vis/kmeans.
```bash
sh scripts/kmeans.sh
```

### Download pre-trained weights
Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).
```bash
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

## Commence training and testing

### train
to be added

### test
Open project with VSCode. Launch Nuplan Test Script from launch.json file (make sure you have adapted the pathes in launch.json accordingly)

or

```
sh tools/test_nuplan.sh #if you are on a machine without flashattention compatibility

sh tools/test_nuplan_remote.sh #if you are on a machine with flashattention compatibility
```

### Visualization
Open project with VSCode. Launch Nuplan Visualize Script from launch.json file (make sure you have adapted the pathes in launch.json accordingly)

or

```
sh tools/visualize_nuplan.sh
```