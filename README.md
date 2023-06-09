### Code structure:
### 2DPASS
contains files on 3D semantic segmentation, and ViTPointFuser
Reference: https://github.com/yanx27/2DPASS
- Refer to https://github.com/yanx27/2DPASS for information on the pretrained network, how to train and test the model
- Note. Add a soft link with name "dataset" to the data root in the bevfusion folder. 
The data root should have the following structure (refer to BEVFusion and 2DPASS github for data preparation steps):
```
2dpass
├── network
├── dataloader
├── config
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
│   │   |   ├── category.json
│   │   |   ├── lidarseg.json
|   |   ├── v1.0-trainval
│   │   |   ├── category.json
│   │   |   ├── lidarseg.json
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
│   │   ├── lidarseg
│   │   |   ├── v1.0-trainval
│   │   |   ├── v1.0-test
```

### ViTPointFuser
Reference: https://github.com/huggingface/transformers
- To train/test ViTPointFuser,
1. Amend the configuration file in config/2DPASS-fuser-nuscenese.yaml
 - set training_detections_fuser: False, if you are training ViTPointFuser on a subset of the train-val dataset. Else, set this value as True when running test to use the entire dataset
 - set split_across_channels: True to split the object proposal feature map along the channel axis. Refer to report 3.1 ViTPointFuser for more information on this. Alternatively, if this is set to False, the object proposal feature map would not be split long the channels. The resulting feature map would be C x H x W, where C is channel width, H is height, and W is width.
 - Change ViT configurations in the vit key at the end of the file

2. Change the variable self.saved_detections_roots = ["results","results_1"] in dataloader/pc_dataset to specify the data directories in which to find the saved object detections data.

3. Run the file train_2dpass_fuser.py<br>
*** Note: the model_name variable saved in the configurations file is used for initialization of the model.
To pretrain the a MLP head, set mlp_size and pretraining: True under vit key in the configs file. Update the model name in the config file to "_2dpass_fuser_variation_1"

To train:
```
cd <root dir of this repo>
python train_2dpass_fuser.py --config config/<config file> --gpu 0 --checkpoint <dir for the pytorch checkpoint>
```
To test:
```
cd <root dir of this repo>
python train_2dpass_fuser.py --config config/<config file> --gpu 0 --test --num_vote 2 --checkpoint <dir for the pytorch checkpoint>
```
Num_vote refers to the number of times to duplicate a sample for test-time augmentation.

#### ViTPointFuser variants
*** Note: there is a are multiple variations of ViTPointFuser.
1. _2dpass_fuser_variation_1.py: 
- ViTPointFuser with MLP head. Amend size of MLP hidden units with the config vit.mlp_size and set vit.pretraining = True. 
- Refer to the config file for reference. 2DPASS-channel-width-128-mlp-hiddensize-512--nuscenese-fuser.yaml
2. _2dpass_fuser_variation_2.py: 
- ViTPointFuser with point features from multiple scales. set vit.point_dim as a list of channel widths of the input feature, and vit.multi_scale_point_features_indices as a list of the indices of the point feature map from the SPVCNN to use. The length of both point_dim and multi_scale_point_features_indices must match. 
- Additionally, set vit.additional_feature_list_channel_width as a list of input width of additional inputs to the ViTPointFuser model.
- Refer to to the config file for reference2DPASS-channel-width-128-6-point-scales-2-attention-head-additional-features-MLP-nuscenese-fuser.yaml

Note: If another model file is added and to be trained with the feature maps from object detection, ensure that the condition check in dataloaders/pc_dataset is satisfied. The condition is "if self.model_name.startswith("_2dpass_fuser")"

### BEVfusion
contains files on 3D object detection
Reference: https://github.com/mit-han-lab/bevfusion
- Refer to https://github.com/mit-han-lab/bevfusion for information on the pretrained network, how to train and test the model
- Note. Add a soft link with name "data" to the data root in the bevfusion folder. 
The data root should have the following structure (refer to BEVFusion github for data preparation steps):
```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
```

- To save object detections:
1. change the data_root in save_data() function in mmdet3d/models/fusion_models/bevfusion.py to the folder in which to save detection files. In the folder, there must be folders with the following structure:
- train
 - data (a list of information on detections for each sample)
 - detections (contains BEV feature map of the sample)
- val
 - data
 - detections
- test
 - data
 - detections
2. In the file tools/save_detections.py, comment out the relevant parts to save detections for train, validation, and test sets.
3. If desired, change thresholds for object detection by changing the variables self.bboxes_thresholds, and self.min_detections (number of detections to obtain by iteratively lowering thresholds according to self.bboxes_thresholds). Found in mmdet3d/models/fusion_models/bevfusion.py file.
4. Run tools/save_detections.py

To train:
```
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --load_from pretrained/<checkpoint file>
```

To test:
```
torchpack dist-run -np 8 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --load_from pretrained/<checkpoint file> --eval bbox
```
Google drive link for saved ViTPointFuser models and configuration files:
https://drive.google.com/drive/folders/1sV1W_y47RlW4HPUJy4a1iNSHX9e-dYFA?usp=share_link# ViTPointFuser

### visualization
Refer to visualization_.ipynb for attention scores heatmap

### Downloading dependencies
The bev2dpass_envt.yaml file provides a conda environment that could be used to quick-start the set-up. However, some required packages may not be downloaded. Refer to the dockerfile in BEVFusion github and 2DPASS readme for the required packages.