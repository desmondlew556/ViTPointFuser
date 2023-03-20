### Code structure:
### 2DPASS
contains files on 3D semantic segmentation, and ViTPointFuser
Reference: https://github.com/yanx27/2DPASS
- Refer to https://github.com/yanx27/2DPASS for information on the pretrained network, how to train and test the model

### ViTPointFuser
Reference: https://github.com/huggingface/transformers
- To train/test ViTPointFuser,
1. Amend the configuration file in config/2DPASS-fuser-nuscenese.yaml
 - set training_detections_fuser: False, if you are training ViTPointFuser on a subset of the train-val dataset
 - set split_across_channels: True to split the object proposal feature map along the channel axis. Refer to report 3.1 ViTPointFuser for more information on this. Alternatively, if this is set to False, the object proposal feature map would not be split long the channels. The resulting feature map would be C x H x W, where C is channel width, H is height, and W is width.
 - Change ViT configurations in the vit key at the end of the file
*** Note: there is a 2nd variation of ViTPointFuser that allows the changing of the size of the MLP in the classifier head. Amend the file in config/2DPASS-fuser-variant-nuscenese.yaml instead. The mlp_size variable is used to control the size of the mlp in the classifier head.

2. Change the variable self.saved_detections_roots = ["results","results_1"] in data_loader/pc_dataset to specify the data directories in which to find the saved object detections data.

3. Run the file train_2dpass_fuser.py
*** Note: the model_name variable saved in the configurations file is used for initialization of the model.


### BEVfusion
contains files on 3D object detection
Reference: https://github.com/mit-han-lab/bevfusion
- Refer to https://github.com/mit-han-lab/bevfusion for information on the pretrained network, how to train and test the model
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

Google drive link for saved ViTPointFuser models and configuration files:
https://drive.google.com/drive/folders/1sV1W_y47RlW4HPUJy4a1iNSHX9e-dYFA?usp=share_link# ViTPointFuser
