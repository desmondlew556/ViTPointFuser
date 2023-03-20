import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import time

from network.basic_block import Lovasz_loss
from network.spvcnn import get_model as SPVCNN
from network.base_model import LightningBaseModel
from network.basic_block import ResNetFCN
from network.arch_2dpass import get_model as _2DPASS
from network.base_model import LightningBaseModel
import torch.nn.functional as F
from network.basic_block import Lovasz_loss

from transformers.src.transformers.configuration_utils import PretrainedConfig
from transformers.src.transformers.models.vit.modeling_vit import ViTModel
from transformers.src.transformers.models.vit.configuration_vit import ViTConfig

class get_model(LightningBaseModel):
    def __init__(self, config):
        super(get_model, self).__init__(config)
        # models
        self.save_hyperparameters()
        self.num_classes = config['model_params']['num_classes']
        self.hiden_size = config['model_params']['hiden_size']
        model = _2DPASS(config)
        self.base_model = model.load_from_checkpoint(config._2dpass_checkpoint)
        self.base_model.freeze()
        
        self.vit_config = ViTConfig(hidden_size=config.vit.hidden_size,
                num_hidden_layers=config.vit.num_hidden_layers,
                num_attention_heads=config.vit.num_attention_heads,
                intermediate_size=config.vit.intermediate_size,
                hidden_act=config.vit.hidden_act,
                hidden_dropout_prob=config.vit.hidden_dropout_prob,
                attention_probs_dropout_prob=config.vit.attention_probs_dropout_prob,
                initializer_range=config.vit.initializer_range,
                layer_norm_eps=config.vit.layer_norm_eps,
                image_size=config.vit.image_size,
                patch_size=config.vit.patch_size,
                num_channels=config.vit.num_channels,
                qkv_bias=config.vit.qkv_bias,
                encoder_stride=config.vit.encoder_stride,
                point_dim=config.vit.point_dim)
        self.vit = ViTModel(self.vit_config,add_pooling_layer=config.vit.pool)
        self.vit_mlp_head = nn.Linear(config.vit.hidden_size, config.vit.mlp_size)
        self.vit_classifier = nn.Linear(config.vit.mlp_size, self.num_classes) if self.num_classes > 0 else nn.Identity()
        # losses
        self.config = config
        self.lambda_lovasz = self.config['train_params'].get('lambda_lovasz', 0.1)
        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(
            weight=seg_labelweights,
            ignore_index=config['dataset_params']['ignore_label']
        )
        self.lovasz_loss = Lovasz_loss(
            ignore=config['dataset_params']['ignore_label']
        )
        # labels
        self.ignore_label = config['dataset_params']['ignore_label']
        self.test = config.test
        
    def forward(self, data_dict):
        # 3D network
        start_time = time.time()
        data_dict = self.base_model(data_dict)

        base_model_time = time.time()

        point_features = self.base_model.model_3d.saved_points_features
        
        data_dict['loss_main_ce_vit']= None
        data_dict['loss_main_lovasz_vit']= None
        data_dict['loss_vit'] = None
        
        # Fuse base model's point features with 2d feature map from BEV object detection and run classification on points within bboxes
        if "feature_maps" in data_dict['detected_obj_data']:      
            
            # indices of each point 
            ref_xyz_indices = torch.tensor([i for i in range(len(data_dict["ref_xyz"]))])
            
            # each iteration goes through predicted bboxes for that sample
            feature_maps = data_dict['detected_obj_data']["feature_maps"]
            object_boundary = data_dict['detected_obj_data']["xy_range"]

            obj_feature_map,points,point_labels,point_indices = self.get_points_inside_detected_boxes(feature_maps,object_boundary,data_dict["ref_xyz"],ref_xyz_indices,data_dict["labels"],point_features[0])

            outputs = self.vit(
                    obj_feature_map,
                    head_mask=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    interpolate_pos_encoding=None,
                    return_dict=True,
                    point_feature=points,
            )
            sequence_output = outputs[0]

            # Classifier head
            logits = self.vit_classifier(sequence_output[:, 0, :])

            # losses 
            loss_ce = self.ce_loss(logits, point_labels.long())
            
            if logits.shape[0]<5:
                return data_dict
            loss_lovasz = self.lovasz_loss(F.softmax(logits, dim=1), point_labels.long())
            loss_combined = loss_ce + loss_lovasz * self.lambda_lovasz

            # save logits
            data_dict['logits_head_alt'] = logits
            data_dict['logits_head_alt_indices'] = point_indices
            
            # add accumulated losses
            data_dict['loss_main_ce_vit']= loss_ce
            data_dict['loss_main_lovasz_vit']= loss_lovasz
            data_dict['loss_vit'] = loss_combined
        
        fuser_time = time.time()
        print("base model took: %.3f sec, vit fuser took: %.3f sec"%(base_model_time-start_time,fuser_time-base_model_time))

        return data_dict
        
    def get_points_inside_detected_boxes(self,feature_maps,point_cloud_xy_ranges,ref_xyz,ref_xyz_indices,point_labels,point_features):
        """
        Returns the point features, feature maps of points within detected bboxes
        """
        obj_feature_maps = torch.ones([0]+list(feature_maps.shape[1:])).to(device="cuda")
        obj_points_features = torch.ones([0]+list(point_features.shape[1:])).to(device="cuda")
        indices = torch.ones((0,))
        labels = torch.ones((0,)).to(device="cuda")
        
        
        for i in range(feature_maps.shape[0]):
            
            feature_map = feature_maps[i]
            point_cloud_xy_range = point_cloud_xy_ranges[i]
            
            # filter points within the detected object's point cloud range
            mask_x =  torch.logical_and(ref_xyz[:,0]>point_cloud_xy_range[1],ref_xyz[:,0]<point_cloud_xy_range[3])
            mask_y = torch.logical_and(ref_xyz[:,1]>point_cloud_xy_range[0],ref_xyz[:,1]<point_cloud_xy_range[2])
            mask = torch.logical_and(mask_x,mask_y)
            
            # get data on the points within the detected object. This inclues feature map, point feature, and labels
            indices_ = ref_xyz_indices[mask] # each element contains index of the point in the original ref_xyz tensor
            obj_points_features_ = point_features[mask]
            labels_ = point_labels[mask]
            obj_feature_maps_ = feature_map.unsqueeze(0)
            obj_feature_maps_ = obj_feature_maps_.repeat(len(indices_),1,1,1)
            
            # Accumulate the data
            obj_feature_maps = torch.cat((obj_feature_maps,obj_feature_maps_),axis = 0)
            obj_points_features = torch.cat((obj_points_features,obj_points_features_),axis = 0)
            indices = torch.cat((indices,indices_),axis = 0)
            labels = torch.cat((labels,labels_),axis = 0)
            
            

        return obj_feature_maps,obj_points_features,labels,indices.long()

    def remove_duplicated_points(self,logits,indices):
        '''
        Remove duplicated values in indices and their corresponding logits. e.g. if indices [0,0,1,2] and logits [0.9,0.1,0.7,0.4], then the return indices will be [0,1,2] and logits is [0.9,0.7,0.4], where the row with higher max logits is kept. The order is not guaranteed.
        
        '''

        # get predicted counts for each point
        index,counts = torch.unique(indices,return_counts=True)
        point_idx_counts = dict([(z.item(),y.item()) for z,y in zip(index,counts)])
        point_logits = dict() # stores logits for duplicated points idx. {0:[0.8,0.1,0.9]}
        point_indices = dict() # stores index for duplicated points idx. {0:[1,4,2]}

        new_indices = torch.zeros((0,),dtype=int).to(device="cuda")
        new_logits = torch.zeros([0]+list(logits.shape[1:])).to(device="cuda")

        # go through each predicted point
        for idx in range(indices.shape[0]):
            point_idx = indices[idx].item()
            if point_idx_counts[point_idx]>1:
                if point_logits.get(point_idx,None)==None:
                    point_logits[point_idx] = []
                    point_indices[point_idx] = []
                point_logits[point_idx].append(logits[idx].max()) 
                point_indices[point_idx].append(idx)
            else:
                new_indices=torch.cat((new_indices,indices[idx].unsqueeze(0).to(device="cuda")), axis = 0)
                new_logits = torch.cat((new_logits,logits[idx].unsqueeze(0).to(device="cuda")), axis = 0)

        for point_idx in point_logits:
            max_idx = torch.argmax(torch.tensor(point_logits[point_idx]))
            idx = point_indices[point_idx][max_idx]
            new_indices=torch.cat((new_indices,indices[idx].unsqueeze(0).to(device="cuda")), axis = 0)
            new_logits = torch.cat((new_logits,logits[idx].unsqueeze(0).to(device="cuda")), axis = 0)
            
        return new_logits,new_indices
    
    def training_step(self, data_dict, batch_idx):
        data_dict = self.forward(data_dict)
        
        base_model_logits = data_dict['logits']
        
        if data_dict.get('logits_head_alt',None)!=None and data_dict['logits_head_alt'].shape[0]>0:
            fuser_logits = data_dict['logits_head_alt']
            main_index = data_dict['logits_head_alt_indices']
            labels = data_dict['labels'][main_index]

            self.train_acc(fuser_logits.argmax(1)[labels != self.ignore_label],
                       labels[labels != self.ignore_label])

        else:
            self.train_acc(base_model_logits.argmax(1)[data_dict['labels'] != self.ignore_label],
                       data_dict['labels'][data_dict['labels'] != self.ignore_label])
        self.log('train/acc', self.train_acc, on_epoch=True)
        if data_dict['loss_main_ce_vit']!=None:
            self.log('train/loss_main_ce', data_dict['loss_main_ce_vit'])
        if data_dict['loss_main_lovasz_vit']!=None:
            self.log('train/loss_main_lovasz', data_dict['loss_main_lovasz_vit'])

        return data_dict['loss_vit']

    def validation_step(self, data_dict, batch_idx):
        indices = data_dict['indices']
        raw_labels = data_dict['raw_labels'].squeeze(1).cpu()
        origin_len = data_dict['origin_len']
        vote_logits = torch.zeros((len(raw_labels), self.num_classes))
        data_dict = self.forward(data_dict)
        
        base_model_logits = data_dict['logits']
        
        if data_dict.get('logits_head_alt',None)!=None and data_dict['logits_head_alt'].shape[0]>0:
            fuser_logits = data_dict['logits_head_alt']
            main_index = data_dict['logits_head_alt_indices']
            
            fuser_logits = data_dict['logits_head_alt']
            main_index = data_dict['logits_head_alt_indices']
            labels = data_dict['labels'][main_index]

            if self.args['test']:
                # only select predictions that are within first sample
                num_points_first_sample = vote_logits.shape[0]
                mask = main_index < num_points_first_sample # the main_index stores indices referring to the point's location in the entire tensor of points
                fuser_logits = fuser_logits[mask]
                labels = labels[mask]
            
            prediction = fuser_logits.argmax(1)

            if self.ignore_label != 0:
                prediction = prediction[labels != self.ignore_label]
                labels = labels[labels != self.ignore_label]
                prediction += 1
                labels += 1

            self.val_acc(prediction[labels != self.ignore_label],
                    labels[labels != self.ignore_label])
            self.log('val/acc', self.val_acc, on_epoch=True)
            self.val_iou(
                prediction.cpu().detach().numpy(),
                labels.cpu().detach().numpy(),
            )
        
        else:
            if self.args['test']:
                vote_logits.index_add_(0, indices.cpu(), base_model_logits.cpu())
                if self.args['dataset_params']['pc_dataset_type'] == 'SemanticKITTI_multiscan':
                    vote_logits = vote_logits[:origin_len]
                    raw_labels = raw_labels[:origin_len]
            else:
                vote_logits = base_model_logits.cpu()
                raw_labels = data_dict['labels'].squeeze(0).cpu()

            prediction = vote_logits.argmax(1)

            if self.ignore_label != 0:
                prediction = prediction[raw_labels != self.ignore_label]
                raw_labels = raw_labels[raw_labels != self.ignore_label]
                prediction += 1
                raw_labels += 1

            self.val_acc(prediction, raw_labels)
            self.log('val/acc', self.val_acc, on_epoch=True)
            self.val_iou(
                prediction.cpu().detach().numpy(),
                raw_labels.cpu().detach().numpy(),
            )

        return data_dict['loss_vit']
        
    def test_step(self, data_dict, batch_idx):
        '''
        The logits of the alternate head (ViT fuser) predictions are inserted into the main head (2dpass) predictions
        '''
        indices = data_dict['indices']
        origin_len = data_dict['origin_len']
        raw_labels = data_dict['raw_labels'].squeeze(1).cpu()
        ref_xyz = data_dict['ref_xyz']
        path = data_dict['path'][0]
        
        data_dict = self.forward(data_dict)
        
        base_model_logits = data_dict['logits']
        
        vote_logits = torch.zeros((len(raw_labels), self.num_classes))
        ref_xyz_new = torch.zeros((len(raw_labels), 3))
        
        start_time = time.time()
        if data_dict.get('logits_head_alt',None)!=None and data_dict['logits_head_alt'].shape[0]>0:
            # compare predictions only for test sample
            fuser_logits = data_dict['logits_head_alt']
            main_index = data_dict['logits_head_alt_indices']
            
            fuser_logits = data_dict['logits_head_alt']
            main_index = data_dict['logits_head_alt_indices']
            labels = data_dict['labels'][main_index]

            if self.args['test']:
                # only select predictions that are within first sample
                num_points_first_sample = vote_logits.shape[0]
                mask = main_index < num_points_first_sample # the main_index stores indices referring to the point's location in the entire tensor of points
                fuser_logits = fuser_logits[mask]
                labels = labels[mask]
            
            prediction = fuser_logits.argmax(1)

            if self.ignore_label != 0:
                prediction = prediction[labels != self.ignore_label]
                labels = labels[labels != self.ignore_label]
                prediction += 1
                labels += 1

            self.val_acc(prediction[labels != self.ignore_label],
                    labels[labels != self.ignore_label])
            self.log('val/acc', self.val_acc, on_epoch=True)
            self.val_iou(
                prediction.cpu().detach().numpy(),
                labels.cpu().detach().numpy(),
            )

            # # this segment combines the results with the base model
            # fuser_logits = data_dict['logits_head_alt']
            # main_index = data_dict['logits_head_alt_indices']
            # fuser_logits,main_index = self.remove_duplicated_points(fuser_logits,main_index)
            
            # # replace logits of base model with that of alternate head when the latter yield higher values
            # base_logits = base_model_logits[main_index]
            # # replace logits of base model with that of alternate head when the latter yield higher values
            # base_logits = base_model_logits[main_index]

            # #preprocessing steps to store indices of the logits separately for indices in each axis
            # index = torch.arange(base_logits.shape[0])
            # first_index = index.unsqueeze(1)
            # first_index = first_index.repeat(1,base_logits.shape[1]).cuda()
            # index = torch.arange(base_logits.shape[1])
            # second_index = index.unsqueeze(0)
            # second_index = second_index.repeat(base_logits.shape[0],1).cuda()

            # # values to change in base logits
            # mask = fuser_logits>base_logits

            # # update base logits
            # filtered_logits = fuser_logits[mask]
            # first_index = first_index[mask]
            # second_index = second_index[mask]
            # base_logits[first_index,second_index]=filtered_logits

            # # update the main base logits
            # base_model_logits[main_index]=base_logits

            # print("Max logits selection took %.3f sec"%(time.time()-start_time))

        
        ref_xyz_new.index_add_(0, indices.cpu(), ref_xyz.cpu())
        vote_logits.index_add_(0, indices.cpu(), base_model_logits.cpu())

        if self.args['dataset_params']['pc_dataset_type'] == 'SemanticKITTI_multiscan':
            vote_logits = vote_logits[:origin_len]
            raw_labels = raw_labels[:origin_len]
            ref_xyz_new = ref_xyz_new[:origin_len]
        prediction = vote_logits.argmax(1)

        if self.ignore_label != 0:
            prediction = prediction[raw_labels != self.ignore_label]
            raw_labels = raw_labels[raw_labels != self.ignore_label]
            prediction += 1
            raw_labels += 1

        if not self.args['submit_to_server']:
            self.val_acc(prediction, raw_labels)
            self.log('val/acc', self.val_acc, on_epoch=True)
            self.val_iou(
                prediction.cpu().detach().numpy(),
                raw_labels.cpu().detach().numpy(),
             )
            # calculate iou for various distances from lidar sweep center
            if self.val_iou_list is not None:
                self.get_val_iou_by_distance(vote_logits,raw_labels,ref_xyz_new,torch.arange(ref_xyz_new.shape[0]))

        else:
            if self.args['dataset_params']['pc_dataset_type'] != 'nuScenes':
                components = path.split('/')
                sequence = components[-3]
                points_name = components[-1]
                label_name = points_name.replace('bin', 'label')
                full_save_dir = os.path.join(self.submit_dir, 'sequences', sequence, 'predictions')
                os.makedirs(full_save_dir, exist_ok=True)
                full_label_name = os.path.join(full_save_dir, label_name)

                if os.path.exists(full_label_name):
                    print('%s already exsist...' % (label_name))
                    pass

                valid_labels = np.vectorize(self.mapfile['learning_map_inv'].__getitem__)
                original_label = valid_labels(vote_logits.argmax(1).cpu().numpy().astype(int))
                final_preds = original_label.astype(np.uint32)
                final_preds.tofile(full_label_name)

            else:
                meta_dict = {
                    "meta": {
                        "use_camera": False,
                        "use_lidar": True,
                        "use_map": False,
                        "use_radar": False,
                        "use_external": False,
                    }
                }
                os.makedirs(os.path.join(self.submit_dir, 'test'), exist_ok=True)
                with open(os.path.join(self.submit_dir, 'test', 'submission.json'), 'w', encoding='utf-8') as f:
                    json.dump(meta_dict, f)
                original_label = prediction.cpu().numpy().astype(np.uint8)

                assert all((original_label > 0) & (original_label < 17)), \
                    "Error: Array for predictions must be between 1 and 16 (inclusive)."

                full_save_dir = os.path.join(self.submit_dir, 'lidarseg/test')
                full_label_name = os.path.join(full_save_dir, path + '_lidarseg.bin')
                os.makedirs(full_save_dir, exist_ok=True)

                if os.path.exists(full_label_name):
                    print('%s already exsist...' % (full_label_name))
                else:
                    original_label.tofile(full_label_name)

        return data_dict['loss_vit']
