#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: base_model.py
@time: 2021/12/7 22:39
'''
import os
import torch
import yaml
import json
import numpy as np
import pytorch_lightning as pl

from datetime import datetime
from pytorch_lightning.metrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from utils.metric_util import IoU
from utils.schedulers import cosine_schedule_with_warmup


class LightningBaseModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.val_iou = IoU(self.args['dataset_params'], compute_on_step=False)
        self.val_iou_lt20 = IoU(self.args['dataset_params'], compute_on_step=False)
        self.val_iou_ge20_lt30 = IoU(self.args['dataset_params'], compute_on_step=False)
        self.val_iou_ge30 = IoU(self.args['dataset_params'], compute_on_step=False)
        self.val_iou_list = [self.val_iou_lt20,self.val_iou_ge20_lt30,self.val_iou_ge30]
        self.coords_boundary = [None, 20,30,65]


        if self.args['submit_to_server']:
            self.submit_dir = os.path.dirname(self.args['checkpoint']) + '/submit_' + datetime.now().strftime(
                '%Y_%m_%d')
            with open(self.args['dataset_params']['label_mapping'], 'r') as stream:
                self.mapfile = yaml.safe_load(stream)

        self.ignore_label = self.args['dataset_params']['ignore_label']

    def configure_optimizers(self):
        if self.args['train_params']['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.args['train_params']["learning_rate"])
        elif self.args['train_params']['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.args['train_params']["learning_rate"],
                                        momentum=self.args['train_params']["momentum"],
                                        weight_decay=self.args['train_params']["weight_decay"],
                                        nesterov=self.args['train_params']["nesterov"])
        else:
            raise NotImplementedError

        if self.args['train_params']["lr_scheduler"] == 'StepLR':
            lr_scheduler = StepLR(
                optimizer,
                step_size=self.args['train_params']["decay_step"],
                gamma=self.args['train_params']["decay_rate"]
            )
        elif self.args['train_params']["lr_scheduler"] == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=self.args['train_params']["decay_rate"],
                patience=self.args['train_params']["decay_step"],
                verbose=True
            )
        elif self.args['train_params']["lr_scheduler"] == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.args['train_params']['max_num_epochs'] - 4,
                eta_min=1e-5,
            )
        elif self.args['train_params']["lr_scheduler"] == 'CosineAnnealingWarmRestarts':
            from functools import partial
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=partial(
                    cosine_schedule_with_warmup,
                    num_epochs=self.args['train_params']['max_num_epochs'],
                    batch_size=self.args['dataset_params']['train_data_loader']['batch_size'],
                    dataset_size=self.args['dataset_params']['training_size'],
                    num_gpu=len(self.args.gpu)
                ),
            )
        else:
            raise NotImplementedError

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step' if self.args['train_params']["lr_scheduler"] == 'CosineAnnealingWarmRestarts' else 'epoch',
            'frequency': 1
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': self.args.monitor,
        }

    def forward(self, data):
        pass

    def training_step(self, data_dict, batch_idx):
        data_dict = self.forward(data_dict)
        self.train_acc(data_dict['logits'].argmax(1)[data_dict['labels'] != self.ignore_label],
                       data_dict['labels'][data_dict['labels'] != self.ignore_label])
        self.log('train/acc', self.train_acc, on_epoch=True)
        self.log('train/loss_main_ce', data_dict['loss_main_ce'])
        self.log('train/loss_main_lovasz', data_dict['loss_main_lovasz'])

        return data_dict['loss']


    def validation_step(self, data_dict, batch_idx):
        indices = data_dict['indices']
        raw_labels = data_dict['raw_labels'].squeeze(1).cpu()
        origin_len = data_dict['origin_len']
        
        
        vote_logits = torch.zeros((len(raw_labels), self.num_classes))
        
        data_dict = self.forward(data_dict)
        
        if self.args['test']:
            vote_logits.index_add_(0, indices.cpu(), data_dict['logits'].cpu())
            
            if self.args['dataset_params']['pc_dataset_type'] == 'SemanticKITTI_multiscan':
                vote_logits = vote_logits[:origin_len]
                raw_labels = raw_labels[:origin_len]
                
        else:
            vote_logits = data_dict['logits'].cpu()
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

        return data_dict['loss']

    def test_step(self, data_dict, batch_idx):
        indices = data_dict['indices']
        origin_len = data_dict['origin_len']
        raw_labels = data_dict['raw_labels'].squeeze(1).cpu()
        ref_xyz = data_dict['ref_xyz'].cpu()
        path = data_dict['path'][0]
    
        data_dict = self.forward(data_dict)
        vote_logits = torch.zeros((len(raw_labels), self.num_classes))
        ref_xyz_new = torch.zeros((len(raw_labels), 3))
        ref_xyz_new.index_add_(0, indices.cpu(), ref_xyz.cpu())
        vote_logits.index_add_(0, indices.cpu(), data_dict['logits'].cpu())

        if self.args['dataset_params']['pc_dataset_type'] == 'SemanticKITTI_multiscan':
            vote_logits = vote_logits[:origin_len]
            raw_labels = raw_labels[:origin_len]
            ref_xyz_new = ref_xyz_new[:origin_len]

        prediction = vote_logits.argmax(1)
        # # save predictions
        # if self.save_N_predictions>0:
        #     prediction_to_save = prediction
        #     self.save_semantic_predictions(data_dict,prediction_to_save)
        #     self.save_N_predictions-=1

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
            # remove predictions for noise label (label 0)
            mask = raw_labels!=0
            prediction = prediction[mask]
            raw_labels = prediction[mask]

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
                original_label = valid_labels(prediction.cpu().numpy().astype(int))# valid_labels(vote_logits.argmax(1).cpu().numpy().astype(int))
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

        return data_dict['loss']

    def validation_epoch_end(self, outputs):
        iou, best_miou = self.val_iou.compute()
        mIoU = np.nanmean(iou)
        str_print = ''
        self.log('val/mIoU', mIoU, on_epoch=True)
        self.log('val/best_miou', best_miou, on_epoch=True)
        str_print += 'Validation per class iou: '

        for class_name, class_iou in zip(self.val_iou.unique_label_str, iou):
            str_print += '\n%s : %.2f%%' % (class_name, class_iou * 100)

        str_print += '\nCurrent val miou is %.3f while the best val miou is %.3f' % (mIoU * 100, best_miou * 100)
        self.print(str_print)

    def test_epoch_end(self, outputs):
        if not self.args['submit_to_server']:
            iou, best_miou = self.val_iou.compute()
            mIoU = np.nanmean(iou)
            str_print = ''
            self.log('val/mIoU', mIoU, on_epoch=True)
            self.log('val/best_miou', best_miou, on_epoch=True)
            str_print += 'Validation per class iou: '

            if self.val_iou_list is not None:
                for i in range(len(self.val_iou_list)):
                    val_iou = self.val_iou_list[i]
                    bound = self.coords_boundary[i]
                    iou_, best_miou_ = val_iou.compute()
                    mIoU_ = np.nanmean(iou_)
                    str_print = ''
                    self.log('val/mIoU %s'%(bound), mIoU_, on_epoch=True)
                    self.log('val/best_miou %s'%(bound), best_miou_, on_epoch=True)

            for class_name, class_iou in zip(self.val_iou.unique_label_str, iou):
                str_print += '\n%s : %.2f%%' % (class_name, class_iou * 100)

            str_print += '\nCurrent val miou is %.3f while the best val miou is %.3f' % (mIoU * 100, best_miou * 100)
            self.print(str_print)

    def on_after_backward(self) -> None:
        """
        Skipping updates in case of unstable gradients
        https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()


    def get_val_iou_by_distance(self,vote_logits,raw_labels,ref_xyz,xyz_indices):
        """
        Updates val iou for each threshold in self.val_iou_list
        """
        self.val_iou_list = [self.val_iou_lt20,self.val_iou_ge20_lt30,self.val_iou_ge30]
        self.coords_boundary = [20,30,65]
        for i in range(len(self.val_iou_list)):
            
            val_iou = self.val_iou_list[i]
            bound = self.coords_boundary[i]
            # get points within boundary
            mask_x = torch.logical_and(ref_xyz[:,0] > -bound,ref_xyz[:,0] < bound)
            mask_y = torch.logical_and(ref_xyz[:,1] > -bound,ref_xyz[:,1] < bound)
            mask = torch.logical_and(mask_x,mask_y)
            
            xyz_indices_ = xyz_indices[mask]
    
            # remove overlapping points
            if i>0:
                combined = torch.cat((xyz_indices_, prev_xyz_indices_))
                uniques, counts = combined.unique(return_counts=True)
                xyz_indices_ = uniques[counts == 1]
                
                prev_xyz_indices_ = torch.cat((prev_xyz_indices_,xyz_indices_))
            else:  
                prev_xyz_indices_ = xyz_indices_

            # calculate  mIOU
            vote_logits_ = vote_logits[xyz_indices_]
            raw_labels_ = raw_labels[xyz_indices_]
            prediction = vote_logits_.argmax(1)

            if self.ignore_label != 0:
                prediction = prediction[raw_labels_ != self.ignore_label]
                raw_labels_ = raw_labels_[raw_labels_ != self.ignore_label]
                prediction += 1
                raw_labels_ += 1

            val_iou(
                prediction.cpu().detach().numpy(),
                raw_labels_.cpu().detach().numpy(),
            )
        return 
    def save_semantic_predictions(self,data_dict,prediction):
        print("saving sample predictions...")
        np.array(prediction).astype(np.uint8).tofile("saved_predictions/sample_"+data_dict['token']+"_lidarseg.bin")
        return