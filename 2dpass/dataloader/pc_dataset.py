import os
import yaml
import numpy as np
import torch

from PIL import Image
from torch.utils import data
from pathlib import Path
from nuscenes.utils import splits

REGISTERED_PC_DATASET_CLASSES = {}

from mmcv import load as mmcv_load


def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]


def absoluteFilePaths(directory, num_vote):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            for _ in range(num_vote):
                yield os.path.abspath(os.path.join(dirpath, f))


@register_dataset
class SemanticKITTI(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)

        self.config = config
        self.num_vote = num_vote
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset

        if imageset == 'train':
            split = semkittiyaml['split']['train']
            if config['train_params'].get('trainval', False):
                split += semkittiyaml['split']['valid']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.proj_matrix = {}

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), num_vote)
            calib_path = os.path.join(data_path, str(i_folder).zfill(2), "calib.txt")
            calib = self.read_calib(calib_path)
            proj_matrix = np.matmul(calib["P2"], calib["Tr"])
            self.proj_matrix[i_folder] = proj_matrix

        seg_num_per_class = config['dataset_params']['seg_labelweights']
        seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
        self.seg_labelweights = np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        origin_len = len(raw_data)
        points = raw_data[:, :3]

        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            instance_label = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            instance_label = annotated_data >> 16
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

            if self.config['dataset_params']['ignore_label'] != 0:
                annotated_data -= 1
                annotated_data[annotated_data == -1] = self.config['dataset_params']['ignore_label']

        image_file = self.im_idx[index].replace('velodyne', 'image_2').replace('.bin', '.png')
        image = Image.open(image_file)
        proj_matrix = self.proj_matrix[int(self.im_idx[index][-22:-20])]

        data_dict = {}
        data_dict['xyz'] = points
        data_dict['labels'] = annotated_data.astype(np.uint8)
        data_dict['instance_label'] = instance_label
        data_dict['signal'] = raw_data[:, 3:4]
        data_dict['origin_len'] = origin_len
        data_dict['img'] = image
        data_dict['proj_matrix'] = proj_matrix

        return data_dict, self.im_idx[index]


@register_dataset
class nuScenes(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        if config.debug:
            version = 'v1.0-mini'
            scenes = splits.mini_train
        else:
            if imageset != 'test':
                version = 'v1.0-trainval'
                if imageset == 'train':
                    scenes = splits.train
                else:
                    scenes = splits.val
            else:
                version = 'v1.0-test'
                scenes = splits.test

        self.split = imageset
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']
        
        self.num_vote = num_vote
        self.data_path = data_path
        self.imageset = imageset
        self.img_view = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
                         'CAM_FRONT_LEFT']
        self.training_detections_fuser = config.training_detections_fuser
        self.saved_detections_roots = ["results","results_1"]
        self.split_across_channels = config.split_across_channels
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
        

        self.get_available_scenes()
        available_scene_names = [s['name'] for s in self.available_scenes]
        scenes = list(filter(lambda x: x in available_scene_names, scenes))
        scenes = set([self.available_scenes[available_scene_names.index(s)]['token'] for s in scenes])
        self.get_path_infos_cam_lidar(scenes)

        print('Total %d scenes in the %s split' % (len(self.token_list), imageset))
        
        # Whether to only use samples with saved feature maps from object detections
        self.skip_samples = not config.test and config.model_params.model_architecture == "_2dpass_fuser"
        self.model_name = config.model_params.model_architecture
        
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.token_list)

    def loadDataByIndex(self, index):
        lidar_sample_token = self.token_list[index]['lidar_token']
        lidar_path = os.path.join(self.data_path,
                                  self.nusc.get('sample_data', lidar_sample_token)['filename'])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))

        if self.split == 'test':
            self.lidarseg_path = None
            annotated_data = np.expand_dims(
                np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            lidarseg_path = os.path.join(self.data_path,
                                         self.nusc.get('lidarseg', lidar_sample_token)['filename'])
            annotated_data = np.fromfile(
                lidarseg_path, dtype=np.uint8).reshape((-1, 1))  # label

        pointcloud = raw_data[:, :4]
        sem_label = annotated_data
        inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        return pointcloud, sem_label, inst_label, lidar_sample_token

    def labelMapping(self, sem_label):
        sem_label = np.vectorize(self.map_name_from_general_index_to_segmentation_index.__getitem__)(
            sem_label)  # n, 1
        assert sem_label.shape[-1] == 1
        sem_label = sem_label[:, 0]
        return sem_label

    def loadImage(self, index, image_id):
        cam_sample_token = self.token_list[index]['cam_token'][image_id]
        cam = self.nusc.get('sample_data', cam_sample_token)
        image = Image.open(os.path.join(self.nusc.dataroot, cam['filename']))
        return image, cam_sample_token

    def get_available_scenes(self):
        # only for check if all the files are available
        self.available_scenes = []
        for scene in self.nusc.scene:
            scene_token = scene['token']
            scene_rec = self.nusc.get('scene', scene_token)
            sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, _, _ = self.nusc.get_sample_data(sd_rec['token'])
                if not Path(lidar_path).exists():
                    scene_not_exist = True
                    break
                else:
                    break

            if scene_not_exist:
                continue
            self.available_scenes.append(scene)

    def get_path_infos_cam_lidar(self, scenes):
        self.token_list = []
        # get lidar tokens
        for sample in self.nusc.sample:
            scene_token = sample['scene_token']
            lidar_token = sample['data']['LIDAR_TOP']  # 360 lidar 
            sample_token = sample['token']

            # skip samples without detections when training fuser to learn feature maps from detections
            if self.training_detections_fuser:
                detections_exist = self.check_exists_detections_file(sample_token)
                if not detections_exist:
                    continue
            if scene_token in scenes:
                # get image tokens
                for _ in range(self.num_vote):
                    cam_token = []
                    for i in self.img_view:
                        cam_token.append(sample['data'][i])

                    self.token_list.append(
                        {'lidar_token': lidar_token,
                         'sample_token':sample_token,
                         'cam_token': cam_token}
                    )
        

    def check_exists_detections_file(self,sample_token):
        '''
        Checks if files exists containing list of detected objects for the sample.
        Returns True/False
        '''
        found_file = False
        for data_dir in self.saved_detections_roots:
            file_1 = os.path.join(data_dir,self.imageset,"data","sample_"+sample_token)
            file_2 = os.path.join(data_dir,self.imageset,"detections","sample_"+sample_token)
            file_exists = os.path.isfile(file_1) and os.path.isfile(file_2)
            if file_exists:
                found_file = True
                break
        return found_file
    def get_detected_objects(self,index,split_across_channels=True):
        '''
        Gets list of detected objects for the sample.
        Returns dict of the extracted feature maps from the object detection network and coordinates
        Args:
            split_across_channels: whether to split feautre across channels or keep it as it is
        '''
        sample_token = self.token_list[index]['sample_token']
        found_file = False
        for data_dir in self.saved_detections_roots:
            file_1 = os.path.join(data_dir,self.imageset,"data","sample_"+sample_token)
            file_2 = os.path.join(data_dir,self.imageset,"detections","sample_"+sample_token)
            file_exists = os.path.isfile(file_1) and os.path.isfile(file_2)
            if file_exists:
                found_file = True
                break
        if not found_file:
            return {}
        
        else:
            try:
                
                with open(file_1,'rb') as f:
                    data = np.load(f)
                    sample_data = data['sample_data']
                with open(file_2,'rb') as f:
                    data = np.load(f)
                    metas = data["meta"]
                    feature_map = torch.tensor(data["feature_map"])
                    scale_factor = torch.tensor((metas[3],metas[4])) # gets scale factor for y and x
                    x_width = int(metas[-1])
                    y_width = int(metas[-2])

                    if split_across_channels:
                        detections_feature_maps = torch.zeros((sample_data.shape[0],int(feature_map.shape[0]/4),4*y_width,4*x_width))
                    else:
                        detections_feature_maps = torch.zeros((sample_data.shape[0],feature_map.shape[0],2*y_width,2*x_width))
                    coords = torch.zeros((sample_data.shape[0],4)) # -y,x,y,x
                    point_cloud_range = torch.tensor((metas[6],metas[5])) # gets min x and y values of point cloud
                    for idx in range(sample_data.shape[0]):
                        x,y,_,_,_,_,_ = torch.tensor(sample_data[idx])
                        BOX_DIM = 3
                        coord = torch.tensor((y-BOX_DIM,x-BOX_DIM,y+BOX_DIM,x+BOX_DIM))
                        # use this
                        transformed_y = int(np.round((y-point_cloud_range[1])*scale_factor[0])) # transform coordinates to index. 0,0 starts from top left. y increases downwards, and x increases rightwards.
                        transformed_x = int(np.round((x-point_cloud_range[0])*scale_factor[1]))

                        fm_tmp = feature_map[:,max(0,transformed_y-y_width):min(transformed_y+y_width,feature_map.shape[1]),max(transformed_x-x_width,0):min(transformed_x+x_width,feature_map.shape[2])]
                        # pad zeros if coordinates are out of boundaries
                        if (transformed_y-y_width<0):
                            # pad zeros to the top
                            fm_tmp = torch.cat((
                                    torch.zeros((fm_tmp.shape[0],abs(transformed_y-y_width),fm_tmp.shape[2])),
                                    fm_tmp,
                                ),axis = 1)
                        elif (transformed_y+y_width-1>feature_map.shape[1]-1):
                            # pad zeros to bottom
                            fm_tmp = torch.cat((
                                    fm_tmp,
                                    torch.zeros((fm_tmp.shape[0],transformed_y+y_width-feature_map.shape[1],fm_tmp.shape[2])),
                                ),axis = 1)

                        if (transformed_x-x_width<0):
                            # pad zeros to the left
                            fm_tmp = torch.cat((
                                    torch.zeros((fm_tmp.shape[0],fm_tmp.shape[1],abs(transformed_x-x_width))),
                                    fm_tmp,
                                ),axis = 2)
                        elif (transformed_x+x_width-1>feature_map.shape[2]-1):
                            # pad zeros to the right
                            fm_tmp = torch.cat((
                                    fm_tmp,
                                    torch.zeros((fm_tmp.shape[0],fm_tmp.shape[1],transformed_x+x_width-feature_map.shape[2])),
                                ),axis = 2)
                        if split_across_channels:
                            fm_1=torch.cat((fm_tmp[:128,:,:],fm_tmp[128:128*2,:,:]),axis=1)
                            fm_2=torch.cat((fm_tmp[128*2:128*3,:,:],fm_tmp[128*3:128*4,:,:]),axis=1)
                            fm_tmp=torch.cat((fm_1,fm_2),axis=2)

                        detections_feature_maps[idx,:]=fm_tmp
                        coords[idx,:]=coord
                    detections_data = {
                        'feature_maps':detections_feature_maps, # shape is N*128*12*12, where N is num detections
                        "xy_range":coords # shape is N*4 [y_min,x_min,y_max,x_max]
                    }
                    return detections_data
            except Exception as e:
                print("failed to open or load file for sample %s. Skipping..."%(sample_token))
                print(e)
                return {}


    def __getitem__(self, index):
        if self.model_name == "_2dpass_fuser":
            detected_obj_data = self.get_detected_objects(index,self.split_across_channels)
        else:
            detected_obj_data = {}
        
        pointcloud, sem_label, instance_label, lidar_sample_token = self.loadDataByIndex(index)
        sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)
        


        # get image feature
        image_id = np.random.randint(6)
        image, cam_sample_token = self.loadImage(index, image_id)

        cam_path, boxes_front_cam, cam_intrinsic = self.nusc.get_sample_data(cam_sample_token)
        pointsensor = self.nusc.get('sample_data', lidar_sample_token)
        cs_record_lidar = self.nusc.get('calibrated_sensor',
                                        pointsensor['calibrated_sensor_token'])
        pose_record_lidar = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        cam = self.nusc.get('sample_data', cam_sample_token)
        cs_record_cam = self.nusc.get('calibrated_sensor',
                                      cam['calibrated_sensor_token'])
        pose_record_cam = self.nusc.get('ego_pose', cam['ego_pose_token'])

        calib_infos = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam['translation'],
            "ego2global_rotation_cam": pose_record_cam['rotation'],
            "cam2ego_translation": cs_record_cam['translation'],
            "cam2ego_rotation": cs_record_cam['rotation'],
            "cam_intrinsic": cam_intrinsic,
        }

        data_dict = {}
        data_dict['xyz'] = pointcloud[:, :3]
        data_dict['img'] = image
        data_dict['calib_infos'] = calib_infos
        data_dict['labels'] = sem_label.astype(np.uint8)
        data_dict['signal'] = pointcloud[:, 3:4]
        data_dict['origin_len'] = len(pointcloud)
        data_dict['detected_obj_data'] = detected_obj_data
        data_dict['num_vote'] = self.num_vote
        data_dict['token'] = self.token_list[index]['sample_token']

        return data_dict, lidar_sample_token


def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name
