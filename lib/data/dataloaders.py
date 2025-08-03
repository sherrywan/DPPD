from torch.utils.data import Dataset, DataLoader
from abc import ABC
from torchvision import transforms
import torch
import os

import pickle
import random
import copy
import pandas as pd
import numpy as np

from const import path
from lib.utils.tools import read_pkl
from lib.data.augmentation import MirrorReflection, RandomRotation, RandomNoise, axis_mask
from lib.learning.utils import compute_class_weights

_TOTAL_SCORES = 3
_MAJOR_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
#                1,   2,  3,  4,  5,  6,  7,  9, 10, 11, 13, 14, 15, 17, 18, 19, 21
_ROOT = 0
_MIN_STD = 1e-4

METADATA_MAP = {'gender': 0, 'age': 1, 'height': 2, 'weight': 3, 'bmi': 4}
    
PD_Gait_Fold = {
    1: 'SUB01',
    2: 'SUB02',
    3: 'SUB03',
    4: 'SUB04',
    5: 'SUB05',
    6: 'SUB06',
    7: 'SUB07',
    8: 'SUB08',
    9: 'SUB11',
    10: 'SUB12',
    11: 'SUB13',
    12: 'SUB14',
    13: 'SUB15',
    14: 'SUB17',
    15: 'SUB18',
    16: 'SUB19',
    17: 'SUB20',
    18: 'SUB21',
    19: 'SUB22',
    20: 'SUB23',
    21: 'SUB24',
    22: 'SUB26'
}  

PD_Gait_Fold_wo_0 = {
    1: 'SUB01',
    5: 'SUB05',
    7: 'SUB07',
    8: 'SUB08',
    9: 'SUB11',
    10: 'SUB12',
    12: 'SUB14',
    13: 'SUB15'
}  

PD_Gait_Fold_with0 = {
    15: 'SUB18',
    17: 'SUB20',
    19: 'SUB22',
    21: 'SUB24',
}  

class MotionDataset(Dataset):
    def __init__(self, data_dir, subset, fold=0): 
        np.random.seed(0)
        self.data_root = data_dir
        self.subset = subset
        
        file_list_all = []
        file_list_train = []
        file_list_train_val = []
        file_list_val = []
        file_list_test = []
        video_name = []
        video_name_exist=False
        if subset=='pdgait':
            if fold>0:
                test_sub = PD_Gait_Fold_with0[fold]
                train_val_sub = [v for k, v in PD_Gait_Fold_with0.items() if k != fold]
                val_sub = random.sample(train_val_sub, 2)           
            data_path = os.path.join(self.data_root, subset)
            if os.path.exists(os.path.join(data_path,'video_name.pkl')):
                video_name_exist = True
                with open(os.path.join(data_path,'video_name.pkl'), "rb") as f:
                    video_name = pickle.load(f)                
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.pkl'):
                        if "video_name.pkl" in file:
                            continue
                        file_path = os.path.join(root, file)
                        if not video_name_exist:
                            motion_file = read_pkl(file_path)        
                            video_name.append(motion_file['video_name'])
                        file_list_all.append(file_path)
                        if test_sub in file_path:
                            file_list_test.append(file_path)
                        elif file_path.split('/')[-3] in train_val_sub: 
                            file_list_train_val.append(file_path)
                            train_f = True
                            for val_s in val_sub:
                                if val_s in file_path:
                                    file_list_val.append(file_path)
                                    train_f = False
                            if train_f:
                                file_list_train.append(file_path)
        
        if subset=='3dgait':
            data_path = os.path.join(self.data_root, subset)
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.pkl'):
                        if "video_name.pkl" in file:
                            continue
                        file_path = os.path.join(root, file)
                        if not video_name_exist:  
                            video_name.append(file[:8])
                        file_list_all.append(file_path)
            files_len = len(file_list_all)
            file_list_test = random.sample(file_list_all, round(files_len/10))
            file_list_train_val = [v for v in file_list_all if v not in file_list_test]
        self.file_list_all = file_list_all
        self.file_list_train = file_list_train
        self.file_list_train_val = file_list_train_val
        self.file_list_val = file_list_val
        self.file_list_test = file_list_test
        self.video_names = list(set(video_name))
    
    def __len__(self):
        raise NotImplementedError  

    def __getitem__(self, index):
        raise NotImplementedError 

class PDMotionDataset3D(MotionDataset):
    def __init__(self, data_dir, params=None, mode='train', fold=0, downstream='pd', transform=None):
        super(PDMotionDataset3D, self).__init__(data_dir, params['dataset'],fold=fold)
        self._params = params
        self._mode = mode
        self.data_dir = data_dir
        self._task = downstream
        self._NMAJOR_JOINTS = len(_MAJOR_JOINTS)

        if self._task == 'pd':
            self._updrs_str = ['normal', 'slight', 'moderate']  # , 'severe']
            self._TOTAL_SCORES = _TOTAL_SCORES

        self.fold = fold
        self.transform = transform

        self._pose_dim = 3 * self._NMAJOR_JOINTS
        
        if self._mode == 'train':
            self.file_list = self.file_list_train
        elif self._mode == 'test':
            self.file_list = self.file_list_test
        elif self._mode == 'val':
            self.file_list = self.file_list_val
        elif self._mode == 'train-eval':
            self.file_list = self.file_list_train_val
        else:
            self.file_list = self.file_list_all
        self.video_name_to_index = {name: index for index, name in enumerate(self.video_names)}
            
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)      
        x = motion_file['pose']
        label = motion_file['label']
        if self._params['dataset'] == "3dgait":
            if self._params['task_type'] == "subtype":
                subtype = motion_file['diag']
                label = subtype
                if label == 3:
                    label = 1
                elif label ==4:
                    label = 2
            video_idx = self.video_name_to_index[file_path.split('/')[-1][:8]] 
        else:
            video_idx = self.video_name_to_index[motion_file['video_name']] 
        residual_feat_pose = []
        residual_feat_motion = []
        if self._params['feature'] == "extracted":
            feat_file_path = file_path.replace('.pkl', '_motioneres.npz')
        elif self._params['feature'] == "extracted_gt":
            feat_file_path = file_path.replace('.pkl', '_gt_motioneres.npz')
        if os.path.exists(feat_file_path):
            residual_feat_motion = np.load(feat_file_path)['arr_0']
        else:
            print(feat_file_path)
            
        if self._params['feature'] == "extracted":
            feat_file_path = file_path.replace('.pkl', '_structureres.npz')
        elif self._params['feature'] == "extracted_gt":
            feat_file_path = file_path.replace('.pkl', '_gt_structureres.npz')
        if os.path.exists(feat_file_path):
            residual_feat_pose = np.load(feat_file_path)['arr_0']
        else:
            print(feat_file_path)
            

        x = np.array(x, dtype=np.float32)

        if len(self._params['metadata']) > 0:
            metadata_idx = [METADATA_MAP[element] for element in self._params['metadata']]
            metadata = motion_file['metadata']
            md = metadata[metadata_idx].astype(np.float32)
        else:
            md = []

        motion_3d = torch.FloatTensor(x)
        residual_feat_pose = torch.FloatTensor(residual_feat_pose)
        residual_feat_motion = torch.FloatTensor(residual_feat_motion)
        label_score = torch.tensor(label)
        video_idx = torch.tensor(video_idx)
        metadata = torch.FloatTensor(md)
        
        return motion_3d, residual_feat_pose, residual_feat_motion, label_score, video_idx, metadata
        
        # # Select sample
        # file_path = self.file_list[index]
        # motion_file = read_pkl(file_path)      
        # x = motion_file['pose']
        # label = motion_file['label']
        # video_idx = self.video_name_to_index[motion_file['video_name']] 
        # residual_feat = []
        # feat_file_path = file_path.replace('pkl', 'npz')
        # if os.path.exists(feat_file_path):
        #     residual_feat = np.load(feat_file_path)['arr_0']

        # if self._params['data_type'] != "GastNet":
        #     joints = self._get_joint_orders()
        #     x = x[:, joints, :]

        # if self._params['in_data_dim'] == 2:
        #     if self._params['simulate_confidence_score']:
        #         # TODO: Confidence score should be a function of depth (probably)
        #         x[..., 2] = 1  # Consider 3rd dimension as confidence score and set to be 1.
        #     else:
        #         x = x[..., :2]  # Make sure it's two-dimensional
        # elif self._params['in_data_dim'] == 3:
        #     x = x[..., :3] # Make sure it's 3-dimensional
                
        # if self._params['merge_last_dim']:
        #     N = np.shape(x)[0]
        #     x = x.reshape(N, -1)   # N x 17 x 3 -> N x 51

        # x = np.array(x, dtype=np.float32)

        # if x.shape[0] > self._params['source_seq_len']:
        #     # If we're reading a preprocessed pickle file that has more frames
        #     # than the expected frame length, we throw away the last few ones.
        #     x = x[:self._params['source_seq_len']]
        # elif x.shape[0] < self._params['source_seq_len']:
        #     raise ValueError("Number of frames in tensor x is shorter than expected one.")
        
        # if len(self._params['metadata']) > 0:
        #     metadata_idx = [METADATA_MAP[element] for element in self._params['metadata']]
        #     metadata = motion_file['metadata']
        #     md = metadata[metadata_idx].astype(np.float32)
        # else:
        #     md = []

        # sample = {
        #     'encoder_inputs': x,
        #     'residual_feat': residual_feat,
        #     'label': label,
        #     'labels_str': self._updrs_str[label],
        #     'video_idx': video_idx,
        #     'metadata': md,
        # }
        # if self.transform:
        #     sample = self.transform(sample)
        
        # return sample
    
    def _get_joint_orders(self):
        joints = _MAJOR_JOINTS
        return joints


def collate_fn(batch):
    """Collate function for data loaders."""
    e_inp = torch.from_numpy(np.stack([e['encoder_inputs'] for e in batch]))
    e_residual = torch.from_numpy(np.stack([e['residual_feat'] for e in batch]))
    labels = torch.from_numpy(np.stack([e['label'] for e in batch]))
    video_idxs = torch.from_numpy(np.stack([e['video_idx'] for e in batch]))
    metadata = torch.from_numpy(np.stack([e['metadata'] for e in batch]))
    return e_inp, e_residual, labels, video_idxs, metadata


def dataset_factory(params, backbone, fold):
    """Defines the datasets that will be used for training and validation."""
    params['n_joints'] = len(_MAJOR_JOINTS)

    data_dir = params['data_path']

    use_validation = params['use_validation']

    train_transform = transforms.Compose([
        PreserveKeysTransform(transforms.RandomApply([MirrorReflection(data_dim=params['in_data_dim'])], p=params['mirror_prob'])),
        PreserveKeysTransform(transforms.RandomApply([RandomRotation(*params['rotation_range'], data_dim=params['in_data_dim'])], p=params['rotation_prob'])),
        PreserveKeysTransform(transforms.RandomApply([RandomNoise(data_dim=params['in_data_dim'])], p=params['noise_prob'])),
        PreserveKeysTransform(transforms.RandomApply([axis_mask(data_dim=params['in_data_dim'])], p=params['axis_mask_prob']))
    ])

    train_dataset = PDMotionDataset3D(data_dir, fold=fold, params=params,
                                            mode='train' if use_validation else 'train-eval', transform=train_transform)
    eval_dataset = PDMotionDataset3D(data_dir, fold=fold, params=params, mode='val') if use_validation else None
    test_dataset = PDMotionDataset3D(data_dir, fold=fold, params=params, mode='test')
    
    train_dataset_fn = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )
    eval_dataset_fn = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    ) if use_validation else None

    test_dataset_fn = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # class_weights = compute_class_weights(train_dataset_fn)
    if params['dataset'] == 'pdgait':
        class_weights = [0.5,0.3,0.2]
    elif params['dataset'] == '3dgait':
        if params['task_type'] == 'pd_score':
            class_weights = [0.26, 0.28, 0.29, 0.17]
        elif params['task_type'] == 'subtype':
            class_weights = [0.33, 0.33, 0.34]

    return train_dataset_fn, test_dataset_fn, eval_dataset_fn, class_weights

class PreserveKeysTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        transformed_sample = self.transform(sample)

        # Ensure all original keys are preserved
        for key in sample.keys():
            if key not in transformed_sample:
                transformed_sample[key] = sample[key]

        return transformed_sample
