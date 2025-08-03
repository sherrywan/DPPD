import torch
import numpy as np
import glob
import os
import io
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from lib.data.augmentation import Augmenter3D
from lib.utils.tools import read_pkl
from lib.utils.utils_data import flip_data

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

class MotionDataset(Dataset):
    def __init__(self, args, subset_list, label_list = ['0','1','2']): 
        np.random.seed(0)
        self.data_root = args.data_root
        self.subset_list = subset_list
        file_list_all = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset)
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    folder_name = os.path.basename(os.path.normpath(root))
                    if folder_name not in label_list:
                        continue
                    if file.endswith('.pkl'):
                        file_list_all.append(os.path.join(root, file))
        self.file_list = file_list_all
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError 

class MotionDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split, score=False, parts_list={"torse":[0,7,8,9,10], "larm":[11,12,13], "rarm":[14,15,16], "lleg":[4,5,6], "rleg":[1,2,3]}, num_joints=17):
        super(MotionDataset3D, self).__init__(args, subset_list, label_list=args.label_list)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.joints_index = args.joints_index
        self.data_split = data_split
        self.subset_list = subset_list
        self.aug = Augmenter3D(args)
        self.score = score
        self.num_parts = len(parts_list)
        self.parts_len = [len(parts_list[i]) for i in parts_list]
        joint2part_index = []
        part2joint_index = list(range(num_joints))
        i_start = 0
        for part_i, part_key in enumerate(parts_list):
            part_index = parts_list[part_key]
            joint2part_index += part_index
            for p in range(len(part_index)):
                part2joint_index[part_index[p]] = i_start
                i_start += 1
        self.joint2part_index = joint2part_index
        self.part2joint_index = part2joint_index
        
        self.name_to_index = {name: index for index, name in PD_Gait_Fold.items()}

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_3d = motion_file["pose"]  
        label_score = motion_file["label"]
        if 'pdgait' in file_path:
            id_participant = self.name_to_index[motion_file["id"]]
        else:
            id_participant = motion_file["id"]
        if self.data_split=="train":
            if self.aug is not None:
                motion_3d = self.aug.augment3D(motion_3d)
            else:
                raise ValueError('Training illegal.') 
        elif self.data_split=="test":                                           
            pass
        else:
            raise ValueError('Data split unknown.')    
        motion_3d = torch.FloatTensor(motion_3d)
        label_score = torch.Tensor([label_score])
        id_participant = torch.Tensor([id_participant])
        
        if self.joints_index == "part":
            motion_3d = motion_3d[:,self.joint2part_index]
        if self.score:
            return motion_3d, label_score, id_participant, file_path
        else:
            return motion_3d