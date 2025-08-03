import os
import sys
import copy
import pickle
import ipdb
import torch
import numpy as np
import argparse
from tqdm import tqdm
from public_pd_datareader import PDReader
os.sys.path.append("/opt/data/private/gait/PerNGR")
from lib.utils.utils_data import split_clips


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='/opt/data/private/dataset/ClinicalPD/', type=str, help='Path to the input folder')
    parser.add_argument('--output_path', default='/opt/data/private/gait/PerNGR/data/motion3d', type=str, help='Path to the input folder')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    input_path_c3dfiles = os.path.join(args.input_path, 'C3Dfiles_processed_new')
    output_path_c3dfiles = os.path.join(args.output_path, 'MB3D_f81s9/pdgait')
    input_metafile = "/opt/data/private/dataset/ClinicalPD/PDGinfo.xlsx"

    if not os.path.exists(input_path_c3dfiles):
        raise FileNotFoundError(f"Input folder '{input_path_c3dfiles}' not found.")

    os.makedirs(output_path_c3dfiles, exist_ok=True)
    
    PD_reader = PDReader(input_path_c3dfiles, input_metafile)
    pose_dict, labels_dict, video_names_list, participant_ID, metadata_dict = PD_reader.pose_dict, PD_reader.labels_dict, PD_reader.video_names, PD_reader.participant_ID, PD_reader.metadata_dict
    
    joints_3d = []
    # normaliza 3d poses
    for video_name in video_names_list:
        joints_3d.append(pose_dict[video_name])
    joints_3d_all = np.vstack(joints_3d)
    print(joints_3d_all.shape)

    max_x, minx_x = np.max(joints_3d_all[:,:,0]), np.min(joints_3d_all[:,:,0])
    max_y, minx_y = np.max(joints_3d_all[:,:,1]), np.min(joints_3d_all[:,:,1])
    max_z, minx_z = np.max(joints_3d_all[:,:,2]), np.min(joints_3d_all[:,:,2])
    print("before normalization")
    print(max_x, minx_x)
    print(max_y, minx_y)
    print(max_z, minx_z)
    ratio = max_y - minx_y

    # normalize y to [-1, 1], then x z are normalized with same ratio to keep body not change
    for video_name in video_names_list:
        joints_3d = pose_dict[video_name]
        joints_3d[:,:,1] = (joints_3d[:,:,1] - minx_y - ratio/2)/ratio*2
        joints_3d[:,:,0] = joints_3d[:,:,0]/ratio * 2
        joints_3d[:,:,2] = joints_3d[:,:,2]/ratio * 2
        pose_dict[video_name] = joints_3d
        
    joints_3d = []
    # normaliza 3d poses
    for video_name in video_names_list:
        joints_3d.append(pose_dict[video_name])
    joints_3d_all = np.vstack(joints_3d)
    print(joints_3d_all.shape)

    max_x, minx_x = np.max(joints_3d_all[:,:,0]), np.min(joints_3d_all[:,:,0])
    max_y, minx_y = np.max(joints_3d_all[:,:,1]), np.min(joints_3d_all[:,:,1])
    max_z, minx_z = np.max(joints_3d_all[:,:,2]), np.min(joints_3d_all[:,:,2])
    print("after normalization")
    print(max_x, minx_x)
    print(max_y, minx_y)
    print(max_z, minx_z)
    
    # save data as same as the structure of motionbert
    clip_idx = 0
    clip_idx_0 = 0
    clip_idx_1 = 0
    clip_idx_2 = 0
    n_frames = 81
    for v_idx, video_name in enumerate(video_names_list):
        joints_3d = pose_dict[video_name]
        label_v = labels_dict[video_name]
        metadata_v = metadata_dict[video_name]
        id_v = participant_ID[v_idx]
        output_path_v = output_path_c3dfiles + f'/{id_v}' + f'/{label_v}'
        os.makedirs(output_path_v, exist_ok=True)
        vid_list = []
        vid_len = joints_3d.shape[0]
        for _ in range(vid_len):
            vid_list.append(0)
        split_id = split_clips(vid_list, n_frames=n_frames, data_stride=9)
        joints_3d_clip = []
        for s_idx in range(len(split_id)):
            joints_3d_clip.append(np.expand_dims(joints_3d[split_id[s_idx]], axis=0))
        if len(split_id) > 0:
            joints_3d_clip = np.vstack(joints_3d_clip)
        joints_3d_clip = np.array(joints_3d_clip)
        print(f"{video_name} clip shape: {joints_3d_clip.shape}")
        for c_idx in range(joints_3d_clip.shape[0]):
            motion = joints_3d_clip[c_idx]
            data_dict = {
                "id": id_v,
                "video_name": video_name,
                "pose": motion,
                "label": label_v,
                "metadata": metadata_v
            }
            with open(os.path.join(output_path_v, "%08d.pkl" % clip_idx), "wb") as myprofile:  
                pickle.dump(data_dict, myprofile)
                print("sucessfully save to {}".format(os.path.join(output_path_v, "%08d.pkl" % clip_idx)))
            clip_idx += 1
            if label_v == 0:
                clip_idx_0 += 1
            elif label_v == 1:
                clip_idx_1 += 1
            elif label_v == 2:
                clip_idx_2 += 1
    print(f"There are totally {clip_idx} clips, in which 0 with {clip_idx_0} clips, 1 with {clip_idx_1} and 2  with {clip_idx_2}.")
        
if __name__ == "__main__":
    main()



