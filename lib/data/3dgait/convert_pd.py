import os
import pickle
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
os.sys.path.append("/opt/data/private/gait/PerNGR")
from lib.utils.utils_data import split_clips

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='/opt/data/private/dataset/3DGait_dataset', type=str, help='Path to the input folder')
    parser.add_argument('--output_path', default='/opt/data/private/gait/PerNGR/data/motion3d', type=str, help='Path to the input folder')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    input_path_files = os.path.join(args.input_path, 'processed_3Dpose')
    output_path_files = os.path.join(args.output_path, 'MB3D_f81s9/3dgait')
    input_metafile = os.path.join(args.input_path, 'label.xlsx')

    if not os.path.exists(input_path_files):
        raise FileNotFoundError(f"Input folder '{input_path_files}' not found.")

    os.makedirs(output_path_files, exist_ok=True)
    os.makedirs(os.path.join(output_path_files, '0'), exist_ok=True)
    os.makedirs(os.path.join(output_path_files, '1'), exist_ok=True)
    os.makedirs(os.path.join(output_path_files, '2'), exist_ok=True)
    os.makedirs(os.path.join(output_path_files, '3'), exist_ok=True)
    joints_3d = []
    for root, dirs, files in os.walk(input_path_files):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                joint_3d = np.load(file_path)
                joints_3d.append(joint_3d)
    joints_3d_all = np.vstack(joints_3d)

    max_x, minx_x = np.max(joints_3d_all[:,:,0]), np.min(joints_3d_all[:,:,0])
    max_y, minx_y = np.max(joints_3d_all[:,:,1]), np.min(joints_3d_all[:,:,1])
    max_z, minx_z = np.max(joints_3d_all[:,:,2]), np.min(joints_3d_all[:,:,2])
    print("before normalization")
    print(max_x, minx_x)
    print(max_y, minx_y)
    print(max_z, minx_z)
    ratio = max_y - minx_y
    
    # load label
    df = pd.read_excel(input_metafile)
    df = df[['vidname', 'score', 'diag', 'patientid']]
        
    # load data and normalize y to [-1, 1], then x z are normalized with same ratio to keep body not change
    joints_3d_list = []
    clip_idx = 0
    clip_idx_0 = 0
    clip_idx_1 = 0
    clip_idx_2 = 0
    clip_idx_3 = 0
    n_frames = 81
    for root, dirs, files in os.walk(input_path_files):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                joint_3d = np.load(file_path)
                vidname = file.split('.')[0]
                subject_rows = df[df['vidname'] == vidname]
                label = subject_rows['score'].values[0]
                diag = subject_rows['diag'].values[0]
                patient_id = subject_rows['patientid'].values[0]
                joint_3d[:,:,1] = (minx_y - joint_3d[:,:,1] + ratio/2)/ratio*2
                joint_3d[:,:,0] = joint_3d[:,:,0]/ratio * 2
                joint_3d[:,:,2] = joint_3d[:,:,2]/ratio * 2
                joints_3d_list.append(joint_3d)
                
                vid_list = []
                vid_len = joint_3d.shape[0]
                for _ in range(vid_len):
                    vid_list.append(0)
                split_id = split_clips(vid_list, n_frames=n_frames, data_stride=9)
                joint_3d_clip = []
                for s_idx in range(len(split_id)):
                    joint_3d_clip.append(np.expand_dims(joint_3d[split_id[s_idx]], axis=0))
                if len(split_id) > 0:
                    joint_3d_clip = np.vstack(joint_3d_clip)
                joint_3d_clip = np.array(joint_3d_clip)
                print(f"{vidname} clip shape: {joint_3d_clip.shape}")
                for c_idx in range(joint_3d_clip.shape[0]):
                    motion = joint_3d_clip[c_idx]
                    data_dict = {
                        "id": patient_id,
                        "pose": motion,
                        "label": label,
                        "diag": diag
                    }
                    with open(os.path.join(output_path_files, "%d/%08d.pkl" % (label,clip_idx)), "wb") as myprofile:  
                        pickle.dump(data_dict, myprofile)
    
                    clip_idx += 1
                    if label == 0:
                        clip_idx_0 += 1
                    elif label == 1:
                        clip_idx_1 += 1
                    elif label == 2:
                        clip_idx_2 += 1
                    elif label == 3:
                        clip_idx_3 += 1
    print(f"There are totally {clip_idx} clips, in which 0 with {clip_idx_0} clips, 1 with {clip_idx_1} and 2 with {clip_idx_2} and 3 with {clip_idx_3}.")
    
    # analysis normalization
    joints_3d_all = np.vstack(joints_3d_list)
    max_x, minx_x = np.max(joints_3d_all[:,:,0]), np.min(joints_3d_all[:,:,0])
    max_y, minx_y = np.max(joints_3d_all[:,:,1]), np.min(joints_3d_all[:,:,1])
    max_z, minx_z = np.max(joints_3d_all[:,:,2]), np.min(joints_3d_all[:,:,2])
    print("after normalization")
    print(max_x, minx_x)
    print(max_y, minx_y)
    print(max_z, minx_z)
   
if __name__ == "__main__":
    main()



