import argparse
import os
import numpy as np
import joblib

from const_pd import H36M_FULL, PD


def convert_pd_h36m(sequence):
    new_keyponts = np.zeros((sequence.shape[0], 17, 3))
    new_keyponts[..., H36M_FULL['B.TORSO'], :] = sequence[..., PD['hip'], :]
    new_keyponts[..., H36M_FULL['L.HIP'], :] = sequence[..., PD['lhip (SMPL)'], :] 
    new_keyponts[..., H36M_FULL['L.KNEE'], :] = sequence[..., PD['lknee'], :]
    new_keyponts[..., H36M_FULL['L.FOOT'], :] = sequence[..., PD['lankle'], :]
    new_keyponts[..., H36M_FULL['R.HIP'], :] = sequence[..., PD['rhip (SMPL)'], :] 
    new_keyponts[..., H36M_FULL['R.KNEE'], :] = sequence[..., PD['rknee'], :]
    new_keyponts[..., H36M_FULL['R.FOOT'], :] = sequence[..., PD['rankle'], :]
    new_keyponts[..., H36M_FULL['U.TORSO'], :] = sequence[..., PD['thorax'], :]
    new_keyponts[..., H36M_FULL['C.TORSO'], :] = sequence[..., PD['Spine (H36M)'], :]
    new_keyponts[..., H36M_FULL['R.SHOULDER'], :] = sequence[..., PD['rshoulder'], :]
    new_keyponts[..., H36M_FULL['R.ELBOW'], :] = sequence[..., PD['relbow'], :] 
    new_keyponts[..., H36M_FULL['R.HAND'], :] = sequence[..., PD['rwrist'], :] 
    new_keyponts[..., H36M_FULL['L.SHOULDER'], :] = sequence[..., PD['lshoulder'], :]
    new_keyponts[..., H36M_FULL['L.ELBOW'], :] = sequence[..., PD['lelbow'], :] 
    new_keyponts[..., H36M_FULL['L.HAND'], :] = sequence[..., PD['lwrist'], :] 
    new_keyponts[..., H36M_FULL['NECK'], :] = sequence[..., PD['neck'], :]
    new_keyponts[..., H36M_FULL['HEAD'], :] = sequence[..., PD['Head (H36M)'], :]
    
    return new_keyponts

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='/opt/data/private/dataset/3DGait_dataset', type=str, help='Path to the input folder')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    input_path_files = os.path.join(args.input_path, 'MAX-GR_90.json')
    output_path_files = os.path.join(args.input_path, 'processed_3Dpose')

    if not os.path.exists(input_path_files):
        raise FileNotFoundError(f"Input folder '{input_path_files}' not found.")

    os.makedirs(output_path_files, exist_ok=True)
    
    data = joblib.load(input_path_files)
    # 遍历每个视频数据
    for vid_name, content in data.items():
        joints3D_data = np.array(content['joints3D'])  # 确保是 numpy array

        # 转换为 h36m 格式
        joints_h36m = convert_pd_h36m(joints3D_data)

        # 保存为 npy 文件
        np.save(os.path.join(output_path_files, f"{vid_name}.npy"), joints_h36m)

        print("transform and save to {}".format(os.path.join(output_path_files, f"{vid_name}.npy")))

if __name__ == "__main__":
    main()