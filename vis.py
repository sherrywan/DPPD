import os
import numpy as np
import argparse
import errno
import math
import pickle
import tensorboardX
from tqdm import tqdm
from time import time
from datetime import datetime
import copy
import random
import prettytable

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.data.augmentation import Augmenter_input
from lib.data.datareader_h36m import DataReaderH36M  
from lib.model.loss import *
from collections import OrderedDict

from lib.model.diffusionpose import *
from lib.utils.vismo import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion/PM_finetune.yaml", help="Path to the config file.")
    parser.add_argument('-Mc', '--motionbertcheckpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-mp', '--mpjpe_loss', default=True, type=bool, help='loss')
    parser.add_argument('-cl', '--contrastive_loss', default=False, type=bool, help='loss')
    parser.add_argument('-fe', '--feature_loss', default=False, type=bool, help='loss')
    parser.add_argument('-ffl', '--first_frame_loss', default=False, type=bool, help='loss')
    # General arguments
    parser.add_argument('-d', '--dataset', default='pdgait', type=str, metavar='NAME', help='target dataset') # pdgait or 3d gait
    parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-str', '--subjects-train', default='S1,S5,S6,S7,S8', type=str, metavar='LIST',
                        help='training subjects separated by comma')
    parser.add_argument('-ste', '--subjects-test', default='S9,S11', type=str, metavar='LIST', help='test subjects separated by comma')
    parser.add_argument('-sun', '--subjects-unlabeled', default='', type=str, metavar='LIST',
                        help='unlabeled subjects separated by comma for self-supervision')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('-Dc', '--D3DPcheckpoint', default='', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('-l', '--log', default='log/default', type=str, metavar='PATH',
                        help='log file directory')
    parser.add_argument('-cf','--checkpoint-frequency', default=20, type=int, metavar='N',
                        help='create a checkpoint every N epochs')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('--nolog', action='store_true', help='forbiden log function')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--render', action='store_true', help='visualize a particular video')
    parser.add_argument('--by-subject', action='store_true', help='break down error by subject (on evaluation)')
    parser.add_argument('--export-training-curves', action='store_true', help='save training curves as .png images')


    # Model arguments
    parser.add_argument('-s', '--stride', default=9, type=int, metavar='N', help='chunk size to use during training')
    parser.add_argument('-e', '--epochs', default=1000, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=1024, type=int, metavar='N', help='batch size in terms of predicted frames')
    parser.add_argument('-drop', '--dropout', default=0., type=float, metavar='P', help='dropout probability')
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=0.993, type=float, metavar='LR', help='learning rate decay per epoch')
    parser.add_argument('--coverlr', action='store_true', help='cover learning rate with assigned during resuming previous model')
    parser.add_argument('-mloss', '--min_loss', default=100000, type=float, help='assign min loss(best loss) during resuming previous model')
    parser.add_argument('-no-da', '--no-data-augmentation', dest='data_augmentation', action='store_false',
                        help='disable train-time flipping')
    parser.add_argument('-cs', default=512, type=int, help='channel size of model, only for trasformer') 
    parser.add_argument('-dep', default=8, type=int, help='depth of model')    
    parser.add_argument('-alpha', default=0.01, type=float, help='used for wf_mpjpe')
    parser.add_argument('-beta', default=2, type=float, help='used for wf_mpjpe')
    parser.add_argument('--postrf', action='store_true', help='use the post refine module')
    parser.add_argument('--ftpostrf', action='store_true', help='For fintune to post refine module')
    # parser.add_argument('-no-tta', '--no-test-time-augmentation', dest='test_time_augmentation', action='store_false',
    #                     help='disable test-time flipping')
    # parser.add_argument('-arc', '--architecture', default='3,3,3', type=str, metavar='LAYERS', help='filter widths separated by comma')
    parser.add_argument('-f', '--number-of-frames', default='81', type=int, metavar='N',
                        help='how many frames used as input')
    # parser.add_argument('--causal', action='store_true', help='use causal convolutions for real-time processing')
    # parser.add_argument('-ch', '--channels', default=1024, type=int, metavar='N', help='number of channels in convolution layers')

    # Experimental
    parser.add_argument('-gpu', default='0', type=str, help='assign the gpu(s) to use')
    parser.add_argument('--subset', default=1, type=float, metavar='FRACTION', help='reduce dataset size by fraction')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor (semi-supervised)')
    parser.add_argument('--warmup', default=1, type=int, metavar='N', help='warm-up epochs for semi-supervision')
    parser.add_argument('--no-eval', action='store_true', help='disable epoch evaluation while training (small speed-up)')
    parser.add_argument('--dense', action='store_true', help='use dense convolutions instead of dilated convolutions')
    parser.add_argument('--disable-optimizations', action='store_true', help='disable optimized model for single-frame predictions')
    parser.add_argument('--linear-projection', action='store_true', help='use only linear coefficients for semi-supervised projection')
    parser.add_argument('--no-bone-length', action='store_false', dest='bone_length_term',
                        help='disable bone length term in semi-supervised settings')
    parser.add_argument('--no-proj', action='store_true', help='disable projection for semi-supervised setting')
    parser.add_argument('--ft', action='store_true', help='use ft 2d(only for detection keypoints!)')
    parser.add_argument('--ftpath', default='checkpoint/exp13_ft2d', type=str, help='assign path of ft2d model chk path')
    parser.add_argument('--ftchk', default='epoch_330.pth', type=str, help='assign ft2d model checkpoint file name')
    parser.add_argument('--no_eval', action='store_true', default=False, help='no_eval')
    
    # Visualization
    parser.add_argument('--viz-subject', type=str, metavar='STR', help='subject to render')
    parser.add_argument('--viz-action', type=str, metavar='STR', help='action to render')
    parser.add_argument('--viz-camera', type=int, default=0, metavar='N', help='camera to render')
    parser.add_argument('--viz-video', type=str, metavar='PATH', help='path to input video')
    parser.add_argument('--viz-skip', type=int, default=0, metavar='N', help='skip first N frames of input video')
    parser.add_argument('--viz-output', type=str, metavar='PATH', help='output file name (.gif or .mp4)')
    parser.add_argument('--viz-export', type=str, metavar='PATH', help='output file name for coordinates')
    parser.add_argument('--viz-bitrate', type=int, default=3000, metavar='N', help='bitrate for mp4 videos')
    parser.add_argument('--viz-no-ground-truth', action='store_true', help='do not show ground-truth poses')
    parser.add_argument('--viz-limit', type=int, default=-1, metavar='N', help='only render first N frames')
    parser.add_argument('--viz-downsample', type=int, default=1, metavar='N', help='downsample FPS by a factor N')
    parser.add_argument('--viz-size', type=int, default=5, metavar='N', help='image size')
    parser.add_argument('--compare', action='store_true', default=False, help='Whether to compare with other methods e.g. Poseformer')
    # parser.add_argument('-comchk', type=str, default='/mnt/data3/home/zjl/workspace/3dpose/PoseFormer/checkpoint/detected81f.bin', help='checkpoint of comparison methods')

    # ft2d.py
    parser.add_argument('-lcs', '--linear_channel_size', type=int, default=1024, metavar='N', help='channel size of the LinearModel')
    parser.add_argument('-depth', type=int, default=4, metavar='N', help='nums of blocks of the LinearModel')
    parser.add_argument('-ldg', '--lr_decay_gap', type=float, default=10000, metavar='N', help='channel size of the LinearModel')

    parser.add_argument('-scale', default=1.0, type=float, help='the scale of SNR')
    parser.add_argument('-timestep', type=int, default=1000, metavar='N', help='timestep')
    parser.add_argument('-timestep_eval', type=int, default=1000, metavar='N', help='timestep_eval')
    parser.add_argument('-sampling_timesteps', type=int, default=5, metavar='N', help='sampling_timesteps')
    parser.add_argument('-num_proposals', type=int, default=300, metavar='N')
    parser.add_argument('--debug', action='store_true', default=False, help='debugging mode')
    parser.add_argument('--p2', action='store_true', default=False, help='using protocol #2, i.e., P-MPJPE')


    parser.set_defaults(bone_length_term=True)
    parser.set_defaults(data_augmentation=True)
    parser.set_defaults(test_time_augmentation=False)
    # parser.set_defaults(test_time_augmentation=False)

    args = parser.parse_args()
    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()
        
    if args.export_training_curves and args.no_eval:
        print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
        exit()
        
    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss' : min_loss
    }, chk_path)

def dtw_pytorch(tensor_a, tensor_b, dist_func=None):
    """
    tensor_a: [T1, J, C]
    tensor_b: [T2, J, C]
    dist_func: function to compute distance (default: L2)

    Returns:
        cost: [T1+1, T2+1] matrix with accumulated cost
        path: list of (i, j) pairs
    """
    T1, J, _ = tensor_a.shape
    T2, _, _ = tensor_b.shape
    device = tensor_a.device

    if dist_func is None:
        dist_func = lambda x, y: torch.norm(x - y, dim=-1).mean(dim=-1)

    cost = torch.zeros((T1 + 1, T2 + 1), device=device) + float('inf')
    cost[0, 0] = 0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            d = dist_func(tensor_a[i - 1], tensor_b[j - 1])
            cost[i, j] = d + torch.min(torch.stack([
                cost[i - 1, j],      # insertion
                cost[i, j - 1],      # deletion
                cost[i - 1, j - 1],  # match
            ]))

    # 回溯路径
    i, j = T1, T2
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        min_val, min_idx = torch.min(torch.stack([
            cost[i - 1, j],      # 上
            cost[i, j - 1],      # 左
            cost[i - 1, j - 1]   # 左上
        ]), dim=0)

        if min_idx.item() == 0:
            i -= 1
        elif min_idx.item() == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    path.reverse()
    return cost[1:, 1:], path

def extract_between_minmax_value(my_dict):
    if not my_dict:
        return {}

    # 先找最小值和最大值
    min_value = min(my_dict.values())
    max_value = max(my_dict.values())

    # 找最小值对应的最大 key
    min_val_max_key = max(k for k, v in my_dict.items() if v == min_value)

    # 找最大值对应的最小 key
    max_val_min_key = min(k for k, v in my_dict.items() if v == max_value)

    # 排序后取中间段
    start_key = min_val_max_key
    end_key = max_val_min_key

    result_dict = {k: my_dict[k] for k in my_dict if start_key <= k <= end_key}

    return min_val_max_key, max_val_min_key, result_dict
    
def p_align(predicteds, targets):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicteds.shape == targets.shape
    predicted = predicteds[0:1]
    target = targets[0:1]
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    # Handle reflection case (if det(R) < 0)
    detR = np.linalg.det(R)
    reflect = detR < 0

    # 修正反射：将 V 的最后一列取反
    V[reflect, :, -1] *= -1
    R[reflect] = np.matmul(V[reflect], U[reflect].transpose(0, 2, 1))
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    # Perform rigid transformation on the input
    predicteds_aligned = a*np.matmul(predicteds, R) + t
    # Return MPJPE
    return predicteds_aligned

     
def evaluate(args, model_pos, test_loader, vis=False, vis_freq=100, writer=None, restore=False, align=False, render=False):
    print('INFO: Testing')
    epoch_loss_3d_valid = 0 
    N = 0
    model_pos.eval()        
    with torch.no_grad():
        for idx, (batch_gt, label_score, id_participant, file_paths) in tqdm(enumerate(test_loader)):    
            batch_input = batch_gt.clone()
            batch_size = len(batch_input)   
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
                batch_gt = batch_gt.cuda()
            batch_input = batch_input - batch_input[:,:,0:1,:]
            # Predict 3D poses
            original_feat, predicted_feat, predicted_3d_pos = model_pos(batch_input)    # (N, T, 17, 3)
            error = mpjpe_diffusion(predicted_3d_pos, batch_input)
            epoch_loss_3d_valid += batch_input.shape[0] * batch_input.shape[1] * error.clone()
            N += batch_input.shape[0] * batch_input.shape[1]
            if restore:
                for batch_i in range(batch_size):
                    file_path = file_paths[batch_i]
                    ori_feat_i = original_feat[batch_i]
                    pred_feat_i = predicted_feat[batch_i, -1, 0]
                    ori_pose_i = batch_input[batch_i]
                    pred_pose_i = predicted_3d_pos[batch_i, -1 ,0]
                    if align:
                        T_a, J, C = ori_feat_i.shape
                        T_b, J, C = pred_feat_i.shape
                        ori_feat_i_sub = ori_feat_i
                        _, path = dtw_pytorch(ori_pose_i, pred_pose_i)
                        dtw_map = dict(path)     
                        del dtw_map[T_a-1]
                        a_start, a_end, sub_dtw_map = extract_between_minmax_value(dtw_map)
                        ori_feat_i_sub = ori_feat_i_sub[a_start:(a_end+1)]
                        b_index = [d_j for d_j in sub_dtw_map.values()]    
                        min_indices_np = np.array(b_index)  
                        min_indices = torch.Tensor(b_index)  
                        
                        # structure residual
                        matched_b = pred_feat_i[min_indices_np]
                        structure_distance = torch.norm(ori_feat_i_sub - matched_b, dim=-1)  # [T_a,J]
                        structure_residual = ori_feat_i_sub - matched_b  # [T_a,J,C]
                        structure_distance_clips = structure_distance.unfold(dimension=0, size=args.stride, step=args.stride).permute(0,2,1) # [T_a/step, step, J]
                        structure_residual_clips = structure_residual.unfold(dimension=0, size=args.stride, step=args.stride).permute(0,3,1,2) # [T_a/step, step, J, C]
                        structure_distance_clips = structure_distance_clips.mean(dim=(1,2))
                        structure_residual_clips = structure_residual_clips.mean(dim=1) # [T_a/step, J, C]
                        structure_max_idx = torch.argmax(structure_distance_clips)
                        
                        # motion residual
                        ori_feat_i_clip = ori_feat_i_sub.unfold(dimension=0, size=args.stride, step=args.stride).permute(0,3,1,2) #[ T_a/step, step, J, C]
                        N = ori_feat_i_clip.shape[0]
                        b_flat = pred_feat_i.view(T_b, -1)
                        b_indices = min_indices.unfold(dimension=0, size=args.stride, step=args.stride)
                        # batch gather: batch indices
                        b_gathered = b_flat[b_indices.long()]  # [T_a/step, step, JC]
                        matched_b_segs = b_gathered.view(N, args.stride, J, C)
                        motion_ori_clip = ori_feat_i_clip[:,1:] - ori_feat_i_clip[:,:-1]
                        motion_matched_pred_clip = matched_b_segs[:,1:] - matched_b_segs[:,:-1]
                        motion_residual = (motion_ori_clip - motion_matched_pred_clip).mean(dim=1).mean(dim=0) #[J, C]

                    np.savez(file_path.replace('.pkl','_motioneres.npz'), motion_residual.cpu().detach().numpy())
                    np.savez(file_path.replace('.pkl','_structureres.npz'), structure_residual_clips[structure_max_idx].cpu().detach().numpy())
            if vis:
                vis_clip=18
                for batch_i in range(batch_size):
                    # if label_score[0].item() == 1.0 and id_participant[0]==26.0:
                    device = original_feat.device
                    file_path = file_paths[batch_i]
                    ori_feat_i = original_feat[batch_i]
                    pred_feat_i = predicted_feat[batch_i, -1, 0]
                    ori_pose_i = batch_input[batch_i]
                    pred_pose_i = predicted_3d_pos[batch_i, -1 ,0]
                    # p normalization
                    ori_pose_i = ori_pose_i.cpu().detach().numpy()
                    pred_pose_i = pred_pose_i.cpu().detach().numpy()
                    ori_pose_i_norm = p_align(ori_pose_i, pred_pose_i)
                    ori_pose_i_norm[...,0] *= -1
                    ori_pose_i = torch.tensor(ori_pose_i_norm, device=device)
                    pred_pose_i = torch.tensor(pred_pose_i, device=device)
                    ori_pose_i = ori_pose_i - ori_pose_i[:,0:1,:]
                    pred_pose_i = pred_pose_i - pred_pose_i[:,0:1,:]
                    if align:
                        T_a, J, C = ori_feat_i.shape
                        T_b, J, C = pred_feat_i.shape
                        ori_feat_i_sub = ori_feat_i
                        _, path = dtw_pytorch(ori_pose_i, pred_pose_i)
                        dtw_map = dict(path)     
                        del dtw_map[T_a-1]
                        a_start, a_end, sub_dtw_map = extract_between_minmax_value(dtw_map)
                        match_b_len = dtw_map[a_start+vis_clip] - dtw_map[a_start]
                        # if match_b_len < vis_clip:
                        ori_feat_i_sub = ori_feat_i_sub[a_start:(a_end+1)]
                        ori_pose_sub = ori_pose_i[a_start:(a_start+vis_clip)]
                        pred_pose_sub = pred_pose_i[dtw_map[a_start] : (dtw_map[a_start+vis_clip])]
                        corresponding_slice = {(key-a_start): (dtw_map[key]-dtw_map[a_start]) for key in dtw_map.keys() if (a_start <= key ) and (key < (a_start+vis_clip))}
                        b_index = [d_j for d_j in sub_dtw_map.values()]    
                        min_indices_np = np.array(b_index)  
                        min_indices = torch.Tensor(b_index)  
                        
                        # structure residual
                        matched_b = pred_feat_i[min_indices_np]
                        structure_distance = torch.norm(ori_feat_i_sub - matched_b, dim=-1)  # [T_a,J]
                        structure_distance_clips = structure_distance.unfold(dimension=0, size=args.stride, step=args.stride).permute(0,2,1) # [T_a/step, step, J]
                        structure_distance_clips = structure_distance_clips.mean(dim=1) #[T,J]
                        s_max_values, s_max_indices = torch.max(structure_distance_clips, dim=1)
                        s_mask = s_max_values > 0.75
                        s_final_indices = torch.full_like(s_max_indices, -1)
                        s_final_indices[s_mask] = s_max_indices[s_mask]
                        structure_max_idx = s_final_indices.unsqueeze(1).expand(-1, args.stride) 
                        structure_max_idx = structure_max_idx.reshape(-1) #[T]
                    
                        # motion residual
                        ori_feat_i_clip = ori_feat_i_sub.unfold(dimension=0, size=args.stride, step=args.stride).permute(0,3,1,2) #[ T_a/step, step, J, C]
                        N = ori_feat_i_clip.shape[0]
                        b_flat = pred_feat_i.view(T_b, -1)
                        b_indices = min_indices.unfold(dimension=0, size=args.stride, step=args.stride)
                        # batch gather: batch indices
                        b_gathered = b_flat[b_indices.long()]  # [T_a/step, step, JC]
                        matched_b_segs = b_gathered.view(N, args.stride, J, C)
                        motion_ori_clip = ori_feat_i_clip[:,1:] - ori_feat_i_clip[:,:-1]
                        motion_matched_pred_clip = matched_b_segs[:,1:] - matched_b_segs[:,:-1]
                        motion_residual = torch.norm(motion_ori_clip - motion_matched_pred_clip, dim=-1).mean(dim=1)
                        m_max_values, m_max_indices = torch.max(motion_residual, dim=1)
                        m_mask = m_max_values > 0.06
                        m_final_indices = torch.full_like(m_max_indices, -1)
                        m_final_indices[m_mask] = m_max_indices[m_mask]
                        motion_max_idx = m_final_indices.unsqueeze(1).expand(-1, args.stride) 
                        motion_max_idx = motion_max_idx.reshape(-1) #[T]                    
                    
                        # visualize the sequence and anomaly
                        file_save_fold = f'res/anomaly/{label_score[0].item()}/{id_participant[0].item()}'
                        os.makedirs(file_save_fold, exist_ok=True)
                        ori_pose_sub_vis = ori_pose_sub.cpu().detach().numpy()
                        pred_pose_sub_vis = pred_pose_sub.cpu().detach().numpy()
                        structure_max_idx_vis = structure_max_idx.cpu().detach().numpy()[:vis_clip]
                        motion_max_idx_vis = motion_max_idx.cpu().detach().numpy()[:vis_clip]
                        
                        file_save_path = os.path.join(file_save_fold, f'visulization.png')
                        # p normalization
                        # ori_pose_sub_vis[...,0] *= -1
                        # ori_pose_sub_vis_norm,_ = p_mpjpe(ori_pose_sub_vis, pred_pose_sub_vis)
                        # ori_pose_sub_vis_norm[...,0] *= -1
                        # pred_len = dtw_map[a_start+vis_clip] - dtw_map[a_start]
                        # pred_pose_sub_vis = pred_pose_sub_vis[:pred_len]
                        # visualize_pose_sequences(ori_pose_sub_vis, ori_pose_sub_vis, pred_pose_sub_vis, motion_max_idx_vis, structure_max_idx_vis, corresponding_frames=corresponding_slice, save_path=file_save_path)
                
                    # render the video
                    if idx%vis_freq == 0 and render:
                        file_save_fold = f'res/generation/{label_score[0].item()}/{id_participant[0].item()}'
                        os.makedirs(file_save_fold, exist_ok=True)
                        predicted_3d_pos_vis = predicted_3d_pos[0,-1].cpu().detach().numpy() # (H,T,J,C) the last diffused pose
                        gt_3d_pose_vis = batch_gt[0:1].cpu().detach().numpy()
                        predicted_3d_pos_vis = predicted_3d_pos_vis - predicted_3d_pos_vis[:,:,0:1]
                        gt_3d_pose_vis = gt_3d_pose_vis - gt_3d_pose_vis[:,:,0:1]
                        hypo_num = predicted_3d_pos_vis.shape[0]
                        # if writer is not None:
                        #     motion_vis = np.concatenate((gt_3d_pose_vis,predicted_3d_pos_vis), axis=0)
                        #     vis_image = render_motion(motion_vis)
                        #     writer.add_image(
                        #                 f"{label_score[0].item()}/motion/{idx}", vis_image.transpose(2, 0, 1))
                                    
                        file_save_path = os.path.join(file_save_fold, f'{idx}.mp4')
                        anomaly_motion_idx = np.full((T_a), -1)
                        anomaly_pose_idx = np.full((T_a), -1)
                        anomaly_motion_idx[a_start:(a_start + N * args.stride)] = motion_max_idx.cpu().detach().numpy()
                        anomaly_pose_idx[a_start:(a_start + N * args.stride)] = structure_max_idx.cpu().detach().numpy()
                        gt_vis = gt_3d_pose_vis[0]
                        predicted_vis = predicted_3d_pos_vis[0]
                        render_and_save_1(gt_vis, predicted_vis, file_save_path, anomaly_motion=anomaly_motion_idx, anomaly_pose=anomaly_pose_idx, keep_imgs=False, fps=30)
                        

    return epoch_loss_3d_valid, N
      
def train_epoch(args, model_pos, train_loader, optimizer):
    epoch_loss_3d_train = 0
    N = 0
    model_pos.train()
    for idx, (batch_gt, label_score, id_participant, _) in tqdm(enumerate(train_loader)):    
        batch_input = batch_gt.clone()
        batch_size = len(batch_input)     
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
            id_participant = id_participant.cuda()
        with torch.no_grad():
            batch_gt = batch_gt - batch_gt[:,:,0:1,:]
            batch_input = batch_input - batch_input[:,:,0:1,:]
        # Predict 3D poses
        predicted_feat, original_feat, predicted_pose = model_pos(batch_input)    # (N, T, 17, 3)
        
        optimizer.zero_grad()
        loss_total = 0
        if args.feature_loss:
            loss_3d_pos = F.mse_loss(predicted_feat, original_feat)
            loss_total += loss_3d_pos
        if args.first_frame_loss:
            loss_3d_first_frame = F.mse_loss(predicted_feat[:,0], original_feat[:,0])
            loss_total += 1000 * loss_3d_first_frame
        if args.mpjpe_loss:
            loss_3d_mpjpe = mpjpe(predicted_pose, batch_input)
            loss_total += loss_3d_mpjpe
        if args.contrastive_loss:
            loss_subj_contrastive = supervised_contrastive_loss(predicted_feat, id_participant)
            loss_total += loss_subj_contrastive
        loss_total.backward()
        optimizer.step()
        epoch_loss_3d_train += batch_gt.shape[0] * batch_gt.shape[1] * loss_total.item()
        N += batch_gt.shape[0] * batch_gt.shape[1]
        if idx%50 == 0:
            if args.feature_loss:
                print('[%d] iter: loss_feat %f' % (
                            idx,
                            loss_3d_pos.item()))
            if args.first_frame_loss:
                print('[%d] iter: loss_fist_frame %f' % (
                            idx,
                            loss_3d_first_frame.item() * 10))
            if args.mpjpe_loss:
                print('[%d] iter: loss_pose %f' % (
                            idx,
                            loss_3d_mpjpe.item()))
    return epoch_loss_3d_train, N

# get the subset
def get_fixed_subset(dataset, ratio=0.2, seed=42):
    np.random.seed(seed)
    total_size = len(dataset)
    subset_size = int(total_size * ratio)
    indices = np.random.choice(total_size, subset_size, replace=False)
    return Subset(dataset, indices)

def train_with_config(args_all, args_motionbert):
    TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
    try:
        os.makedirs(os.path.join(args_all.checkpoint, "logs"))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', args_all.checkpoint)
    writer = tensorboardX.SummaryWriter(os.path.join(args_all.checkpoint, f"logs/{TIMESTAMP}"))
    writer.add_text('command', 'python ' + ' '.join(sys.argv))

    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args_all.batch_size,
          'shuffle': True,
          'num_workers': 12,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    
    testloader_params = {
          'batch_size': args_all.batch_size,
          'shuffle': False,
          'num_workers': 12,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }

    train_dataset = MotionDataset3D(args_motionbert, args_motionbert.subset_list, "test", score=True) # test = no augmentation
    test_dataset = MotionDataset3D(args_motionbert, args_motionbert.subset_list, "test", score=True)
    train_subset = get_fixed_subset(train_dataset, ratio=args_motionbert.subset_ratio, seed=123)
    
    train_loader_3d = DataLoader(train_subset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)
        
    model_backbone = load_backbone(args_motionbert)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable backbone parameter count:', model_params)
    if not args_all.nolog:
        writer.add_text(args_all.checkpoint+'_'+TIMESTAMP + '/Trainable backbone parameter count', str(model_params/1000000) + ' Million')

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()
        
    chk_filename = args_all.motionbertcheckpoint
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    
    joints_left = [4,5,6,11,12,13]
    joints_right = [1,2,3,14,15,16]
    model_pos_train = D3DP(args_all, joints_left, joints_right, backbone=model_backbone, is_train=True)
    model_pos_test_temp = D3DP(args_all,joints_left, joints_right, backbone=model_backbone, is_train=False, sampling_timesteps=args_all.sampling_timesteps)
    
    model_params = 0
    for parameter in model_pos_train.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params/1000000, 'Million')
    if not args_all.nolog:
        writer.add_text(args_all.checkpoint+'_'+TIMESTAMP + '/Trainable D3DP parameter count', str(model_params/1000000) + ' Million')

    # make model parallel
    if torch.cuda.is_available():
        model_pos_train = nn.DataParallel(model_pos_train)
        model_pos_train = model_pos_train.cuda()
        model_pos_test_temp = nn.DataParallel(model_pos_test_temp)
        model_pos_test_temp = model_pos_test_temp.cuda()
    
    if args_all.resume or args_all.evaluate:    
        chk_filename = os.path.join(args_all.D3DPcheckpoint, args_all.resume if args_all.resume else args_all.evaluate)
        # chk_filename = args.resume or args.evaluate
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))
        model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
        model_pos_test_temp.load_state_dict(checkpoint['model_pos'], strict=False)

    if not args_all.evaluate: 
        lr_decay = args_all.lr_decay
        lr = args_all.learning_rate
        backbone_params = list(model_pos_train.module.backbone.parameters())
        all_params = list(model_pos_train.parameters())
        backbone_param_ids = set(id(p) for p in backbone_params)
        other_params = [p for p in all_params if id(p) not in backbone_param_ids]
        param_groups = [
            {
                "params": [p for p in backbone_params if p.requires_grad],
                "lr": args_motionbert.learning_rate
            },
            {
                "params": [p for p in other_params if p.requires_grad],
                "lr": lr
            }
        ]
        optimizer = optim.AdamW(param_groups, weight_decay=0.1)
        st = 0
        if args_all.resume:
            st = checkpoint['epoch']
            # if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            #     optimizer.load_state_dict(checkpoint['optimizer'])
            # else:
            #     print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            # if not args_all.coverlr:
            #     lr = checkpoint['lr']
        
        losses_3d_train = []
        losses_3d_valid = []
        
        min_loss = 1000
        best_epoch = 0
        initial_momentum = 0.1
        final_momentum = 0.001
        print('** Note: reported losses are averaged over all frames.')
        print('** The final evaluation will be carried out after the last training epoch.')
        print('INFO: Training on {}(3D) batches'.format(len(train_loader_3d)))
        
        # Training
        for epoch in range(st, args_all.epochs):
            print('Training epoch %d.' % epoch)
            start_time = time()
            epoch_loss_3d_train, N_train = train_epoch(args_all, model_pos_train, train_loader_3d, optimizer) 
            elapsed = (time() - start_time) / 60
            print('[%d] time %.2f lr %f 3d_train (loss) %f' % (
                epoch + 1,
                elapsed,
                lr,
                epoch_loss_3d_train / N_train))
            losses_3d_train.append(epoch_loss_3d_train / N_train)
            
            model_pos_test_temp.load_state_dict(model_pos_train.state_dict(), strict=False)
            
            epoch_loss_3d_val, N_val = evaluate(args_all, model_pos_test_temp, test_loader)
            elapsed = (time() - start_time) / 60
            losses_3d_valid.append(epoch_loss_3d_val / N_val)
            print('[%d] time %.2f lr %f 3d_val (mpjpe) %f' % (
                epoch + 1,
                elapsed,
                lr,
                epoch_loss_3d_val * 1000 / N_val))
            
            writer.add_scalar('training loss (L2Loss)', epoch_loss_3d_train / N_train, epoch + 1)
            writer.add_scalar('validation loss (mpjpe)', epoch_loss_3d_val * 1000/N_val, epoch + 1)
                
            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            # Save checkpoints
            chk_path_latest = os.path.join(args_all.checkpoint, 'latest_epoch.bin')
            chk_path_best = os.path.join(args_all.checkpoint, 'best_epoch.bin'.format(epoch))
            
            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos_train, min_loss)
            if losses_3d_valid[-1] * 1000 < min_loss:
                min_loss = losses_3d_valid[-1] * 1000
                best_epoch = epoch
                save_checkpoint(chk_path_best, epoch, lr, optimizer, model_pos_train, min_loss)
    
    else:
        start_time = time()
        epoch_loss_3d_val, N_val = evaluate(args_all, model_pos_test_temp, test_loader, vis=True, vis_freq=100, writer=writer, restore=False, align=True, render=True)          
        elapsed = (time() - start_time) / 60
            
if __name__ == "__main__":
    args_all = parse_args()
    set_random_seed(args_all.seed)
    args_motionbert = get_config(args_all.config)
    args_all.checkpoint = "checkpoint/diff"
    os.makedirs(args_all.checkpoint, exist_ok=True)
    
    train_with_config(args_all, args_motionbert)