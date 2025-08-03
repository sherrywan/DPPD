import os
import numpy as np
import argparse
import errno
import tensorboardX
from tqdm import tqdm
from time import time
import random

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

from lib.utils.tools import *
from lib.utils.learning import *
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.data.augmentation import Augmenter_input
from lib.model.loss import *
from lib.model.diffusionpose import *
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('--chamfer', default=False, type=bool, metavar='Opt', help='choose to caculate chamfer distance')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-mt', '--modeltype', default='motionbert', type=str, metavar='MODEL', help='model type')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    
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
    parser.add_argument('-l', '--log', default='log/default', type=str, metavar='PATH',
                        help='log file directory')
    parser.add_argument('-cf','--checkpoint-frequency', default=20, type=int, metavar='N',
                        help='create a checkpoint every N epochs')
    parser.add_argument('--nolog', action='store_true', help='forbiden log function')
    parser.add_argument('--render', action='store_true', help='visualize a particular video')
    parser.add_argument('--by-subject', action='store_true', help='break down error by subject (on evaluation)')
    parser.add_argument('--export-training-curves', action='store_true', help='save training curves as .png images')


    # Model arguments
    parser.add_argument('-s', '--stride', default=9, type=int, metavar='N', help='chunk size to use during training')
    parser.add_argument('-e', '--epochs', default=1000, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='batch size in terms of predicted frames')
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
    parser.add_argument('-f', '--number-of-frames', default='81', type=int, metavar='N',
                        help='how many frames used as input')
    
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
        
    opts = parser.parse_args()
    return opts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    
def evaluate_motionbert(args, model_pos, test_loader):
    print('INFO: Testing')
    results_all = defaultdict(lambda:defaultdict(lambda: defaultdict(list)))
    model_pos.eval()            
    with torch.no_grad():
        for input, label, id, file_paths in tqdm(test_loader):
            B, T = input.shape[:2]
            if args.rootrel:
                input = input - input[:,:,0:1,:]
            predicted_feat = model_pos(input, return_rep=True)
            for b_i in range(B):
                feat_i = predicted_feat[b_i].cpu().numpy()
                label_i = int(label[b_i].item())
                id_i = int(id[b_i].item())
                results_all[id_i][label_i]['feature'].append(feat_i)
                results_all[id_i][label_i]['path'].append(file_paths[b_i])
    return results_all

def evaluate_D3DP(args, model_pos, test_loader, data_type='BC', chamfer=False, align_method="original", dis_type="eu"):
    print('INFO: Testing')
    original_feats = []
    residual_feats = []
    step=9
    structure_chamfer = []
    motion_chamfer = []
    seq_chamfer = []
    labels = []
    model_pos.eval()            
    with torch.no_grad():
        for input, label, id, file_paths in tqdm(test_loader):
            B, T = input.shape[:2]
            if args.rootrel:
                input = input - input[:,:,0:1,:]
            original_feat, predicted_feat, predicted_pose = model_pos(input)
            input = input.to(predicted_pose.device)
            for b_i in range(B):
                file_paths_i = file_paths[b_i]
                ori_feat_i = original_feat[b_i]
                pred_feat_i = predicted_feat[b_i, -1, 0]
                ori_pose_i = input[b_i]
                pred_pose_i = predicted_pose[b_i, -1 ,0]
                if chamfer:
                    T_a, J, C = ori_feat_i.shape
                    T_b, J, C = pred_feat_i.shape
                    ori_feat_i_sub = ori_feat_i
                    if "dtw" in align_method:
                        # dtw alignment
                        _, path = dtw_pytorch(ori_pose_i, pred_pose_i)
                        dtw_map = dict(path)     
                        del dtw_map[T_a-1]
                        a_start, a_end, sub_dtw_map = extract_between_minmax_value(dtw_map)
                        ori_feat_i_sub = ori_feat_i_sub[a_start:(a_end+1)]
                        b_index = [d_j for d_j in sub_dtw_map.values()]    
                        min_indices_np = np.array(b_index)  
                        min_indices = torch.Tensor(b_index)  
                    elif "chamfer" in align_method:         
                        # chamfer alignment
                        a_expand = ori_pose_i.unsqueeze(1)  # [T_a, 1, J, C]
                        b_expand = pred_pose_i[:(T_b-step)].unsqueeze(0)  # [1, T_b-step, J, C]
                        dists = torch.norm(a_expand - b_expand, dim=-1).mean(dim=-1)  # [ T_a, T_b-step]
                        # find nearest frame to ori_feat_i in pred_feat_i
                        min_indices = torch.argmin(dists, dim=1)  # [T_a]
                        min_indices_np = min_indices.cpu().numpy()
                    else:
                        # no alignment
                        min_indices = torch.tensor(range(T_b))  # [T_a]
                        min_indices_np = min_indices.cpu().numpy()    
                    
                    # structure residual
                    matched_b = pred_feat_i[min_indices_np]
                    if dis_type=="inner":
                        structure_residual = torch.sum(ori_feat_i_sub * matched_b, dim=-1)  # [T_a,J]
                    else:
                        structure_residual = torch.norm(ori_feat_i_sub - matched_b, dim=-1)  # [T_a,J]
                    structure_residual_clips = structure_residual.unfold(dimension=0, size=step, step=step).permute(0,2,1) # [T_a/step, step, J]
                    structure_residual_clips = structure_residual_clips.mean(dim=1)
                    structure_chamfer.append(max(structure_residual_clips.reshape(-1)).item())
                    
                    # motion residual
                    ori_feat_i_clip = ori_feat_i_sub.unfold(dimension=0, size=step, step=step).permute(0,3,1,2) #[ N, step, J, C]
                    N = ori_feat_i_clip.shape[0]
                    b_flat = pred_feat_i.view(T_b, -1)
                    if "chamfer" in align_method:
                        # batch gather: indices in batch
                        min_indices_clip = min_indices[::step] # [N]
                        offsets = torch.arange(step, device=ori_feat_i.device).view(1, step)  # [1, step]
                        b_indices = min_indices_clip.unsqueeze(-1) + offsets  # [N, step]
                    else:
                        b_indices = min_indices.unfold(dimension=0, size=step, step=step)
                    # batch gather: batch indices
                    b_gathered = b_flat[b_indices.long()]  # [N, step, JC]
                    matched_b_segs = b_gathered.view(N, step, J, C)
                    motion_ori_clip = ori_feat_i_clip[:,1:] - ori_feat_i_clip[:,:-1]
                    motion_matched_pred_clip = matched_b_segs[:,1:] - matched_b_segs[:,:-1]
                    if dis_type=="inner":
                        motion_residual = torch.sum(motion_ori_clip * motion_matched_pred_clip, dim=-1).mean(dim=1).mean(dim=0)
                    else:
                        motion_residual = torch.norm(motion_ori_clip - motion_matched_pred_clip, dim=-1).mean(dim=1).mean(dim=0)
                    motion_chamfer.append(max(motion_residual).item())
                    
                    # sequence residual
                    if dis_type=="inner":
                        sequence_residual = torch.sum(ori_feat_i_clip * matched_b_segs, dim=-1).mean(dim=1).mean(dim=0)
                    else:
                        sequence_residual = torch.norm(ori_feat_i_clip - matched_b_segs, dim=-1).mean(dim=1).mean(dim=0)
                    seq_chamfer.append(max(sequence_residual).item())
                    
                ori_feat_i = ori_feat_i.cpu().detach().numpy()
                pred_feat_i = pred_feat_i.cpu().detach().numpy()
                res_feat_i = ori_feat_i - pred_feat_i
                ori_feat_i = dataype_trans(ori_feat_i, datatype=data_type)
                res_feat_i = dataype_trans(res_feat_i, datatype=data_type)
                labels.append(int(label[b_i].item()))
                original_feats.append(ori_feat_i)
                residual_feats.append(res_feat_i)
    return labels, original_feats, residual_feats, structure_chamfer, motion_chamfer, seq_chamfer

def flatten_feats(feat_dict):
    feats = []
    labels = []
    for id in feat_dict:
        for label in feat_dict[id]:
            for feat in feat_dict[id][label]:
                feats.append(feat)
                labels.append(label)
    return np.array(feats), np.array(labels)

def dataype_trans(feat, datatype='BC'):
    if datatype == "BC":
        return np.mean(feat, axis=(0,1)).reshape(-1)
    elif datatype == "BJC":
        return np.mean(feat, axis=0).reshape(-1)
    elif datatype == "BTC":
        return np.mean(feat, axis=1).reshape(-1)

def find_nearest_in_label0(feat, pool):
    """Find most similar vector in pool (label=0) using cosine similarity"""
    if len(pool) == 0:
        return np.zeros_like(feat)
    sims = cosine_similarity(feat.reshape(1, -1), np.stack(pool))  # (1, N)
    best_idx = np.argmax(sims)
    return pool[best_idx]

def subtract_feat(feat_dict):
    new_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for id in feat_dict:
        label0_feats = feat_dict[id].get(0, [])
        if len(label0_feats) == 0 :
            continue
        for label in feat_dict[id]:
            for feat in feat_dict[id][label]['feature']:  
                new_dict['BC'][id][label].append(np.mean(feat, axis=(0,1)).reshape(-1))
                new_dict['BJC'][id][label].append(np.mean(feat, axis=0).reshape(-1))
                new_dict['BTC'][id][label].append(np.mean(feat, axis=1).reshape(-1))
    return new_dict

def save_residual(feat_dict):
    global_label0_feat = []
    pca = PCA(n_components=10)
    # for id in feat_dict:
    #     label0_feats = feat_dict[id].get(0, [])
    #     if len(label0_feats) == 0 :
    #         continue
    #     features = feat_dict[id][0]['feature']
    #     features = np.array(features)
    #     B,T,J,C = features.shape
    #     features_reshaped = features.reshape(B, -1)  # [B, T*J*C]
    #     features_reshaped = features_reshaped.transpose(1,0)

    #     # 对 batch 维度做 PCA，压缩到 1024 个样本
    #     features_pca = pca.fit_transform(features_reshaped)  # [B=B, -> 10]
    #     features_pca = features_pca.transpose(1,0)
    #     global_label0_feat += features_pca.reshape(-1,T,J,C).tolist()
    # global_label0_feat = np.array(global_label0_feat)
    local_label0_feats = []
    for id in feat_dict:
        print(f"----------------{id} feature extraction----------------")
        label0_feats = feat_dict[id].get(0, [])
        for label in feat_dict[id]:
            if len(label0_feats) > 0: 
                local_label0_feats  = np.array(label0_feats['feature'])
                b_label0,T,J,C = local_label0_feats.shape
                if b_label0 > 100:
                    local_label0_feats = local_label0_feats[::4]
                    b_label0,T,J,C = local_label0_feats.shape
                    features_reshaped = local_label0_feats.reshape(b_label0, -1) 
                    features_reshaped = features_reshaped.transpose(1,0)
                    features_pca = pca.fit_transform(features_reshaped)
                    local_label0_feats = features_pca.transpose(1,0).reshape(-1,T,J,C)
                    b_label0 = local_label0_feats.shape[0]
            for pa, feat in zip(feat_dict[id][label]['path'], feat_dict[id][label]['feature']):
                feat = np.array(feat)
                sims = cosine_similarity(feat.reshape(1, -1), local_label0_feats.reshape(b_label0,-1))  # (1, N)
                best_idx = np.argmax(sims)
                ref = local_label0_feats[best_idx]
                structure_res = (feat - ref) 
                T,J,C=structure_res.shape
                norms = np.linalg.norm(structure_res.reshape(T, -1), axis=1) 
                max_batch_idx= np.argmax(norms)
                max_structure_res = structure_res[max_batch_idx]  
                
                motion_res = (feat[1:] - feat[:-1]) - (ref[1:] - ref[:-1])
                mean_motion_res = motion_res.mean(axis=0)
                np.savez(pa.replace('.pkl','_gt_structureres.npz'), max_structure_res)
                np.savez(pa.replace('.pkl','_gt_motioneres.npz'), mean_motion_res)


def subtract_label0_feats(feat_dict):
    new_dict = {}

    for id in feat_dict:
        label0_feats = feat_dict[id].get(0, [])
        if len(label0_feats) == 0 :
            continue
        new_dict[id] = {}

        for label in feat_dict[id]:
            new_dict[id][label] = []
            for feat in feat_dict[id][label]:
                ref = find_nearest_in_label0(feat, label0_feats)
                diff = feat - ref
                new_dict[id][label].append(diff)
    return new_dict


def subtract_global_label0(feat_dict):
    pool_feats = []
    for id in feat_dict:
        for feat in feat_dict[id].get(0, []):
            pool_feats.append(feat)
    pool_feats = np.stack(pool_feats) if pool_feats else np.zeros((1, 1))

    new_dict = {}
    for id in feat_dict:
        new_dict[id] = {}
        for label in feat_dict[id]:
            new_dict[id][label] = []
            for feat in feat_dict[id][label]:
                best_ref = find_nearest_in_label0(feat, pool_feats)
                residual = feat - best_ref
                new_dict[id][label].append(residual)
    return new_dict

def analysis_chamfer(distances, labels, title="Structure", save_dir="res", save_fold="chamfer_bins"):
    # 转成 numpy 方便操作
    save_dir = save_dir + '/' + save_fold
    os.makedirs(save_dir, exist_ok=True)
    distances = np.array(distances)
    labels = np.array(labels)
    cmap = {
        0: '#e2ecd5',  # green
        1: '#97c4cf',  # blue
        2: '#f9dcae'   # yellow
        }
    # 分类绘图
    plt.figure(figsize=(8, 5))
    bins = np.linspace(distances.min(), distances.max(), 50)  # 设置 bin

    for label in [2,1,0]:
        mask = labels == label
        plt.hist(distances[mask], bins, alpha=0.7, label=str(label), density=True, color=cmap[label])

    plt.xlabel("Distance", fontfamily='Time New Roman')
    plt.ylabel("Density", fontfamily='Time New Roman')
    plt.title(f"{title} Distance Distribution by Label", fontfamily='Time New Roman')
    plt.legend(title="gait score", loc="best", fontsize='small')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,title+'chamfer.png'),dpi=400)

def compute_tsne(feats, labels, title="t-SNE", figsize=(6,5), save_dir="res"):
    """
    feats: ndarray, shape [N, D]
    labels: ndarray/list, shape [N], int or str
    """
    tsne = TSNE(n_components=2, metric='cosine', perplexity=10, init='pca', random_state=0)
    tsne_feats = tsne.fit_transform(feats)

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    cmap = {
        0: '#e2ecd5',  # green
        1: '#97c4cf',  # blue
        2: '#f9dcae'   # yellow
        }

    plt.figure(figsize=figsize)
    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(tsne_feats[idx, 0], tsne_feats[idx, 1],
                    label=str(label),
                    alpha=0.7,
                    s=5,
                    color=cmap[label])
    
    # plt.title(title, fontfamily='Time New Roman')
    plt.xlabel("TSNE-1", fontfamily='Time New Roman', fontsize=14)
    plt.ylabel("TSNE-2", fontfamily='Time New Roman', fontsize=14)
    plt.xlim(-150, 150) 
    plt.ylim(-150, 150)
    # plt.legend(title="gait score", loc="upper right", fontsize=14)
    # 设置 legend
    legend = plt.legend(
        title="gait score",
        loc="upper right",
        fontsize=14,               # 图例标签字体大小
        title_fontsize=14,         # 图例标题字体大小
        frameon=True               # 显示图例边框（可选）
    )
    # 设置 title 加粗
    legend.get_title().set_weight("bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,title+'.png'),dpi=400)
    
def analy_with_config(args, opts):
    print(args)
    type_paint = ['BJC']
    print('Loading dataset...')

    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }

    test_dataset = MotionDataset3D(args, args.subset_list, "test", score=True)
    test_loader = DataLoader(test_dataset, **testloader_params)
    
    model_backbone = load_backbone(args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()
    if opts.modeltype == "D3DP":
        joints_left = [4,5,6,11,12,13]
        joints_right = [1,2,3,14,15,16]
        model_pos = D3DP(opts, joints_left, joints_right, backbone=model_backbone, is_train=False, sampling_timesteps=opts.sampling_timesteps)
        if torch.cuda.is_available():
            model_pos = nn.DataParallel(model_pos)
            model_pos = model_pos.cuda()
    else:
        model_pos = model_backbone
    model_params = 0
    for parameter in model_pos.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    
    chk_filename = os.path.join(opts.pretrained, opts.selection)
    print('Loading checkpoint', chk_filename)
    # state_dict = torch.load(chk_filename, map_location='cpu')['model_pos']
    # # 如果是多卡保存的，去掉 module. 前缀
    # if any(k.startswith('module.') for k in state_dict.keys()):
    #     state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # missing_keys, unexpected_keys = model_pos.load_state_dict(state_dict, strict=False)
    # print("Missing keys:", missing_keys)
    # print("Unexpected keys:", unexpected_keys)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'], strict=True)

    save_dir_root = f'res/TSNE/{opts.modeltype}'
    if opts.modeltype == "motionbert":
        save_feats = False
        results_all = evaluate_motionbert(args, model_pos, test_loader)
        if save_feats:
            save_residual(results_all)
        results_restricted = subtract_feat(results_all)
        for type_i in type_paint:
            results_all = results_restricted[type_i]
            save_dir = os.path.join(save_dir_root,type_i)
            os.makedirs(save_dir, exist_ok=True)
            # TSNE of original features
            feats, labels = flatten_feats(results_all)
            compute_tsne(feats, labels, title="original feature", save_dir=save_dir)
            
            # TSNE of residual features
            # residual feature from the nearest normal feature of the same participant
            feat_dict_residual = subtract_label0_feats(results_all)
            feats_res, labels_res = flatten_feats(feat_dict_residual)
            compute_tsne(feats_res, labels_res, title="residual feature (local)", save_dir=save_dir)
            # residual feature from the nearest normal feature of all participant
            feat_dict_residual_global = subtract_global_label0(results_all)
            feats_res, labels_res = flatten_feats(feat_dict_residual_global)
            compute_tsne(feats_res, labels_res, title="residual feature (global)", save_dir=save_dir)
    elif opts.modeltype == "D3DP":
        for type_i in type_paint:
            save_dir = os.path.join(save_dir_root,type_i)
            os.makedirs(save_dir, exist_ok=True)
            save_fold = 'dtw_bins_wo_temporalloss'
            labels_res, original_feat_res, residual_feat_res, structure_chamfer, motion_chamfer, seq_chamfer = evaluate_D3DP(args, model_pos, test_loader, data_type=type_i, chamfer=opts.chamfer, align_method=save_fold, dis_type="eu")
            if opts.chamfer:
                analysis_chamfer(structure_chamfer, labels_res, title='Structure', save_fold=save_fold)
                analysis_chamfer(seq_chamfer, labels_res, title='Sequence', save_fold=save_fold)
                analysis_chamfer(motion_chamfer, labels_res, title='Motion', save_fold=save_fold)
            # compute_tsne(original_feat_res, labels_res, title="original feature", save_dir=save_dir)
            # compute_tsne(residual_feat_res, labels_res, title="residual feature", save_dir=save_dir)
               
if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    analy_with_config(args, opts)