import torch
import logging
import random
import numpy as np
import json
import os
import pdb
import re
from torch.utils.data import Dataset
from torch.nn import functional as F


def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())

def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
# 1, 4, 10, 12
type_selected = [
  'behind_long_vehicle',
  # 'changing_lane', #1
  'following_lane_with_lead',
  'high_lateral_acceleration', #3
  # 'high_magnitude_speed', #4
  'low_magnitude_speed', #5
  'near_multiple_vehicles', #6
  'starting_left_turn', #7
  'starting_right_turn',
  'starting_straight_traffic_light_intersection_traversal',
  # 'stationary_in_traffic', #10
  'stopping_with_lead',
  # 'traversing_pickup_dropoff', #12
  'waiting_for_pedestrian_to_cross',
]
"""

test_types = [
  'behind_long_vehicle',
  'changing_lane',
  'following_lane_with_lead',
  'high_lateral_acceleration',
  'high_magnitude_speed',
  'low_magnitude_speed',
  'near_multiple_vehicles',
  'starting_left_turn',
  'starting_right_turn',
  'starting_straight_traffic_light_intersection_traversal',
  'stationary_in_traffic',
  'stopping_with_lead',
  'traversing_pickup_dropoff',
  'waiting_for_pedestrian_to_cross'
]


# root of the processed nuPlan data (scenario-type json files, npz features)
DATA_ROOT = os.environ.get("LILO_DATA_ROOT", "/data/nuplan")

def get_scenario_type():
    scenario_type = {}
    for name in ["score/_scenario_type_train14.json",
                 "score/scenario_type_valtest_token.json",
                 "scenario_split/scenario_type_train10_200.json"]:
        path = os.path.join(DATA_ROOT, name)
        if os.path.exists(path):
            with open(path) as f:
                scenario_type.update(dict(json.load(f)))
    return scenario_type

scenario_type = get_scenario_type()

# Stage-0 classes J: the ten common nuPlan scenario types used to train the Scene Encoder.
type_selected = [
  'behind_long_vehicle',
  'following_lane_with_lead',
  'high_lateral_acceleration',
  'low_magnitude_speed',
  'starting_left_turn',
  'starting_right_turn',
  'starting_straight_traffic_light_intersection_traversal',
  'stationary_in_traffic',
  'stopping_with_lead',
  'waiting_for_pedestrian_to_cross',
]

# Scenario types added to memory in stages t = 1..4, in this order (H, N, C, T).
# They are never used to train the encoder.
added_types = [
  'high_magnitude_speed',        # H
  'near_multiple_vehicles',      # N
  'changing_lane',               # C
  'traversing_pickup_dropoff',   # T
]


# split version
class DrivingData_timing(Dataset):
    def __init__(self, data_dir, n_neighbors, time):
        self.data_list = data_dir
        self._n_neighbors = n_neighbors
        self.scenario_type = self.get_scenario_score()
        self.time = time

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])

        ego_past = data['ego_agent_past']
        neighbors_past = data['neighbor_agents_past']
        ego_future = data['ego_agent_future']
        neighbors_future = data['neighbor_agents_future']
        ep, ef = ego_past[:, :3], ego_future[:, :3]
        ap, af = neighbors_past[:, :, :3], neighbors_future[:, :, :3]
        ego = np.concatenate((ep, ef), axis=0)
        agents = np.concatenate((ap, af), axis=1)
        ego_window = ego[self.time : self.time + 21, :]
        agents_window = agents[:, self.time : self.time + 21, :]
        route_lanes = data['route_lanes']
        map_lanes = data['lanes']
        map_crosswalks = data['crosswalks']

        # get class_gt from token
        token = data['token']
        class_gt = type_selected.index(self.scenario_type[token.item()])

        return ego_window, agents_window, map_lanes, map_crosswalks, route_lanes, class_gt, token
    
    def get_scenario_score(self):
        with open(os.path.join(DATA_ROOT, "score/scenario_type_valtest_token.json")) as f:
            scenario_type = json.load(f)
        return scenario_type

class TrainDataset(Dataset):
    def __init__(self, data_dir, n_neighbors):
        self.data_list = data_dir
        self._n_neighbors = n_neighbors

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]

        file_type = "npz"
        if file_type == "npz":
            data = np.load(file_name, allow_pickle=True)#.tolist()
            ego = data['ego_agent_past']#[:, :3]
            neighbors = data['neighbor_agents_past']#[:, :, :3]
            route_lanes = data['route_lanes'] 
            map_lanes = data['lanes']#data['map_lanes']
            map_crosswalks = data['crosswalks']#data['map_crosswalks']
        elif file_type == "npy":
            data = np.load(file_name, allow_pickle=True).tolist()
            ego = data['ego_agent_past']#[:, :3]
            neighbors = data['neighbor_agents_past']#[:, :, :3]
            route_lanes = data['route_lanes'] 
            map_lanes = data['map_lanes']
            map_crosswalks = data['map_crosswalks']

        # get class_gt from token
        token = re.findall("([0-9a-z]{16})", file_name)[0]
        
        # class_gt = all_types.index(self.scenario_type[token])
        class_gt = type_selected.index(scenario_type[token])

        return ego, neighbors, map_lanes, map_crosswalks, route_lanes, class_gt  #, token
    
    
class TestDataset(Dataset):
    def __init__(self, data_dir, n_neighbors):
        self.data_list = data_dir
        self._n_neighbors = n_neighbors

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]

        file_type = "npz"
        if file_type == "npz":
            data = np.load(file_name, allow_pickle=True)#.tolist()
            ego = data['ego_agent_past']#[:, :3]
            neighbors = data['neighbor_agents_past']#[:, :, :3]
            route_lanes = data['route_lanes'] 
            map_lanes = data['lanes']#data['map_lanes']
            map_crosswalks = data['crosswalks']#data['map_crosswalks']
        elif file_type == "npy":
            data = np.load(file_name, allow_pickle=True).tolist()
            ego = data['ego_agent_past']#[:, :3]
            neighbors = data['neighbor_agents_past']#[:, :, :3]
            route_lanes = data['route_lanes'] 
            map_lanes = data['map_lanes']
            map_crosswalks = data['map_crosswalks']

        # get class_gt from token
        # token = data['token']
        token = re.findall("([0-9a-z]{16})", file_name)[0]
        
        # class_gt = type_selected.index(scenario_type[token])
        try:
            class_gt = test_types.index(scenario_type[token])
        except KeyError:
            print(f"KeyError: {token}")
            class_gt = 0
        
        # scores = np.array(self.scenario_score[token.item()]) # 1) from .npz [default, straight, turn, across, stationary]

        return ego, neighbors, map_lanes, map_crosswalks, route_lanes, class_gt, token
    

def imitation_loss(gmm, scores, ground_truth):
    B, N = gmm.shape[0], gmm.shape[1]
    distance = torch.norm(gmm[:, :, :, :, :2] - ground_truth[:, :, None, :, :2], dim=-1)
    best_mode = torch.argmin(distance.mean(-1), dim=-1)

    mu = gmm[..., :2]
    best_mode_mu = mu[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, :, None]]
    best_mode_mu = best_mode_mu.squeeze(2)
    dx = ground_truth[..., 0] - best_mode_mu[..., 0]
    dy = ground_truth[..., 1] - best_mode_mu[..., 1]

    cov = gmm[..., 2:]
    best_mode_cov = cov[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, :, None]]
    best_mode_cov = best_mode_cov.squeeze(2)
    log_std_x = torch.clamp(best_mode_cov[..., 0], -2, 2)
    log_std_y = torch.clamp(best_mode_cov[..., 1], -2, 2)
    std_x = torch.exp(log_std_x)
    std_y = torch.exp(log_std_y)

    gmm_loss = log_std_x + log_std_y + 0.5 * (torch.square(dx/std_x) + torch.square(dy/std_y))
    gmm_loss = torch.mean(gmm_loss)

    score_loss = F.cross_entropy(scores.permute(0, 2, 1), best_mode, reduction='none') #label_smoothing=0.2
    score_loss = score_loss * torch.ne(ground_truth[:, :, 0, 0], 0)
    score_loss = torch.mean(score_loss)
    
    loss = gmm_loss + score_loss

    return loss, best_mode_mu, best_mode


def level_k_loss(outputs, ego_future, neighbors_future, neighbors_future_valid):
    loss: torch.tensor = 0
    levels = len(outputs.keys()) // 2 
    gt_future = torch.cat([ego_future[:, None], neighbors_future], dim=1)

    for k in range(levels):
        trajectories = outputs[f'level_{k}_interactions']
        scores = outputs[f'level_{k}_scores']
        predictions = trajectories[:, 1:] * neighbors_future_valid[:, :, None, :, 0, None]
        plan = trajectories[:, :1]
        trajectories = torch.cat([plan, predictions], dim=1)
        il_loss, future, best_mode = imitation_loss(trajectories, scores, gt_future)
        loss += il_loss 

    return loss, future


def planning_loss(plan, ego_future):
    loss = F.smooth_l1_loss(plan, ego_future)
    loss += F.smooth_l1_loss(plan[:, -1], ego_future[:, -1])

    return loss


def motion_metrics(plan_trajectory, prediction_trajectories, ego_future, neighbors_future, neighbors_future_valid):
    prediction_trajectories = prediction_trajectories * neighbors_future_valid
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ego_future[:, :, :2], dim=-1)
    prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - neighbors_future[:, :, :, :2], dim=-1)
    heading_error = torch.abs(torch.fmod(plan_trajectory[:, :, 2] - ego_future[:, :, 2] + np.pi, 2 * np.pi) - np.pi)

    # planning
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(plan_distance[:, -1])
    plannerAHE = torch.mean(heading_error)
    plannerFHE = torch.mean(heading_error[:, -1])
    
    # prediction
    predictorADE = torch.mean(prediction_distance, dim=-1)
    predictorADE = torch.masked_select(predictorADE, neighbors_future_valid[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_distance[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, neighbors_future_valid[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)

    return plannerADE.item(), plannerFDE.item(), plannerAHE.item(), plannerFHE.item(), predictorADE.item(), predictorFDE.item()


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=[0.5, 0.5, 0.5], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表, 三分类中第0类权重0.2, 第1类权重0.3, 第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).to("cuda")
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class DiceLoss(torch.nn.Module):
    def __init__(self, alpha: float = 1.0, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.smooth) / (probs_with_factor + 1 + self.smooth)

        if self.reduction == "mean":
            return loss.mean()


def compute_recall(predicted, labels, num):
    '''
    # compute F1
    zeros = torch.zeros([predicted.shape[0]]).to(predicted.device)
    ones = torch.ones([predicted.shape[0]]).to(predicted.device)
    p1 = predicted == ones
    p0 = predicted == zeros
    gt1 = labels == ones
    gt0 = labels == zeros
    TP = (p1 & gt1).sum().item()
    FP = (p1 & gt0).sum().item()
    TN = (p0 & gt0).sum().item()
    FN = (p0 & gt1).sum().item()
    return TP, FP, TN, FN
    '''
    
    # compute recall
    tp = [0 for i in range(num)]
    total = [0 for i in range(num)]

    all_true = [i * torch.ones([predicted.shape[0]]).to(predicted.device) for i in range(num)]

    for i in range(num):
        tp[i] += ((predicted == all_true[i]) & (labels == all_true[i])).sum().item()
        total[i] += (labels == all_true[i]).sum().item()
        
    return tp, total