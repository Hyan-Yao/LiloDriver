import os
import re
import random
import csv
import json
import argparse
import pdb
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from GameFormer.predictor import ClassFormer
from GameFormer.train_utils import *

from tensorboardX import SummaryWriter
global logger



class PrototypeMarginLoss(nn.Module):
    """
    Prototype-based margin loss L_PRO (Sec. 3.3).
    D_ij = 1 - cos(z_i, P_j);
    l_ij = y_ij * max(D_ij - m_p, 0) + (1 - y_ij) * max(m_n - D_ij, 0);
    L_PRO = 1 / (|B| J) * sum_i sum_j l_ij.
    """
    def __init__(self, mp=0.1, mn=0.5):
        super(PrototypeMarginLoss, self).__init__()
        assert 0 <= mp < mn
        self.mp = mp  # margin to the own prototype
        self.mn = mn  # margin to all other prototypes

    def forward(self, features, targets, prototypes):
        normalized_features = F.normalize(features, p=2, dim=1)       # [B, d]
        normalized_prototypes = F.normalize(prototypes, p=2, dim=1)   # [J, d]
        distance = 1 - normalized_features @ normalized_prototypes.t()  # [B, J] cosine distance

        y = F.one_hot(targets, num_classes=prototypes.shape[0]).float()
        positive_loss = y * torch.clamp(distance - self.mp, min=0)
        negative_loss = (1 - y) * torch.clamp(self.mn - distance, min=0)

        return (positive_loss + negative_loss).mean()  # mean over |B| x J


def train_meta(dataloader, model, optimizer, scheduler, classification_loss_fn, prototype_loss_fn, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        running_cls_loss = 0.0
        running_pro_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch in dataloader:
            # prepare data
            inputs = {
                'ego_agent_past': batch[0].to(args.device),
                'neighbor_agents_past': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'route_lanes': batch[4].to(args.device)
            }

            labels = batch[5].to(args.device)

            optimizer.zero_grad()

            # forward: class logits and scene embedding z from the same pass
            outputs, features = model(inputs, return_features=True)
            prototypes = model.get_prototypes()

            cls_loss = classification_loss_fn(outputs, labels)
            pro_loss = prototype_loss_fn(features, labels, prototypes)

            # L = L_PRO + lambda * L_CLS
            total_loss = pro_loss + args.cls_weight * cls_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            running_loss += total_loss.item()
            running_cls_loss += cls_loss.item()
            running_pro_loss += pro_loss.item()
        
        # reduce learning rate
        scheduler.step()

        # save model at the end of epoch
        torch.save(model.state_dict(), f'ckpt/{args.name}/model_epoch_{epoch+1}.pth')
        logging.info(f"Model saved in ckpt/{args.name}\n")

        epoch_loss = running_loss / len(dataloader)
        epoch_cls_loss = running_cls_loss / len(dataloader)
        epoch_pro_loss = running_pro_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_samples * 100

        logger.add_scalar("train_loss", epoch_loss, global_step=epoch + 1)
        logger.add_scalar("train_cls_loss", epoch_cls_loss, global_step=epoch + 1)
        logger.add_scalar("train_pro_loss", epoch_pro_loss, global_step=epoch + 1)
        logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, "
                     f"Classification Loss: {epoch_cls_loss:.4f}, Prototype Loss: {epoch_pro_loss:.4f}, "
                     f"Accuracy: {epoch_accuracy:.2f}%")


# 假设 dataloader 已经准备好
def train_epoch(data_loader, model, optimizer, loss_function, epoch):
    model.train()
    epoch_loss = []
    total = 0
    correct = 0

    for batch in data_loader:
        # prepare data
        inputs = {
            'ego_agent_past': batch[0].to(args.device),
            'neighbor_agents_past': batch[1].to(args.device),
            'map_lanes': batch[2].to(args.device),
            'map_crosswalks': batch[3].to(args.device),
            'route_lanes': batch[4].to(args.device)
        }

        labels = batch[5].to(args.device)

        # call the mdoel
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

        # loss backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        # compute metrics
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        epoch_loss.append(loss.item())

    acc_best = correct / total
    logger.add_scalar("train_loss", np.mean(epoch_loss), global_step=epoch)
    logger.add_scalar("train_acc_best", acc_best, global_step=epoch)
    print('train_loss: {:.4f}, train_acc_best: {:.4f}'.format(np.mean(epoch_loss), acc_best))


def valid_epoch(data_loader, model, loss_function, epoch, dataset_name=""):
    model.eval()
    epoch_loss = []

    total = 0
    correct = 0
    recall_tp = [0 for i in range(len(all_types))]
    recall_total = [0 for i in range(len(all_types))]

    for batch in data_loader:
        # prepare data
        inputs = {
            'ego_agent_past': batch[0].to(args.device),
            'neighbor_agents_past': batch[1].to(args.device),
            'map_lanes': batch[2].to(args.device),
            'map_crosswalks': batch[3].to(args.device),
            'route_lanes': batch[4].to(args.device)
        }

        labels = batch[5].to(args.device)

        # call the model
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

        # compute metrics
        _, predicted = torch.max(outputs.data, dim=1) #[bs]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        _tp, _total = compute_recall(predicted, labels, len(all_types))
        for i in range(len(all_types)):
            recall_tp[i] += _tp[i]
            recall_total[i] += _total[i]

        # print("valid:", predicted, labels)
        epoch_loss.append(loss.item())
        
    acc_best = correct / total
    logger.add_scalar(f"valid_loss_{dataset_name}", np.mean(epoch_loss), global_step=epoch)
    logger.add_scalar(f"valid_acc_best_{dataset_name}", acc_best, global_step=epoch)
    for i in range(len(all_types)):
        logger.add_scalar(f"valid_recall_class{i}_{dataset_name}", (recall_tp[i]+1) / (recall_total[i]+1), global_step=epoch)

    print(dataset_name, ' -- valid_loss: {:.4f}, valid_acc: {:.4f}'.format(np.mean(epoch_loss), acc_best))
    print(dataset_name, f' -- recall_tp: {recall_tp}, recall_total: {recall_total}')


def read_data_config(all_dir, num_scenarios=None, seed=0):
    """
    Collect training-split scenarios of the stage-0 types and sample num_scenarios of them.
    """
    all_files = sorted(os.listdir(all_dir))
    obj_files = []
    for f in tqdm(all_files):
        token = re.findall("([0-9a-z]{16})", f)[0]
        try:
            class_name = scenario_type[token]
        except KeyError:
            print("keyerror:", token)
            continue 
        if class_name in type_selected:
            obj_files.append(os.path.join(all_dir, f))

    if num_scenarios is not None and len(obj_files) > num_scenarios:
        rng = random.Random(seed)
        obj_files = rng.sample(obj_files, num_scenarios)
            
    return obj_files

def model_training():
    # Logging
    log_path = f"./log/{args.name}/"
    ckpt_path = f"./ckpt/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use device: {}".format(args.device))

    # set seed
    set_seed(args.seed)

    # set up model
    gameformer = ClassFormer(encoder_layers=args.encoder_layers, token_dim=args.embed_dim, heads=args.heads,
                             num_classes=len(type_selected), only_xyh=False)
    gameformer = gameformer.to(args.device)
    logging.info("Model Params: {}".format(sum(p.numel() for p in gameformer.parameters())))

    # set up optimizer
    optimizer = optim.AdamW(gameformer.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 12, 14, 16, 18], gamma=0.5)
    
    classification_loss_fn = nn.CrossEntropyLoss()
    prototype_loss_fn = PrototypeMarginLoss(mp=args.margin_pos, mn=args.margin_neg)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    
    # set up data loaders
    train_set = TrainDataset(read_data_config(args.train_dir, args.num_train_scenarios, args.seed), args.num_neighbors)

    # test_loader_entire = DataLoader(test_set_entire, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    logging.info("Dataset Prepared: {} train data\n".format(len(train_set)))

    # begin training
    train_meta(train_loader, gameformer, optimizer, scheduler, classification_loss_fn, prototype_loss_fn, train_epochs)

    '''
    for epoch in tqdm(range(train_epochs)):
        logging.info(f"Epoch {epoch+1}/{train_epochs}")
        train_epoch(train_loader, gameformer, optimizer, loss_function, epoch + 1)

        # reduce learning rate
        scheduler.step()

        # save model at the end of epoch
        torch.save(gameformer.state_dict(), f'ckpt/{args.name}/model_epoch_{epoch+1}.pth')
        logging.info(f"Model saved in ckpt/{args.name}\n")
    '''


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name', default="scene_encoder_stage0")
    parser.add_argument('--train_dir', type=str, help='processed nuPlan training-split npz files',
                        default=os.path.join(DATA_ROOT, "data/train14"))
    parser.add_argument('--type_json', type=str, help='{token: scenario_type} written by data_process.py',
                        default=None)
    parser.add_argument('--num_train_scenarios', type=int, help='number of stage-0 training scenarios', default=12000)
    parser.add_argument('--seed', type=int, help='fix random seed', default=24601)
    parser.add_argument('--encoder_layers', type=int, help='number of Transformer encoder layers', default=3)
    parser.add_argument('--heads', type=int, help='number of attention heads', default=8)
    parser.add_argument('--embed_dim', type=int, help='embedding dimension d', default=256)
    parser.add_argument('--margin_pos', type=float, help='margin m_p', default=0.1)
    parser.add_argument('--margin_neg', type=float, help='margin m_n', default=0.5)
    parser.add_argument('--cls_weight', type=float, help='weight lambda of L_CLS', default=0.5)
    parser.add_argument('--num_neighbors', type=int, help='number of neighbor agents to predict', default=20)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=30)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 64)', default=64)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 1e-4)', default=1e-4)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    args = parser.parse_args()

    if args.type_json is not None:
        with open(args.type_json) as f:
            scenario_type.update(json.load(f))  # shared with GameFormer.train_utils

    # Run
    logger = SummaryWriter(log_dir=f"./log/{args.name}")
    model_training()