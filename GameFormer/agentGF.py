import argparse
import numpy as np
import torch
from GameFormer.predictor import ClassFormer
from GameFormer.train_utils import *
from shared_memory_dict import SharedMemoryDict

# Scene Encoder service: the planner sends vectorized scene inputs through shared memory
# and receives the scene embedding z (feature_out=True) or the stage-0 class.
parser = argparse.ArgumentParser(description='Scene Encoder service')
parser.add_argument('--ckpt', type=str, help='trained Scene Encoder checkpoint',
                    default="ckpt/scene_encoder_stage0/model_epoch_30.pth")
parser.add_argument('--encoder_layers', type=int, default=3)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--embed_dim', type=int, default=256)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

MemoryName = "classifier"
device = torch.device(args.device)

print("starting", MemoryName)
gameformer = ClassFormer(encoder_layers=args.encoder_layers, token_dim=args.embed_dim, heads=args.heads,
                         num_classes=len(type_selected), only_xyh=False)
print("loading ckpt")
state_dict = torch.load(args.ckpt, map_location=torch.device("cpu"))
gameformer = gameformer.to(device)
gameformer.load_state_dict(state_dict)
gameformer.eval() # the encoder is frozen after stage 0
print("Model Params: {}".format(sum(p.numel() for p in gameformer.parameters())))

def inference(data, feature_out = False):
    inputs = {
            'ego_agent_past': torch.Tensor(np.array([data['ego_agent_past']])).to(device), # [:, :3]
            'neighbor_agents_past': torch.Tensor(np.array([data['neighbor_agents_past']])).to(device), # [:, :, :3]
            # 'ego_agent_past': torch.Tensor(np.array([data['ego_agent_past']])).to(device),
            # 'neighbor_agents_past': torch.Tensor(np.array([data['neighbor_agents_past']])).to(device),
            'map_lanes': torch.Tensor(np.array([data['map_lanes']])).to(device),
            'map_crosswalks': torch.Tensor(np.array([data['map_crosswalks']])).to(device),
            'route_lanes': torch.Tensor(np.array([data['route_lanes']])).to(device)
    }

    if not feature_out:
        with torch.no_grad(): # important !
            outputs = gameformer(inputs)
        # outputs[0][[1, 4, 10, 12]] = - 2.0 # 消融类型
        _, predicted = torch.max(outputs.data, dim=1)
        return int(predicted[0])
    else:
        with torch.no_grad(): # important !
            feature = gameformer(inputs, feature_out = True)
        return np.array(feature[0].cpu())

print("creating SHM")
smd = SharedMemoryDict(name=MemoryName, size=204800)
smd['input'] = 0 # initialize
smd['return'] = 0
smd['llm_ready'] = 1
smd['llm_output'] = 0

smd['feature_out'] = False
smd['feature'] = np.zeros(args.embed_dim, dtype=np.float32)

print("Initial state: ", smd['input'], smd['return'], smd['llm_ready'], smd['llm_output'])

while True:
    llm_input = {}
    while True:
        if smd['input'] == 1 and smd['return'] == 0:
            print("\n-- Reading input to", MemoryName)
            llm_input['ego_agent_past'] = smd['ego_agent_past']
            llm_input['neighbor_agents_past'] = smd['neighbor_agents_past']
            llm_input['map_lanes'] = smd['map_lanes']
            llm_input['map_crosswalks'] = smd['map_crosswalks']
            llm_input['route_lanes'] = smd['route_lanes']
            is_feature_out = smd['feature_out']
            break

    if is_feature_out:
        llm_output = inference(llm_input, feature_out = True)
        smd['feature'] = llm_output
    else:
        llm_output = inference(llm_input, feature_out = False)
        smd['llm_output'] = llm_output
    smd['input'] = 0
    smd['return'] = 1
    
    print("-- End state", smd['input'], smd['return'])
