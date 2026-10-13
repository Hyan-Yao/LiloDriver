import torch
import math
import pdb
try:
    from .predictor_modules import *
except ImportError:  # run as a script from this directory
    from predictor_modules import *

class Encoder(nn.Module):
    def __init__(self, dim=256, layers=6, heads=8, dropout=0.1, only_xyh=False, map_segment_len=10):
        super(Encoder, self).__init__()
        if only_xyh == True:
            agent_dim = 3
            ego_dim = 3
        else:
            agent_dim = 11
            ego_dim = 7

        self._lane_len = 50
        self._lane_feature = 7
        self._crosswalk_len = 30
        self._crosswalk_feature = 3
        self.agent_encoder = AgentEncoder(agent_dim=agent_dim)
        self.ego_encoder = AgentEncoder(agent_dim=ego_dim)
        self.lane_encoder = VectorMapEncoder(self._lane_feature, self._lane_len, map_segment_len)
        self.crosswalk_encoder = VectorMapEncoder(self._crosswalk_feature, self._crosswalk_len, map_segment_len)
        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                     activation="gelu", dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers) #enable_nested_tensor=False

    def forward(self, inputs):
        # agents
        ego = inputs['ego_agent_past']
        neighbors = inputs['neighbor_agents_past']
        actors = torch.cat([ego[:, None, :, :5], neighbors[..., :5]], dim=1)

        # agent encoding
        encoded_ego = self.ego_encoder(ego)
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)
        actors_mask[:, 0] = False # the ego vehicle sits at the origin of its own frame and is never padding

        # vector maps
        map_lanes = inputs['map_lanes']
        map_crosswalks = inputs['map_crosswalks']

        # map encoding
        encoded_map_lanes, lanes_mask = self.lane_encoder(map_lanes)
        encoded_map_crosswalks, crosswalks_mask = self.crosswalk_encoder(map_crosswalks)

        # attention fusion encoding
        input = torch.cat([encoded_actors, encoded_map_lanes, encoded_map_crosswalks], dim=1)
        mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)

        encoding = self.fusion_encoder(input, src_key_padding_mask=mask)

        # outputs
        encoder_outputs = {
            'actors': actors,
            'encoding': encoding,
            'mask': mask,
            'route_lanes': inputs['route_lanes']
        }
        return encoder_outputs


class Decoder(nn.Module):
    def __init__(self, neighbors=10, modalities=6, levels=3):
        super(Decoder, self).__init__()
        self.levels = levels
        future_encoder = FutureEncoder()

        # initial level
        self.initial_predictor = InitialPredictionDecoder(modalities, neighbors)

        # level-k reasoning
        self.interaction_stage = nn.ModuleList([InteractionDecoder(modalities, future_encoder) for _ in range(levels)])

    def forward(self, encoder_outputs):
        decoder_outputs = {}
        current_states = encoder_outputs['actors'][:, :, -1]
        encoding, mask = encoder_outputs['encoding'], encoder_outputs['mask']

        # level 0 decode
        last_content, last_level, last_score = self.initial_predictor(current_states, encoding, mask)
        decoder_outputs['level_0_interactions'] = last_level
        decoder_outputs['level_0_scores'] = last_score
        
        # level k reasoning
        for k in range(1, self.levels+1):
            interaction_decoder = self.interaction_stage[k-1]
            last_content, last_level, last_score = interaction_decoder(current_states, last_level, last_score, last_content, encoding, mask)
            decoder_outputs[f'level_{k}_interactions'] = last_level
            decoder_outputs[f'level_{k}_scores'] = last_score
        
        env_encoding = last_content[:, 0]

        return decoder_outputs, env_encoding


class NeuralPlanner(nn.Module):
    def __init__(self):
        super(NeuralPlanner, self).__init__()
        self._future_len = 80
        self.route_fusion = CrossTransformer()
        self.plan_decoder = nn.Sequential(nn.Linear(512, 256), nn.ELU(), nn.Dropout(0.1), nn.Linear(256, self._future_len*2))
        self.route_encoder = VectorMapEncoder(3, 50)

    def dynamics_layer(self, controls, initial_state):       
        dt = 0.1 # discrete time period [s]
        max_a = 5 # vehicle's accleration limits [m/s^2]
        max_d = 0.5 # vehicle's steering limits [rad]
        
        vel_init = torch.hypot(initial_state[..., 3], initial_state[..., 4])
        vel = vel_init[:, None] + torch.cumsum(controls[..., 0].clamp(-max_a, max_a) * dt, dim=-1)
        vel = torch.clamp(vel, min=0)

        yaw_rate = controls[..., 1].clamp(-max_d, max_d) * vel
        yaw = initial_state[:, None, 2] + torch.cumsum(yaw_rate * dt, dim=-1)
        yaw = torch.fmod(yaw, 2*math.pi)

        vel_x = vel * torch.cos(yaw)
        vel_y = vel * torch.sin(yaw)

        x = initial_state[:, None, 0] + torch.cumsum(vel_x * dt, dim=-1)
        y = initial_state[:, None, 1] + torch.cumsum(vel_y * dt, dim=-1)

        return torch.stack((x, y, yaw), dim=-1)

    def forward(self, env_encoding, route_lanes, initial_state):
        route_lanes, mask = self.route_encoder(route_lanes)
        mask[:, 0] = False
        route_encoding = self.route_fusion(env_encoding, route_lanes, route_lanes, mask)
        env_route_encoding = torch.cat([env_encoding, route_encoding], dim=-1)
        env_route_encoding = torch.max(env_route_encoding, dim=1)[0] # max pooling over modalities
        control = self.plan_decoder(env_route_encoding)
        control = control.reshape(-1, self._future_len, 2)
        plan = self.dynamics_layer(control, initial_state)

        return plan
    
    
class GameFormer(nn.Module):
    def __init__(self, encoder_layers=6, decoder_levels=3, modalities=6, neighbors=10):
        super(GameFormer, self).__init__()
        self.encoder = Encoder(layers=encoder_layers)
        self.decoder = Decoder(neighbors, modalities, decoder_levels)
        self.planner = NeuralPlanner()

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        route_lanes = encoder_outputs['route_lanes']
        initial_state = encoder_outputs['actors'][:, 0, -1]
        decoder_outputs, env_encoding = self.decoder(encoder_outputs)
        ego_plan = self.planner(env_encoding, route_lanes, initial_state)

        return decoder_outputs, ego_plan

'''
class ClassFormer(nn.Module):
    def __init__(self, encoder_layers=6, token_dim = 256, num_classes=5, only_xyh = False):
        super(ClassFormer, self).__init__()
        self.encoder = Encoder(layers=encoder_layers, dim=token_dim, only_xyh=only_xyh) # [batch, 236, 256]
        self.AvgPool = nn.AvgPool1d(236, stride=236) # 沿 dim=1 方向一维池化 -> [batch, 256]
        self.classifier = nn.Linear(token_dim, num_classes) # [batch, num_class]

    def forward(self, inputs, feature_out = False):
        encoder_outputs = self.encoder(inputs)['encoding']

        encoder_outputs = encoder_outputs.transpose(1, 2)
        feature = self.AvgPool(encoder_outputs).squeeze(-1)
        out = self.classifier(feature)

        if feature_out:
            return feature
        return out
'''


class ClassFormer(nn.Module):
    """
    Scene Encoder of LiloDriver (Sec. 3.3).
    Agent histories are encoded by LSTMs, each map polyline by an MLP max-pooled over its waypoints,
    and a Transformer encoder fuses the tokens. The contextualized ego token is the scene embedding z.
    A linear softmax classifier on z and one learnable prototype per stage-0 class are trained jointly.
    """
    def __init__(self, encoder_layers=3, token_dim=256, heads=8, num_classes=10, reps_per_class=1, only_xyh=False):
        super(ClassFormer, self).__init__()
        self.encoder = Encoder(dim=token_dim, layers=encoder_layers, heads=heads, only_xyh=only_xyh, map_segment_len=None)
        self.classifier = nn.Linear(token_dim, num_classes)  # [batch, num_class]

        self.num_known_classes = num_classes
        self.embedding_dim = token_dim
        self.reps_per_class = reps_per_class

        # learnable prototypes P_j, optimized jointly with the encoder
        self.representatives = nn.Parameter(
            torch.zeros(self.num_known_classes * self.reps_per_class, self.embedding_dim)
        )
        nn.init.normal_(self.representatives)

    def get_features(self, inputs):
        """
        Scene embedding z: the contextualized ego token of the Transformer encoder.
        """
        encoding = self.encoder(inputs)['encoding']  # [batch, tokens, token_dim], token 0 is the ego vehicle
        return encoding[:, 0]

    def forward(self, inputs, feature_out=False, return_features=False):
        feature = self.get_features(inputs)
        if feature_out:
            return feature
        out = self.classifier(feature)
        if return_features:
            return out, feature
        return out

    def get_prototypes(self):
        """
        Learnable prototypes used as class centers
        """
        return self.representatives
