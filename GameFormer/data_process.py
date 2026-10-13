import os
import yaml
import argparse
import json
import re
from tqdm import tqdm
from common_utils import *
from GameFormer.data_utils import *
import matplotlib.pyplot as plt
import pdb
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from multiprocessing import Pool, cpu_count
import time

# define data processor
class DataProcessor(object):
    def __init__(self, scenarios=None):
        self._scenarios = scenarios

        self.past_time_horizon = 2 # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon 
        self.future_time_horizon = 8 # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon
        self.num_agents = 20

        self._map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK'] # name of map features to be extracted.
        self._max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': 50, 'ROUTE_LANES': 50, 'CROSSWALK': 30} # maximum number of points per feature to extract per feature layer.
        self._radius = 60 # [m] query radius scope relative to the current pose.
        self._interpolation_method = 'linear' # Interpolation method to apply when interpolating to maintain fixed size map elements.

    def get_ego_agent(self):
        self.anchor_ego_state = self.scenario.initial_ego_state
        
        past_ego_states = self.scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        )

        sampled_past_ego_states = list(past_ego_states) + [self.anchor_ego_state]
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(sampled_past_ego_states)

        past_time_stamps = list(
            self.scenario.get_past_timestamps(
                iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
            )
        ) + [self.scenario.start_time]

        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

        return past_ego_states_tensor, past_time_stamps_tensor
    
    def get_neighbor_agents(self):
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]

        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        past_tracked_objects_tensor_list, past_tracked_objects_types = \
              sampled_tracked_objects_to_tensor_list(sampled_past_observations)

        return past_tracked_objects_tensor_list, past_tracked_objects_types

    def get_map(self):        
        ego_state = self.scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(0)

        coords, traffic_light_data = get_neighbor_vector_set_map(
            self.map_api, self._map_features, ego_coords, self._radius, route_roadblock_ids, traffic_light_data
        )

        vector_map = map_process(ego_state.rear_axle, coords, traffic_light_data, self._map_features, 
                                 self._max_elements, self._max_points, self._interpolation_method)

        return vector_map

    def get_speed_limit(self):
        """Speed limit [m/s] of the lane under the initial ego position (nan if unknown); used in the scene description."""
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        ego_state = self.scenario.initial_ego_state
        layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        objects = self.map_api.get_proximal_map_objects(ego_state.center.point, 2.0, layers)
        for layer in layers:
            for lane in objects[layer]:
                if lane.contains_point(ego_state.center.point) and lane.speed_limit_mps is not None:
                    return float(lane.speed_limit_mps)
        return float('nan')

    def get_ego_agent_future(self):
        current_absolute_state = self.scenario.initial_ego_state

        trajectory_absolute_states = self.scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
        )

        # Get all future poses of the ego relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
        )

        return trajectory_relative_poses
    
    def get_neighbor_agents_future(self, agent_index):
        current_ego_state = self.scenario.initial_ego_state
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects

        # Get all future poses of of other agents
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
            )
        ]

        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_tensor_list(sampled_future_observations)
        agent_futures = agent_future_process(current_ego_state, future_tracked_objects_tensor_list, self.num_agents, agent_index)

        return agent_futures
    
    def plot_scenario(self, data):
        # Create map layers
        create_map_raster(data['lanes'], data['crosswalks'], data['route_lanes'])

        # Create agent layers
        create_ego_raster(data['ego_agent_past'][-1])
        create_agents_raster(data['neighbor_agents_past'][:, -1])

        # Draw past and future trajectories
        draw_trajectory(data['ego_agent_past'], data['neighbor_agents_past'])
        draw_trajectory(data['ego_agent_future'], data['neighbor_agents_future'])

        plt.xlim((-50,50))
        plt.ylim((-50,50))
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()
        plt.savefig(f"input/test14_thin/{data['token']}.png")

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)

    def is_exist(self, token):
        fn = re.search("(([\w\-]+){}.npz)".format(token), str(self.all_files))
        if fn == None:
            return False
        else:
            return True
    def work(self, save_dir):
        self.all_files = os.listdir(save_dir)
        for scenario in tqdm(self._scenarios):
            map_name = scenario._map_name
            token = scenario.token

            if self.is_exist(token) == True:
                print(f"Exist {token} in save dir, pass to load next one ......")
                continue

            self.scenario = scenario
            self.map_api = scenario.map_api        

            # get agent past tracks
            ego_agent_past, time_stamps_past = self.get_ego_agent()
            neighbor_agents_past, neighbor_agents_types = self.get_neighbor_agents()
            ego_agent_past, neighbor_agents_past, neighbor_indices = \
                agent_past_process(ego_agent_past, time_stamps_past, neighbor_agents_past, neighbor_agents_types, self.num_agents)

            # get vector set map
            vector_map = self.get_map()

            # get agent future tracks
            ego_agent_future = self.get_ego_agent_future()
            neighbor_agents_future = self.get_neighbor_agents_future(neighbor_indices)

            # gather data
            data = {"map_name": map_name, "token": token, "ego_agent_past": ego_agent_past, "ego_agent_future": ego_agent_future,
                    "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future,
                    "speed_limit": self.get_speed_limit()}
            data.update(vector_map)
            
            # save to disk
            self.save_to_disk(save_dir, data)

            # visualization
            #self.plot_scenario(data)
    def get_scenario_type(self, save_file):
        """Save {token: scenario_type} of the processed scenarios."""
        result = {}
        for scenario in tqdm(self._scenarios):
            result[scenario.token] = scenario.scenario_type

        with open(save_file, "w") as f:
            json.dump(result, f)

def AtomWork(args):
    self = DataProcessor()
    save_dir, scenario = args

    map_name = scenario._map_name
    token = scenario.token
    self.scenario = scenario
    self.map_api = scenario.map_api        

    # get agent past tracks
    ego_agent_past, time_stamps_past = self.get_ego_agent()
    neighbor_agents_past, neighbor_agents_types = self.get_neighbor_agents()
    ego_agent_past, neighbor_agents_past, neighbor_indices = \
        agent_past_process(ego_agent_past, time_stamps_past, neighbor_agents_past, neighbor_agents_types, self.num_agents)

    # get vector set map
    vector_map = self.get_map()
    
    # get agent future tracks
    ego_agent_future = self.get_ego_agent_future()
    neighbor_agents_future = self.get_neighbor_agents_future(neighbor_indices)

    # gather data
    data = {"map_name": map_name, "token": token, "ego_agent_past": ego_agent_past, "ego_agent_future": ego_agent_future,
            "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future,
            "speed_limit": self.get_speed_limit()}
    data.update(vector_map)
    
    # save to disk
    self.save_to_disk(save_dir, data)
    

def ParallelWork(scenarios, save_dir):
    num_workers = 2 # min(cpu_count(), len(scenarios))  # use available CPUs or limit to the number of scenarios
    with Pool(num_workers) as pool:
        args = [(save_dir, scenario) for scenario in scenarios]
        for _ in tqdm(pool.imap_unordered(AtomWork, args), total=len(scenarios)):
            pass  # process the results if needed, currently just looping through for progress


def load_yaml_as_dict(file_path):
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process nuPlan scenarios into vectorized npz files')
    parser.add_argument('--data_path', type=str, required=True, help='nuplan-v1.1 split directory, e.g. .../nuplan-v1.1/splits/trainval')
    parser.add_argument('--map_path', type=str, required=True, help='nuPlan maps directory')
    parser.add_argument('--scenario_filter', type=str, required=True, help='scenario filter yaml (tuplan_garage/.../scenario_filter/*.yaml)')
    parser.add_argument('--save_path', type=str, required=True, help='output directory of the npz files')
    args = parser.parse_args()

    cfg = load_yaml_as_dict(args.scenario_filter)
    cfg.pop('_target_')
    cfg.pop('_convert_')

    # create save folder
    os.makedirs(args.save_path, exist_ok=True)

    # get scenarios
    map_version = "nuplan-maps-v1.0"
    sensor_root = None
    db_files = None
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version, scenario_mapping=scenario_mapping)
    scenario_filter = ScenarioFilter(**cfg)
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    print("loading scenarios")
    scenarios = builder.get_scenarios(scenario_filter, worker) # nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.NuPlanScenario
    print(f"Total number of scenarios: {len(scenarios)}")
    
    # process data
    del worker, builder, scenario_filter, scenario_mapping
    processor = DataProcessor(scenarios)
    processor.get_scenario_type(os.path.join(args.save_path, "scenario_types.json"))
    processor.work(args.save_path)
