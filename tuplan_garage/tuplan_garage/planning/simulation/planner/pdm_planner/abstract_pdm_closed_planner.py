from typing import List, Optional
import pdb
import numpy as np
import os
import logging

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from tuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_planner import (
    AbstractPDMPlanner,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.pdm_generator import (
    PDMGenerator,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.pdm_proposal import (
    PDMProposalManager,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_emergency_brake import (
    PDMEmergencyBrake,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    parallel_discrete_path,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

from tuplan_garage.planning.simulation.planner.pdm_planner.diffusion_utils import convert_to_local
from tuplan_garage.planning.simulation.planner.pdm_planner.helpers import (
    Vehicle, 
    VehicleGraph,
    Lane,
    LaneGraph
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
import matplotlib.pyplot as plt
import textwrap

logger = logging.getLogger(__name__)


from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import sampled_past_ego_states_to_tensor, sampled_past_timestamps_to_tensor
from tuplan_garage.planning.simulation.planner.data_utils import agent_past_process, map_process, get_neighbor_vector_set_map, sampled_tracked_objects_to_tensor_list


class DataProcessor(object):
    def __init__(self, scenario, current_input: PlannerInput, pre_time: float, ego_future_prediction: list[EgoState] = None):
        self.scenario = scenario
        self.current_input = current_input

        self.past_time_horizon = 2 # [seconds]
        self.pre_time = pre_time # 0: -2.0 ~ 0.0s, 10: -1.0 ~ 1.0s, 20: 0.0 ~ 2.0s
        self.num_past_poses = 10 * self.past_time_horizon
        self.time_interval = 0.1
        self.num_agents = 20

        self._map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK'] # name of map features to be extracted.
        self._max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': 50, 'ROUTE_LANES': 50, 'CROSSWALK': 30} # maximum number of points per feature to extract per feature layer.
        self._radius = 60 # [m] query radius scope relative to the current pose.
        self._interpolation_method = 'linear' # Interpolation method to apply when interpolating to maintain fixed size map elements.
        
        self.ego_future_prediction = ego_future_prediction

    def get_ego_agent(self):
        '''
        self.anchor_ego_state = self.scenario.initial_ego_state
        past_ego_states = self.scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        )
        sampled_past_ego_states = list(past_ego_states) + [self.anchor_ego_state]
        '''
        if self.ego_future_prediction == None:
            sampled_past_ego_states = self.current_input.history.ego_states[- self.num_past_poses - 1 : ]
        else:
            timing_ego_states = self.current_input.history.ego_states[- self.num_past_poses - 1 : ] + self.ego_future_prediction
            sampled_past_ego_states = timing_ego_states[self.pre_time : self.pre_time + self.num_past_poses + 1]

        past_ego_states_tensor = sampled_past_ego_states_to_tensor(sampled_past_ego_states)
        past_time_stamps = list(
            self.scenario.get_past_timestamps(
                iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
            )
        ) + [self.scenario.start_time]

        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

        return past_ego_states_tensor, past_time_stamps_tensor
    
    def get_neighbor_agents(self):
        '''
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]
        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        '''
        
        observations = self.current_input.history.observations[- self.num_past_poses - 1 : ]
        sampled_past_observations = [observation.tracked_objects for observation in observations]

        past_tracked_objects_tensor_list, past_tracked_objects_types = \
              sampled_tracked_objects_to_tensor_list(sampled_past_observations)
    
        '''
        timing_tracked_objects_tensor_list = past_tracked_objects_tensor_list
        past_length = len(timing_tracked_objects_tensor_list)
        for t in range(int(self.pre_time)):
            next_states = timing_tracked_objects_tensor_list[past_length + t - 1]
            for idx in range(len(next_states)):
                vx = next_states[idx][1]
                vy = next_states[idx][2]
                next_states[idx][6] += vx * self.time_interval # x
                next_states[idx][7] += vy * self.time_interval # y
            timing_tracked_objects_tensor_list.append(next_states)
        return timing_tracked_objects_tensor_list[self.pre_time : self.pre_time + past_length], past_tracked_objects_types
        '''
        return past_tracked_objects_tensor_list, past_tracked_objects_types

    def get_map(self):        
        # ego_state = self.scenario.initial_ego_state
        ego_state, _ = self.current_input.history.current_state

        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(self.current_input.iteration.index)

        coords, traffic_light_data = get_neighbor_vector_set_map(
            self.map_api, self._map_features, ego_coords, self._radius, route_roadblock_ids, traffic_light_data
        )

        vector_map = map_process(ego_state.rear_axle, coords, traffic_light_data, self._map_features, 
                                 self._max_elements, self._max_points, self._interpolation_method)

        return vector_map

    def work(self):
        self.map_api = self.scenario.map_api        

        # get agent past tracks
        ego_agent_past, time_stamps_past = self.get_ego_agent()
        neighbor_agents_past, neighbor_agents_types = self.get_neighbor_agents()
        ego_agent_past, neighbor_agents_past, neighbor_indices = \
            agent_past_process(ego_agent_past, time_stamps_past, neighbor_agents_past, neighbor_agents_types, self.num_agents)

        # get vector set map
        vector_map = self.get_map()

        # gather data
        data = {"ego_agent_past": ego_agent_past, "neighbor_agents_past": neighbor_agents_past}
        data.update(vector_map)
        
        return data


class AbstractPDMClosedPlanner(AbstractPDMPlanner):
    """
    Interface for planners incorporating PDM-Closed. Used for PDM-Closed and PDM-Hybrid.
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        map_radius: float,
        ttc_threshold: float,
    ):
        """
        Constructor for AbstractPDMClosedPlanner
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        """

        super(AbstractPDMClosedPlanner, self).__init__(map_radius)

        assert (
            trajectory_sampling.interval_length == proposal_sampling.interval_length
        ), "AbstractPDMClosedPlanner: Proposals and Trajectory must have equal interval length!"

        # config parameters
        self._trajectory_sampling: int = trajectory_sampling
        self._proposal_sampling: int = proposal_sampling
        self._idm_policies: BatchIDMPolicy = idm_policies
        self._lateral_offsets: Optional[List[float]] = lateral_offsets

        # observation/forecasting class
        self._observation = PDMObservation(
            trajectory_sampling, proposal_sampling, map_radius
        )

        # proposal/trajectory related classes
        self._generator = PDMGenerator(trajectory_sampling, proposal_sampling)
        self._simulator = PDMSimulator(proposal_sampling)
        self._scorer = PDMScorer(proposal_sampling)
        self._emergency_brake = PDMEmergencyBrake(trajectory_sampling, time_to_infraction_threshold=ttc_threshold)

        # lazy loaded
        self._proposal_manager: Optional[PDMProposalManager] = None

    def _update_proposal_manager(self, ego_state: EgoState):
        """
        Updates or initializes PDMProposalManager class
        :param ego_state: state of ego-vehicle
        """

        current_lane = self._get_starting_lane(ego_state)

        # TODO: Find additional conditions to trigger re-planning
        create_new_proposals = self._iteration == 0

        if create_new_proposals:
            proposal_paths: List[PDMPath] = self._get_proposal_paths(current_lane)

            self._proposal_manager = PDMProposalManager(
                lateral_proposals=proposal_paths,
                longitudinal_policies=self._idm_policies,
            )

        # update proposals
        self._proposal_manager.update(current_lane.speed_limit_mps)

    def _get_proposal_paths(
        self, current_lane: LaneGraphEdgeMapObject
    ) -> List[PDMPath]:
        """
        Returns a list of path's to follow for the proposals. Inits a centerline.
        :param current_lane: current or starting lane of path-planning
        :return: lists of paths (0-index is centerline)
        """
        centerline_discrete_path = self._get_discrete_centerline(current_lane)
        self._centerline = PDMPath(centerline_discrete_path)

        # 1. save centerline path (necessary for progress metric)
        output_paths: List[PDMPath] = [self._centerline]

        # 2. add additional paths with lateral offset of centerline
        if self._lateral_offsets is not None:
            for lateral_offset in self._lateral_offsets:
                offset_discrete_path = parallel_discrete_path(
                    discrete_path=centerline_discrete_path, offset=lateral_offset
                )
                output_paths.append(PDMPath(offset_discrete_path))

        return output_paths

    def _get_closed_loop_trajectory(
        self,
        current_input: PlannerInput,
        pred_paras: dict = None,
    ) -> InterpolatedTrajectory:
        """
        Creates the closed-loop trajectory for PDM-Closed planner.
        :param current_input: planner input
        :return: trajectory
        """
        self.ego_state, observation = current_input.history.current_state

        # 1. Environment forecast and observation update
        self._observation.update(
            self.ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )

        # 2. Centerline extraction and proposal update
        self._update_proposal_manager(self.ego_state)

        # 3. Generate/Unroll proposals
        proposals_array = self._generator.generate_proposals(
            self.ego_state, self._observation, self._proposal_manager
        )

        # 4. Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(
            proposals_array, self.ego_state
        )

        # 5. Score proposals
        proposal_scores = self._scorer.score_proposals(
            simulated_proposals_array,
            self.ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._drivable_area_map,
            self._map_api,
        )

        # 6.a Apply brake if emergency is expected
        trajectory = self._emergency_brake.brake_if_emergency(
            self.ego_state, proposal_scores, self._scorer
        )

        # 6.b Otherwise, extend and output best proposal
        if trajectory is None:
            trajectory = self._generator.generate_trajectory(np.argmax(proposal_scores))

        if pred_paras != None:
            self.scenario = pred_paras['scenario']
            self.update_graphs(current_input)
            self.vectorize_scenario(current_input, trajectory, pred_paras['pre_time'])
            self.build_representation()
            return proposal_scores, simulated_proposals_array, self.prompt_input, self.vector_input
        else:
            return trajectory
    

    def build_representation(self):
        scenario_description = textwrap.dedent("""You are driving on a road with multiple lanes. 
        """)
        # The image shows the road layout, where multiple lanes are separated by black lines. The purple lane represents the current lane, while the yellow lane is the left lane and the orange lane is the right lane.
        # The rectangles represents vehicles. The blue rectangle is the ego autonomous vehicles you are driving. The red vehicles are human drivers around you. Several numbers are scattered on the vehicles and lanes to indicate lane identifiers and vehicle identifiers.\n

        ego_speed = self.ego_vehicle.speed
        scenario_description += textwrap.dedent(f"""- You are currently driving at speed {round(ego_speed,1)} m/s({self.ego_vehicle.get_speed_level()}).\n""")
        
        close_description = ""
        medium_description = ""
        far_description = ""
        for vidx, vehicle in enumerate(self.vehicle_graph.get_vehicles()):
            dis_to_ego = vehicle.distance_to(self.ego_vehicle)
            if dis_to_ego < 50:
                agent_speed = vehicle.speed
                
                is_ahead = vehicle.is_ahead_of(self.ego_vehicle)
                left_right = self.ego_vehicle.relative_position(vehicle)

                if dis_to_ego < 10:
                    description = f'''- Vehicle {vidx} is driving on the {left_right} lane, {is_ahead} of you, moving at {round(agent_speed,1)} m/s({vehicle.get_speed_level()}), and nearby ego vehicle (only {round(dis_to_ego,1)}m away).'''
                    close_description += description + "\n"
                elif dis_to_ego < 30:
                    description = f'''- Vehicle {vidx} is driving on the {left_right} lane, {is_ahead} of you, moving at {round(agent_speed,1)} m/s({vehicle.get_speed_level()}), and within medium range with ego vehicle ({round(dis_to_ego,1)}m away).'''
                    medium_description += description + "\n"
                else:
                    description = f'''- Vehicle {vidx} is driving on the {left_right} lane, {is_ahead} of you, moving at {round(agent_speed,1)} m/s({vehicle.get_speed_level()}), and far away with ego vehicle ({round(dis_to_ego,1)}m away).'''
                    far_description += description + "\n"

        # 总结各个距离区间的车辆情况
        if close_description:
            scenario_description += "Close vehicles (within 10 meters):\n" + close_description + "\n"
        else:
            scenario_description += "There are no vehicles nearby ego vehicle.\n\n"

        if medium_description:
            scenario_description += "Medium-range vehicles (between 10 and 30 meters):\n" + medium_description + "\n"
        else:
            scenario_description += "There are no vehicles in the medium range on ego vehicle.\n\n"

        if far_description:
            scenario_description += "Far vehicles (between 30 and 50 meters):\n" + far_description + "\n"
        else:
            scenario_description += "There are no vehicles far away.\n\n"

        self.prompt_input = {
            'scenario_description': scenario_description,
        }

        history_dir = os.environ.get("LILO_HISTORY_DIR")
        if history_dir is not None:
            history_file = os.path.join(history_dir, self.scenario.token + ".txt")
            with open(history_file, "w") as f:
                f.write(scenario_description)
            

    def vectorize_scenario(self, current_input: PlannerInput, ego_proposal: InterpolatedTrajectory, pre_time: float):
        ego_future_prediction = ego_proposal.get_sampled_trajectory()

        DataPro = DataProcessor(self.scenario, current_input, pre_time, ego_future_prediction)
        self.vector_input = DataPro.work()

        ego_agent_past_flat = self.vector_input['ego_agent_past'].flatten()
        neighbor_agents_past_flat = self.vector_input['neighbor_agents_past'].flatten()
        map_lanes_flat = self.vector_input['map_lanes'].flatten()
        map_crosswalks_flat = self.vector_input['map_crosswalks'].flatten()
        route_lanes_flat = self.vector_input['route_lanes'].flatten()

        # 合并所有展平后的数组
        self.flatten_vector = np.concatenate([  # (20717)
            ego_agent_past_flat,
            neighbor_agents_past_flat,
            map_lanes_flat,
            map_crosswalks_flat,
            route_lanes_flat
        ])
        

    def update_graphs(self, current_input):
        self.lane_graph, lane_object_to_node = self.generate_lane_graph()
        try:
            self.current_lane = lane_object_to_node[self._get_starting_lane(self.ego_state).id]
        except KeyError:
            print("key Error")
            # self.current_lane = lane_object_to_node[list(lane_object_to_node.keys())[0]] # pseudo lane
        self.ego_vehicle, self.vehicle_graph = self.generate_vehicle_graph(current_input)

        for vehicle in self.vehicle_graph.get_vehicles():
            closest_lane = vehicle.get_closest_lane(self.lane_graph)
            closest_lane.vehicles.append(vehicle)
    
    def generate_lane_graph(self):
        map_object_types = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR] # self._map_api.get_available_map_objects()
        map_objects = self._map_api.get_proximal_map_objects(self.ego_state.center.point, 100.0, layers=map_object_types)
        nodes = []
        for map_object_type in map_object_types:
            for map_object in map_objects[map_object_type]:
                # if map_object.id in self._route_lane_dict:
                nodes.append(map_object)
    
        # Parse map objects to get lane graph
        nodes = [Lane(node) for node in nodes]
        lane_object_to_node = {node.lane_object.id: node for node in nodes}

        # Extract connectivity information from NuPlan map objects
        for node in nodes:
            node.incoming = [
                lane_object_to_node[obj.id] for obj in node.lane_object.incoming_edges
                if obj.id in lane_object_to_node
            ]
            node.outgoing = [
                lane_object_to_node[obj.id] for obj in node.lane_object.outgoing_edges
                if obj.id in lane_object_to_node
            ]
            node.adjacent = [
                lane_object_to_node[obj.id] if (obj is not None) and (obj.id in lane_object_to_node) else None
                for obj in node.lane_object.adjacent_edges
            ]

        lane_graph = LaneGraph(nodes)
        return lane_graph, lane_object_to_node

    def generate_vehicle_graph(self, current_input):
        observation = current_input.history.current_state[1]
        vehicles = []
        for object in observation.tracked_objects:
            if object.tracked_object_type == TrackedObjectType.VEHICLE:
                vehicles.append(object)
        vehicles = [Vehicle(obj) for obj in vehicles]
        
        ego_vehicle = Vehicle(self.ego_state.agent)
        vehicles = [vehicle for vehicle in vehicles if vehicle.distance_to(ego_vehicle) < 100]
        vehicle_graph = VehicleGraph(vehicles)
        return ego_vehicle, vehicle_graph
    
    def save_map_figure(self, scenario_token):
        fig = plt.figure(figsize=(20,20))

        # Lane graph
        for i, lane in enumerate(self.lane_graph.get_lanes()):
            centerline = convert_to_local(self.ego_state, lane.centerline)
            plt.plot(centerline[:,0], centerline[:,1], color='black', linewidth=2)
            median_idx = centerline.shape[0] // 2
            # plt.text(centerline[median_idx,0], centerline[median_idx,1], f'{i}')

            lane.median_idx = median_idx

        left_lane = self.current_lane.get_left_lane()
        right_lane = self.current_lane.get_right_lane()

        lanes = [self.current_lane, left_lane, right_lane]
        colors = ['purple', 'yellow', 'orange']

        for color, lane in zip(colors, lanes):
            if lane is None:
                continue
            centerline = convert_to_local(self.ego_state, lane.extend_centerline())
            plt.plot(centerline[:,0], centerline[:,1], color=color, linewidth=2)

        # Vehicle graph
        for vidx, vehicle in enumerate(self.vehicle_graph.get_vehicles()):
            position = convert_to_local(self.ego_state, vehicle.position)
            plt.scatter(position[0], position[1], s=50, marker='s', color='red')
            plt.text(position[0], position[1], str(vidx))

        ego_position = convert_to_local(self.ego_state, self.ego_vehicle.position)
        plt.scatter(ego_position[0], ego_position[0], s=50, marker='s', color='blue')
        plt.text(ego_position[0], ego_position[1], 'ego')

        # scale = 20
        # plt.xlim(ego_position[0] - scale, ego_position[0] + scale)
        # plt.ylim(ego_position[1] - scale, ego_position[1] + scale)

        map_path = os.path.join(os.environ.get("LILO_HISTORY_DIR", "."), scenario_token + ".png")
        plt.savefig(map_path)
        plt.close(fig)
