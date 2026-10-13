import gc
import logging
import warnings
from typing import Dict, List, Optional, Type

import numpy as np
import requests

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

from tuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_planner import AbstractPDMPlanner
from tuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_closed_planner import DataProcessor
from tuplan_garage.planning.simulation.planner.lilo_configs import (
    FALLBACK_ID,
    build_planner,
    config_to_text,
    get_config,
)
from tuplan_garage.planning.simulation.planner.prompt_generator import (
    build_prompt,
    describe_scene,
    parse_answer,
)
from tuplan_garage.planning.simulation.planner.driving_memory import ScenarioMemory, flat_embedding

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)


class llm_planner(AbstractPDMPlanner):
    """
    LiloDriver: memory-augmented LLM agent that selects a PDM-Closed configuration (Sec. 3.5).

    Every T_LLM seconds the current scene is embedded, the K_r nearest memory entries are retrieved,
    the configurations of their clusters form the candidate set, and a selector picks one candidate:
        llm     the LLM chooses among the candidates (full system)
        greedy  the candidate whose cluster has the highest stored closed-loop score (non-LLM selector, Table 3 row 1)
        none    no memory, the fixed fallback configuration (Table 3 row 2)
    The selected configuration drives the behavior planner at 10 Hz until the next query.
    """
    requires_scenario: bool = True

    def __init__(
        self,
        map_radius: float,
        scenario: AbstractScenario,
        memory_path: str,
        selector: str = "llm",
        embedding: str = "encoder",
        top_k: int = 3,
        llm_period: float = 15.0,
        llm_url: str = "http://127.0.0.1:5000/predict",
        selection_log: Optional[str] = None,
    ):
        """
        :param map_radius: radius around ego to consider
        :param scenario: simulated scenario (used to build the vectorized scene inputs)
        :param memory_path: directory of the memory built by driving_memory.py
        :param selector: llm / greedy / none
        :param embedding: encoder (Scene Encoder service, agentGF.py) / flat (flattened perception features)
        :param top_k: number of retrieved examples K_r
        :param llm_period: LLM query period T_LLM [s]
        :param llm_url: endpoint of the LLM server (llama_server.py)
        :param selection_log: optional file to append the per-query selections to
        """
        super(llm_planner, self).__init__(map_radius)
        assert selector in ["llm", "greedy", "none"]
        assert embedding in ["encoder", "flat"]

        self.scenario = scenario
        self.selector = selector
        self.embedding = embedding
        self.top_k = top_k
        self.llm_period = llm_period
        self.llm_url = llm_url
        self.selection_log = selection_log

        self.memory = ScenarioMemory(memory_path) if selector != "none" else None
        self.agent_name = "classifier"  # shared memory of the Scene Encoder service
        self.P: Dict[int, AbstractPDMPlanner] = {}

    def initialize(self, initialization: PlannerInitialization) -> None:
        """ Inherited, see superclass. """
        self._map_api = initialization.map_api
        self._load_route_dicts(initialization.route_roadblock_ids)
        gc.collect()
        self.initialization = initialization
        self.query_interval = None
        self.best_planner = None

    def name(self) -> str:
        """ Inherited, see superclass. """
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """ Inherited, see superclass. """
        return DetectionsTracks  # type: ignore

    # ------------------------------------------------------------------
    # perception and scene encoding
    # ------------------------------------------------------------------
    def vector_request(self, vector_input: dict, feature_out = False):
        """ Send the vectorized scene to the Scene Encoder service and wait for the result. """
        from shared_memory_dict import SharedMemoryDict

        smd = SharedMemoryDict(name=self.agent_name, size=204800)
        while True:
            if smd['llm_ready'] == 1:
                smd['llm_ready'] = 0

                if smd['input'] == 0 and smd['return'] == 0: # ready for a new input
                    smd['ego_agent_past'] = vector_input['ego_agent_past']
                    smd['neighbor_agents_past'] = vector_input['neighbor_agents_past']
                    smd['map_lanes'] = vector_input['map_lanes']
                    smd['map_crosswalks'] = vector_input['map_crosswalks']
                    smd['route_lanes'] = vector_input['route_lanes']

                    smd['feature_out'] = feature_out
                    smd['input'] = 1 # request processing

                while True: # wait until the result is returned
                    if smd['input'] == 0 and smd['return'] == 1:
                        output = smd['feature'] if feature_out else smd["llm_output"]
                        smd['input'] = 0 # result read, ready for the next input
                        smd['return'] = 0
                        break

                smd['llm_ready'] = 1
                break

        return output

    def perceive(self, current_input: PlannerInput):
        """ Vectorized scene inputs x, scene embedding z, and the textual scene description. """
        vector_input = DataProcessor(self.scenario, current_input, pre_time=0).work()

        ego_state, _ = current_input.history.current_state
        try:
            speed_limit = self._get_starting_lane(ego_state).speed_limit_mps
        except Exception:
            speed_limit = None

        if self.embedding == "encoder":
            z = np.asarray(self.vector_request(vector_input, feature_out=True), dtype=np.float32)
        else:
            z = flat_embedding(vector_input)

        return vector_input, z, describe_scene(vector_input, speed_limit)

    # ------------------------------------------------------------------
    # retrieval and selection
    # ------------------------------------------------------------------
    def retrieve(self, z: np.ndarray):
        """ K_r nearest memory entries and the configurations of their clusters as candidates. """
        neighbors = self.memory.query(z, top_k=self.top_k)

        candidates: List[int] = []   # config ids, in order of retrieval
        scores: Dict[int, Optional[float]] = {}
        examples = []
        for neighbor in neighbors:
            metadata = neighbor["metadata"]
            config_id, score = self.memory.cluster_config(metadata["cluster"])
            if config_id not in candidates:
                candidates.append(config_id)
                scores[config_id] = score
            examples.append({
                "similarity": 1.0 - neighbor["distance"],
                "scenario_type": metadata["scenario_type"],
                "scenario_description": metadata["scenario_description"],
                "candidate_index": candidates.index(config_id),
                "config_text": config_to_text(get_config(config_id)),
                "score": score,
            })

        return candidates, scores, examples

    def llm_select(self, scene_description: str, candidates: List[int], examples: List[dict]):
        """ Ask the LLM for a candidate index; None if no valid index can be parsed. """
        prompt = build_prompt(
            scene_description,
            examples,
            [config_to_text(get_config(c)) for c in candidates],
            self.llm_period,
        )
        try:
            r = requests.post(self.llm_url, json={"prompt": prompt}).json()
            answer = r["prediction"] if r.get("success") else ""
        except Exception as e:
            logger.warning(f"LLM request failed: {e}")
            answer = ""

        return parse_answer(answer, len(candidates))

    def select_configuration(self, current_input: PlannerInput) -> int:
        if self.selector == "none":
            return FALLBACK_ID

        vector_input, z, scene_description = self.perceive(current_input)
        candidates, scores, examples = self.retrieve(z)

        if self.selector == "greedy":
            # highest stored closed-loop score from planner search; noise entries have no score
            index = int(np.argmax([-np.inf if scores[c] is None else scores[c] for c in candidates]))
        else:
            index = self.llm_select(scene_description, candidates, examples)
            if index is None:
                index = 0  # configuration of the nearest retrieved memory entry

        return candidates[index]

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def compute_planner_trajectory(self, current_input: PlannerInput) -> List[AbstractTrajectory]:
        '''
        The LLM is queried every T_LLM seconds; in between, the selected behavior planner runs at 10 Hz.
        current_input.iteration.index: Iteration in the simulation, starting from 0.
        '''
        if self.query_interval is None:
            self.query_interval = max(1, int(round(self.llm_period / self.scenario.database_interval)))

        if current_input.iteration.index % self.query_interval == 0:
            config_id = self.select_configuration(current_input)

            if self.selection_log is not None:
                with open(self.selection_log, "a") as f:
                    f.write(f"{self.scenario.token} {current_input.iteration.index} {config_id}\n")

            if config_id not in self.P:
                self.P[config_id] = build_planner(get_config(config_id))
            self.best_planner = self.P[config_id]
            self.best_planner.initialize(self.initialization)

        trajectory = self.best_planner.compute_planner_trajectory(current_input)
        return trajectory
