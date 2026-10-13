"""
Closed-loop planner search of LiloDriver (Sec. 3.4, Algorithm 1, step 2).

Every configuration theta in the grid Theta (2 x 3^4 = 162) is simulated with nuPlan's non-reactive
closed-loop simulation on the training-split scenarios of the clusters that need a search
(written by `driving_memory.py pending`). The per-scenario scores are saved to
<score_dir>/config_XXX.json and turned into theta_k* by `driving_memory.py search`.
"""
import os
import json
import argparse
import subprocess

import yaml
import pandas as pd
from tqdm import tqdm

from tuplan_garage.planning.simulation.planner.lilo_configs import (
    THETA,
    HEADWAY_TIME,
    ACCELERATION_EXPONENT,
    SPEED_LIMIT_FRACTION,
    FALLBACK_TARGET_VELOCITY,
    MAP_RADIUS,
    lateral_offsets,
)

CONFIG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuplan_garage/planning/script/config")
PLANNER_YAML = os.path.join(CONFIG_ROOT, "simulation/planner/grid_planner.yaml")
FILTER_YAML = os.path.join(CONFIG_ROOT, "common/scenario_filter/memory_search.yaml")
CHALLENGE = "closed_loop_nonreactive_agents"


def generate_planner_yaml(theta):
    config = {
        'grid_planner': {
            '_target_': 'tuplan_garage.planning.simulation.planner.pdm_planner.grid_planner.PDMClosedPlanner',
            '_convert_': 'all',
            'trajectory_sampling': {
                '_target_': 'nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling',
                '_convert_': 'all',
                'num_poses': 80,
                'interval_length': 0.1,
            },
            'proposal_sampling': {
                '_target_': 'nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling',
                '_convert_': 'all',
                'num_poses': 40,
                'interval_length': 0.1,
            },
            'idm_policies': {
                '_target_': 'tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy.BatchIDMPolicy',
                '_convert_': 'all',
                'speed_limit_fraction': SPEED_LIMIT_FRACTION,
                'fallback_target_velocity': FALLBACK_TARGET_VELOCITY,
                'min_gap_to_lead_agent': float(theta['s0']),   # s0 [m]
                'headway_time': HEADWAY_TIME,                  # T_hw [s]
                'accel_max': float(theta['a_m']),              # a_m [m/s^2]
                'decel_max': float(theta['b']),                # b [m/s^2]
                'acceleration_exponent': ACCELERATION_EXPONENT,  # delta
            },
            'lateral_offsets': lateral_offsets(theta),  # null: centerline only; [-1, 1]: centerline and +-1 m
            'map_radius': MAP_RADIUS,
            'ttc_threshold': float(theta['th']),        # th [s]
        }
    }
    with open(PLANNER_YAML, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)


def generate_filter_yaml(tokens):
    """Scenario filter with the training-split scenarios to be searched."""
    config = {
        '_target_': 'nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter',
        '_convert_': 'all',
        'scenario_types': None,
        'scenario_tokens': list(tokens),
        'log_names': None,
        'map_names': None,
        'num_scenarios_per_type': None,
        'limit_total_scenarios': None,
        'timestamp_threshold_s': 15,
        'ego_displacement_minimum_m': None,
        'ego_start_speed_threshold': None,
        'ego_stop_speed_threshold': None,
        'speed_noise_tolerance': None,
        'expand_scenarios': None,
        'remove_invalid_goals': True,
        'shuffle': False,
    }
    with open(FILTER_YAML, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def run_simulation(experiment_name, exp_root, threads):
    nuplan_devkit_root = os.environ.get("NUPLAN_DEVKIT_ROOT", os.path.expanduser("~") + "/nuplan-devkit")
    os.environ["NUPLAN_EXP_ROOT"] = exp_root

    command = [
        "python", f"{nuplan_devkit_root}/nuplan/planning/script/run_simulation.py",
        f"+simulation={CHALLENGE}",
        "planner=grid_planner",
        "scenario_filter=memory_search",
        "scenario_builder=nuplan",  # nuPlan training split
        f"worker.threads_per_node={threads}",
        "verbose=true",
        f"experiment_uid={experiment_name}",
        "hydra.searchpath=[pkg://tuplan_garage.planning.script.config.common,"
        " pkg://tuplan_garage.planning.script.config.simulation,"
        " pkg://nuplan.planning.script.config.common,"
        " pkg://nuplan.planning.script.experiments]"
    ]
    subprocess.run(command, check=True)


def get_scenario_scores(exp_root, experiment_name):
    """{token: closed-loop score} from the aggregator metric of one simulation run."""
    parquet_path = os.path.join(exp_root, "exp/simulation", CHALLENGE, experiment_name, "aggregator_metric/")
    parquet_file = os.path.join(parquet_path, os.listdir(parquet_path)[0])
    df = pd.read_parquet(parquet_file)
    df = df[df['scenario'] != 'final_score']
    return {str(row['scenario']): float(row['score']) for _, row in df.iterrows()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closed-loop grid search over Theta")
    parser.add_argument("--tokens", type=str, required=True, help="json list of scenario tokens (driving_memory.py pending)")
    parser.add_argument("--exp_root", type=str, required=True, help="nuPlan experiment root for the simulations")
    parser.add_argument("--score_dir", type=str, required=True, help="output directory of config_XXX.json")
    parser.add_argument("--threads", type=int, default=256)
    args = parser.parse_args()

    with open(args.tokens) as f:
        tokens = json.load(f)
    if len(tokens) == 0:
        print("no cluster needs a planner search")
        raise SystemExit(0)
    os.makedirs(args.score_dir, exist_ok=True)
    generate_filter_yaml(tokens)

    for config_id, theta in enumerate(tqdm(THETA)):
        experiment_name = f"grid_config_{config_id:03d}_{len(tokens)}"
        generate_planner_yaml(theta)
        run_simulation(experiment_name, args.exp_root, args.threads)

        # merge with the scores of earlier memory updates
        res_file = os.path.join(args.score_dir, f"config_{config_id:03d}.json")
        scores = {}
        if os.path.exists(res_file):
            with open(res_file) as f:
                scores = json.load(f)["scores"]
        scores.update(get_scenario_scores(args.exp_root, experiment_name))
        with open(res_file, 'w') as f:
            json.dump({"config_id": config_id, "theta": theta, "scores": scores}, f)
