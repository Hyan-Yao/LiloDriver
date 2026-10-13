"""
Template-based prompt generator of the Reasoning and Execution module (Sec. 3.5, Appendix B).

The scene description is computed from the vectorized perception features (Sec. 3.2), all in the
ego frame at the current step:
    ego_agent_past        [T_h, 7]       x, y, heading, vx, vy, ax, ay
    neighbor_agents_past  [N_a, T_h, 11] x, y, heading, vx, vy, yaw rate, length, width, one-hot type
    map_lanes             [40, 50, 7]    x, y, heading, one-hot traffic light (green, yellow, red, unknown)
    map_crosswalks        [5, 30, 3]     x, y, heading
    route_lanes           [10, 50, 3]    x, y, heading
The same generator describes memory entries and the current scene, so both use one format.
"""
import re
from typing import Dict, List, Optional

import numpy as np

AGENT_TYPES = ["vehicle", "pedestrian", "cyclist"]
TRAFFIC_LIGHT_STATES = ["green", "yellow", "red", "unknown"]
NUM_DESCRIBED_AGENTS = 5     # closest N road agents in the description
STATIONARY_SPEED = 0.5       # [m/s]
TURNING_YAW_CHANGE = 0.1     # [rad] heading change over the last second
INTERSECTION_TURN = 0.5      # [rad] heading of the route ahead that counts as a turn
LANE_CHANGE_OFFSET = 1.75    # [m] lateral offset of the closest route lane that counts as a lane change
ROUTE_LOOKAHEAD = 40.0       # [m]

SYSTEM_PROMPT = (
    "You are the decision module of an autonomous vehicle in closed-loop simulation. "
    "A rule-based behavior planner (PDM-Closed with IDM) drives the car. "
    "Your only job is to choose ONE planner configuration from the candidate list for the next {period:g} seconds. "
    "Each configuration sets: lateral centerline offsets lo [m], minimum gap s0 [m], maximum acceleration a_m [m/s^2], "
    "comfortable deceleration b [m/s^2], and time-to-collision braking threshold th [s]. "
    "Prefer safety (no collisions, sufficient TTC), then progress along the route, then comfort. "
    "Think step by step, then output your final answer on the last line exactly as: Answer: <index>"
)

INSTRUCTION = (
    "Step 1: Identify the main risk in the current scene (e.g., lead vehicle, crossing pedestrian, merging traffic, stop line).\n"
    "Step 2: Compare the current scene with each example: which one is most similar in road layout, ego motion, and agent interaction?\n"
    "Step 3: Decide whether the situation calls for more cautious, similar, or more assertive behavior than that example.\n"
    "Step 4: Select the candidate configuration that best matches your decision.\n"
    "Answer: <index>\n"
)


def normalize_vector_input(data) -> Dict[str, np.ndarray]:
    """Accept both the simulation keys (map_lanes, ...) and the processed npz keys (lanes, ...)."""
    def get(*keys):
        for k in keys:
            if k in data:
                return np.asarray(data[k])
        raise KeyError(keys)

    return {
        "ego_agent_past": get("ego_agent_past"),
        "neighbor_agents_past": get("neighbor_agents_past"),
        "map_lanes": get("map_lanes", "lanes"),
        "map_crosswalks": get("map_crosswalks", "crosswalks"),
        "route_lanes": get("route_lanes"),
    }


def _valid_points(polyline: np.ndarray) -> np.ndarray:
    return polyline[np.abs(polyline[:, :2]).sum(-1) > 0]


def _ego_text(ego: np.ndarray) -> str:
    vx, vy, ax, ay = ego[-1, 3], ego[-1, 4], ego[-1, 5], ego[-1, 6]
    speed = float(np.hypot(vx, vy))
    accel = float(ax)  # longitudinal acceleration in the ego frame
    yaw_change = float(ego[-1, 2] - ego[max(0, len(ego) - 11), 2])
    yaw_change = (yaw_change + np.pi) % (2 * np.pi) - np.pi
    if yaw_change > TURNING_YAW_CHANGE:
        heading = "turning left"
    elif yaw_change < -TURNING_YAW_CHANGE:
        heading = "turning right"
    else:
        heading = "straight"
    return f"Ego: speed {speed:.1f} m/s, acceleration {accel:.1f} m/s^2, heading {heading}."


def _route_text(route_lanes: np.ndarray, speed_limit: Optional[float]) -> str:
    lanes = [_valid_points(lane) for lane in route_lanes]
    lanes = [lane for lane in lanes if len(lane) > 0]
    n_ahead = sum(1 for lane in lanes if np.any(lane[:, 0] > 0))

    maneuver = "keep lane"
    if len(lanes) > 0:
        points = np.concatenate(lanes, axis=0)
        closest = points[np.argmin(np.linalg.norm(points[:, :2], axis=-1))]
        ahead = points[(points[:, 0] > 0) & (np.linalg.norm(points[:, :2], axis=-1) < ROUTE_LOOKAHEAD)]
        if abs(closest[1]) > LANE_CHANGE_OFFSET:
            maneuver = "lane change left" if closest[1] > 0 else "lane change right"
        elif len(ahead) > 0:
            headings = (ahead[:, 2] + np.pi) % (2 * np.pi) - np.pi
            if headings.max() > INTERSECTION_TURN:
                maneuver = "left turn at intersection"
            elif headings.min() < -INTERSECTION_TURN:
                maneuver = "right turn at intersection"

    limit = f"{speed_limit:.1f} m/s" if speed_limit is not None else "unknown"
    return f"Route: {n_ahead} route lanes ahead; next maneuver {maneuver}; speed limit {limit}."


def _traffic_light_text(map_lanes: np.ndarray, route_lanes: np.ndarray) -> str:
    route_points = [_valid_points(lane)[:, :2] for lane in route_lanes]
    route_points = [p for p in route_points if len(p) > 0]
    if len(route_points) == 0:
        return "Traffic lights: none."
    route_points = np.concatenate(route_points, axis=0)

    best = None
    for lane in map_lanes:
        lane = _valid_points(lane)
        if len(lane) == 0:
            continue
        state = int(np.argmax(lane[0, 3:7]))
        if lane[0, 3:7].sum() == 0 or TRAFFIC_LIGHT_STATES[state] == "unknown":
            continue
        # the lane must lie on the ego route and start ahead of the ego vehicle
        dist_to_route = np.linalg.norm(lane[:, None, :2] - route_points[None], axis=-1).min()
        if dist_to_route > 0.5 or lane[0, 0] < 0:
            continue
        distance = float(np.linalg.norm(lane[0, :2]))
        if best is None or distance < best[1]:
            best = (TRAFFIC_LIGHT_STATES[state], distance)

    if best is None:
        return "Traffic lights: none."
    return f"Traffic lights: {best[0]} at {best[1]:.1f} m."


def _crosswalk_text(map_crosswalks: np.ndarray) -> str:
    crosswalks = [_valid_points(c) for c in map_crosswalks]
    crosswalks = [c for c in crosswalks if len(c) > 0]
    if len(crosswalks) == 0:
        return "Crosswalks: 0."
    nearest = min(float(np.linalg.norm(c[:, :2], axis=-1).min()) for c in crosswalks)
    return f"Crosswalks: {len(crosswalks)}, nearest at {nearest:.1f} m."


def _agents_text(ego: np.ndarray, neighbors: np.ndarray) -> str:
    current = neighbors[:, -1]
    valid = np.abs(current[:, :2]).sum(-1) > 0
    agents = current[valid]
    if len(agents) == 0:
        return "Nearby agents: none."

    distance = np.linalg.norm(agents[:, :2], axis=-1)
    order = np.argsort(distance)[:NUM_DESCRIBED_AGENTS]
    ego_velocity = ego[-1, 3:5]

    lines = [f"Nearby agents (closest {len(order)} by distance):"]
    for i in order:
        x, y, vx, vy = agents[i, 0], agents[i, 1], agents[i, 3], agents[i, 4]
        agent_type = AGENT_TYPES[int(np.argmax(agents[i, 8:11]))]
        speed = float(np.hypot(vx, vy))
        if speed < STATIONARY_SPEED:
            motion = "stationary"
        else:
            relative_velocity = np.array([vx, vy]) - ego_velocity
            radial = float(np.dot(relative_velocity, [x, y]) / max(distance[i], 1e-3))
            motion = "approaching" if radial < 0 else "receding"
        longitudinal = "ahead" if x >= 0 else "behind"
        lateral = "left" if y >= 0 else "right"
        lines.append(
            f"  - {agent_type} at {abs(x):.1f} m {longitudinal}, {abs(y):.1f} m {lateral}, speed {speed:.1f} m/s, {motion}"
        )
    return "\n".join(lines)


def describe_scene(vector_input, speed_limit: Optional[float] = None) -> str:
    """Textual scene description (part ii of the prompt)."""
    data = normalize_vector_input(vector_input)
    return "\n".join([
        _ego_text(data["ego_agent_past"]),
        _route_text(data["route_lanes"], speed_limit),
        _traffic_light_text(data["map_lanes"], data["route_lanes"]),
        _crosswalk_text(data["map_crosswalks"]),
        _agents_text(data["ego_agent_past"], data["neighbor_agents_past"]),
    ])


def build_prompt(scene_description: str, examples: List[Dict], candidates: List[str], period: float) -> str:
    """
    Four-part prompt: (i) system prompt, (ii) current scene, (iii) K_r retrieved examples with the
    configuration of their cluster, (iv) chain-of-thought instruction. The candidate list is shown with indices.
    :param examples: dicts with keys similarity, scenario_type, scenario_description, candidate_index,
                     config_text, score (None for noise entries that use the fallback configuration)
    :param candidates: text of each candidate configuration, in index order
    """
    indent = lambda text: "\n".join("  " + line for line in text.splitlines())

    prompt = "[System]\n" + SYSTEM_PROMPT.format(period=period) + "\n\n"
    prompt += "[Current scene]\n" + scene_description + "\n\n"

    prompt += "[Similar past scenarios from memory]\n"
    for j, example in enumerate(examples, start=1):
        score = "n/a (fallback configuration)" if example["score"] is None else f"{example['score']:.2f}"
        prompt += f"Example {j} (similarity {example['similarity']:.2f}, scenario type {example['scenario_type']}):\n"
        prompt += indent(example["scenario_description"]) + "\n"
        prompt += (
            f"  Best configuration for its cluster: [{example['candidate_index']}] {example['config_text']}, "
            f"mean closed-loop score {score}.\n"
        )
    prompt += "\n"

    prompt += "[Candidate configurations]\n"
    for i, candidate in enumerate(candidates):
        prompt += f"[{i}] {candidate}\n"
    prompt += "\n"

    prompt += "[Instruction]\n" + INSTRUCTION
    return prompt


def parse_answer(text: str, num_candidates: int) -> Optional[int]:
    """Parse 'Answer: <index>' from the last non-empty line; None if no valid index is present."""
    lines = [line.strip() for line in str(text).strip().splitlines() if line.strip()]
    if len(lines) == 0:
        return None
    match = re.search(r"Answer:\s*\[?(\d+)\]?", lines[-1])
    if match is None:
        return None
    index = int(match.group(1))
    return index if 0 <= index < num_candidates else None
