"""
Behavior-planner configurations of LiloDriver (Sec. 3.5 and Appendix A).

A configuration theta = (lo, s0, a_m, b, th) parameterizes a PDM-Closed planner:
    lo   lateral centerline offsets of the proposals [m]: {0} or {-1, 0, 1}
    s0   minimum gap to the leading agent [m]
    a_m  maximum acceleration [m/s^2]
    b    comfortable deceleration [m/s^2]
    th   time-to-collision threshold of the emergency brake [s]
IDM uses a fixed time headway T_hw = 1.5 s and acceleration exponent delta = 4.
The grid Theta has 2 x 3^4 = 162 configurations; a configuration is referred to by its index in Theta.
"""
import itertools
from typing import Dict, List, Optional

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy

# parameter grid Theta
LATERAL_OFFSET_SETS = [[0.0], [-1.0, 0.0, 1.0]]  # [m]
MIN_GAP_LIST = [1.0, 2.0, 3.0]                    # s0 [m]
ACCEL_MAX_LIST = [1.0, 1.5, 2.0]                  # a_m [m/s^2]
DECEL_LIST = [2.0, 3.0, 4.0]                      # b [m/s^2]
TTC_LIST = [1.0, 2.0, 3.0]                        # th [s]

# fixed IDM parameters
HEADWAY_TIME = 1.5          # T_hw [s]
ACCELERATION_EXPONENT = 4.0 # delta

# PDM-Closed settings shared by all configurations
SPEED_LIMIT_FRACTION = [0.2, 0.4, 0.6, 0.8, 1.0]
FALLBACK_TARGET_VELOCITY = 15.0
MAP_RADIUS = 50.0

THETA: List[Dict] = [
    {"lo": lo, "s0": s0, "a_m": a_m, "b": b, "th": th}
    for lo, s0, a_m, b, th in itertools.product(
        LATERAL_OFFSET_SETS, MIN_GAP_LIST, ACCEL_MAX_LIST, DECEL_LIST, TTC_LIST
    )
]
assert len(THETA) == 162

# Fixed fallback configuration of the executor, used for DBSCAN noise points and
# for the memory-free ablation. It is not the published PDM-Closed baseline.
FALLBACK_CONFIG = {"lo": [-1.0, 0.0, 1.0], "s0": 1.0, "a_m": 1.5, "b": 3.0, "th": 2.0}
FALLBACK_ID = THETA.index(FALLBACK_CONFIG)


def get_config(config_id: int) -> Dict:
    return THETA[int(config_id)]


def lateral_offsets(theta: Dict) -> Optional[List[float]]:
    """PDM-Closed always rolls out the centerline; only the non-zero offsets are passed."""
    offsets = [float(o) for o in theta["lo"] if o != 0.0]
    return offsets if len(offsets) > 0 else None


def build_planner(theta: Dict) -> PDMClosedPlanner:
    """Instantiate the PDM-Closed behavior planner of configuration theta."""
    trajectory_sampling = TrajectorySampling(num_poses=80, interval_length=0.1)
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    idm_policies = BatchIDMPolicy(
        speed_limit_fraction=SPEED_LIMIT_FRACTION,
        fallback_target_velocity=FALLBACK_TARGET_VELOCITY,
        min_gap_to_lead_agent=float(theta["s0"]),
        headway_time=HEADWAY_TIME,
        accel_max=float(theta["a_m"]),
        decel_max=float(theta["b"]),
        acceleration_exponent=ACCELERATION_EXPONENT,
    )
    return PDMClosedPlanner(
        trajectory_sampling,
        proposal_sampling,
        idm_policies,
        lateral_offsets(theta),
        MAP_RADIUS,
        float(theta["th"]),
    )


def behavior_tag(theta: Dict) -> str:
    """Short tag (cautious / nominal / assertive) derived from the parameters; each term is in {-1, 0, 1}."""
    assertiveness = (
        (theta["a_m"] - 1.5) / 0.5      # stronger acceleration
        + (2.0 - theta["s0"])           # smaller minimum gap
        + (2.0 - theta["th"])           # later emergency braking
        + (theta["b"] - 3.0)            # harder comfortable braking, closer following
    )
    if assertiveness <= -2.0:
        return "cautious"
    if assertiveness >= 2.0:
        return "assertive"
    return "nominal"


def config_to_text(theta: Dict) -> str:
    lo = "{" + ", ".join(f"{o:g}" for o in theta["lo"]) + "}"
    return (
        f"lo={lo}, s0={theta['s0']:g}, a_m={theta['a_m']:g}, b={theta['b']:g}, th={theta['th']:g} "
        f"({behavior_tag(theta)})"
    )
