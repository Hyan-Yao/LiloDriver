# Staged memory construction of LiloDriver (Sec. 3.1, 3.4).
# Stage 0: the ten common types; stages 1-4 add H, N, C, T in this order. Only training-split scenarios are used.
# The Scene Encoder is trained once on stage 0 (GameFormer/train_prototype.py) and kept frozen.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MEMORY=${LILO_MEMORY_PATH:?set LILO_MEMORY_PATH}
NPZ_DIR=${NPZ_DIR:?processed training-split scenarios (*.npz)}
TYPE_JSON=${TYPE_JSON:?json mapping token to scenario type}
ENCODER_CKPT=${ENCODER_CKPT:?Scene Encoder checkpoint}
EMBEDDING=${EMBEDDING:-encoder}
EXP_ROOT=${EXP_ROOT:-${MEMORY}_grid_exp}
SCORE_DIR=${SCORE_DIR:-${MEMORY}_grid_scores}
MEM="python $ROOT/tuplan_garage/planning/simulation/planner/driving_memory.py"

STAGES=(
  "behind_long_vehicle following_lane_with_lead high_lateral_acceleration low_magnitude_speed starting_left_turn starting_right_turn starting_straight_traffic_light_intersection_traversal stationary_in_traffic stopping_with_lead waiting_for_pedestrian_to_cross"
  "high_magnitude_speed"
  "near_multiple_vehicles"
  "changing_lane"
  "traversing_pickup_dropoff"
)
LAST_STAGE=${LAST_STAGE:-4}

for t in $(seq 0 $LAST_STAGE); do
  # add D_t, re-run DBSCAN, mark new clusters and clusters whose membership changed by more than 10%
  $MEM add --db $MEMORY --stage $t --npz_dir $NPZ_DIR --type_json $TYPE_JSON --types ${STAGES[$t]} \
      --embedding $EMBEDDING --encoder_ckpt $ENCODER_CKPT
  $MEM pending --db $MEMORY --out $MEMORY/pending_stage$t.json
  # closed-loop search over the 162 configurations, only on the scenarios of the marked clusters
  python $ROOT/grid_search.py --tokens $MEMORY/pending_stage$t.json --exp_root $EXP_ROOT --score_dir $SCORE_DIR
  $MEM search --db $MEMORY --score_dir $SCORE_DIR
  # snapshot of the memory after stage t; evaluate it with LILO_MEMORY_PATH=${MEMORY}_stage$t simulation/llm_sim.sh (Table 2)
  rm -rf ${MEMORY}_stage$t && cp -r $MEMORY ${MEMORY}_stage$t
done
