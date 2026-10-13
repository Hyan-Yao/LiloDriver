# LiloDriver closed-loop simulation (non-reactive CLS).
# Start the Scene Encoder service (GameFormer/agentGF.py) and the LLM server (llama_server.py) first.
CHALLENGE=closed_loop_nonreactive_agents
SPLIT=${SPLIT:-test14-hard}      # test14-hard / val14_split
SELECTOR=${SELECTOR:-llm}        # llm (LiloDriver) / greedy (non-LLM selector) / none (fixed fallback, no memory)
EMBEDDING=${EMBEDDING:-encoder}  # encoder / flat (flattened perception features)
SEED=${SEED:-0}                  # results are averaged over three runs
NUPLAN_DEVKIT_ROOT=${NUPLAN_DEVKIT_ROOT:-$HOME/nuplan-devkit}
export LILO_MEMORY_PATH=${LILO_MEMORY_PATH:?set LILO_MEMORY_PATH to the memory built by memory_pipeline.sh}

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=llm_planner \
    planner.llm_planner.selector=$SELECTOR \
    planner.llm_planner.embedding=$EMBEDDING \
    scenario_filter=$SPLIT \
    scenario_builder=nuplan_challenge \
    worker.threads_per_node=256 \
    verbose=true \
    experiment_uid=lilodriver_${SPLIT}_${SELECTOR}_${EMBEDDING}_s${SEED} \
    hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
