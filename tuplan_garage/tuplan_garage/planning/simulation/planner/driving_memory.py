"""
Memory and Planner Generation module of LiloDriver (Sec. 3.4, Algorithm 1).

The memory M stores one entry per training-split scenario: its embedding z_i, its textual scene
description, its scenario-type label, and (through its cluster) a planner configuration.
Evaluation scenarios are never added to memory.

Offline workflow for a stage t (t = 0 adds the ten common types, t = 1..4 add H, N, C, T):
    1. add      embed D_t with the frozen encoder, insert it, re-run DBSCAN over the whole memory,
                and mark clusters that are new or whose membership changed by more than 10%
    2. pending  list the training scenarios of the marked clusters
    3. grid_search.py --tokens <pending.json>  runs closed-loop simulation for every theta in Theta
    4. search   theta_k* = argmax_theta mean_{s in C_k} Score(s; theta) for every marked cluster
Unmarked clusters keep their configurations; DBSCAN noise points use the fixed fallback configuration.
"""
import os
import re
import json
import argparse
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

try:
    from tuplan_garage.planning.simulation.planner.lilo_configs import THETA, FALLBACK_ID
    from tuplan_garage.planning.simulation.planner.prompt_generator import describe_scene, normalize_vector_input
except ImportError:  # run as a script from this directory
    from lilo_configs import THETA, FALLBACK_ID
    from prompt_generator import describe_scene, normalize_vector_input

DBSCAN_EPS = 0.15          # cosine distance
DBSCAN_MIN_PTS = 5
RESEARCH_THRESHOLD = 0.10  # re-search a cluster whose membership changes by more than 10%
NOISE = -1


def flat_embedding(vector_input) -> np.ndarray:
    """Flattened, L2-normalized perception features (ablation without the Scene Encoder, Table 3 row 3)."""
    data = normalize_vector_input(vector_input)
    flat = np.concatenate([data[k].astype(np.float32).ravel() for k in
                           ["ego_agent_past", "neighbor_agents_past", "map_lanes", "map_crosswalks", "route_lanes"]])
    return flat / max(np.linalg.norm(flat), 1e-6)


class ScenarioMemory:
    """External memory: a cosine-distance vector store plus the per-cluster planner configurations."""

    def __init__(self, db_path: str, collection_name: str = "scenario_memory"):
        import chromadb

        os.makedirs(db_path, exist_ok=True)
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # retrieval by cosine distance
        )
        self.state_file = os.path.join(db_path, "memory_state.json")
        self.state = self._load_state()

    # ----- persistent cluster state -----
    def _load_state(self) -> Dict:
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                return json.load(f)
        return {"stage": -1, "clusters": {}}

    def save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=1)

    def cluster_config(self, cluster_id: int):
        """(config_id, mean closed-loop score) of a cluster; noise points use the fallback without a score."""
        cluster = self.state["clusters"].get(str(int(cluster_id)))
        if int(cluster_id) == NOISE or cluster is None or cluster["config_id"] is None:
            return FALLBACK_ID, None
        return int(cluster["config_id"]), float(cluster["score"])

    # ----- entries -----
    def __len__(self):
        return self.collection.count()

    def add_entries(self, tokens: List[str], embeddings: List[np.ndarray], metadatas: List[Dict], batch: int = 1000):
        for i in range(0, len(tokens), batch):
            self.collection.upsert(
                ids=tokens[i:i + batch],
                embeddings=[np.asarray(e, dtype=np.float32).tolist() for e in embeddings[i:i + batch]],
                metadatas=metadatas[i:i + batch],
            )

    def get_all(self):
        result = self.collection.get(include=["embeddings", "metadatas"])
        return result["ids"], np.asarray(result["embeddings"], dtype=np.float32), result["metadatas"]

    def query(self, embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """K_r nearest memory entries by cosine distance, nearest first."""
        result = self.collection.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        return [
            {"token": token, "distance": float(distance), "metadata": metadata}
            for token, distance, metadata in zip(result["ids"][0], result["distances"][0], result["metadatas"][0])
        ]


# ----------------------------------------------------------------------------------------------
# offline memory construction
# ----------------------------------------------------------------------------------------------
def load_encoder(ckpt: str, device: str = "cuda", encoder_layers: int = 3, heads: int = 8, embed_dim: int = 256,
                 num_classes: int = 10):
    import torch
    try:
        from tuplan_garage.planning.simulation.planner.predictor import ClassFormer
    except ImportError:
        from predictor import ClassFormer

    model = ClassFormer(encoder_layers=encoder_layers, token_dim=embed_dim, heads=heads, num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval()


def encode(model, data: Dict[str, np.ndarray], device: str = "cuda") -> np.ndarray:
    import torch

    inputs = {k: torch.as_tensor(np.asarray(v, dtype=np.float32))[None].to(device) for k, v in data.items()}
    with torch.no_grad():
        return model(inputs, feature_out=True)[0].cpu().numpy()


def add_stage(memory: ScenarioMemory, npz_files: List[str], scenario_types: Dict[str, str], stage: int,
              embedding: str = "encoder", model=None, device: str = "cuda"):
    """Embed the scenarios of D_t with the frozen encoder and insert them into memory."""
    tokens, embeddings, metadatas = [], [], []
    for file_name in tqdm(npz_files, desc=f"stage {stage}"):
        token = re.findall("([0-9a-z]{16})", os.path.basename(file_name))[0]
        raw = np.load(file_name, allow_pickle=True)
        data = normalize_vector_input(raw)
        speed_limit = float(raw["speed_limit"]) if "speed_limit" in raw else None
        speed_limit = None if speed_limit is None or np.isnan(speed_limit) else speed_limit
        z = flat_embedding(data) if embedding == "flat" else encode(model, data, device)
        tokens.append(token)
        embeddings.append(z)
        metadatas.append({
            "scenario_description": describe_scene(data, speed_limit),
            "scenario_type": scenario_types[token],
            "stage": int(stage),
            "cluster": NOISE,
        })
    memory.add_entries(tokens, embeddings, metadatas)
    memory.state["stage"] = int(stage)
    memory.save_state()


def recluster(memory: ScenarioMemory, eps: float = DBSCAN_EPS, min_pts: int = DBSCAN_MIN_PTS):
    """
    Re-run DBSCAN over the whole memory. A new cluster keeps the configuration of the previous cluster it
    overlaps most if its membership changed by at most 10%; otherwise it is marked for planner search.
    """
    from sklearn.cluster import DBSCAN

    tokens, embeddings, metadatas = memory.get_all()
    labels = DBSCAN(eps=eps, min_samples=min_pts, metric="cosine", algorithm="brute").fit_predict(embeddings)

    old_clusters = {cid: set(c["members"]) for cid, c in memory.state["clusters"].items()}
    new_clusters = {}
    for label in sorted(set(labels) - {NOISE}):
        members = {t for t, l in zip(tokens, labels) if l == label}
        best_old, best_overlap = None, 0
        for cid, old_members in old_clusters.items():
            overlap = len(members & old_members)
            if overlap > best_overlap:
                best_old, best_overlap = cid, overlap

        entry = {"members": sorted(members), "config_id": None, "score": None}
        if best_old is not None:
            old = memory.state["clusters"][best_old]
            change = len(members ^ old_clusters[best_old]) / len(old_clusters[best_old])
            if change <= RESEARCH_THRESHOLD and old["config_id"] is not None:
                entry["config_id"], entry["score"] = old["config_id"], old["score"]
        new_clusters[str(int(label))] = entry

    for metadata, label in zip(metadatas, labels):
        metadata["cluster"] = int(label)
    for i in range(0, len(tokens), 1000):
        memory.collection.update(ids=tokens[i:i + 1000], metadatas=metadatas[i:i + 1000])

    memory.state["clusters"] = new_clusters
    memory.save_state()

    n_search = sum(1 for c in new_clusters.values() if c["config_id"] is None)
    print(f"{len(new_clusters)} clusters, {int(np.sum(labels == NOISE))} noise points, {n_search} clusters to search")


def pending_tokens(memory: ScenarioMemory) -> List[str]:
    """Training scenarios of the clusters that still need a planner search."""
    return sorted(t for c in memory.state["clusters"].values() if c["config_id"] is None for t in c["members"])


def load_grid_scores(score_dir: str) -> Dict[int, Dict[str, float]]:
    """{config_id: {token: closed-loop score}} written by grid_search.py."""
    scores = {}
    for file_name in os.listdir(score_dir):
        match = re.match(r"config_(\d+)\.json$", file_name)
        if match is None:
            continue
        with open(os.path.join(score_dir, file_name)) as f:
            scores[int(match.group(1))] = json.load(f)["scores"]
    return scores


def planner_search(memory: ScenarioMemory, score_dir: str):
    """theta_k* = argmax_theta 1/|C_k| sum_{s in C_k} Score(s; theta) for every cluster without a configuration."""
    if all(c["config_id"] is not None for c in memory.state["clusters"].values()):
        return
    grid_scores = load_grid_scores(score_dir)
    missing = [i for i in range(len(THETA)) if i not in grid_scores]
    assert len(missing) == 0, f"grid search results missing for configurations {missing}"

    for cid, cluster in memory.state["clusters"].items():
        if cluster["config_id"] is not None:
            continue
        best_id, best_score = None, -np.inf
        for config_id in range(len(THETA)):
            scores = [grid_scores[config_id][t] for t in cluster["members"] if t in grid_scores[config_id]]
            assert len(scores) == len(cluster["members"]), f"cluster {cid}: scenarios missing in config {config_id}"
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_id, best_score = config_id, mean_score
        cluster["config_id"], cluster["score"] = best_id, best_score
        print(f"cluster {cid} ({len(cluster['members'])} scenarios): config {best_id} {THETA[best_id]}, score {best_score:.2f}")
    memory.save_state()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiloDriver memory construction")
    parser.add_argument("command", choices=["add", "pending", "search"])
    parser.add_argument("--db", type=str, required=True, help="memory directory")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--npz_dir", type=str, help="processed training-split scenarios (*.npz)")
    parser.add_argument("--type_json", type=str, help="{token: scenario_type} of the training split")
    parser.add_argument("--types", type=str, nargs="+", help="scenario types of D_t")
    parser.add_argument("--embedding", type=str, default="encoder", choices=["encoder", "flat"])
    parser.add_argument("--encoder_ckpt", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="pending_tokens.json")
    parser.add_argument("--score_dir", type=str, help="grid search results (config_XXX.json)")
    args = parser.parse_args()

    memory = ScenarioMemory(args.db)

    if args.command == "add":
        with open(args.type_json) as f:
            scenario_types = dict(json.load(f))
        npz_files = []
        for file_name in sorted(os.listdir(args.npz_dir)):
            found = re.findall("([0-9a-z]{16})", file_name)
            if len(found) > 0 and scenario_types.get(found[0]) in args.types:
                npz_files.append(os.path.join(args.npz_dir, file_name))
        model = load_encoder(args.encoder_ckpt, args.device) if args.embedding == "encoder" else None
        add_stage(memory, npz_files, scenario_types, args.stage, args.embedding, model, args.device)
        recluster(memory)

    elif args.command == "pending":
        tokens = pending_tokens(memory)
        with open(args.out, "w") as f:
            json.dump(tokens, f)
        print(f"{len(tokens)} scenarios to simulate, written to {args.out}")

    elif args.command == "search":
        planner_search(memory, args.score_dir)
