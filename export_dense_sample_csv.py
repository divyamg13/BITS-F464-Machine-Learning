from __future__ import annotations

import argparse
import heapq
import pickle
from collections import defaultdict
from pathlib import Path

import dgl
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a dense sampled subgraph from a DGL .gpickle file into node and edge CSV files."
    )
    parser.add_argument("rows", type=int, help="Number of nodes to keep in the sampled graph.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the .gpickle file. If omitted, the script looks for one .gpickle file in the current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where the CSV files will be written.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional output filename prefix. Defaults to '<gpickle_stem>_dense'.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Degree bias used by the dense sampler. Larger values favor higher-degree nodes more strongly.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print progress every N sampled nodes.",
    )
    return parser.parse_args()


def resolve_gpickle_path(input_path: Path | None) -> Path:
    if input_path is not None:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return input_path

    matches = sorted(Path(".").glob("*.gpickle"))
    if not matches:
        raise FileNotFoundError("No .gpickle file found in the current directory.")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"Multiple .gpickle files found. Pass --input explicitly. Found: {names}")
    return matches[0]


def load_graph(path: Path):
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, dgl.DGLGraph):
        raise TypeError(f"Expected a DGLGraph inside {path}, got {type(graph)!r}")
    return graph


def get_positive_nodes(graph: dgl.DGLGraph) -> list[int]:
    if "label" not in graph.ndata:
        raise ValueError("Graph does not contain a 'label' node tensor.")
    labels = graph.ndata["label"].detach().cpu().numpy().reshape(-1)
    return np.flatnonzero(labels == 1).astype(int).tolist()


def dense_sample_nodes(graph: dgl.DGLGraph, sample_size: int, alpha: float, log_every: int) -> list[int]:
    num_nodes = graph.num_nodes()
    if sample_size <= 0:
        raise ValueError("rows must be a positive integer.")
    if sample_size > num_nodes:
        raise ValueError(f"rows={sample_size} is larger than the graph node count ({num_nodes}).")

    in_degree = graph.in_degrees().numpy()
    out_degree = graph.out_degrees().numpy()
    total_degree = in_degree + out_degree
    degree_rank = total_degree.argsort()[::-1]

    selected_mask = [False] * num_nodes
    selected_nodes: list[int] = []
    frontier_hits: dict[int, int] = defaultdict(int)
    heap: list[tuple[float, int, int]] = []
    rank_cursor = 0

    def push(node_id: int) -> None:
        hits = frontier_hits[node_id]
        priority = hits + alpha * float(total_degree[node_id])
        heapq.heappush(heap, (-priority, node_id, hits))

    def next_seed() -> int:
        nonlocal rank_cursor
        while rank_cursor < len(degree_rank):
            node_id = int(degree_rank[rank_cursor])
            rank_cursor += 1
            if not selected_mask[node_id]:
                return node_id
        raise RuntimeError("Unable to find an unseen seed node.")

    def add_node(node_id: int) -> None:
        if selected_mask[node_id]:
            return

        selected_mask[node_id] = True
        selected_nodes.append(node_id)

        succ = graph.successors(node_id).numpy()
        pred = graph.predecessors(node_id).numpy()
        if len(succ) == 0 and len(pred) == 0:
            return

        neighbor_ids = np.concatenate([succ, pred])
        unique_neighbors, counts = np.unique(neighbor_ids, return_counts=True)
        for neighbor_id, count in zip(unique_neighbors.tolist(), counts.tolist()):
            if selected_mask[neighbor_id]:
                continue
            frontier_hits[neighbor_id] += int(count)
            push(int(neighbor_id))

    positive_nodes = get_positive_nodes(graph)
    if len(positive_nodes) > sample_size:
        raise ValueError(
            f"rows={sample_size} is too small because the graph contains {len(positive_nodes)} nodes with label=1. "
            "Increase rows to at least that many."
        )

    positive_nodes = sorted(positive_nodes, key=lambda node_id: total_degree[node_id], reverse=True)
    for node_id in positive_nodes:
        add_node(int(node_id))

    print(f"Included {len(positive_nodes):,} positive-label nodes before dense expansion.")

    while len(selected_nodes) < sample_size:
        next_node: int | None = None

        while heap:
            _, candidate, stored_hits = heapq.heappop(heap)
            if selected_mask[candidate]:
                continue
            if frontier_hits[candidate] != stored_hits:
                continue
            next_node = int(candidate)
            break

        if next_node is None:
            next_node = next_seed()

        add_node(next_node)

        if log_every > 0 and len(selected_nodes) % log_every == 0:
            print(f"Sampled {len(selected_nodes):,} / {sample_size:,} nodes...")

    return selected_nodes


def tensor_columns(name: str, values) -> dict[str, object]:
    array = values.detach().cpu().numpy()
    if array.ndim == 1:
        return {name: array}

    flat = array.reshape(array.shape[0], -1)
    return {f"{name}_{idx}": flat[:, idx] for idx in range(flat.shape[1])}


def build_nodes_frame(subgraph: dgl.DGLGraph) -> pd.DataFrame:
    original_ids = subgraph.ndata[dgl.NID].detach().cpu().numpy()
    data: dict[str, object] = {"node_id": original_ids}

    if "label" in subgraph.ndata:
        data.update(tensor_columns("label", subgraph.ndata["label"]))

    for key, values in subgraph.ndata.items():
        if key in {dgl.NID, "_ID", "label"}:
            continue
        data.update(tensor_columns(str(key), values))

    return pd.DataFrame(data)


def build_edges_frame(subgraph: dgl.DGLGraph) -> pd.DataFrame:
    original_ids = subgraph.ndata[dgl.NID].detach().cpu().numpy()
    src_local, dst_local = subgraph.edges(order="eid")
    src = original_ids[src_local.detach().cpu().numpy()]
    dst = original_ids[dst_local.detach().cpu().numpy()]
    return pd.DataFrame({"src_node_id": src, "dst_node_id": dst})


def main() -> None:
    args = parse_args()
    input_path = resolve_gpickle_path(args.input)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = load_graph(input_path)
    print(f"Loaded graph: {input_path.name}")
    print(f"Nodes: {graph.num_nodes():,}, edges: {graph.num_edges():,}")

    sampled_nodes = dense_sample_nodes(graph, args.rows, args.alpha, args.log_every)
    subgraph = dgl.node_subgraph(graph, sampled_nodes)

    nodes_df = build_nodes_frame(subgraph)
    edges_df = build_edges_frame(subgraph)

    prefix = args.prefix or f"{input_path.stem}_dense"
    nodes_path = output_dir / f"{prefix}_nodes_{len(nodes_df)}.csv"
    edges_path = output_dir / f"{prefix}_edges_{len(nodes_df)}.csv"

    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    print(f"Wrote {len(nodes_df):,} nodes to {nodes_path}")
    print(f"Wrote {len(edges_df):,} edges to {edges_path}")


if __name__ == "__main__":
    main()
