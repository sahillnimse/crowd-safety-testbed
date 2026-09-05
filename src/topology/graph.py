"""
Camera topology graph module for cross-camera fusion.

Maintains a directed graph of cameras, corridor capacities, schematic positions,
calibrated clock offsets, and inter-camera travel times.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config_io import load_yaml_dict

logger = logging.getLogger(__name__)

DEFAULT_TOPOLOGY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "topology.yaml")
)
GENERATED_TOPOLOGY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "topology.generated.yaml")
)


@dataclass
class CameraPosition:
    x: float = 0.0
    y: float = 0.0

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}


@dataclass
class CameraNode:
    id: str
    name: str
    corridor_capacity_pax_min: float
    position: CameraPosition = field(default_factory=CameraPosition)
    clock_offset_sec: float = 0.0
    holding_capacity_pax: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "corridor_capacity_pax_min": self.corridor_capacity_pax_min,
            "position": self.position.to_dict(),
            "clock_offset_sec": self.clock_offset_sec,
        }
        if self.holding_capacity_pax is not None:
            d["holding_capacity_pax"] = self.holding_capacity_pax
        return d


@dataclass
class TopologyEdge:
    from_cam: str
    to_cam: str
    travel_time_sec: float

    def to_dict(self) -> dict:
        return {
            "from": self.from_cam,
            "to": self.to_cam,
            "travel_time_sec": self.travel_time_sec,
        }


class CameraTopology:
    """Directed graph of cameras and corridor travel times."""

    def __init__(self, config_path: Optional[str] = None):
        self._lock = threading.RLock()
        if config_path:
            self.config_path = os.path.abspath(config_path)
        elif os.path.exists(GENERATED_TOPOLOGY_PATH):
            self.config_path = GENERATED_TOPOLOGY_PATH
        else:
            self.config_path = DEFAULT_TOPOLOGY_PATH

        self.staleness_threshold_sec: float = 5.0
        self.fusion_tick_sec: float = 1.0
        self.density_threshold: float = 2.5
        self.cameras: Dict[str, CameraNode] = {}
        self.edges: List[TopologyEdge] = []
        
        # Adjacency indices for O(1) lookups
        self._upstream_map: Dict[str, List[TopologyEdge]] = {}
        self._downstream_map: Dict[str, List[TopologyEdge]] = {}

        if os.path.exists(self.config_path):
            self.load_from_file(self.config_path)

    def load_from_file(self, path: str) -> None:
        """Load topology definition from a YAML file."""
        data = load_yaml_dict(path)
        self.update_from_dict(data)

    def update_from_dict(self, data: dict) -> None:
        """Update topology configuration from a dictionary (thread-safe)."""
        with self._lock:
            self.staleness_threshold_sec = float(data.get("staleness_threshold_sec", 5.0))
            self.fusion_tick_sec = float(data.get("fusion_tick_sec", 1.0))
            self.density_threshold = float(data.get("density_threshold", 2.5))

            cams_raw = data.get("cameras", {})
            new_cameras: Dict[str, CameraNode] = {}
            for cam_id, info in cams_raw.items():
                pos_raw = info.get("position", {})
                pos = CameraPosition(
                    x=float(pos_raw.get("x", 0.0)),
                    y=float(pos_raw.get("y", 0.0)),
                )
                holding_cap = info.get("holding_capacity_pax")
                node = CameraNode(
                    id=str(cam_id),
                    name=str(info.get("name", cam_id)),
                    corridor_capacity_pax_min=float(info.get("corridor_capacity_pax_min", 400.0)),
                    position=pos,
                    clock_offset_sec=float(info.get("clock_offset_sec", 0.0)),
                    holding_capacity_pax=float(holding_cap) if holding_cap not in (None, "") else None,
                )
                new_cameras[str(cam_id)] = node

            edges_raw = data.get("edges", [])
            new_edges: List[TopologyEdge] = []
            upstream_map: Dict[str, List[TopologyEdge]] = {cid: [] for cid in new_cameras}
            downstream_map: Dict[str, List[TopologyEdge]] = {cid: [] for cid in new_cameras}

            for e in edges_raw:
                u = str(e.get("from"))
                v = str(e.get("to"))
                t_sec = float(e.get("travel_time_sec", 0.0))

                if u not in new_cameras:
                    logger.warning("Edge references unknown source camera '%s'", u)
                if v not in new_cameras:
                    logger.warning("Edge references unknown target camera '%s'", v)

                edge_obj = TopologyEdge(from_cam=u, to_cam=v, travel_time_sec=t_sec)
                new_edges.append(edge_obj)

                if v in upstream_map:
                    upstream_map[v].append(edge_obj)
                if u in downstream_map:
                    downstream_map[u].append(edge_obj)

            self.cameras = new_cameras
            self.edges = new_edges
            self._upstream_map = upstream_map
            self._downstream_map = downstream_map
            logger.info("Loaded camera topology: %d cameras, %d edges", len(self.cameras), len(self.edges))

    def upstream_of(self, cam_id: str) -> List[Tuple[str, float]]:
        """Return list of (upstream_cam_id, travel_time_sec) feeding into cam_id."""
        with self._lock:
            edges = self._upstream_map.get(cam_id, [])
            return [(e.from_cam, e.travel_time_sec) for e in edges]

    def downstream_of(self, cam_id: str) -> List[Tuple[str, float]]:
        """Return list of (downstream_cam_id, travel_time_sec) leading out of cam_id."""
        with self._lock:
            edges = self._downstream_map.get(cam_id, [])
            return [(e.to_cam, e.travel_time_sec) for e in edges]

    def get_camera(self, cam_id: str) -> Optional[CameraNode]:
        """Lookup camera node by ID."""
        with self._lock:
            return self.cameras.get(cam_id)

    def all_cameras(self) -> List[CameraNode]:
        with self._lock:
            return list(self.cameras.values())

    def all_edges(self) -> List[TopologyEdge]:
        with self._lock:
            return list(self.edges)

    @property
    def is_generated(self) -> bool:
        with self._lock:
            return os.path.abspath(self.config_path) == GENERATED_TOPOLOGY_PATH

    def reset_to_default(self) -> None:
        """Discard generated topology file and revert to deployment baseline."""
        with self._lock:
            if os.path.exists(GENERATED_TOPOLOGY_PATH):
                try:
                    os.remove(GENERATED_TOPOLOGY_PATH)
                    logger.info("Removed generated topology: %s", GENERATED_TOPOLOGY_PATH)
                except Exception as e:
                    logger.warning("Could not delete generated topology file: %s", e)
            self.config_path = DEFAULT_TOPOLOGY_PATH
            if os.path.exists(DEFAULT_TOPOLOGY_PATH):
                self.load_from_file(DEFAULT_TOPOLOGY_PATH)

    def to_dict(self) -> dict:
        """Serialize complete topology graph for API/Frontend."""
        with self._lock:
            return {
                "staleness_threshold_sec": self.staleness_threshold_sec,
                "fusion_tick_sec": self.fusion_tick_sec,
                "density_threshold": self.density_threshold,
                "cameras": {cid: node.to_dict() for cid, node in self.cameras.items()},
                "edges": [edge.to_dict() for edge in self.edges],
                "is_generated": self.is_generated,
                "source_file": os.path.basename(self.config_path),
            }


def build_topology_from_route(
    cameras: List[dict],
    edges: List[dict],
    defaults: Optional[dict] = None,
) -> dict:
    """
    Construct a complete topology dictionary from an explicit camera list and edge list.
    
    Supports arbitrary directed acyclic graphs: linear chains, merges, and forks/splits.
    Computes deterministic, non-overlapping schematic coordinates using a Layered DAG Layout.
    """
    if not cameras:
        raise ValueError("Route topology must define at least one camera.")

    # 1. Validate cameras
    cam_nodes: Dict[str, dict] = {}
    cam_ids: List[str] = []
    for c in cameras:
        cid = str(c.get("camera_id") or c.get("id") or "").strip()
        if not cid:
            raise ValueError("Every camera must have a non-empty 'camera_id'.")
        if cid in cam_nodes:
            raise ValueError(f"Duplicate camera ID '{cid}' in route definition.")
        default_cap = float((defaults or {}).get("corridor_capacity_pax_min", 400.0))
        cap_val = c.get("corridor_capacity_pax_min")
        cap = float(cap_val) if cap_val not in (None, "", 0) else default_cap
        cam_node_data = {
            "name": str(c.get("name") or cid),
            "corridor_capacity_pax_min": cap,
            "clock_offset_sec": float(c.get("clock_offset_sec", 0.0)),
        }
        holding_val = c.get("holding_capacity_pax")
        if holding_val not in (None, ""):
            cam_node_data["holding_capacity_pax"] = float(holding_val)
        cam_nodes[cid] = cam_node_data
        cam_ids.append(cid)

    # 2. Validate edges
    valid_edges: List[dict] = []
    seen_edges = set()
    adj: Dict[str, List[str]] = {cid: [] for cid in cam_ids}
    in_degree: Dict[str, int] = {cid: 0 for cid in cam_ids}

    for e in edges:
        u = str(e.get("from") or e.get("from_cam") or "").strip()
        v = str(e.get("to") or e.get("to_cam") or "").strip()
        if not u or not v:
            raise ValueError("Every edge must specify 'from' and 'to' camera IDs.")
        if u not in cam_nodes:
            raise ValueError(f"Edge references unknown source camera '{u}'.")
        if v not in cam_nodes:
            raise ValueError(f"Edge references unknown target camera '{v}'.")
        if u == v:
            raise ValueError(f"Self-loop edge on camera '{u}' is not allowed.")
        if (u, v) in seen_edges:
            raise ValueError(f"Duplicate edge from '{u}' to '{v}'.")

        travel_time = float(e.get("travel_time_sec", 0.0))
        if travel_time <= 0:
            raise ValueError(
                f"Edge from '{u}' to '{v}' has invalid travel time ({travel_time}s). "
                "Travel time must be strictly positive and surveyed."
            )

        seen_edges.add((u, v))
        valid_edges.append({"from": u, "to": v, "travel_time_sec": travel_time})
        adj[u].append(v)
        in_degree[v] += 1

    # 3. Cycle Detection & Topological Order (Kahn's Algorithm)
    in_deg_copy = dict(in_degree)
    queue = [cid for cid in cam_ids if in_deg_copy[cid] == 0]
    topological_order: List[str] = []
    while queue:
        curr = queue.pop(0)
        topological_order.append(curr)
        for succ in adj[curr]:
            in_deg_copy[succ] -= 1
            if in_deg_copy[succ] == 0:
                queue.append(succ)

    if len(topological_order) != len(cam_ids):
        raise ValueError("Cycle detected in route topology: crowd flow must be a directed acyclic graph (DAG).")

    # 4. Layered DAG Layout: assign layer depth L(v)
    layer_depth: Dict[str, int] = {cid: 0 for cid in cam_ids}
    for u in topological_order:
        for v in adj[u]:
            layer_depth[v] = max(layer_depth[v], layer_depth[u] + 1)

    # Group nodes by layer depth
    from collections import defaultdict
    layers: Dict[int, List[str]] = defaultdict(list)
    for cid in cam_ids:
        layers[layer_depth[cid]].append(cid)

    # Assign coordinates: X increases with layer depth; Y centered within layer
    for depth, nodes_in_layer in layers.items():
        k = len(nodes_in_layer)
        for i, cid in enumerate(nodes_in_layer):
            x = 140.0 + depth * 260.0
            y = 240.0 + (i - (k - 1) / 2.0) * 160.0
            cam_nodes[cid]["position"] = {"x": round(x, 1), "y": round(y, 1)}

    def_dict = defaults or {}
    return {
        "staleness_threshold_sec": float(def_dict.get("staleness_threshold_sec", 5.0)),
        "fusion_tick_sec": float(def_dict.get("fusion_tick_sec", 1.0)),
        "density_threshold": float(def_dict.get("density_threshold", 2.5)),
        "cameras": cam_nodes,
        "edges": valid_edges,
    }


TOPOLOGY = CameraTopology()

