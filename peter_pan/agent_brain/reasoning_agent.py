"""
Rule-based reasoning agent (Sprint PRD Goal B).
Priority: safety (edge) > goal toward climbable > shadow preference > reroute.
Deterministic JSON output; no LLM in closed loop.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import heapq

from peter_pan.config import load_config

from .reasoning_common import env_hash, save_reasoning_output_file, write_reasoning_log


@dataclass
class ReasoningAgentConfig:
    edge_margin: float = 36.0  # table units
    grid_step: float = 16.0
    shadow_bias_weight: float = 12.0  # extra cost when far from shadow (lower = prefer shadow)
    climbable_margin: float = 22.0  # distance to book edge to count as "at edge"


class RuleBasedReasoningAgent:
    """
    FR-B1..B6: reads environment dict, outputs reasoning_output.json (strict).
    Minimum behaviors: walk_to_waypoint, avoid_edge, approach_book_edge, climb_book, wait_in_shadow.
    """

    def __init__(self, cfg: Optional[ReasoningAgentConfig] = None, config: Optional[dict] = None):
        self._yaml = config or load_config()
        ra = self._yaml.get("reasoning_agent", {})
        self.cfg = cfg or ReasoningAgentConfig(
            edge_margin=float(ra.get("edge_margin", 36)),
            grid_step=float(ra.get("grid_step", 16)),
            shadow_bias_weight=float(ra.get("shadow_bias_weight", 12)),
            climbable_margin=float(ra.get("climbable_margin", 22)),
        )
        self._decision_counter = 0

    def _next_id(self) -> str:
        self._decision_counter += 1
        return f"dec_{self._decision_counter:04d}"

    def decision(self, env: dict[str, Any], *, log: bool = True) -> dict[str, Any]:
        """Single planning step; returns PRD reasoning_output dict. Set log=False when a wrapper logs."""
        t0 = time.time()
        table = env.get("table_boundary", [[0, 0], [640, 0], [640, 720], [0, 720]])
        tx0 = min(p[0] for p in table)
        ty0 = min(p[1] for p in table)
        tx1 = max(p[0] for p in table)
        ty1 = max(p[1] for p in table)

        agent = env.get("agent", {})
        pos = tuple(agent.get("position", [120, 640]))
        gx, gy = env.get("goal", {}).get("position", [580, 180])
        goal = (float(gx), float(gy))

        objects = env.get("objects", [])
        shadows = env.get("shadow_regions", [])

        obstacles = self._obstacle_polys(objects)
        climbable = [o for o in objects if "climbable" in o.get("affordance", [])]
        shadow_polys = [s["polygon"] for s in shadows if s.get("polygon")]

        rule_trace: list[str] = []

        # Priority 1: edge
        if self._dist_to_boundary(pos, (tx0, ty0, tx1, ty1)) < self.cfg.edge_margin:
            rule_trace.append("P1: near_table_edge -> avoid_edge")
            inward = self._inward_target(pos, (tx0, ty0, tx1, ty1))
            path = self._astar(pos, inward, (tx0, ty0, tx1, ty1), obstacles, shadow_polys, goal)
            out = self._pack(
                env,
                "avoid_edge",
                None,
                inward,
                "avoiding_edge",
                path,
                "Agent is within edge margin; move inward for safety.",
                rule_trace,
            )
            if log:
                self._log(env, out, rule_trace, t0)
            return out

        # Priority 1b: wait in shadow if already inside shadow and goal achieved (optional)
        if self._point_in_any_shadow(pos, shadow_polys) and self._dist(pos, goal) < 40:
            rule_trace.append("P3: inside shadow near goal -> wait_in_shadow")
            out = self._pack(
                env,
                "wait_in_shadow",
                None,
                list(pos),
                "waiting",
                [list(pos)],
                "Near goal inside shadow preferred zone; hold position.",
                rule_trace,
            )
            if log:
                self._log(env, out, rule_trace, t0)
            return out

        # Priority 2: climbable book if blocking path
        for book in climbable:
            poly = book.get("polygon", [])
            if len(poly) < 3:
                continue
            if self._segment_intersects_poly(pos, goal, poly):
                edge_pt = self._nearest_edge_point_toward_goal(poly, pos, goal)
                dist_edge = self._dist(pos, edge_pt)
                if dist_edge < self.cfg.climbable_margin:
                    rule_trace.append("P2: at climbable book edge -> climb_book")
                    ctr = self._polygon_centroid(poly)
                    center = book.get("center")
                    if isinstance(center, (list, tuple)) and len(center) >= 2:
                        tp = [float(center[0]), float(center[1])]
                    else:
                        tp = [ctr[0], ctr[1]]
                    out = self._pack(
                        env,
                        "climb_book",
                        book.get("id"),
                        tp,
                        "climbing",
                        [list(pos), list(edge_pt)],
                        f"{book.get('id')} is climbable and advances toward goal.",
                        rule_trace,
                    )
                    if log:
                        self._log(env, out, rule_trace, t0)
                    return out
                rule_trace.append("P2: approach climbable book edge")
                path = self._astar(pos, edge_pt, (tx0, ty0, tx1, ty1), obstacles, shadow_polys, goal)
                out = self._pack(
                    env,
                    "approach_book_edge",
                    book.get("id"),
                    edge_pt,
                    "approach_edge",
                    path,
                    f"Approach book {book.get('id')} edge to climb toward goal.",
                    rule_trace,
                )
                if log:
                    self._log(env, out, rule_trace, t0)
                return out

        # Default: walk toward goal with shadow preference in A*
        rule_trace.append("P2/P3: walk_to_waypoint with shadow bias")
        path = self._astar(pos, goal, (tx0, ty0, tx1, ty1), obstacles, shadow_polys, goal)
        out = self._pack(
            env,
            "walk_to_waypoint",
            None,
            list(goal),
            "walking",
            path,
            "Move toward goal; path cost prefers shadow corridors when available.",
            rule_trace,
        )
        if log:
            self._log(env, out, rule_trace, t0)
        return out

    def _pack(
        self,
        env: dict,
        high_level: str,
        target_object: Optional[str],
        target_point: list | tuple,
        next_state: str,
        path: list[list[float]],
        reason: str,
        rule_trace: list[str],
    ) -> dict[str, Any]:
        return {
            "decision_id": self._next_id(),
            "high_level_action": high_level,
            "target_object": target_object,
            "target_point": [float(target_point[0]), float(target_point[1])],
            "next_state": next_state,
            "path": path,
            "reason": reason,
            "meta": {
                "rule_trace": rule_trace,
                "env_hash": env_hash(env),
                "backend": "rules",
            },
        }

    def _log(self, env: dict, out: dict, rule_trace: list[str], t0: float) -> None:
        """NFR-B3: debug log."""
        log_dir = self._yaml.get("reasoning_agent", {}).get("log_dir", "outputs/reasoning_logs")
        write_reasoning_log(
            env,
            out,
            rule_trace,
            t0,
            log_dir=log_dir,
            backend="rules",
        )

    def save_reasoning_output(self, out: dict[str, Any], path: str | Path) -> None:
        save_reasoning_output_file(out, path)

    # --- geometry helpers ---

    def _obstacle_polys(self, objects: list[dict]) -> list[list[list[float]]]:
        """Book / block footprints block ground navigation."""
        polys = []
        for o in objects:
            poly = o.get("polygon", [])
            if len(poly) < 3:
                continue
            aff = o.get("affordance", [])
            if "obstacle" in aff or o.get("type") == "book":
                polys.append(poly)
        return polys

    @staticmethod
    def _polygon_centroid(poly: list[list[float]]) -> tuple[float, float]:
        n = len(poly)
        if n == 0:
            return 0.0, 0.0
        sx = sum(p[0] for p in poly)
        sy = sum(p[1] for p in poly)
        return sx / n, sy / n

    def _dist_to_boundary(self, pos: tuple[float, float], rect: tuple[float, float, float, float]) -> float:
        """Distance from pos to nearest side of axis-aligned rect (inside or outside)."""
        x0, y0, x1, y1 = rect
        px, py = pos
        if x0 <= px <= x1 and y0 <= py <= y1:
            return min(px - x0, x1 - px, py - y0, y1 - py)
        dx = max(x0 - px, 0.0, px - x1)
        dy = max(y0 - py, 0.0, py - y1)
        return math.hypot(dx, dy)

    def _inward_target(self, pos: tuple[float, float], rect: tuple[float, float, float, float]) -> tuple[float, float]:
        x0, y0, x1, y1 = rect
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        vx, vy = cx - pos[0], cy - pos[1]
        n = math.hypot(vx, vy) or 1.0
        step = 80.0
        return pos[0] + vx / n * step, pos[1] + vy / n * step

    def _dist(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _point_in_poly(self, p: tuple[float, float], poly: list[list[float]]) -> bool:
        # ray casting
        x, y = p
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i][0], poly[i][1]
            xj, yj = poly[j][0], poly[j][1]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def _point_in_any_shadow(self, p: tuple[float, float], polys: list[list[list[float]]]) -> bool:
        for poly in polys:
            if len(poly) >= 3 and self._point_in_poly(p, poly):
                return True
        return False

    def _segment_intersects_poly(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        poly: list[list[float]],
    ) -> bool:
        """Rough: midpoint inside polygon or line crosses."""
        if len(poly) < 3:
            return False
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        if self._point_in_poly(mid, poly):
            return True
        return False

    def _nearest_edge_point_toward_goal(
        self,
        poly: list[list[float]],
        pos: tuple[float, float],
        goal: tuple[float, float],
    ) -> tuple[float, float]:
        """Pick vertex of poly closest to segment pos->goal projection."""
        best = (poly[0][0], poly[0][1])
        best_d = 1e18
        for i in range(len(poly)):
            p = (poly[i][0], poly[i][1])
            d = self._dist(pos, p) + 0.3 * self._dist(p, goal)
            if d < best_d:
                best_d = d
                best = p
        return best

    def _shadow_cost(self, p: tuple[float, float], shadow_polys: list[list[list[float]]]) -> float:
        """Extra cost when far from any shadow (prefer lower cost near shadow)."""
        dists = [self._dist_to_poly(p, poly) for poly in shadow_polys if len(poly) >= 3]
        if not dists:
            return 0.0
        dmin = min(dists)
        return self.cfg.shadow_bias_weight * min(1.0, dmin / 200.0)

    def _dist_to_poly(self, p: tuple[float, float], poly: list[list[float]]) -> float:
        if self._point_in_poly(p, poly):
            return 0.0
        dmin = 1e18
        n = len(poly)
        for i in range(n):
            a = (poly[i][0], poly[i][1])
            b = (poly[(i + 1) % n][0], poly[(i + 1) % n][1])
            d = self._dist_point_to_segment(p, a, b)
            dmin = min(dmin, d)
        return dmin

    @staticmethod
    def _dist_point_to_segment(
        p: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        px, py = p
        ax, ay = a
        bx, by = b
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab2 = abx * abx + aby * aby + 1e-9
        t = max(0, min(1, (apx * abx + apy *aby) / ab2))
        qx, qy = ax + abx * t, ay + aby * t
        return math.hypot(px - qx, py - qy)

    def _cell_blocked(self, px: float, py: float, obstacles: list[list[list[float]]]) -> bool:
        for poly in obstacles:
            if len(poly) >= 3 and self._point_in_poly((px, py), poly):
                return True
        return False

    def _astar(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        bounds: tuple[float, float, float, float],
        obstacles: list[list[list[float]]],
        shadow_polys: list[list[list[float]]],
        goal_ref: tuple[float, float],
    ) -> list[list[float]]:
        """Grid A* with shadow cost bias."""
        x0, y0, x1, y1 = bounds
        step = self.cfg.grid_step
        nx = max(2, int((x1 - x0) / step))
        ny = max(2, int((y1 - y0) / step))

        def to_cell(p: tuple[float, float]) -> tuple[int, int]:
            cx = int(round((p[0] - x0) / step))
            cy = int(round((p[1] - y0) / step))
            return max(0, min(nx - 1, cx)), max(0, min(ny - 1, cy))

        def to_world(c: tuple[int, int]) -> tuple[float, float]:
            return x0 + (c[0] + 0.5) * step, y0 + (c[1] + 0.5) * step

        s = to_cell(start)
        g = to_cell(goal)

        def neighbors(c: tuple[int, int]):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                yield c[0] + dx, c[1] + dy

        def passable(c: tuple[int, int]) -> bool:
            wx, wy = to_world(c)
            if wx < x0 or wx > x1 or wy < y0 or wy > y1:
                return False
            return not self._cell_blocked(wx, wy, obstacles)

        open_set = [(0.0, s)]
        came = {}
        gscore = {s: 0.0}

        def h(c: tuple[int, int]) -> float:
            wx, wy = to_world(c)
            return math.hypot(wx - goal[0], wy - goal[1])

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur == g:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return [[*to_world(c)] for c in path]

            for nb in neighbors(cur):
                if not passable(nb):
                    continue
                wx, wy = to_world(nb)
                tentative = gscore[cur] + math.hypot(
                    to_world(cur)[0] - wx,
                    to_world(cur)[1] - wy,
                ) + self._shadow_cost((wx, wy), shadow_polys)
                if tentative < gscore.get(nb, 1e18):
                    came[nb] = cur
                    gscore[nb] = tentative
                    heapq.heappush(open_set, (tentative + h(nb), nb))

        return [list(start), list(goal)]
