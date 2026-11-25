"""SparseAdaptive fallback for joint path reconstruction."""
from dataclasses import dataclass
import numpy as np
from ..protocol.math3d import angle

MAX_IK_CALLS = 10_000

MAX_CANDIDATES = 24

BEST_EFFORT_KEEP = 6

LEVELS = (
    ("L1", 0.0005, 0.006),
    ("L2", 0.0010, 0.010),
    ("L3", 0.0020, 0.020),
    ("L4", 0.0050, 0.050),
)

PHASE_CAPS = {"strict": 2000, "relaxed": 5000, "max_relaxed": 7500, "bridge": 10000}

@dataclass
class Candidate:
    q: np.ndarray
    pe: float
    re: float
    level: str
    accepted: bool
    origin: str

class SparseAdaptive:
    def __init__(self, robot, refs, cfg, seed=1907):
        self.robot, self.refs, self.cfg = robot, np.asarray(refs), cfg
        self.rng = np.random.default_rng(seed)
        self.layers = [[] for _ in refs]
        self.layers[0] = [Candidate(robot.q0.copy(), 0.0, 0.0, "L0", True, "fixed_initial_q")]
        self.best = [[] for _ in refs]
        self.calls = 0
        self.accepted_calls = 0
        self.phase_calls = {key: 0 for key in PHASE_CAPS}

    def residual(self, q, layer):
        actual = self.robot.fk(q)[1]
        return float(np.linalg.norm(actual[:3, 3] - self.refs[layer][:3, 3])), float(angle(actual, self.refs[layer]))

    @staticmethod
    def score(pe, re):
        return pe / 0.005 + re / 0.05

    def add_best(self, layer, candidate):
        pool = self.best[layer]
        if any(np.linalg.norm(row.q-candidate.q) < 0.008 for row in pool):
            return
        pool.append(candidate)
        pool.sort(key=lambda row: self.score(row.pe, row.re))
        del pool[BEST_EFFORT_KEEP:]

    def solve(self, layer, seed, level, ptol, rtol, phase, origin):
        if self.calls >= PHASE_CAPS[phase]:
            return
        cfg = dict(self.cfg["planner"]["ik_config"])
        damping, weight = ((0.02, 0.25), (0.01, 0.15), (0.05, 0.50))[self.calls % 3]
        cfg.update(ik_max_iterations=260, ik_position_tolerance_m=ptol,
                   ik_orientation_tolerance_rad=rtol, ik_damping=damping,
                   ik_orientation_weight=weight)
        result = self.robot.solve(self.refs[layer], np.clip(seed, self.robot.lower, self.robot.upper), cfg)
        self.calls += 1
        self.phase_calls[phase] += 1
        q = np.clip(np.asarray(result.q), self.robot.lower, self.robot.upper)
        if not np.isfinite(q).all():
            return
        pe, re = self.residual(q, layer)
        accepted = pe <= ptol and re <= rtol
        row = Candidate(q, pe, re, level, accepted, origin)
        self.add_best(layer, row)
        if accepted and len(self.layers[layer]) < MAX_CANDIDATES:
            if not any(np.linalg.norm(old.q-q) < 0.008 for old in self.layers[layer]):
                self.layers[layer].append(row)
                self.accepted_calls += 1

    def seed_bank(self, layer, limit=18):
        seeds = []
        for neighbor in (layer-1, layer+1):
            if 0 <= neighbor < len(self.layers):
                for row in (self.layers[neighbor] or self.best[neighbor])[:6]:
                    seeds.extend((row.q, row.q+self.rng.normal(0, 0.04, len(row.q)),
                                  row.q+self.rng.normal(0, 0.12, len(row.q))))
        seeds.extend((self.robot.q0, self.rng.uniform(self.robot.lower, self.robot.upper)))
        return seeds[:limit]

    def coverage_phase(self, phase, levels, attempts_per_layer):
        for attempt in range(attempts_per_layer):
            for layer in range(1, len(self.refs)):
                if self.calls >= PHASE_CAPS[phase]:
                    return
                level, ptol, rtol = levels[min(attempt * len(levels) // attempts_per_layer, len(levels)-1)]
                bank = self.seed_bank(layer)
                seed = bank[attempt % len(bank)] if bank else self.rng.uniform(self.robot.lower, self.robot.upper)
                self.solve(layer, seed, level, ptol, rtol, phase, "round_robin")

    def active(self, layer):
        return self.layers[layer] if self.layers[layer] else self.best[layer]

    def dp(self, max_step, accepted_only=False):
        pools = [self.layers[i] if accepted_only else self.active(i) for i in range(len(self.refs))]
        if any(not pool for pool in pools):
            return None, None
        costs = [np.full(len(pool), np.inf) for pool in pools]
        parents = [np.full(len(pool), -1, int) for pool in pools]
        costs[0][0] = 0.0
        for layer in range(1, len(pools)):
            for j, cur in enumerate(pools[layer]):
                node = self.score(cur.pe, cur.re) + (0 if cur.accepted else 20)
                for i, prev in enumerate(pools[layer-1]):
                    step = float(np.max(np.abs(cur.q-prev.q)))
                    if step > max_step or not np.isfinite(costs[layer-1][i]):
                        continue
                    value = costs[layer-1][i] + np.linalg.norm(cur.q-prev.q) + node
                    if value < costs[layer][j]:
                        costs[layer][j], parents[layer][j] = value, i
        if not np.isfinite(costs[-1]).any():
            return None, None
        j = int(np.argmin(costs[-1])); indices = [j]
        for layer in range(len(pools)-1, 0, -1):
            j = int(parents[layer][j]); indices.append(j)
        indices.reverse()
        selected = [pools[i][j] for i,j in enumerate(indices)]
        return np.asarray([row.q for row in selected]), selected

    def bridge_phase(self):
        for _ in range(80):
            if self.calls >= PHASE_CAPS["bridge"]:
                return
            path, selected = self.dp(0.5)
            if path is not None:
                return
            empties = [i for i in range(1, len(self.refs)) if not self.active(i)]
            targets = empties
            if not targets:
                jumps = []
                for i in range(1, len(self.refs)):
                    a, b = self.active(i-1), self.active(i)
                    jump = min(np.max(np.abs(x.q-y.q)) for x in a for y in b)
                    jumps.append((jump, i))
                targets = [i for _,i in sorted(jumps, reverse=True)[:6]]
            for layer in targets:
                for seed in self.seed_bank(layer, 24):
                    self.solve(layer, seed, "L4", 0.005, 0.050, "bridge", "bridge_targeted")
                    if self.calls >= PHASE_CAPS["bridge"]:
                        return

    def run(self):
        self.coverage_phase("strict", LEVELS[:1], 24)
        self.coverage_phase("relaxed", LEVELS[1:3], 84)
        self.coverage_phase("max_relaxed", LEVELS[3:], 70)
        self.bridge_phase()
        strict_path, strict_rows = self.dp(0.5, accepted_only=True)
        best_path, best_rows = (strict_path, strict_rows) if strict_path is not None else self.dp(0.5)
        if best_path is None:
            best_path, best_rows = self.dp(np.inf)
        return best_path, best_rows
