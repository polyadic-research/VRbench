"""Append-only layered graph; cached NumPy edge blocks and lexicographic DP.

Graph score follows IKLink: minimize reconfigurations, then joint L2 movement.
Reconfiguration edges add one to the first cost and no joint movement cost.
Sparse guide edges are never returned as a dense full-path solution.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class GraphPath:
    indices: list
    reconfigurations: int
    joint_movement: float
    sparse_edges: int

    def score(self):
        return self.reconfigurations, self.joint_movement


class LayeredGraph:
    def __init__(self, n, ndof, times, velocities, supersede_factor=1.1):
        self.layers = [np.empty((0, ndof), dtype=float) for _ in range(n)]
        self.origins = [[] for _ in range(n)]
        self.times = np.asarray(times, float)
        self.velocities = np.asarray(velocities, float)
        if n < 2 or self.times.shape != (n,) or not np.all(np.isfinite(self.times)) or np.any(np.diff(self.times) <= 0):
            raise ValueError('Graph needs at least two strictly increasing finite times')
        if self.velocities.shape != (ndof,) or not np.isfinite(self.velocities).all() or np.any(self.velocities <= 0):
            raise ValueError('Explicit positive finite velocity vector required')
        self.supersede_factor = supersede_factor
        self.cache = {}
        self.pair_evaluations = 0
        self.sparse_edges_superseded = 0
        self.tracking_blocked = set()

    def append(self, layer, q, origin):
        q = np.asarray(q, float)
        if q.shape != self.layers[layer].shape[1:] or not np.isfinite(q).all():
            raise ValueError('Invalid graph candidate')
        self.layers[layer] = np.vstack((self.layers[layer], q))
        self.origins[layer].append(origin)

    def edge_block(self, left, right):
        """Reuse old-old pair costs; evaluate only newly appended rows/columns."""
        a, b = self.layers[left], self.layers[right]
        key = left, right
        old_d, old_v = self.cache.get(key, (np.empty((0, 0)), np.empty((0, 0), dtype=bool)))
        if old_d.shape == (len(a), len(b)):
            return old_d, old_v
        d = np.empty((len(a), len(b)))
        valid = np.empty(d.shape, dtype=bool)
        rows, cols = old_d.shape
        d[:rows, :cols], valid[:rows, :cols] = old_d, old_v
        dt = self.times[right]-self.times[left]
        def fill(aa, bb):
            if not len(aa) or not len(bb): return np.empty((len(aa), len(bb))), np.empty((len(aa), len(bb)), bool)
            delta = bb[None, :, :]-aa[:, None, :]
            self.pair_evaluations += len(aa)*len(bb)
            return np.linalg.norm(delta, axis=2), np.all(abs(delta) <= self.velocities*dt+1e-12, axis=2)
        d[rows:, :], valid[rows:, :] = fill(a[rows:], b)
        d[:rows, cols:], valid[:rows, cols:] = fill(a[:rows], b[cols:])
        self.cache[key] = d, valid
        return d, valid

    def dense_cost(self, left, right):
        """Min-plus dense path costs for sparse-edge supersession checks."""
        if any(not len(self.layers[i]) for i in range(left, right+1)):
            return np.full((len(self.layers[left]), len(self.layers[right])), np.inf)
        d, valid = self.edge_block(left, left+1)
        cost = np.where(valid, d, np.inf)
        for i in range(left+1, right):
            d, valid = self.edge_block(i, i+1)
            edge = np.where(valid, d, np.inf)
            chunks = [np.min(cost[k:k+16, :, None]+edge[None, :, :], axis=1)
                      for k in range(0, len(cost), 16)]
            cost = np.vstack(chunks)
        return cost

    def search(self, sparse_step=None, *, tracking_feasible=False):
        """sparse_step=None: dense solution DP. Otherwise: hybrid guide DP."""
        n = len(self.layers)
        if not len(self.layers[0]) or not len(self.layers[-1]): return None
        rc = [np.full(len(q), np.inf) for q in self.layers]
        movement = [np.full(len(q), np.inf) for q in self.layers]
        parents = [np.full((len(q), 2), -1, dtype=int) for q in self.layers]
        rc[0][:] = 0.; movement[0][:] = 0.
        for right in range(1, n):
            if not len(self.layers[right]): continue
            predecessors = [right-1]
            if sparse_step is not None:
                if sparse_step < 2: raise ValueError('sparse_step must be >=2')
                if right >= sparse_step:
                    predecessors.append(right-sparse_step)
                if right == n-1 and right % sparse_step:
                    predecessors.append(right-right % sparse_step)
            for left in sorted(set(predecessors), reverse=True):
                if left < 0 or left >= right or not len(self.layers[left]): continue
                if not np.isfinite(rc[left]).any(): continue
                distance, valid = self.edge_block(left, right)
                reconfig = rc[left][:, None]+(~valid).astype(int)
                total = movement[left][:, None]+np.where(valid, distance, 0.)
                if tracking_feasible:
                    for a, ai, b, bi in self.tracking_blocked:
                        if a == left and b == right:
                            reconfig[ai, bi] = np.inf
                            total[ai, bi] = np.inf
                if right-left > 1:
                    active = ~(self.dense_cost(left, right) < self.supersede_factor*distance)
                    self.sparse_edges_superseded += int(np.sum(~active))
                    reconfig = np.where(active, reconfig, np.inf)
                    total = np.where(active, total, np.inf)
                min_rc = reconfig.min(axis=0)
                total = np.where(reconfig == min_rc[None, :], total, np.inf)
                selected = np.argmin(total, axis=0)
                min_movement = total[selected, np.arange(total.shape[1])]
                improve = (min_rc < rc[right]) | ((min_rc == rc[right]) & (min_movement < movement[right]))
                rc[right][improve] = min_rc[improve]
                movement[right][improve] = min_movement[improve]
                parents[right][improve, 0] = left
                parents[right][improve, 1] = selected[improve]
        if not np.isfinite(rc[-1]).any(): return None
        best_rc = rc[-1].min()
        end = int(np.argmin(np.where(rc[-1] == best_rc, movement[-1], np.inf)))
        layer, candidate = n-1, end
        nodes = [(layer, candidate)]
        while layer:
            layer, candidate = map(int, parents[layer][candidate])
            if layer < 0: return None
            nodes.append((layer, candidate))
        nodes.reverse()
        return GraphPath(nodes, int(best_rc), float(movement[-1][end]),
                         sum(b[0]-a[0]>1 for a, b in zip(nodes[:-1], nodes[1:])))

    def joints(self, path):
        return np.array([self.layers[i][j] for i, j in path.indices])
