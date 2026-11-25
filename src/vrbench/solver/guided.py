"""Guided Anytime graph search with damped least-squares inverse kinematics.

Adapted from Wang & Gleicher (RA-L 2025) and IKLink/anytime (MIT). See NOTICE.
"""
import time
import numpy as np
from ..protocol.math3d import angle
from ..protocol.math3d import lerp_pose
from .graph import LayeredGraph
from .path import interpolation_samples


def assess_tracking(robot, qs, refs, planner_cfg, replay_cfg):
    """Feasibility only: fixed start, bounds, FK tracking and sampled interpolation.

    Never calls task scoring, obstacle geometry or a second IK solver.
    """
    qs = np.asarray(qs, float)
    if qs.shape != (len(refs), len(robot.arm)) or not np.isfinite(qs).all():
        return dict(valid=False, reason='incomplete_or_invalid_joint_array')
    start_error = float(np.linalg.norm(qs[0]-robot.q0))
    violation = float(np.maximum(np.maximum(robot.lower-qs, qs-robot.upper), 0).max())
    max_p = max_r = 0.; samples = 0
    invalid_edges = set()
    for i, q in enumerate(qs):
        actual = robot.fk(q)[1]
        p, r = float(np.linalg.norm(actual[:3, 3]-refs[i][:3, 3])), angle(actual, refs[i])
        max_p, max_r = max(max_p, p), max(max_r, r)
        if p > planner_cfg['path_position_tolerance_m'] or r > planner_cfg['path_orientation_tolerance_rad']:
            invalid_edges.update(j for j in (i-1, i) if 0 <= j < len(qs)-1)
    error = None
    try:
        for i in range(1, len(qs)):
            for q, ref in interpolation_samples(robot, qs[i-1], qs[i], refs[i-1], refs[i], replay_cfg):
                samples += 1
                if samples > replay_cfg['max_samples']:
                    raise ValueError('Tracking sample budget exceeded')
                actual = robot.fk(q)[1]
                p, r = float(np.linalg.norm(actual[:3, 3]-ref[:3, 3])), angle(actual, ref)
                max_p, max_r = max(max_p, p), max(max_r, r)
                if p > planner_cfg['path_position_tolerance_m'] or r > planner_cfg['path_orientation_tolerance_rad']:
                    invalid_edges.add(i-1)
    except ValueError as exc:
        error = str(exc)
        invalid_edges.add(i-1)
    return dict(valid=bool(error is None and start_error < 1e-10 and violation < 1e-10 and
        max_p <= planner_cfg['path_position_tolerance_m'] and max_r <= planner_cfg['path_orientation_tolerance_rad']),
        position_max_m=max_p, rotation_max_rad=max_r, initial_q_error_rad=start_error,
        joint_violation_max_rad=violation, interpolation_samples=samples, error=error,
        invalid_reference_edges=sorted(invalid_edges),
        continuous_collision_certificate=False, dynamics_checked=False)


class GuidedAnytimeSolver:
    def __init__(self, robot, refs, times, velocities, config, replay_config, seed=421):
        self.robot, self.refs, self.cfg, self.replay_cfg = robot, refs, config, replay_config
        self.rng = np.random.default_rng(seed)
        for key in ('time_budget_seconds', 'max_iterations', 'max_ik_calls', 'sparse_step',
                    'initial_candidates_per_anchor', 'initial_attempts_per_anchor', 'max_candidates_per_layer'):
            if config[key] <= 0: raise ValueError('Positive planner setting required: '+key)
        if config['sparse_step'] < 2: raise ValueError('sparse_step must be >=2')
        self.graph = LayeredGraph(len(refs), len(robot.arm), times, velocities, config['sparse_supersede_factor'])
        self.graph.append(0, robot.q0.copy(), 'fixed_actual_initial_q')
        self.random_counts = np.zeros(len(refs), dtype=int)
        self.guided_counts = np.zeros(len(refs), dtype=int)
        self.calls = []
        self.ik_seconds = 0.
        self.graph_seconds = 0.
        self.tracking_seconds = 0.
        self.history = []
        self.best_full = self.best_valid = None
        self.best_full_assessment = self.best_valid_assessment = None
        self.best_full_joints = self.best_valid_joints = None
        self.started = None
        self.lazy_tracking_searches = 0
        self.tracking_edge_checked = set()
        self.propagated_reference_edges = set()

    def expired(self):
        return (time.perf_counter()-self.started >= self.cfg['time_budget_seconds'] or
                len(self.calls) >= self.cfg['max_ik_calls'])

    def add_ik(self, layer, seed, origin):
        if layer == 0 or self.expired() or len(self.graph.layers[layer]) >= self.cfg['max_candidates_per_layer']:
            return False
        seed = np.clip(np.asarray(seed, float), self.robot.lower, self.robot.upper)
        begin = time.perf_counter()
        result = self.robot.solve(self.refs[layer], seed, self.cfg['ik_config'])
        elapsed = time.perf_counter()-begin
        self.ik_seconds += elapsed
        row = dict(layer=layer, origin=origin, success=bool(result.success),
            iterations=result.iterations, position_m=result.position_error,
            orientation_rad=result.orientation_error, seconds=elapsed, accepted=False)
        self.calls.append(row)
        if not result.success or not np.isfinite(result.q).all(): return False
        if np.any(result.q < self.robot.lower-1e-10) or np.any(result.q > self.robot.upper+1e-10): return False
        candidates = self.graph.layers[layer]
        if len(candidates) and np.min(np.linalg.norm(candidates-result.q, axis=1)) < self.cfg['dedup_distance_rad']:
            row['duplicate'] = True
            return False
        self.graph.append(layer, result.q.copy(), origin)
        row['accepted'] = True
        if origin in ('random', 'initial_random', 'initial_propagation'):
            self.random_counts[layer] += 1
        else:
            self.guided_counts[layer] += 1
        return True

    def initialize_sparse(self):
        anchors = sorted(set([0, *range(self.cfg['sparse_step'], len(self.refs), self.cfg['sparse_step']), len(self.refs)-1]))
        previous = 0
        for layer in anchors[1:]:
            if self.expired(): break
            for q in self.graph.layers[previous][-self.cfg['initial_candidates_per_anchor']:]:
                self.add_ik(layer, q, 'initial_propagation')
            attempts = 0
            while (len(self.graph.layers[layer]) < self.cfg['initial_candidates_per_anchor'] and
                   attempts < self.cfg['initial_attempts_per_anchor'] and not self.expired()):
                self.add_ik(layer, self.rng.uniform(self.robot.lower, self.robot.upper), 'initial_random')
                attempts += 1
            previous = layer
        return anchors

    def guided_sample(self, guide):
        if guide is None: return 0
        before = int(self.guided_counts.sum())
        for (left, li), (right, ri) in zip(guide.indices[:-1], guide.indices[1:]):
            if right-left <= 1: continue
            qa, qb = self.graph.layers[left][li].copy(), self.graph.layers[right][ri].copy()
            offsets = self.rng.uniform(-self.cfg['offset_range_rad'], self.cfg['offset_range_rad'],
                                       (self.cfg['guided_offsets_per_edge'], len(qa)))
            for layer in range(left, right+1):
                if self.expired(): break
                if layer == 0: continue
                u = (self.graph.times[layer]-self.graph.times[left])/(self.graph.times[right]-self.graph.times[left])
                seed = (1-u)*qa+u*qb
                if layer not in (left, right): self.add_ik(layer, seed, 'guide_interpolation')
                for offset in offsets:
                    self.add_ik(layer, seed+offset, 'guide_perturbation')
        return int(self.guided_counts.sum())-before

    def random_sample(self, attempts):
        for _ in range(attempts):
            if self.expired(): break
            available = np.array([i for i in range(1, len(self.refs))
                                  if len(self.graph.layers[i]) < self.cfg['max_candidates_per_layer']])
            if not len(available): break
            counts = self.random_counts[available]
            weights = np.exp(-(counts-counts.min()).astype(float))
            layer = int(self.rng.choice(available, p=weights/weights.sum()))
            self.add_ik(layer, self.rng.uniform(self.robot.lower, self.robot.upper), 'random')

    def search(self, sparse=False, tracking_feasible=False):
        begin = time.perf_counter()
        result = self.graph.search(self.cfg['sparse_step'] if sparse else None,
                                   tracking_feasible=tracking_feasible)
        self.graph_seconds += time.perf_counter()-begin
        return result

    def consider(self, dense):
        if dense is None: return None
        qs = self.graph.joints(dense)
        new_full = self.best_full is None or dense.score() < self.best_full.score()
        begin = time.perf_counter()
        assessment = assess_tracking(self.robot, qs, self.refs, self.cfg, self.replay_cfg)
        self.tracking_seconds += time.perf_counter()-begin
        assessment['num_reconfigurations'] = dense.reconfigurations
        assessment['valid'] = bool(assessment['valid'] and dense.reconfigurations == 0 and len(qs) == len(self.refs))
        if new_full:
            self.best_full, self.best_full_joints, self.best_full_assessment = dense, qs.copy(), assessment
        if assessment['valid'] and (self.best_valid is None or dense.score() < self.best_valid.score()):
            self.best_valid, self.best_valid_joints, self.best_valid_assessment = dense, qs.copy(), assessment
        return assessment

    def search_trackable(self, path, assessment):
        """Lazy TCP-edge validation, never collision/goal feedback or IK repair.

        Preserve the original graph and raw optimum for diagnostics; only the
        secondary feasible-path DP excludes confirmed bad candidate edges.
        """
        for _ in range(self.cfg.get('max_tracking_searches_per_iteration', 64)):
            if path is None or assessment is None or assessment.get('valid'):
                return assessment
            bad = assessment.get('invalid_reference_edges', [])
            if not bad or self.expired(): return assessment
            for left in bad:
                a, ai = path.indices[left]
                b, bi = path.indices[left+1]
                self.graph.tracking_blocked.add((a, ai, b, bi))
            fresh = [left for left in bad if left not in self.propagated_reference_edges]
            if fresh:
                left = min(fresh)
                self.propagated_reference_edges.add(left)
                seed = self.graph.layers[path.indices[left][0]][path.indices[left][1]].copy()
                stop = min(len(self.refs), left+1+self.cfg.get('tracking_propagation_nodes', 16))
                for layer in range(left+1, stop):
                    before = len(self.graph.layers[layer])
                    self.add_ik(layer, seed, 'tracking_targeted_propagation')
                    if len(self.graph.layers[layer]) > before:
                        seed = self.graph.layers[layer][-1].copy()
                    elif len(self.graph.layers[layer]):
                        seed = self.graph.layers[layer][np.argmin(np.linalg.norm(self.graph.layers[layer]-seed, axis=1))].copy()
                    if self.expired(): break
            begin = time.perf_counter()
            for left in bad:
                for ai, qa in enumerate(self.graph.layers[left]):
                    for bi, qb in enumerate(self.graph.layers[left+1]):
                        key = left, ai, left+1, bi
                        if key in self.tracking_edge_checked or self.expired(): continue
                        ok = edge_tracks(self.robot, qa, qb, self.refs[left], self.refs[left+1], self.cfg, self.replay_cfg)
                        self.tracking_edge_checked.add(key)
                        if not ok: self.graph.tracking_blocked.add(key)
            self.tracking_seconds += time.perf_counter()-begin
            path = self.search(tracking_feasible=True)
            self.lazy_tracking_searches += 1
            assessment = self.consider(path)
        return assessment

    def solve(self, callback=None):
        self.started = time.perf_counter()
        anchors = self.initialize_sparse()
        reason = 'maximum_iterations'
        for iteration in range(self.cfg['max_iterations']):
            if self.expired():
                reason = 'search_budget_exhausted'; break
            guide = self.search(sparse=True)
            added = self.guided_sample(guide)
            balance = int(self.guided_counts.sum()-self.random_counts.sum())
            self.random_sample(max(self.cfg['minimum_random_attempts_per_iteration'], balance))
            dense = self.search()
            assessment = self.consider(dense)
            assessment = self.search_trackable(dense, assessment)
            row = dict(iteration=iteration, elapsed_seconds=time.perf_counter()-self.started,
                ik_calls=len(self.calls), accepted_candidates=sum(map(len, self.graph.layers)),
                guide_nodes=len(guide.indices) if guide else 0,
                guide_sparse_edges=guide.sparse_edges if guide else None,
                targeted_candidates_added=added, dense_solution_found=dense is not None,
                candidate_score=list(dense.score()) if dense else None,
                best_dense_score=list(self.best_full.score()) if self.best_full else None,
                best_valid_score=list(self.best_valid.score()) if self.best_valid else None,
                tracking_assessment=assessment)
            self.history.append(row)
            if callback: callback(row, self.best_valid_joints if self.best_valid is not None else self.best_full_joints)
            if self.best_valid is not None and self.cfg['stop_after_first_valid_path']:
                reason = 'first_trackable_path_found'; break
        if self.best_full is None:
            self.consider(self.search())
        best = self.best_valid if self.best_valid is not None else self.best_full
        qs = self.best_valid_joints if self.best_valid is not None else self.best_full_joints
        assessment = self.best_valid_assessment if self.best_valid is not None else self.best_full_assessment
        report = dict(implementation='guided_anytime_python_DLS_variant', official_Rust_executed=False,
            status='trackable_candidate_found' if self.best_valid is not None else 'full_candidate_tracking_failed' if best else 'no_full_candidate',
            stop_reason=reason, trackable_path_found=self.best_valid is not None,
            total_planning_seconds=time.perf_counter()-self.started,
            timing=dict(ik_seconds=self.ik_seconds, graph_seconds=self.graph_seconds, tracking_seconds=self.tracking_seconds),
            sparse_anchor_indices=anchors, fixed_initial_q=True, candidate_counts=list(map(len, self.graph.layers)),
            ik_calls=len(self.calls), accepted_candidate_count=sum(map(len, self.graph.layers)),
            edge_pair_evaluations=self.graph.pair_evaluations,
            cached_edge_blocks=len(self.graph.cache), sparse_supersede_decisions=self.graph.sparse_edges_superseded,
            tracking_blocked_candidate_edges=len(self.graph.tracking_blocked),
            lazy_tracking_searches=self.lazy_tracking_searches,
            tracking_edge_pair_checks=len(self.tracking_edge_checked),
            selected_score=list(best.score()) if best else None, selected_tracking=assessment,
            history=self.history, collision_aware_planning=False, final_task_score_used_for_selection=False,
            failure_is_not_unreachability_proof=True, no_training_or_pretrained_models=True,
            precision='NumPy/SciPy float64; no RelaxedIK/PANOC equivalence claimed')
        return qs, report


def edge_tracks(robot, qa, qb, ra, rb, planner_cfg, replay_cfg):
    """Same discrete tracking rule as assess_tracking, for one immutable edge."""
    try:
        mid_actual = robot.fk((qa+qb)/2)[1]
        mid_ref = lerp_pose(ra, rb, .5)
        if (np.linalg.norm(mid_actual[:3, 3]-mid_ref[:3, 3]) > planner_cfg['path_position_tolerance_m'] or
                angle(mid_actual, mid_ref) > planner_cfg['path_orientation_tolerance_rad']):
            return False
        samples = [(qa, ra), *interpolation_samples(robot, qa, qb, ra, rb, replay_cfg)]
        if len(samples) > replay_cfg['max_samples']: return False
        for q, ref in samples:
            actual = robot.fk(q)[1]
            if (np.linalg.norm(actual[:3, 3]-ref[:3, 3]) > planner_cfg['path_position_tolerance_m'] or
                    angle(actual, ref) > planner_cfg['path_orientation_tolerance_rad']):
                return False
        return True
    except ValueError:
        return False
