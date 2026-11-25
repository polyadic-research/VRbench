"""Uniform virtual execution time; independent of the predicted path."""
import numpy as np


def virtual_times(robot, n, config):
    if config.get('policy') != 'max_joint_range_over_explicit_velocity':
        raise ValueError('Unsupported virtual time policy')
    if set(config['velocity_by_joint']) != set(robot.arm):
        raise ValueError('Explicit velocity for each active joint is required; no guessing')
    v = np.array([config['velocity_by_joint'][j] for j in robot.arm], float)
    width = robot.upper-robot.lower
    if np.any(width <= 0) or not np.isfinite(width).all() or np.any(v <= 0) or not np.isfinite(v).all():
        raise ValueError('Finite bounded joints and positive explicit velocities required')
    step = float(np.max(width/v))
    times = np.arange(n, dtype=float)*step
    return times, v, dict(policy=config['policy'], step_virtual_seconds=step,
        velocity_values=v.tolist(), joint_names=robot.arm, joint_ranges_used=width.tolist(),
        source=config['source'], physical_time_prediction=False, velocity_score_enabled=False,
        interpretation='Whole joint range can be traversed within each virtual interval. This intentionally makes velocity feasibility nonrestrictive; TCP interpolation tracking must be checked separately.')
