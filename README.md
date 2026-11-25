# VRBench (VirtualRoboBench)

VRBench evaluates action-level manipulation planning from three initial RGB views, an instruction, camera calibration, and the robot's initial state. A vision-language model predicts 36 world-frame 6-DoF TCP waypoints and gripper events. The task evaluator measures interaction and placement goals in the required order. The robot evaluator reconstructs joint motion and estimates tracking and collision risk.

[Dataset on Hugging Face][dataset] · [Project page][project]

[dataset]: https://huggingface.co/datasets/cyx712/VirtualRoboBench
[project]: https://yuxuanmath.github.io/VRBench/

## Install

```sh
python -m pip install -e '.[hub,collision]'
```

Python 3.10 or later is required. MuJoCo and Python-FCL support scene replay and geometric contact checks.

## Dataset

The 50 RoboCasa tasks comprise 25 atomic and 25 composite tasks. Each includes three 1024 × 1024 initial observations, a robot specification, replay assets and declarative scoring rules. Download and verify the published [VirtualRoboBench dataset][dataset] at this fixed revision:

```sh
vrbench download --repo cyx712/VirtualRoboBench \
  --revision 610704005c1f7a1921ecf19a2a12e89ed086246c --output dataset
vrbench validate --dataset dataset
```

If Hugging Face rate-limits an anonymous download (HTTP 429), let the client retry. If the command exits, rerun it with the same output directory to reuse downloaded files.

The evaluator reads a task index and checksum inventory, with paths relative to the dataset. Prediction receives the task instruction, initial state and three images. Scene states and expert references are used by geometric replay.

## Evaluate

Copy the appropriate example in `configs/models` and set the environment variable named by `api_key_env`. The Responses example uses `VRBENCH_API_KEY`. Both transports call the configured endpoint directly. `responses` accepts a Responses API base URL, and `chat_completions` accepts an OpenAI-compatible Chat Completions base URL. Requests expose zero tools.

```sh
vrbench predict --dataset dataset --model model.json --output predictions --tasks A01 --execute
vrbench solve --dataset dataset --task A01 --prediction predictions/A01 --output paths/A01
vrbench score --dataset dataset --task A01 --prediction predictions/A01 --path paths/A01 --output scores/A01
```

Omit `--tasks` to select all tasks. Use `--max-attempts` (default 3) and `--max-concurrency` (default 1) to control execution. A quaternion-only error is normalized immediately and the repaired result is accepted after validation. Exhausted invalid predictions receive an explicit zero-score record. Authentication, quota and billing blocks pause queued requests. Without `--execute`, prediction prepares the model input files.

To solve a whole dataset, pass the output directory of `vrbench predict`. Add another `--prediction` for each retry run. Use `--tasks A01,A02` for a subset. Solver concurrency defaults to one process.

```sh
vrbench solve --dataset dataset --prediction predictions --prediction retries \
  --max-concurrency 6 --output paths
```

All input runs must use the same dataset, model, reasoning effort and transport. Missing tasks are reported before solving starts. Use a new output directory. Each task writes its usual joint path and result under `paths/<task>/`, and `paths/summary.json` records the selected inputs and progress.

Score the solved batch and compute overall, atomic and composite-task means with one command:

```sh
vrbench score --dataset dataset --path paths --max-concurrency 4 --output scores
```

Batch scoring reads the exact prediction selected for each task from `paths/summary.json`. It reuses the saved joint paths. Omit `--tasks` for the whole dataset or pass a subset such as `--tasks A01,C01`. Use a new output directory. Each task writes `scores/<task>/result.json`; `scores/summary.json` records progress and the final group means over the selected tasks. An exhausted prediction contributes zero. A scoring error is reported separately and makes the command exit unsuccessfully.

The solver uses Guided Anytime with SparseAdaptive fallback. The scorer records selected stage witnesses, joint tracking errors and eligibility-filtered collisions. Stages with positive task credit are combined as

```text
S = sum_i weight_i * R_i * [0.10 + 0.90 * C_i * W_i * I(C_i >= 0.30, W_i >= 0.40)]
```

The normalized stage weights sum to one. C and W are stage-local coefficients on [0, 1]. Whole-path quantities are diagnostics. The result is a geometric estimate of plan quality.
