# Research Note - Leader Persona Maze Experiment

## Research Question

작은 미로에서 여러 LLM 에이전트가 토론으로 이동 방향을 정할 때,
어떤 페르소나가 리더로 먼저 제안할 때 성공률이 가장 높은가?

## Motivation

10x10 미로에서는 실행 시간이 길고 성공률이 낮아 발표용 비교 데이터가 안정적으로 나오기 어렵다.
따라서 미로 크기를 5x5로 낮추고, 에이전트 수 자체보다 리더 페르소나의 영향을 비교한다.

## Experimental Setup

| Item | Value |
| --- | --- |
| Maze size | 5x5 and 7x7 |
| Dataset | `data/mazes/index_5x5.json`, `data/mazes/index_7x7.json`, or `data/mazes/index_5x5_7x7.json` |
| Agent count | 4 |
| Decision mode | `deliberation` |
| Leader personas | `goal_seeker`, `explorer`, `backtracker`, `balanced_strategist` |
| Follower personas | Same persona pool, fixed per leader condition |
| View ratio | 0.3 |
| Override | Off by default |

## Leader Definition

The leader is the first speaker in every deliberation step.
When `leader_persona` is set, the runner disables random first-speaker rotation and keeps that persona first.

Example:

| Leader | Agent order |
| --- | --- |
| goal_seeker | `[0, 1, 2, 3]` |
| explorer | `[1, 0, 2, 3]` |
| backtracker | `[2, 0, 1, 3]` |
| balanced_strategist | `[3, 0, 1, 2]` |

## Metrics

Primary metric:

- `success_rate` by leader persona

Secondary metrics:

- `avg_steps`
- `avg_deadlocks`
- `avg_llm_calls`
- `avg_decision_time_s`

## Run

```bash
python scripts/run_experiment.py \
  --index data/mazes/index_5x5.json \
  --n_agents 4 \
  --modes deliberation \
  --leader_personas all \
  --max_steps 100 \
  --out_dir results/leader_5x5
```

For the larger small-maze condition:

```bash
python scripts/run_experiment.py \
  --index data/mazes/index_7x7.json \
  --n_agents 4 \
  --modes deliberation \
  --leader_personas all \
  --max_steps 150 \
  --out_dir results/leader_7x7
```

To run both sizes in one batch:

```bash
python scripts/run_experiment.py \
  --index data/mazes/index_5x5_7x7.json \
  --n_agents 4 \
  --modes deliberation \
  --leader_personas all \
  --max_steps 150 \
  --out_dir results/leader_5x5_7x7
```

Summarize:

```bash
python scripts/summarize_leader_results.py \
  --summary results/leader_5x5/summary.json \
  --out results/leader_5x5/leader_summary.csv
```

## Interpretation

If one leader persona has higher success rate with lower average steps or deadlocks,
we can argue that leader role assignment matters in multi-agent maze navigation.
If success rates are similar, compare secondary metrics: a leader may not improve success,
but may reduce cost or stabilize deliberation.
