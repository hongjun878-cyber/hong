# Multi-Agent Maze Solving

**연구 질문**: 미로처럼 경우의 수가 많은 문제를 풀 때, LLM 에이전트가 많을수록 정말 더 잘 풀까?

---

## 목차

1. [연구 배경](#1-연구-배경)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [빠른 시작](#3-빠른-시작)
4. [미로 데이터 생성](#4-미로-데이터-생성)
5. [실험 실행](#5-실험-실행)
6. [에이전트 관찰 방식](#6-에이전트-관찰-방식)
7. [의사결정 방식 2가지](#7-의사결정-방식-2가지)
8. [결과 구조 및 분석](#8-결과-구조-및-분석)
9. [평가 메트릭](#9-평가-메트릭)

---

## 1. 연구 배경

### 문제 의식

Multi-Agent 시스템에서 여러 LLM 에이전트가 협력해 미로를 탐색할 때, 갈랫길이 많은 복잡한 단계에서는 **에이전트 수가 많을수록 오히려 역효과**가 날 수 있다.

- 서로 다른 의견을 조율하는 데 토큰과 시간이 낭비된다
- 의견 수가 많을수록 합의에 도달하기 어려워 Deadlock이 발생한다

### 가설

> 미로에서 갈랫길이 많을수록, 에이전트 수가 **적은** 쪽이 더 빠르고 정확하게 풀 것이다.

### 실험 설계

| 변수 | 값 |
|------|-----|
| 모델 | Qwen3-8B (로컬, GPU) |
| 에이전트 수 N | 1, 2, 3, 5 |
| 미로 크기 | 5×5, 10×10, 15×15, 20×20 |
| 의사결정 방식 | 다수결 / 토론 합의 |
| 시야 범위 | 미로 크기의 30% partial view |

---

## 2. 프로젝트 구조

```
MAZE/
├── src/
│   ├── maze/
│   │   ├── generator.py   # DFS 미로 생성, BFS로 goal 탐색
│   │   └── io.py          # JSON 저장·로드, PNG 이미지 저장
│   └── agent/
│       ├── env.py         # MazeEnv: 에이전트-환경 상호작용, 관찰 생성
│       ├── llm_backend.py # Qwen3-8B 로드 및 추론
│       ├── decision.py    # 다수결 / 토론 의사결정 구현
│       └── runner.py      # 에피소드 실행, 실험 배치 처리
├── scripts/
│   ├── generate_dataset.py   # 미로 데이터 생성
│   └── run_experiment.py     # 실험 실행
├── data/mazes/            # 생성된 미로 파일 (JSON + PNG)
└── results/               # 실험 결과 파일
```

---

## 3. 빠른 시작

### 환경 설정

```bash
pip install torch transformers
```

### 동작 확인 (smoke test)

```bash
python scripts/run_experiment.py --dry_run
```

5×5 미로 1개, 에이전트 1명, 다수결 방식으로 1 에피소드를 실행해 전체 파이프라인을 검증한다.

---

## 4. 미로 데이터 생성

```bash
python scripts/generate_dataset.py --sizes 5 10 15 20 --seeds 10
```

| 인수 | 의미 |
|------|------|
| `--sizes 5 10 15 20` | 생성할 미로 크기 |
| `--seeds 10` | 각 크기마다 seed 0~9로 10개씩 생성 → 총 40개 미로 |
| `--no_image` | PNG 저장 건너뛰기 (선택) |

생성된 파일 구조:

```
data/mazes/
├── 5x5/
│   ├── maze_5x5_seed0.json
│   ├── maze_5x5_seed0.png
│   └── ...
├── 10x10/ ...
├── index.json    # 전체 미로 목록 (실험에서 참조)
└── index.csv
```

미로 JSON에 저장되는 정보:

```jsonc
{
  "sizex": 10,
  "sizey": 10,
  "seed": 0,
  "start": [0, 0],
  "goal": [8, 5],
  "path_length": 81,   // 최단 경로 길이 → 난이도 지표
  "junctions": 9,      // 분기점(3방향 이상 열린 셀) 수 → 복잡도 지표
  "Walls": [ ... ]     // 각 셀의 상하좌우 벽 정보
}
```

---

## 5. 실험 실행

### 전체 실험

```bash
python scripts/run_experiment.py \
    --n_agents 1 2 3 5 \
    --modes majority deliberation \
    --out_dir results/exp01
```

### 단일 미로 디버깅

```bash
python scripts/run_experiment.py \
    --single data/mazes/10x10/maze_10x10_seed0.json \
    --n_agents 3 \
    --modes deliberation \
    --verbose
```

| 인수 | 기본값 | 의미 |
|------|--------|------|
| `--n_agents` | `1 2 3 5` | 에이전트 수 (핵심 독립변수) |
| `--modes` | `majority deliberation` | 의사결정 방식 |
| `--view_ratio` | `0.3` | 시야 범위 비율 |
| `--max_steps` | `300` | 스텝 제한 (초과 시 실패) |
| `--out_dir` | `results/exp01` | 결과 저장 경로 |

---

## 6. 에이전트 관찰 방식

모든 에이전트는 미로 전체가 아닌 **현재 위치 중심의 partial view**만 제공받는다.

- `view_ratio=0.3` + 10×10 미로 → 반경 1 → **3×3 윈도우**
- N명의 에이전트 모두 **동일한 관찰 정보**를 공유

에이전트에게 전달되는 텍스트 예시:

```
You are navigating a 10×10 maze.
Current position : (3, 1)
Goal position    : (8, 5)  [→ right and up]
Step             : 14  |  Visited: 12.0%
⚠ CYCLING DETECTED (8 times). You are going in circles.

--- LOCAL VIEW ---
  (2,2)[LR]  (3,2)[LR]  (4,2)[LD]
  (2,1)[RD]  (3,1)★[LU]  (4,1)[UD]
  (2,0)[LUR]  (3,0)[LR]  (4,0)·[LR]

Moves to UNVISITED cells (prefer these): ['up']
All available moves: ['left', 'up']

Reply with exactly one word: left / up / right / down
```

모델(Qwen3-8B)은 `left / up / right / down` 중 한 단어로 응답한다.

---

## 7. 의사결정 방식 2가지

### 다수결 (majority)

N명이 각자 독립적으로 행동을 선택하고, 가장 많은 표를 받은 방향으로 이동한다. 동점 시 action index가 낮은 방향 우선(결정론적).

```
에이전트 0 → "right"  ┐
에이전트 1 → "right"  ├─ 다수결 → "right" 채택
에이전트 2 → "up"     ┘
```

### 토론 합의 (deliberation)

에이전트들이 공유 스레드에서 토론하고 만장일치 합의가 되면 이동한다. `max_rounds`(기본 4) 안에 합의 불발 시 다수결 fallback.

```
[Round 1]
  에이전트 0: "PROPOSE: right  Reason: goal is to the right"
  에이전트 1: "AGREE: right"
  에이전트 2: "PROPOSE: up  Reason: right path looks visited"

[Round 2]
  에이전트 0: "AGREE: up"
  에이전트 1: "AGREE: up"
  → 만장일치 → "up" 채택
```

### Deadlock Circuit-Breaker

두 방식 모두, 같은 위치를 8회 이상 순환 감지 시 모델 응답을 무시하고 **방문 횟수가 가장 적은 인접 셀**로 강제 이동한다. 이 개입 횟수는 `override_count`로 기록된다.

---

## 8. 결과 구조 및 분석

결과 파일 구조:

```
results/exp01/
├── maze_5x5_seed0_n1_majority.json
├── maze_5x5_seed0_n1_deliberation.json
├── ...
└── summary.json    # 전체 에피소드 결과 통합
```

Python으로 분석하기:

```python
import pandas as pd, json

with open("results/exp01/summary.json") as f:
    df = pd.json_normalize(json.load(f))

# 에이전트 수 × 의사결정 방식별 성공률
print(df.groupby(["n_agents", "decision_mode"])["success"].mean())

# 에이전트 수별 평균 deadlock 횟수
print(df.groupby("n_agents")["deadlock_count"].mean())
```

---

## 9. 평가 메트릭

| 메트릭 | 필드 | 의미 |
|--------|------|------|
| **성공 여부** | `success` | 미로를 목표까지 완주했는가 |
| **스텝 수** | `steps_taken` | 목표 도달까지 이동 횟수 (300이면 실패) |
| **Deadlock 횟수** | `deadlock_count` | 순환 감지 횟수 |
| **강제 개입** | `override_count` | Circuit-breaker 개입 횟수 |
| **의사결정 시간** | `total_decision_time_s` | 에이전트 응답 생성에 쓴 총 시간 |
