# KIAS Maze Multi-Agent Experiment

이 폴더는 KIAS Jupyter Lab에서 실행할 수 있는 Maze path finding multi-agent 실험 코드입니다.

## 실험 질문

핵심 질문은 다음입니다.

> Agent 수가 많아질수록 Maze path finding accuracy가 좋아지는가?  
> 좋아진다면 추가 cost를 정당화할 만큼 좋아지는가?

비교 변수는 세 가지입니다.

1. Agent 수: `1, 3, 5, 10`
2. Maze 난이도: `easy, medium, hard`
3. 합의 방식: `majority_vote`, `debate_vote`, `judge_consensus`

측정값은 다음입니다.

- `success`: goal에 도달했는지
- `valid_steps`: 벽을 통과하지 않고 합법적으로 움직였는지
- `optimal`: 최단 경로인지
- `total_tokens`: prompt + completion token 수
- `generation_time_sec`: 생성에 걸린 시간

## 파일 설명

- `maze_agents_experiment.py`: 미로 생성, 정답 생성, agent 호출, 합의 방식, 평가, 결과 저장까지 들어있는 핵심 코드
- `KIAS_maze_agents_experiment.ipynb`: Jupyter Lab에서 순서대로 실행하는 노트북
- `requirements.txt`: 필요한 Python 패키지 목록

## KIAS Jupyter Lab 실행 순서

1. 이 폴더의 파일들을 KIAS 서버 작업 폴더에 업로드합니다.
2. Jupyter Lab에서 `KIAS_maze_agents_experiment.ipynb`를 엽니다.
3. 첫 번째 셀에서 필요한 패키지를 설치합니다.

```python
%pip install -U -r requirements.txt
```

서버에 PyTorch가 이미 잘 설치되어 있다면, CUDA 버전 충돌을 피하기 위해 아래처럼 설치해도 됩니다.

```python
%pip install -U transformers accelerate pandas matplotlib bitsandbytes
```

4. 먼저 `MockLLM` smoke test를 실행해서 코드가 돌아가는지 확인합니다.
5. 그 다음 Qwen 모델을 한 번만 로드합니다.
6. 작은 easy maze 하나로 single-agent 테스트를 합니다.
7. 문제가 없으면 full experiment 셀을 실행합니다.

## GPU 메모리 관련 핵심 주의점

Agent 수가 10개라고 해서 Qwen 모델을 10번 로드하면 안 됩니다.

이 프로젝트 코드는 모델을 한 번만 로드하고, agent마다 다른 prompt를 보내는 방식입니다.

```python
llm = LocalHFLLM(...)
llm.load()
```

위 코드는 실험 시작 전에 한 번만 실행해야 합니다.

## 추천 시작 설정

처음부터 큰 실험을 돌리지 말고 아래처럼 시작하는 것을 추천합니다.

```python
DIFFICULTIES = {
    "easy": {"sizex": 5, "sizey": 5},
    "medium": {"sizex": 8, "sizey": 8},
    "hard": {"sizex": 10, "sizey": 10},
}
SEEDS = [1, 2, 3]
AGENT_COUNTS = [1, 3, 5]
```

서버가 안정적이면 이후에 다음처럼 확장합니다.

```python
DIFFICULTIES = {
    "easy": {"sizex": 5, "sizey": 5},
    "medium": {"sizex": 10, "sizey": 10},
    "hard": {"sizex": 15, "sizey": 15},
}
SEEDS = [1, 2, 3, 4, 5]
AGENT_COUNTS = [1, 3, 5, 10]
```

## 합의 방식 정의

### `single`

Agent 1개가 혼자 경로를 제출합니다.  
이것은 baseline입니다.

### `majority_vote`

여러 agent가 서로 보지 않고 독립적으로 답을 제출합니다.  
가장 많이 나온 경로를 최종 답으로 선택합니다.

### `debate_vote`

여러 agent가 먼저 답을 제출합니다.  
그 뒤 서로의 답을 보고 한 번 더 답을 수정합니다.  
마지막 답들에 대해 majority vote를 적용합니다.

중요: 토론 중에는 코드가 계산한 정답 여부를 agent에게 알려주지 않습니다. Agent들은 서로의 후보 답만 보고 판단합니다.

### `judge_consensus`

여러 agent가 후보 경로를 제출합니다.  
별도의 judge agent가 후보 중 하나를 고릅니다.

중요: judge도 정답 경로를 보지 않습니다. 정답 검증은 실험이 끝난 뒤 accuracy 계산에만 사용됩니다.

## 결과 파일

실험을 실행하면 `results/` 폴더가 생깁니다.

- `results/experiment_results.csv`: 모든 실험 결과 표
- `results/summary.csv`: method, difficulty, agent_count별 요약
- `results/accuracy_easy.png` 등: accuracy 그래프
- `results/cost_easy.png` 등: token cost 그래프
- `results/transcripts/*.json`: 각 agent의 실제 응답 기록
- `results/mazes/*.png`: 생성된 maze 이미지

## 발표에서 사용할 수 있는 해석 방향

결과 분석은 다음 순서로 보면 됩니다.

1. 난이도가 올라갈수록 single-agent accuracy가 떨어지는가?
2. agent 수가 늘면 accuracy가 올라가는가?
3. accuracy 증가량에 비해 token/time cost가 얼마나 증가하는가?
4. majority vote, debate vote, judge consensus 중 어느 방식이 cost 대비 효율적인가?
5. `agent 수가 많을수록 좋은가?`에 대해 accuracy-cost tradeoff 관점에서 결론을 낸다.
