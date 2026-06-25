# Research Note — Multi-Agent Maze Solving

---

## 연구 질문

미로를 탐색하는 도중 갈랫길(junction)이 나왔을 때,  
에이전트 수가 많을수록 합의가 어려워져 오히려 성능이 떨어지는가?

---

## 핵심 직관

> **사공이 많으면 배가 산으로 간다**
>
> 에이전트가 많을수록 junction에서 의견이 갈리고,  
> 합의에 더 많은 라운드가 필요하거나 합의 자체가 안 된다.  
> 결과적으로 맴돌다가(deadlock) 시간을 낭비하게 된다.

---

## 인과 구조

```
[원인]
junction 수가 많은 미로
    → 에이전트가 어느 방향이 맞는지 진짜로 모르는 상황이 자주 생김
    → 에이전트마다 추론 결과가 다름

[매개]
에이전트 수 N이 클수록
    → junction에서 의견 충돌 횟수 증가 (split_ratio ↑)
    → 합의까지 필요한 토론 라운드 수 증가 (avg_disc_rounds ↑)
    → 합의 실패 → deadlock (deadlock_rate ↑)

[결과]
성공률 하락 (success ↓)
걸린 스텝 수 증가 (steps_taken ↑)
```

단, 이 효과는 **junction에서만** 집중적으로 나타나야 한다.  
직선 복도(선택지 2개 이하)에서는 N이 커도 합의가 어렵지 않기 때문이다.  
이 **차등 효과(junction vs non-junction)** 가 인과 증거가 된다.

---

## 변수 정의

### 고정 변수 — 인과 관계를 오염시키는 요소를 제거하기 위해 고정

| 변수 | 값 | 고정 이유 |
|------|-----|---------|
| 미로 크기 | 10×10 | 크기가 달라지면 junction 수도 자동으로 달라짐. 크기 효과와 junction 효과가 섞이지 않도록 고정 |
| 최대 스텝 | 300 | 에피소드 길이가 다르면 deadlock 횟수 자체가 달라짐 |
| 시야 범위 | 30% (3×3 윈도우) | 모든 에이전트가 받는 정보량을 동일하게 유지 |
| 모델 | Qwen3-8B | 추론 능력 자체를 통제 |
| 온도 | 0.7 | LLM 샘플링 확률성 수준을 통제 |
| override | OFF | 코드가 중간에 개입하면 에이전트 결정이 아님. 순수 LLM 판단만 측정 |

### 독립 변수

**1. 에이전트 수 N — 핵심 조작 변수**

- `n = 2` vs `n = 5`
- 이 두 조건에서 junction 행동이 얼마나 다른지를 측정

**2. 미로 난이도 — junction 수 기반 층화**

- 크기 10×10 고정 후 junction 수로만 난이도를 구분
- junction 수가 미로 크기에 종속되는 문제를 해결하기 위해 크기를 먼저 고정

| Tier | Junction 범위 | 의미 |
|------|-------------|------|
| low | 6 – 8 | 갈랫길이 거의 없음 |
| medium | 9 – 11 | 중간 |
| high | 12 – 14 | 갈랫길이 많음 |

**3. 미로 다양성 — random seed**

- 같은 tier 안에서도 seed마다 미로 형태가 다름
- 특정 미로 모양에 결과가 편향되지 않도록 tier당 여러 seed 수집

**4. 첫 제안자 편향 제거 — 랜덤 rotate**

- deliberation에서 첫 번째로 말하는 에이전트가 가장 큰 영향을 줌
- 매 스텝마다 첫 제안자를 **랜덤하게** 배정해 특정 에이전트 편향 제거
- 규칙적 rotate(0→1→2→0...)는 교번 패턴을 만들어 인위적 deadlock을 유발함

**5. 에이전트 간 기억 공유 — shared_map**

- 에이전트들이 에피소드 전체에서 어디를 몇 번 방문했는지를 공유
- 이것 없이는 에이전트들이 이미 탐색한 길을 모르고 같은 실수를 반복

**6. 페르소나 — 에이전트 응답 다양성 확보**

- 같은 모델에게 같은 프롬프트를 N번 주면 비슷한 답이 나옴
- 에이전트마다 다른 탐색 전략을 system prompt로 부여

| Agent ID | 페르소나 | 전략 |
|----------|---------|------|
| 0 | Goal-seeker | 목표 방향으로 직진 |
| 1 | Explorer | 안 가본 셀 우선 탐색 |
| 2 | Backtracker | 막히면 역추적 |
| 3 | Contrarian | 최근에 덜 간 방향 제안 |
| 4+ | 반복 | 4명 초과 시 순환 |

### Ablation — 의사결정 방식

| 방식 | 설명 |
|------|------|
| Majority | N명이 독립적으로 선택 → 다수결. LLM 호출 수 = N |
| Deliberation | N명이 토론으로 합의 → 최대 4라운드 → 실패 시 다수결 fallback. LLM 호출 수 = N ~ N×4 |

비교 시 총 LLM 호출 수(`total_llm_calls`)로 정규화해서 계산량이 다른 문제를 보정한다.

---

## 측정 지표

### 에피소드 전체 지표

| 지표 | 필드 | 가설과의 관계 |
|------|------|-------------|
| 성공 여부 | `success` | n=5가 n=2보다 낮아야 함 (특히 high tier) |
| 총 스텝 수 | `steps_taken` | n=5가 n=2보다 많아야 함 |
| 전체 deadlock 수 | `deadlock_count` | n=5가 n=2보다 많아야 함 |
| 총 LLM 호출 수 | `total_llm_calls` | 공정 비교를 위한 정규화 기준 |

### Junction 전용 지표 — 인과 증거의 핵심

| 지표 | 필드 | 가설과의 관계 |
|------|------|-------------|
| junction에서 의견 불일치율 | `junction_split_ratio` | n=5 > n=2 이어야 함 |
| 직선에서 의견 불일치율 | `non_junction_split_ratio` | n=5 ≈ n=2 이어야 함 (차등 효과 확인) |
| junction에서 평균 토론 라운드 | `junction_avg_disc_rounds` | n=5 > n=2 이어야 함 |
| 직선에서 평균 토론 라운드 | `non_junction_avg_disc_rounds` | n=5 ≈ n=2 이어야 함 |
| junction 이후 deadlock 발생률 | `junction_deadlock_rate` | n=5 > n=2 이어야 함 |

**핵심**: junction 지표의 n=2 vs n=5 차이가 non-junction 지표의 차이보다 **명확하게 커야** 인과 관계가 성립한다. 둘 다 비슷하게 차이난다면 junction이 원인이 아니라 단순히 에이전트 수 자체의 효과일 수 있다.

### 페르소나 다양성 검증 지표

| 지표 | 필드 | 의미 |
|------|------|------|
| 의견 분열 스텝 비율 | `split_ratio` | 페르소나가 실제로 다른 의견을 만들어냈는가 |

---

## 파이프라인

```
[1단계] 미로 데이터 생성
─────────────────────────────────────────────────
seed 0, 1, 2, ... 순서로 build_maze(10, 10, seed) 실행
    → junction 수 계산
    → 6–8이면 low, 9–11이면 medium, 12–14이면 high 버킷에 넣기
    → 각 버킷이 per_tier개 찰 때까지 반복
    → data/mazes/10x10/{tier}_{idx}.json 저장
    → data/mazes/index.json 으로 전체 목록 관리


[2단계] 실험 실행
─────────────────────────────────────────────────
index.json 에서 미로 목록 불러오기

for 미로 in [low×K, medium×K, high×K]:
    for n_agents in [2, 5]:
        for mode in [majority, deliberation]:

            MazeEnv 초기화
            shared_map = {}   ← 에피소드 전체 공유 탐색 기록

            while 목표 미도달 and 스텝 < 300:

                현재 위치 → shared_map 방문 횟수 +1

                available = 지금 갈 수 있는 방향들
                is_junction = (len(available) >= 3)   ← 핵심 플래그

                offset = random.randint(0, n-1)       ← 첫 제안자 랜덤 배정
                rotated_ids = agent_ids[offset:] + agent_ids[:offset]

                for 각 에이전트:
                    obs_to_prompt(obs, agent_id, shared_map)
                        → [페르소나 역할]
                        → 현재 위치 / 목표 / 시야 3×3
                        → 공유 탐색 지도 (인접 셀 방문 횟수)
                        → 최근 경로 6칸
                    Qwen3-8B 호출 → 방향 단어 반환

                if majority:
                    N명 득표 집계 → 다수결
                if deliberation:
                    제안 → AGREE/PROPOSE 반복 최대 4라운드
                    만장일치 달성 or 초과 → fallback 다수결

                env.step(action)   ← 이동
                deadlock 감지

                스텝 기록:
                    is_junction / votes / disc_rounds /
                    deadlock_count / agent_log (발화 원문)

            에피소드 결과 저장 (junction_stats 포함)


[3단계] 분석
─────────────────────────────────────────────────
summary.json 집계

핵심 비교:
    junction_split_ratio      : n=2 vs n=5, per tier
    junction_avg_disc_rounds  : n=2 vs n=5, per tier
    junction_deadlock_rate    : n=2 vs n=5, per tier
    (비교군) non_junction 동일 지표 → 차등 효과 확인

가설 지지 조건:
    high tier에서 n=5의 junction_split_ratio > n=2의 junction_split_ratio
    AND
    high tier에서 n=5의 junction_deadlock_rate > n=2의 junction_deadlock_rate
    AND
    이 차이가 non_junction 지표에서의 차이보다 유의미하게 큼
```

---

## 방법론으로의 발전 방향

가설이 검증되면 다음 방법론이 정당화된다:

```
미로 탐색 중 junction 감지
    → 현재 에이전트 수 N에서 2로 줄여 빠른 합의 유도
직선 / 단순 구간
    → 에이전트 수 유지 또는 증가 (병렬 탐색)
```

이를 **Adaptive Agent Count** 방법론이라 부를 수 있다.
