## Keyrwords

1. Multi-Agent Coordination 
2. Maze Solving


## 프로젝트 문제 의식

1. Multi-Agent Coordination에서 미로 찾기를 할 때 경우의 수가 많은 복잡한 단계에서는 에이전트 수가 많은 것이 오히려 불리하다. 왜나햐면 오히려 의견의 수가 많을 수록 합의에 도달하기 어렵기 때문이다. 즉, 이러한 경우에 Deadlock과 같은 문제를 일으킨다.

## 프로젝트 가설

1. Maze를 풀 때 경우의 수가 많은 경우, 예를 들어, Maze에서 갈랫길이 많은 경우, 에이전트의 수가 많은 것보다 적은 것이 더 좋을 것이다. 

## 검증 셋업 리스트

1. Qwen 3.5 8B Agent의 개수 N  
2. Maze Solving Generation Pipeline
3. Multi-Agent와 Maze Solving 간의 상호 작용 파이프라인
4. Mutli-Agent의 Maze Solving 스텝별 의사결정 방식 2가지. 다수결 방식, 서로 의견을 내고 합의가 될 때까지 토론하는 방식
5. 미로 전체크기의 일정 비율 크기의 view만 제공해 이동하면 view도 같이 이동하는 형태로 모든 에이전트들에게 동일한 정보를 제공

## 평가 메트릭

1. Accuracy: 미로 찾기 성공 횟수
2. Cost: Deadlock 지연 시간 측정