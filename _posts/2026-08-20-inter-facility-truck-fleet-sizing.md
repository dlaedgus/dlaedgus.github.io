---
title: "Fleet Sizing of Trucks for an Inter-Facility Material Handling System Using Closed Queueing Networks (ORP 2022)"
date: 2026-08-20 12:50:00 +0900
categories: [paper_review, OR]
tags: [fleet-sizing, closed-queueing-network, mean-value-analysis, MINLP, simulation]
math: true
---

# Paper Review — *Fleet Sizing of Trucks for an Inter-Facility Material Handling System Using Closed Queueing Networks* (ORP 2022)

- **1저자:** Mohamed Amjath  
- **공저자:** Laoucine Kerbache, James MacGregor Smith, Adel Elomri  
- **제목:** Fleet Sizing of Trucks for an Inter-Facility Material Handling System Using Closed Queueing Networks  
- **저널:** *Operations Research Perspectives*, Vol. 9, Article 100245  
- **년도:** 2022  
- **DOI:** [10.1016/j.orp.2022.100245](https://doi.org/10.1016/j.orp.2022.100245)  
- **원문:** [ScienceDirect Open Access](https://www.sciencedirect.com/science/article/pii/S2214716022000185)

---

## 1) High-Level Summary (3–5 sentences)

본 논문은 Storage Yard와 생산시설 사이에서 여러 종류의 원자재를 반복 운반하는 **동종 Truck Fleet**의 최소 규모와 제품 Class별 배차대수를 결정합니다. Truck을 Closed Queueing Network(CQN)의 Customer로, Loading·Weighing·Unloading Station을 Queueing Node로 모델링하고, Mean Value Analysis(MVA)로 Fleet 규모에 따른 Cycle Time·Queue·Utilization을 계산합니다. 이 성능값을 Mixed-Integer Nonlinear Programming(MINLP)에 연결하고 Sequential Quadratic Programming(SQP)으로 근사해를 찾은 뒤, AnyLogic Discrete-Event Simulation(DES)으로 검증합니다. 실제 철강 제조 사례의 네 가지 원료조합에서 분석모형과 Simulation이 동일한 최소 Fleet 9·10·10·12대를 산출했고, Queue·Utilization·Cycle Time의 차이는 대체로 $\pm7\%$ 범위였습니다. 이 연구는 **고정된 Class별 순환경로의 Fleet Sizing과 물류병목 분석**이며, 차량이 여러 후보경로 중 하나를 고르는 Route Optimization은 아닙니다.

---

## 2) 문제 설정: 몇 대를 투입해야 충분하면서도 과하지 않은가

### 2.1 운영 상황

생산시설은 매일 Bill of Materials(BOM)에 따라 서로 다른 원자재 조합을 요구합니다. 제3자 물류업체에서 빌린 Truck은 원료별 Storage Yard에서 적재한 뒤 Weighbridge와 검사·대기·하역 공정을 거쳐 생산시설에 원료를 전달하고, 다시 Yard로 돌아와 같은 순환을 반복합니다.

Fleet이 너무 작으면 12시간 Shift 안에 수요를 충족하지 못합니다. 반대로 Truck을 계속 늘린다고 처리량이 선형으로 증가하지도 않습니다. Loading 또는 Unloading Station은 한 번에 한 Truck만 처리할 수 있으므로 Truck 수가 많아질수록 Queue가 길어지고, Cycle Time이 커져 Truck 한 대의 회전횟수는 감소합니다.

즉, 핵심 Trade-off는 다음과 같습니다.

$$
\text{Fleet 증가}
\Rightarrow
\begin{cases}
\text{동시에 운반 가능한 물량 증가}\\
\text{Station Queue와 Cycle Time 증가}
\end{cases}
$$

논문은 “정해진 시간 안에 Class별 수요를 모두 운반할 수 있는 최소 Truck 수”를 찾습니다.

### 2.2 이 논문이 풀지 않는 문제

사례의 Class A, B, C는 각각 정해진 Node Sequence를 반복합니다. 어떤 Truck이 교차로에서 어느 길을 선택할지, 최단경로를 어떻게 찾을지, 충돌을 어떻게 피할지는 최적화하지 않습니다.

- **결정함:** 총 Fleet Size, Class별 Truck Allocation
- **성능평가:** Cycle Time, Throughput, Queue Response Time, Station Utilization
- **고정됨:** Class별 물리 이동경로와 Service Process Sequence

또한 사례 산업은 Semiconductor가 아니라 **Steel Manufacturing**입니다. 논문의 기여는 특정 산업의 공정지식보다, 고정경로 Inter-facility Material Transfer를 `CQN + MINLP + Simulation`으로 분석하는 방법에 있습니다.

---

## 3) Closed Queueing Network 모델

### 3.1 Truck이 “Customer”인 폐쇄형 Network

일반적인 Queueing Model에서 Customer는 시스템에 도착했다가 떠납니다. 이 문제에서는 Fleet의 Truck 수가 고정되어 있고 Truck이 순환경로를 계속 반복하므로, Truck을 Network 안에서 사라지지 않는 Customer로 보는 **Closed Queueing Network**가 자연스럽습니다.

- **Customer:** Truck
- **Customer Class $k$:** Truck이 운반하는 원자재 종류
- **Resource Node:** Gate, Loading Dock, Weighbridge, Unloading Dock 등
- **Circulation Node:** Resource 사이 이동구간
- **Network Population $N_k$:** Class $k$에 배정한 Truck 수

Resource Node는 FCFS(First-Come, First-Served)의 $M/M/1$ Queue로, 이동구간은 동시 이동에 실질적인 Capa 제약이 없다는 가정 아래 $M/M/\infty$ Node로 모델링합니다. 모든 Queue는 무한 Buffer이고, Server Failure와 Breakdown 확률은 0으로 가정합니다.

### 3.2 주요 기호

- $j=1,\ldots,s$: Service Station index
- $k=1,\ldots,p$: Product Class index
- $N$: 전체 Truck 수
- $N_k$: Class $k$ Truck 수
- $T_j$: Station $j$의 평균 Service Time
- $\mu_j^k$: Class $k$가 Station $j$에서 받는 평균 Service Rate
- $V_j^k$: Class $k$가 한 Cycle 동안 Station $j$를 방문하는 평균 횟수
- $Y_j^k(N)$: Class $k$의 Station $j$ 평균 Response Time
- $L_j^k(N)$: Class $k$의 Station $j$ 평균 Queue Length
- $\Theta_k(N)$: Class $k$의 Throughput
- $CT_k$: Class $k$ Truck의 한 Cycle 소요시간

### 3.3 Mean Value Analysis

Arrival Theorem에 따르면 Closed Network의 한 Customer가 Station에 도착했을 때 보는 시스템상태는, 같은 Network에서 Customer 하나가 빠진 상태의 평균과 연결됩니다. 단일 Class, Single-Server Station의 Response Time은 원문 식 (1)에서 다음과 같습니다.

$$
Y_j(N)
=T_j+T_jL_j(N-1)
=T_j\left[1+L_j(N-1)\right]
$$

Service Time에 “도착 시 이미 기다리는 평균 Customer 수”를 곱한 대기항이 더해지는 구조입니다. 원문은 Multi-class 확장을 다음과 같이 적습니다.

$$
Y_j^k(N)
=
\frac{1}{\mu_j^k}
\left[1+L_j^k(N-1)\right]
$$

Class $k$의 한 Cycle Time은 방문횟수로 가중한 Node Response Time의 합입니다.

$$
CT_k(N)
=
\sum_{j=1}^{s}V_j^kY_j^k(N)
$$

Class Population이 $N_k$이면 Little’s Law에 따라 의도된 Class Throughput은

$$
\Theta_k(N)
=
\frac{N_k}{\sum_{j=1}^{s}V_j^kY_j^k(N)}
=
\frac{N_k}{CT_k(N)}
$$

이고, Station별 평균 Class Queue Length는

$$
L_j^k(N)
=
\Theta_k(N)V_j^kY_j^k(N)
$$

으로 갱신됩니다. 이 계산을 작은 Population에서 시작해 Fleet Vector가 목표값에 도달할 때까지 반복하면, 각 Fleet Allocation에 대한 $CT_k$, $\Theta_k$, $L_j^k$, Utilization을 얻을 수 있습니다.

### 3.4 원문 MVA 식의 표기상 주의점

원문 식 (3)은 Class $k$ Throughput의 분자에 $N_k$가 아니라 $N$을 인쇄하고, 식 (4)은 Little’s Law의 Throughput 자리에 $\Theta_k(N)$가 아니라 $\mu_j^k(N)$를 인쇄합니다. 그러나 원문의 기호정의, “식 (3)은 Class $k$의 Throughput, 식 (4)는 Class $k$의 Queue Length”라는 본문 설명, 표준 MVA 관계를 함께 보면 위에서 정리한 $N_k$와 $\Theta_k(N)$ 형태가 수리적으로 자연스럽습니다.

또한 엄밀한 Multi-class MVA에서는 Class $k$ Customer 한 대가 빠진 Population Vector $N-e_k$를 사용하지만, 원문은 이를 단순히 $N-1$로 표기합니다. 공유 FCFS Station의 Response Time도 일반적으로 다른 Class의 Queue를 함께 보아야 하는데, 원문 식 (2)는 $L_j^k$만 사용하므로 Tailored Approximation의 범위를 확인해야 합니다. 따라서 재현할 때는 논문 PDF의 식을 그대로 Coding하기보다, 사용한 MVA Algorithm의 Class별 Population Update와 Shared-node 처리방식을 다시 확인해야 합니다.

---

## 4) Fleet Sizing MINLP

### 4.1 결정변수와 보조변수

$$
x_i^k=
\begin{cases}
1,&\text{Truck }i\text{를 Class }k\text{에 배정}\\
0,&\text{그 외}
\end{cases}
$$

- $N_k$: Class $k$에 배정한 Truck 수
- $CT_k$: MVA가 계산한 Class $k$ Cycle Time
- $Z_i^k$: Truck $i$가 Shift 동안 수행하는 Class $k$ Trip 수
- $F_k$: Class $k$ 원료의 Full Truckload 중량
- $D_k$: Class $k$의 운송수요
- $T$: Shift 길이
- $\beta$: 사용할 수 있는 총 Truck 수 상한

### 4.2 목적함수: 총 Truck 수 최소화

$$
\min N
=
\sum_{i=1}^{t}\sum_{k=1}^{p}x_i^k
$$

Truck이 동종이고 조달비용이 대수에 비례한다는 설정이므로, 별도의 차량별 비용 대신 총 사용대수를 최소화합니다.

### 4.3 Class별 수요충족

$$
N_k\cdot\frac{T}{CT_k}\cdot F_k
\ge D_k,
\qquad k=1,\ldots,p
$$

- $T/CT_k$: Truck 한 대가 한 Shift에 수행할 수 있는 평균 Cycle 수
- $N_k(T/CT_k)$: Class $k$ Fleet의 총 Trip Capacity
- 여기에 Trip당 적재량 $F_k$를 곱하면 총 운송가능량

Fleet 수가 늘면 $N_k$는 커지지만 Queueing 때문에 $CT_k$도 함께 변합니다. 따라서 이 제약은 단순 선형 Capacity 식이 아니라, MVA 성능값이 내재된 Nonlinear Constraint입니다.

### 4.4 Fleet 상한과 Shift 시간

$$
\sum_{i=1}^{t}\sum_{k=1}^{p}x_i^k\le\beta
$$

$$
Z_i^kCT_k\le T,
\qquad \forall i,k
$$

첫 식은 Budget 또는 조달 가능한 Truck 수를 제한하고, 두 번째 식은 한 Truck의 총 운행시간이 Shift 길이를 넘지 않도록 합니다.

### 4.5 변수 Domain

$$
N_k\in\mathbb Z_+,
\qquad
CT_k\in\mathbb R_+,
\qquad
x_i^k\in\{0,1\}
$$

정의상 다음 Linking Relation도 필요합니다.

$$
N_k=\sum_{i=1}^{t}x_i^k
$$

다만 이 관계는 원문에서 기호정의상 암묵적으로 사용되고, 제약식으로 별도 인쇄되지는 않습니다.

### 4.6 원문 식 (6)의 부호 오류

원문은 Truck별 Class 배정을 다음과 같이 인쇄합니다.

$$
\sum_{k=1}^{p}x_i^k\ge1,
\qquad \forall i
\tag{원문 식 (6)}
$$

그러나 바로 뒤 본문은 이 식이 “Truck을 하나의 Product Class에 배정하거나 사용하지 않도록 한다”고 설명합니다. 그 설명을 만족하려면 수리적으로는

$$
\sum_{k=1}^{p}x_i^k\le1,
\qquad \forall i
$$

이어야 합니다. 원문의 $\ge1$은 모든 후보 Truck을 최소 한 Class에 강제로 배정할 뿐 아니라 여러 Class 동시배정도 막지 못해, Fleet 최소화 목적과 모순됩니다. 따라서 이는 단순한 방향 부호의 Typographical Error로 보는 것이 가장 타당하지만, 원문에 Corrigendum이 제시된 것은 아니므로 “저자의 확정 수정식”이 아니라 **본문 설명에 따른 수리적 해석**으로 구분해야 합니다.

---

## 5) 해법: MVA를 내장한 SQP와 DES 검증

### 5.1 왜 문제가 어려운가

$x_i^k$와 $N_k$는 이산변수이고, $CT_k$는 Fleet Allocation에 따라 Queueing Network에서 비선형적으로 변합니다. 논문은 Multi-class FCFS CQN에 이산 Fleet Allocation이 결합된 문제를 NP-hard 범주의 MINLP로 설명합니다.

### 5.2 Analytical Optimization

논문의 계산흐름은 다음과 같습니다.

1. 후보 Fleet Allocation $N=(N_1,\ldots,N_p)$ 생성
2. MVA로 Class별 $CT_k$, Queue Length, Utilization, Throughput 계산
3. 수요충족 및 Shift 제약 평가
4. SQP가 목적과 제약의 Local Quadratic Approximation을 구성
5. 다음 Fleet 후보로 이동하고 수렴할 때까지 반복

### Figure 3 — MVA를 내장한 SQP 계산흐름

<img width="1100" alt="fleet 후보에서 MVA로 cycle time과 throughput을 계산하고 SQP로 갱신하는 flowchart" src="/assets/img/paper-reviews/2026-08-20/amjath-fig3.svg" />

> Source: Amjath et al. (2022), Figure 3, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 원문 그림을 크롭했습니다. [Open-access article](https://doi.org/10.1016/j.joitmc.2022.100015)

각 Fleet 후보에 대해 MVA가 Cycle Time과 Throughput을 계산하고, 수요제약을 만족하면 SQP가 다음 해를 탐색합니다. 즉, Queueing Network의 성능평가가 별도 사후분석이 아니라 최적화 반복 안에 들어갑니다.

Nonlinear Optimization과 MVA는 Fortran 90으로 구현했고, IMSL Library의 `NNLPF` Subroutine이 Sequential Equality-Constrained Quadratic Programming을 수행합니다. 실험 Hardware는 Intel Core i3-7100U 2.40 GHz, RAM 4 GB입니다.

다만 일반적인 SQP는 연속비선형계획 기법입니다. 논문은 이를 MINLP의 Approximation Method로 사용한다고 설명하지만, Binary·Integer Variable을 매 iteration에서 어떻게 강제하거나 보정했는지에 대한 세부 Pseudocode는 충분히 제공하지 않습니다. 따라서 결과표는 재현 가능한 Target이지만 Solver Layer를 완전히 동일하게 복원하려면 추가 구현가정이 필요합니다.

### 5.3 Discrete-Event Simulation 검증

AnyLogic으로 별도의 DES를 만들고 Analytical Model의 해를 검증합니다.

- Simulation Horizon: 43,200 time units
- Warm-up: 1,000 time units
- Replication: 30회
- Confidence Interval: 95%
- Scenario별 Optimization Iteration: 5,000회
- Optimization Engine: AnyLogic OptQuest

Analytical Model은 평균값 기반으로 빠르게 Fleet 후보를 평가하고, DES는 개별 Event와 Queue의 시간진행을 표현해 결과가 현실적인지 확인하는 역할을 합니다.

---

## 6) Steel Manufacturing Case Study

### 6.1 시스템과 원자재 Class

사례기업은 연간 640만 톤의 Steel Rebar를 생산하는 지역 철강사이며, 본 연구는 Storage Yard에서 Billet Plant까지 세 원자재를 운반하는 내부물류를 다룹니다. 모든 Truck은 3PL에서 조달한 동종차량이지만, 적재하는 원료의 밀도가 달라 Class별 Full Truckload가 다릅니다.

| Class | Full Truckload |
|---|---:|
| A | 15 ton |
| B | 33 ton |
| C | 30 ton |

### 6.2 고정된 Class별 경로

각 Class의 Node Sequence는 사전에 정해져 있습니다.

$$
A: 1\rightarrow2\rightarrow3\rightarrow4\rightarrow5\rightarrow6
\rightarrow7\rightarrow8\rightarrow9\rightarrow10\rightarrow11\rightarrow12\rightarrow1
$$

$$
B: 1\rightarrow2\rightarrow13\rightarrow14\rightarrow15\rightarrow6
\rightarrow16\rightarrow17\rightarrow18\rightarrow10\rightarrow11\rightarrow19\rightarrow1
$$

$$
C: 1\rightarrow2\rightarrow13\rightarrow20\rightarrow15\rightarrow6
\rightarrow16\rightarrow17\rightarrow21\rightarrow22\rightarrow23\rightarrow19\rightarrow1
$$

예를 들어 Node 4는 Class A Loading, Node 14는 Class B Loading, Node 20은 Class C Loading입니다. Node 10은 A와 B가 공유하는 Unloading Preparation Station입니다. 주요 평균 Service Time은 Loading A 8분, Loading B 19분, Loading C 8분, Node 10의 Unloading Preparation 8분입니다.### Figure 6 — 실제 사례의 Multi-class Closed Queueing Network

<img width="1100" alt="원자재 class A B C별 truck route와 loading unloading shared resource node network" src="/assets/img/paper-reviews/2026-08-20/amjath-fig6.svg" />

> Source: Amjath et al. (2022), Figure 6, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 원문 그림을 크롭했습니다. [Open-access article](https://doi.org/10.1016/j.joitmc.2022.100015)

초록색은 이동·순환 Node, 황토색은 한 Class만 사용하는 Resource, 살구색은 여러 Class가 공유하는 Resource입니다. 특히 공유 Node의 Queue가 Fleet 증가에 따라 커지므로, Truck을 늘린다고 Throughput이 선형으로 증가하지 않습니다.



### 6.3 네 가지 BOM Scenario

각 Scenario는 12시간 Shift 동안 총 2,000 ton을 Billet Plant에 전달해야 하지만 Class Mix가 다릅니다.

| Scenario | A(ton) | B(ton) | C(ton) | 합계 |
|---|---:|---:|---:|---:|
| 1 | 350 | 550 | 1,100 | 2,000 |
| 2 | 500 | 700 | 800 | 2,000 |
| 3 | 650 | 800 | 550 | 2,000 |
| 4 | 800 | 800 | 400 | 2,000 |

총수요가 같아도 Class별 Truckload, Route, Loading Time, 공유병목 사용량이 다르므로 필요한 Fleet 구성도 달라집니다.

---

## 7) 실험 결과

### 7.1 최적 Fleet Size와 계산시간

| Scenario | Analytical Allocation $(A,B,C)$ | Analytical Time | DES Allocation $(A,B,C)$ | DES Time | 총 Truck |
|---|---:|---:|---:|---:|---:|
| 1 | $(3,2,4)$ | 3초 | $(3,2,4)$ | 23초 | 9 |
| 2 | $(4,3,3)$ | 3초 | $(4,3,3)$ | 24초 | 10 |
| 3 | $(5,3,2)$ | 3초 | $(5,3,2)$ | 24초 | 10 |
| 4 | $(7,3,2)$ | 3초 | $(7,3,2)$ | 25초 | 12 |

네 Scenario 모두에서 MVA–SQP와 AnyLogic Optimization이 동일한 정수 Fleet Allocation을 찾았습니다. 이 사례에서는 Analytical Method가 약 3초, DES가 23–25초로 Analytical Method가 더 빨랐습니다. 다만 Simulation 계산시간은 5,000회 iteration이라는 논문의 설정에 종속됩니다.

### 7.2 Utilization과 물류병목

선택한 주요 Node의 Analytical Utilization을 보면 다음과 같습니다.

| Scenario | Node 4: Load A | Node 10: Unload Prep A/B | Node 14: Load B | Node 23: Unload C |
|---|---:|---:|---:|---:|
| 1 | 0.342 | 0.564 | 0.526 | 0.150 |
| 2 | 0.420 | 0.737 | 0.755 | 0.119 |
| 3 | 0.494 | 0.798 | 0.724 | 0.083 |
| 4 | 0.607 | 0.885 | 0.661 | 0.083 |

Node 10은 Scenario 2–4에서 약 0.74–0.89의 높은 Utilization을 보이고, Node 14도 Scenario 2–3에서 혼잡합니다. 논문은 두 Station을 핵심 Bottleneck으로 식별합니다. Analytical과 DES의 Station Utilization 차이는 모든 비교 Instance에서 $\pm6\%$ 이내입니다.

### 7.3 Queue Response Time

분석모형과 Simulation의 Queue Response Time 차이는 보고한 모든 비교에서 $\pm7\%$ 이내입니다. 특히 이동구간인 Node 7, 13, 15, 16, 19의 차이는 $\pm3.5\%$ 이내였습니다. 이는 무한 Server로 모델링한 Circulation Node의 평균값이 DES와 비교적 가깝게 나온 결과입니다.

### 7.4 Class별 Cycle Time

| Scenario | A: Ana./Sim. | B: Ana./Sim. | C: Ana./Sim. |
|---|---:|---:|---:|
| 1 | 70.09 / 67.24분 | 72.37 / 75.30분 | 66.85 / 65.07분 |
| 2 | 76.27 / 72.86분 | 75.63 / 78.93분 | 63.22 / 64.72분 |
| 3 | 81.06 / 77.20분 | 78.85 / 82.00분 | 59.96 / 62.35분 |
| 4 | 92.33 / 88.01분 | 86.34 / 83.04분 | 60.03 / 63.64분 |

절대 Percentage Difference의 최댓값은 표에서 6%입니다. Fleet이 늘면 전체 Trip 수는 증가하지만 Truck 한 대당 평균 Trip 수는 감소합니다. 일정 수준 이후에는 Queue 때문에 Throughput의 한계증가가 작아지고 Cycle Time이 빠르게 커지므로, 차량만 추가하는 방식에는 명확한 Diminishing Return이 있습니다.

---

## 8) Sensitivity Analysis

### 8.1 Truck Capacity $\pm20\%$

Scenario 1과 2에서 Class별 적재중량을 동시에 $\pm20\%$ 변경합니다.

| Scenario | 변화 | 새 Allocation $(A,B,C)$ | 총 Truck | 기존 대비 |
|---|---:|---:|---:|---:|
| 1 | Capacity $-20\%$ | $(3,3,5)$ | 11 | $+22.22\%$ |
| 1 | Capacity $+20\%$ | $(2,2,3)$ | 7 | $-22.22\%$ |
| 2 | Capacity $-20\%$ | $(5,3,3)$ | 11 | $+10.00\%$ |
| 2 | Capacity $+20\%$ | $(3,2,2)$ | 7 | $-30.00\%$ |

Capacity 변화율과 Fleet 변화율은 일대일로 비례하지 않습니다. Class Mix와 Queueing Delay가 함께 바뀌기 때문입니다. Analytical Method와 DES는 이 Sensitivity Test에서도 동일한 Allocation을 산출합니다.

### 8.2 Bottleneck Service Time 개선

Class A와 B가 공유하는 Node 10의 Service Time을 8분에서 4분으로 50% 단축합니다.

| Scenario | 기존 Allocation | 개선 후 Allocation | 총 Fleet 변화 | Node 10 Ana. Utilization |
|---|---:|---:|---:|---:|
| 3 | $(5,3,2)$ | $(4,3,2)$ | $10\rightarrow9$ | $0.798\rightarrow0.423$ |
| 4 | $(7,3,2)$ | $(5,3,2)$ | $12\rightarrow10$ | $0.885\rightarrow0.505$ |

Bottleneck Station을 개선하면 단지 그 Node의 대기만 줄어드는 것이 아니라 전체 Cycle Time이 짧아져 Truck 한 대의 회전수가 증가합니다. 그 결과 동일 2,000 ton을 더 적은 Fleet으로 운반할 수 있습니다. 이 결과는 차량 증차와 공정 Service 개선을 동일한 Capa 관점에서 비교할 수 있음을 보여줍니다.

---

## 9) 핵심 기여, 강점과 한계

### 9.1 핵심 기여

1. **Fleet를 CQN Population으로 표현**  
   Truck 대수가 곧 Closed Network의 Customer 수이므로 Fleet Sizing과 Queueing Performance가 직접 연결됩니다.

2. **Multi-class와 Shared Station을 함께 고려**  
   원료마다 Route와 Truckload가 다르고, 일부 Loading·Weighing·Unloading Station은 여러 Class가 공유합니다.

3. **Demand Constraint를 Queueing Cycle Time과 결합**  
   단순히 `운송량 = Truck 수 × 고정 회전수`로 계산하지 않고, Fleet 증가로 변하는 $CT_k(N)$를 반영합니다.

4. **Analytical Optimization을 DES로 검증**  
   빠른 평균값 모형과 Event-based Simulation을 병행해 Fleet Size뿐 아니라 Utilization·Queue·Cycle Time까지 비교합니다.

5. **Sensitivity를 통한 병목개선 효과 정량화**  
   Truck Capacity와 Bottleneck Service Time이 최소 Fleet에 미치는 영향을 각각 평가합니다.

### 9.2 강점

- 실제 철강사 Process Map, Service Time, BOM Scenario를 사용한 산업 사례입니다.
- 총수요가 같아도 Product Mix에 따라 필요한 Fleet가 달라짐을 보여줍니다.
- Fleet 증차의 Queueing Diminishing Return을 수리적으로 반영합니다.
- 분석모형의 정수 Allocation이 네 Scenario 모두 DES와 일치합니다.
- Bottleneck Utilization을 Fleet 의사결정과 함께 산출하므로 개선 우선순위를 찾을 수 있습니다.

### 9.3 한계

- **Steel 사례:** 특정 사례는 Semiconductor Fab이 아니라 Storage Yard–Billet Plant 간 Bulk Material Transfer입니다.
- **경로선택 없음:** Class별 Route는 고정되어 있으며 Vehicle Routing, 최단경로, 충돌회피를 풀지 않습니다.
- **Homogeneous Fleet:** 차량별 Capacity·속도·비용 차이가 없습니다.
- **Deterministic Daily Demand:** 확률적 수요변동이나 Rolling Re-planning을 다루지 않습니다.
- **Queueing 가정:** Resource Node는 $M/M/1$, 이동은 $M/M/\infty$, FCFS, 무한 Buffer, 고장 없음으로 단순화합니다.
- **Blocking 미반영:** 통로 Capa, 교차로, 제한 Buffer로 생기는 Blocking과 Deadlock을 명시적으로 모델링하지 않습니다.
- **비용구조 단순화:** Truck 비용이 동질적이라 차량 수 최소화가 비용 최소화와 같다는 가정입니다.
- **SQP의 이산처리 설명 부족:** Integer·Binary Variable을 SQP 안에서 어떻게 처리했는지 상세 Algorithm이 충분하지 않습니다.
- **수식 표기 오류:** 식 (3), (4), (6)과 $N_k=\sum_i x_i^k$의 명시 여부 때문에 그대로 구현하면 다른 문제가 될 수 있습니다.
- **단일 Shift Steady-State:** 장기간의 Maintenance, Fleet 계약, 교대 간 연결은 포함하지 않습니다.

저자들은 향후 연구로 Heterogeneous Fleet와 차량별 Cost, Billet 이후 Finished Product Distribution, Open Queueing Network, Demand Fluctuation, Breakdown·Repair·Maintenance를 제시합니다.

---

## 10) 결론

본 논문은 Inter-facility Material Handling의 Fleet Sizing을 단순한 산술식이 아니라 **Fleet–Queue–Cycle Time–Demand가 서로 연결된 최적화 문제**로 다룹니다. Truck을 CQN의 고정 Population으로 두고 MVA로 Cycle Time을 계산한 뒤, “12시간 안에 Class별 Demand를 충족하는 최소 Fleet”을 MINLP로 정식화합니다. 네 개의 Steel BOM Scenario에서 Analytical Method와 DES가 동일한 9·10·10·12대의 Fleet를 찾았고, Station Performance도 대체로 $\pm7\%$ 범위에서 일치했습니다.

이 연구의 중요한 통찰은 차량 수만 늘리는 것이 항상 효율적이지 않다는 점입니다. Fleet이 커지면 Single-Server Loading·Unloading Station의 Queue가 길어져 한 대당 회전수가 감소합니다. 실제 Sensitivity에서도 Node 10의 Service Time을 절반으로 줄이자 Scenario 3은 10대에서 9대, Scenario 4는 12대에서 10대로 최소 Fleet가 감소했습니다.

다만 논문의 적용범위를 정확히 읽어야 합니다. 이는 **Steel Industry의 고정경로 Truck Fleet Sizing 연구**이며, 반도체 물류 실증도, 차량의 Route Choice Optimization도 아닙니다. 가장 정확한 한 줄 요약은 다음과 같습니다.

> 고정된 다품종 순환 물류망에서 Queueing으로 변하는 Cycle Time을 계산하고, 정해진 시간 내 수요를 충족하는 최소 차량대수와 병목을 함께 찾는 OR 연구이다.
