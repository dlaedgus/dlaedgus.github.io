---
title: "Fleet Sizing of Trucks for an Inter-Facility Material Handling System Using Closed Queueing Networks (ORP 2022)"
date: 2026-08-20 12:50:00 +0900
categories: [paper_review, OR]
tags: [fleet-sizing, closed-queueing-network, mean-value-analysis, MINLP, simulation]
math: true
---

# Paper Review — *Fleet Sizing of Trucks for an Inter-Facility Material Handling System Using Closed Queueing Networks* (ORP 2022)

- **제목:** Fleet Sizing of Trucks for an Inter-Facility Material Handling System Using Closed Queueing Networks  
- **1저자:** Mohamed Amjath  
- **공저자:** Laoucine Kerbache, James MacGregor Smith, Adel Elomri  
- **저널:** *Operations Research Perspectives*, Vol. 9, Article 100245  
- **년도:** 2022  
- **DOI:** [10.1016/j.orp.2022.100245](https://doi.org/10.1016/j.orp.2022.100245)  
- **원문:** [ScienceDirect Open Access](https://www.sciencedirect.com/science/article/pii/S2214716022000185)

---

## 1) High-Level Summary (3–5 sentences)

Storage Yard와 생산시설 사이를 반복해서 도는 Truck을 **Closed Queueing Network(CQN)의 Customer**로 놓고, 정해진 시간 안에 수요를 처리할 최소 Fleet를 찾는 연구. Loading·Weighing·Unloading Station은 Queueing Node, 이동구간은 Infinite-server Node로 표현하고 Mean Value Analysis(MVA)로 Fleet 수에 따라 달라지는 Cycle Time을 계산함. 이 값을 MINLP에 넣어 원료 Class별 Truck 수를 정하고 AnyLogic DES로 검증. 실제 철강 사례의 네 BOM Scenario에서 Analytical Model과 Simulation이 모두 9·10·10·12대를 선택했고, Cycle Time/Queue/Utilization도 대체로 $\pm7\%$ 안에서 일치. 차량 경로는 고정되어 있으며 Vehicle Routing이 아니라 **Fleet Sizing + Bottleneck Analysis** 문제다.

---

## 2) 문제: Truck을 늘리면 처리량도 계속 선형으로 늘어나는가?

생산시설은 BOM에 따라 A/B/C 원자재를 요구한다. Truck은 Yard에서 적재 → Weighing/Inspection → 생산시설 하역 → Yard 복귀를 반복.

Fleet이 작으면 12시간 Shift 안에 수요를 못 맞춘다. 그런데 Truck을 계속 추가한다고 운송량이 같은 비율로 늘지는 않음. Loading이나 Unloading Station은 한 번에 한 대만 처리할 수 있기 때문에 Fleet이 커질수록 Queue와 Cycle Time도 커진다.

$$
\text{Fleet 증가}
\rightarrow
\begin{cases}
\text{동시에 운반하는 Truck 증가}\\
\text{Station Queue 증가}\\
\text{Truck 한 대의 회전수 감소}
\end{cases}
$$

따라서 단순한

$$
\text{운송량}=\text{Truck 수}\times\text{고정 회전수}\times\text{적재량}
$$

으로 계산할 수 없다. 회전수에 들어가는 $CT_k$ 자체가 Truck 수 $N$의 함수이기 때문.

논문의 결정:

- 전체 Fleet Size
- 원자재 Class별 Truck Allocation $(N_A,N_B,N_C)$

고정되는 것:

- Class별 이동 Route
- Station 방문순서
- 동종 Truck의 속도/Capacity 구조

---

## 3) Closed Queueing Network

Truck은 운송을 끝내도 시스템 밖으로 사라지지 않고 다시 Yard로 돌아온다. 그래서 외부에서 Customer가 들어왔다 나가는 Open Network보다, 고정 Population이 계속 도는 Closed Network가 맞음.

- **Customer:** Truck
- **Class $k$:** Truck이 운반하는 원자재
- **Resource Node:** Gate, Loading Dock, Weighbridge, Unloading Dock
- **Circulation Node:** Node 사이 이동구간
- **Population $N_k$:** Class $k$에 배정된 Truck 수

Resource Node는 FCFS $M/M/1$, 이동구간은 $M/M/\infty$. Buffer는 무한, Breakdown은 없다고 가정한다.

주요 기호:

- $j=1,\ldots,s$: Station
- $k=1,\ldots,p$: Product Class
- $N$: 총 Truck 수
- $N_k$: Class $k$ Truck 수
- $T_j$: 평균 Service Time
- $\mu_j^k$: Class $k$의 Station $j$ Service Rate
- $V_j^k$: 한 Cycle 중 Station $j$ 방문횟수
- $Y_j^k(N)$: Station Response Time
- $L_j^k(N)$: 평균 Queue Length
- $\Theta_k(N)$: Class Throughput
- $CT_k$: 한 Cycle의 시간

### Mean Value Analysis

Single-class, Single-server Station의 Response Time:

$$
Y_j(N)
=T_j+T_jL_j(N-1)
=T_j\left[1+L_j(N-1)\right]
$$

Service Time $T_j$에, 도착했을 때 앞에 있는 평균 Customer를 처리하는 시간이 더해진다.

논문의 Multi-class 식:

$$
Y_j^k(N)
=
\frac1{\mu_j^k}
\left[1+L_j^k(N-1)\right]
$$

Class $k$ Cycle Time은 각 Station Response Time을 방문횟수로 합한 값.

$$
CT_k(N)
=
\sum_{j=1}^{s}V_j^kY_j^k(N)
$$

그 다음 Little's Law로 Throughput과 Queue를 갱신한다.

$$
\Theta_k(N)
=
\frac{N_k}{\sum_{j=1}^{s}V_j^kY_j^k(N)}
=
\frac{N_k}{CT_k(N)}
$$

$$
L_j^k(N)
=
\Theta_k(N)V_j^kY_j^k(N)
$$

Population 1부터 목표 Fleet Vector까지 반복하면 Allocation마다 $CT$, Throughput, Queue, Utilization을 계산할 수 있다.

여기서 핵심 관계:

$$
N_k\uparrow
\Rightarrow
L_j^k\uparrow
\Rightarrow
Y_j^k\uparrow
\Rightarrow
CT_k\uparrow
$$

$N_k$가 늘면 분자도 늘지만 $CT_k(N)$도 늘어난다. 그래서 $\Theta_k=N_k/CT_k$는 Diminishing Return을 가짐.

### 원문 식에서 주의할 부분

원문 식 (3)은 Class $k$ Throughput 분자에 $N_k$ 대신 $N$을, 식 (4)은 Little's Law의 Throughput 자리에 $\Theta_k(N)$ 대신 $\mu_j^k(N)$를 인쇄한다. 기호정의와 표준 MVA 관계를 보면 위의 $N_k$, $\Theta_k$ 형태가 자연스럽다.

엄밀한 Multi-class MVA라면 Class $k$ 한 대가 빠진 $N-e_k$를 사용해야 하는데 원문은 $N-1$로 표기. 공유 FCFS Node에서도 다른 Class의 Queue를 같이 봐야 하지만 식 (2)는 $L_j^k$만 사용한다. 논문 PDF의 식을 그대로 옮기기보다 Tailored MVA가 Shared Node를 어떻게 처리했는지 확인이 필요한 부분.

---

## 4) Fleet Sizing MINLP

Truck $i$의 Class 배정변수:

$$
x_i^k=
\begin{cases}
1,&\text{Truck }i\text{를 Class }k\text{에 배정}\\
0,&\text{그 외}
\end{cases}
$$

- $F_k$: Trip당 적재량
- $D_k$: 운송수요
- $T$: Shift 길이
- $Z_i^k$: Truck $i$의 Shift 내 Trip 수
- $\beta$: Fleet 상한

목적함수는 사용 Truck 수 최소화.

$$
\min N
=
\sum_{i=1}^{t}\sum_{k=1}^{p}x_i^k
$$

Class별 Truck 수 연결:

$$
N_k=\sum_{i=1}^{t}x_i^k
$$

원문에는 이 식이 따로 인쇄되지는 않고 정의상 사용됨.

가장 중요한 수요제약:

$$
N_k\cdot\frac{T}{CT_k}\cdot F_k
\ge D_k,
\qquad k=1,\ldots,p
$$

$T/CT_k$가 한 대의 평균 회전수, $N_k(T/CT_k)$가 전체 Trip Capacity. 그런데 $CT_k=CT_k(N)$이므로 이 제약은 Nonlinear.

Fleet 상한과 Shift 제약:

$$
\sum_{i=1}^{t}\sum_{k=1}^{p}x_i^k\le\beta
$$

$$
Z_i^kCT_k\le T,
\qquad \forall i,k
$$

Domain:

$$
N_k\in\mathbb Z_+,
\qquad CT_k\in\mathbb R_+,
\qquad x_i^k\in\{0,1\}
$$

### 식 (6)의 부등호

원문에는 Truck별 Class 배정이

$$
\sum_{k=1}^{p}x_i^k\ge1,
\qquad \forall i
\tag{원문 식 (6)}
$$

로 인쇄되어 있다. 그런데 본문 설명은 “Truck을 한 Class에 배정하거나 사용하지 않음”. 이 설명에 맞는 식은

$$
\sum_{k=1}^{p}x_i^k\le1,
\qquad \forall i
$$

이다. $\ge1$이면 모든 후보 Truck을 강제로 쓰게 되고 한 Truck의 여러 Class 배정도 막지 못해서 최소 Fleet 목적과 모순. 재현할 때는 본문을 따른 수정식으로 구현하되, 저자 Corrigendum이 아니라 해석에 따른 수정임을 남겨야 한다.

---

## 5) MVA–SQP와 DES

문제에는 Integer Fleet Allocation과 $CT_k(N)$의 Nonlinearity가 같이 들어감. 논문의 Analytical 흐름:

1. 후보 Fleet Vector $N=(N_1,\ldots,N_p)$ 생성
2. MVA로 $CT_k$, Queue, Utilization, Throughput 계산
3. 수요/Shift Constraint 평가
4. SQP가 Local Quadratic Approximation 구성
5. 다음 Fleet 후보로 이동

Fortran 90과 IMSL `NNLPF`를 사용. Hardware는 Intel Core i3-7100U 2.40GHz, RAM 4GB.

SQP는 기본적으로 Continuous NLP 방법인데 논문은 MINLP Approximation으로 사용한다고 설명한다. Binary/Integer를 Iteration마다 어떻게 강제했는지는 Pseudocode가 충분하지 않음. 따라서 같은 결과표를 Target으로 삼을 수는 있어도 Solver 내부까지 완전히 동일하게 복원하기는 어렵다.

### Figure 3 — MVA를 내장한 SQP 계산흐름

<img width="1100" alt="fleet 후보에서 MVA로 cycle time과 throughput을 계산하고 SQP로 갱신하는 flowchart" src="/assets/img/paper-reviews/2026-08-20/amjath-fig3.svg" />

> Source: Amjath et al. (2022), Figure 3, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 원문 그림을 크롭했습니다. [Open-access article](https://doi.org/10.1016/j.orp.2022.100245)

MVA가 사후분석에 붙는 것이 아니라 Optimization Loop 안에서 후보 Fleet의 $CT$와 Throughput을 계속 다시 계산한다.

검증은 AnyLogic DES로 수행.

- Horizon: 43,200 time units
- Warm-up: 1,000
- Replication: 30회
- Confidence Interval: 95%
- Scenario별 Optimization Iteration: 5,000회
- Engine: AnyLogic OptQuest

Analytical Model은 평균값으로 후보를 빠르게 평가하고, DES는 개별 Event와 Queue의 시간흐름을 확인하는 역할.

---

## 6) Steel Manufacturing Case

연간 640만 톤 Steel Rebar 생산기업의 Storage Yard–Billet Plant 내부물류. Truck은 모두 동종이지만 원료밀도가 달라 Full Truckload가 다름.

| Class | Full Truckload |
|---|---:|
| A | 15 ton |
| B | 33 ton |
| C | 30 ton |

Class별 Route는 고정.

$$
A:1\rightarrow2\rightarrow3\rightarrow4\rightarrow5\rightarrow6
\rightarrow7\rightarrow8\rightarrow9\rightarrow10\rightarrow11\rightarrow12\rightarrow1
$$

$$
B:1\rightarrow2\rightarrow13\rightarrow14\rightarrow15\rightarrow6
\rightarrow16\rightarrow17\rightarrow18\rightarrow10\rightarrow11\rightarrow19\rightarrow1
$$

$$
C:1\rightarrow2\rightarrow13\rightarrow20\rightarrow15\rightarrow6
\rightarrow16\rightarrow17\rightarrow21\rightarrow22\rightarrow23\rightarrow19\rightarrow1
$$

Node 4/14/20은 A/B/C Loading. Node 10은 A와 B가 공유하는 Unloading Preparation. 주요 Service Time은 Load A 8분, Load B 19분, Load C 8분, Node 10은 8분.

### Figure 6 — 실제 사례의 Multi-class Closed Queueing Network

<img width="1100" alt="원자재 class A B C별 truck route와 loading unloading shared resource node network" src="/assets/img/paper-reviews/2026-08-20/amjath-fig6.svg" />

> Source: Amjath et al. (2022), Figure 6, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 원문 그림을 크롭했습니다. [Open-access article](https://doi.org/10.1016/j.orp.2022.100245)

초록색은 이동 Node, 황토색은 한 Class 전용 Resource, 살구색은 Shared Resource. Fleet가 늘 때 주로 문제가 되는 부분은 여러 Class가 같이 사용하는 Single-server Node.

12시간 동안 총 2,000 ton을 운반하되 Mix가 다른 네 Scenario:

| Scenario | A(ton) | B(ton) | C(ton) | 합계 |
|---|---:|---:|---:|---:|
| 1 | 350 | 550 | 1,100 | 2,000 |
| 2 | 500 | 700 | 800 | 2,000 |
| 3 | 650 | 800 | 550 | 2,000 |
| 4 | 800 | 800 | 400 | 2,000 |

총량은 같지만 Truckload, Route, Service Time, Shared-node 사용량이 달라 필요한 Fleet도 달라진다.

---

## 7) Result

### Fleet Size

| Scenario | Analytical $(A,B,C)$ | Ana. Time | DES $(A,B,C)$ | DES Time | 총 Truck |
|---|---:|---:|---:|---:|---:|
| 1 | $(3,2,4)$ | 3초 | $(3,2,4)$ | 23초 | 9 |
| 2 | $(4,3,3)$ | 3초 | $(4,3,3)$ | 24초 | 10 |
| 3 | $(5,3,2)$ | 3초 | $(5,3,2)$ | 24초 | 10 |
| 4 | $(7,3,2)$ | 3초 | $(7,3,2)$ | 25초 | 12 |

네 Scenario 모두 같은 Integer Allocation. Analytical 3초, DES 23~25초. DES 시간은 5,000 Iteration이라는 설정에 종속된 값.

### Utilization과 Bottleneck

| Scenario | Node 4: Load A | Node 10: Unload Prep A/B | Node 14: Load B | Node 23: Unload C |
|---|---:|---:|---:|---:|
| 1 | 0.342 | 0.564 | 0.526 | 0.150 |
| 2 | 0.420 | 0.737 | 0.755 | 0.119 |
| 3 | 0.494 | 0.798 | 0.724 | 0.083 |
| 4 | 0.607 | 0.885 | 0.661 | 0.083 |

Node 10은 Scenario 2~4에서 0.74~0.89, Node 14도 Scenario 2~3에서 높음. 두 Station이 주요 Bottleneck. Analytical/DES Utilization 차이는 $\pm6\%$ 안.

Queue Response Time 차이는 보고된 비교에서 $\pm7\%$ 안, 이동구간 Node 7/13/15/16/19는 $\pm3.5\%$ 안.

### Cycle Time

| Scenario | A: Ana./Sim. | B: Ana./Sim. | C: Ana./Sim. |
|---|---:|---:|---:|
| 1 | 70.09 / 67.24분 | 72.37 / 75.30분 | 66.85 / 65.07분 |
| 2 | 76.27 / 72.86분 | 75.63 / 78.93분 | 63.22 / 64.72분 |
| 3 | 81.06 / 77.20분 | 78.85 / 82.00분 | 59.96 / 62.35분 |
| 4 | 92.33 / 88.01분 | 86.34 / 83.04분 | 60.03 / 63.64분 |

최대 Absolute Percentage Difference는 6%. Fleet가 늘수록 총 Trip은 증가하지만 Truck 한 대당 Trip 수는 감소하는 패턴이 확인됨.

---

## 8) Sensitivity

Truck Capacity를 $\pm20\%$ 바꾼 결과:

| Scenario | 변화 | 새 Allocation $(A,B,C)$ | 총 Truck | 기존 대비 |
|---|---:|---:|---:|---:|
| 1 | Capacity $-20\%$ | $(3,3,5)$ | 11 | $+22.22\%$ |
| 1 | Capacity $+20\%$ | $(2,2,3)$ | 7 | $-22.22\%$ |
| 2 | Capacity $-20\%$ | $(5,3,3)$ | 11 | $+10.00\%$ |
| 2 | Capacity $+20\%$ | $(3,2,2)$ | 7 | $-30.00\%$ |

Capacity 변화와 Fleet 변화가 1:1로 비례하지 않는다. Class Mix와 Queueing이 같이 바뀌기 때문.

더 흥미로운 것은 Node 10의 Service Time을 8분에서 4분으로 줄인 실험.

| Scenario | 기존 Allocation | 개선 후 Allocation | 총 Fleet 변화 | Node 10 Utilization |
|---|---:|---:|---:|---:|
| 3 | $(5,3,2)$ | $(4,3,2)$ | $10\rightarrow9$ | $0.798\rightarrow0.423$ |
| 4 | $(7,3,2)$ | $(5,3,2)$ | $12\rightarrow10$ | $0.885\rightarrow0.505$ |

Station Service Time 감소 → Queue 감소 → 전체 $CT_k$ 감소 → 한 대의 회전수 증가 → 필요한 Fleet 감소.

즉, 차량을 추가하는 것과 Station을 개선하는 것을 같은 Capacity 관점에서 비교할 수 있다.

---

## 9) 읽으면서 남는 부분

이 논문의 장점은 Fleet Size와 Queueing Performance를 따로 계산하지 않는다는 점. Truck 수가 바뀔 때마다 MVA가 $CT_k(N)$을 다시 계산하고, 그 값이 바로 수요제약으로 돌아간다.

$$
N
\rightarrow CT(N)
\rightarrow \text{Trip Capacity}
\rightarrow \text{Demand Feasibility}
\rightarrow N
$$

실제 Process Map, Service Time, BOM Mix를 썼고, Analytical Allocation이 네 Scenario 모두 DES와 일치한 것도 강점. Node Utilization까지 나오기 때문에 “Truck이 부족한가, Loading/Unloading Service가 부족한가”를 같이 볼 수 있다.

반면 적용범위는 다음과 같음.

- Steel Yard–Billet Plant 사례
- Class별 Route 고정, Vehicle Routing/Conflict 없음
- Homogeneous Fleet
- Daily Demand 고정, Rolling Re-planning 없음
- $M/M/1$, $M/M/\infty$, FCFS, Infinite Buffer, No Breakdown 가정
- Blocking/Deadlock과 통로 Capa 미반영
- SQP에서 Integer/Binary를 처리하는 세부절차 부족
- Multi-class MVA 식과 식 (6)에 표기상 모호성 존재
- Single-shift Steady-state 분석

---

## 10) 정리

단순히 Truck 수를 최소화하는 문제처럼 보이지만 실제 구조는 다음과 같다.

$$
\text{Fleet Allocation}
\rightarrow
\text{Queue Length}
\rightarrow
\text{Cycle Time}
\rightarrow
\text{Shift 내 운송량}
$$

Truck을 CQN Population으로 두고 MVA로 $CT_k(N)$을 계산한 뒤, 12시간 안에 Class별 수요를 채우는 최소 $N_k$를 찾는다. 네 BOM Scenario에서는 9·10·10·12대가 선택됐고 DES도 같은 Allocation을 냄.

Sensitivity에서 Node 10 개선만으로 Scenario 3/4의 Fleet가 10→9대, 12→10대로 줄어든 결과가 핵심. 차량 증차 이전에 Shared Station의 Service Capacity를 먼저 봐야 한다는 것을 수리적으로 보여준다.
