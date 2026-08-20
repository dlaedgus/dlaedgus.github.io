---
title: "Route Planning for Two Wafer Fabs with Capacity-Sharing Mechanisms (IJPR 2009)"
date: 2026-08-20 12:40:00 +0900
categories: [paper_review, OR]
tags: [semiconductor-manufacturing, capacity-sharing, linear-programming, genetic-algorithm, queueing-network]
math: true
---

# Paper Review — *Route Planning for Two Wafer Fabs with Capacity-Sharing Mechanisms* (IJPR 2009)

- **제목:** Route Planning for Two Wafer Fabs with Capacity-Sharing Mechanisms  
- **1저자:** Muh-Cherng Wu  
- **공저자:** Chen-Fu Chen, Chang-Fu Shih  
- **저널:** *International Journal of Production Research*, 47(20), 5843–5856  
- **년도:** 2009  
- **DOI:** [10.1080/00207540802172029](https://doi.org/10.1080/00207540802172029)  
- **원문:** [National Yang Ming Chiao Tung University Repository](https://ir.lib.nycu.edu.tw/items/df5811e0-1f8f-43d9-b14d-d3fcfb4bdcba)

---

## 1) High-Level Summary (3–5 sentences)

서로 이웃한 두 Wafer FAB이 Capa를 공유할 때 제품별 공정분할점(cut-off point)과 네 가지 생산 Route의 물량비율을 정하는 연구. LP로 Cross-FAB LOT이 적은 Cut-off를 먼저 고르고, Queueing Network와 GA로 목표 Cycle Time을 만족하는 Route Ratio를 찾음. HP fab 변형 데이터에서 중앙 Cut-off 또는 Cross-FAB 금지보다 Cycle Time과 Throughput이 좋아짐. 여기서 Route는 차량의 물리경로가 아니라 제품의 Process Route를 어느 FAB가 담당할지에 대한 구분이다.

---

## 2) 이 논문에서 말하는 Route Planning

먼저 Route의 의미를 구분해야 한다. 이 논문은 AGV/OHT가 어느 통로로 갈지, 두 FAB 사이 최단경로를 어떻게 찾을지 다루지 않음.

여기서 Route는 **제품의 공정 sequence를 Fab A와 Fab B에 어떻게 나눌지**를 뜻한다.

결정하는 것:

1. 제품 공정을 어느 지점에서 나눌지: cut-off point $\pi_i$
2. 네 가지 생산 route에 물량을 얼마나 보낼지: route ratio $r_i$

물리적인 운송 path는 두 station 사이에 하나만 있다고 가정한다. 따라서 이 연구는 physical routing이 아니라 **process route + cross-fab volume allocation** 문제.

두 FAB이 같은 제품을 모두 만들 수 있어도 Product Mix와 장비 대수가 다르면 한쪽 Workstation은 막히고 다른 쪽은 남을 수 있다. 제품 Route 일부를 다른 FAB에서 처리하면 이 불균형을 줄일 수 있지만, 너무 많은 LOT을 넘기면 운송과 Queue가 커짐. 논문은 이 둘을 같이 보려 한다.

전체 흐름:

$$
\text{Cut-off 후보}
\xrightarrow{LP}
\text{Cross-FAB LOT이 작은 }\Pi^*
\xrightarrow{Queueing+GA}
\text{Route Ratio }R^*
$$

같은 Throughput에서 Cycle Time을 줄이거나, 같은 Cycle Time에서 Throughput을 늘리는 것이 최종 목표다.

---

## 3) Dual-FAB 문제 정의

제품 $i$의 cut-off point를 $\pi_i$라고 하면 네 가지 Route가 가능.

- $A$: 전 공정을 Fab A에서 수행
- $B$: 전 공정을 Fab B에서 수행
- $A\rightarrow B$: 앞 구간은 A, 뒤 구간은 B
- $B\rightarrow A$: 앞 구간은 B, 뒤 구간은 A

제품별 Route Ratio:

$$
r_i=(a_i,b_i,c_i,d_i)
$$

$a_i,b_i,c_i,d_i$는 각각 위 네 Route에 배정되는 제품 $i$의 물량비율.

논문에는 세 가지 큰 가정이 있다.

- **Functional comprehensiveness:** 두 FAB 모두 도움 없이 각 제품을 끝까지 생산 가능
- **One cut-off point:** 제품 하나가 FAB을 바꾸는 횟수는 최대 한 번
- **Unique transportation path:** Workstation/Buffer 사이 물리경로는 고정

정리하면 $\Pi=[\pi_1,\ldots,\pi_n]$와 $R=[r_1,\ldots,r_n]$을 정하고, 평균 Cycle Time $CT\le CT_0$에서 두 FAB의 총 Throughput을 높이는 문제다.

---

## 4) Module 1 — LP로 Cut-off Point 고르기

첫 단계에서는 Queue와 시간축을 빼고 Workstation Capa만 본다. 특정 $\Pi$가 주어졌을 때 목표 생산량 $Q$를 처리하면서 FAB 경계를 넘는 LOT 수를 최소화.

**기호**

- $i=1,\ldots,n$: 제품
- $g=1,\ldots,m_a$: Fab A Workstation
- $h=1,\ldots,m_b$: Fab B Workstation
- $Q$: 높은 가동률에서 처리할 목표 Throughput
- $P_i$: Product Mix 비율, $\sum_iP_i=1$
- $C_g,C_h$: Workstation별 Available Machine Hour
- $W^a_{ig}$: 제품 $i$를 Route A로 생산할 때 Fab A의 $g$에서 필요한 LOT당 시간
- $W^c_{ig},W^d_{ig}$: Cross-FAB Route가 Fab A의 $g$에서 사용하는 LOT당 시간
- $W^b_{ih},W^c_{ih},W^d_{ih}$: Fab B의 Workload

$W$는 단일 Operation Time이 아니라 해당 Workstation을 재방문하는 시간을 모두 합친 LOT당 Workload.

### 목적함수

$$
\min Z(\Pi)
=
\sum_{i=1}^{n}QP_i(c_i+d_i)
$$

$c_i+d_i$가 FAB을 한 번 넘는 물량비율. 목표량 $Q$ 중 Cross-FAB LOT 수를 줄이는 식이다.

왜 이동량을 목적함수로 썼나? 같은 Capa feasibility라면 FAB을 넘는 LOT이 적을수록 추가 운송시간과 운영 복잡성이 작다고 본 것. 단, 이 단계에서는 운송 Queue를 직접 계산하지 않는다.

### 물량보존

$$
a_i+b_i+c_i+d_i=1,
\qquad i=1,\ldots,n
$$

제품별 전체 물량을 네 Route에 나눔. 각 비율은 $[0,1]$.

### Fab A Capa

$$
\sum_{i=1}^{n}QP_i
\left(
a_iW^a_{ig}
+d_iW^d_{ig}
+c_iW^c_{ig}
\right)
\le C_g,
\qquad g=1,\ldots,m_a
$$

Fab A에서 끝까지 처리하는 물량, $A\rightarrow B$의 앞부분, $B\rightarrow A$의 뒷부분이 모두 Fab A Capa를 사용.

### Fab B Capa

$$
\sum_{i=1}^{n}QP_i
\left(
b_iW^b_{ih}
+d_iW^d_{ih}
+c_iW^c_{ih}
\right)
\le C_h,
\qquad h=1,\ldots,m_b
$$

Fab B도 같은 방식.

이 LP가 답하는 질문은 다음과 같다.

> Product Mix와 목표량 $Q$가 주어졌을 때, 두 FAB의 Workstation Capa를 넘지 않으면서 물량을 네 Route에 어떻게 나눌 수 있는가? 그중 FAB을 넘는 LOT이 가장 적은 해는 무엇인가?

$Q$는 이미 주어져 있으므로 Module 1 자체가 Throughput을 maximize하는 것은 아님. $Q$를 처리할 수 있는 범위에서 Cross-FAB LOT을 줄이는 Surrogate Problem이다.

---

## 5) Cut-off Search + Queueing Network + GA

### Cut-off Point Search

Route가 150~172개 Operation이면 모든 Cut-off 조합을 직접 탐색하기 어렵다. 각 제품의 현재 후보구간에서 1/4, 3/4 지점 두 개를 후보로 만들고 한 Iteration에 $2^n$개 조합을 평가한다.

1. 제품별 Cut-off 후보 2개 생성
2. $2^n$개 조합마다 LP 풀이
3. $Z(\Pi)$가 작은 $\Pi_i^*$ 선택
4. 선택된 Cut-off가 포함된 절반 구간만 유지
5. 원하는 해상도까지 반복

제품 Route 길이 $m$이 $2^{x-1}<m\le2^x$이면 LP 실행횟수는

$$
N_{LP}=x\cdot2^n
$$

Operation 수에는 Log Scale이지만 제품 수 $n$에는 Exponential.

### Queueing Network

Module 1은 운송시간 0, 운송 Capa 무한대로 놓는다. 두 번째 단계에서는 Workstation뿐 아니라 각 고정 운송 Path도 **Capa 1인 Conveyor Machine**으로 모델링한다.

$$
CT=f(TH,R,\Pi)
$$

- $TH$: 총 Throughput
- $R$: Route Ratio
- $\Pi$: Cut-off Point
- $CT$: 평균 Cycle Time

$TH$가 커지면 장비와 운송구간의 Traffic Intensity가 올라가고 Queueing Delay도 증가. 목표 $CT_0$와 $(R,\Pi^*)$가 주어지면 $TH$를 Binary Search해서

$$
f(TH,R,\Pi^*)=CT_0
$$

에 해당하는 최대 $TH$를 찾고, 이 값을 GA Fitness로 사용한다.

### GA로 Route Ratio Search

Chromosome:

$$
R=[r_1,\ldots,r_n],
\qquad r_i=(a_i,b_i,c_i,d_i)
$$

각 제품의 네 비율은 합이 1이어야 한다.

- Initialization: feasible chromosome $N_p$개 생성
- Crossover: Parent의 일부 Gene을 교환하고 합이 1이 되도록 복구
- Mutation: 한 제품 안에서 Gene 두 개 교환
- Selection: 기존/신규 Population 중 Fitness가 높은 $N_p$개 유지
- Stop: Best Solution이 일정 기간 그대로이거나 최대 Iteration 도달

논문 설정은 $P_0=100$, $P_{cr}=0.8$, $P_m=0.1$, $T_b=1000$, $T_u=30$. 다만 본문이 설명하는 두 종료기호와 수치의 크기 순서가 자연스럽지 않아, 원 코드 없이 그대로 재현하기에는 모호한 부분이 있다.

### 왜 두 단계로 나눴나

$$
\underbrace{\text{LP: static workload}}_{\Pi\text{ 선택}}
\rightarrow
\underbrace{\text{Queueing+GA: congestion}}_{R\text{ 선택}}
$$

모든 Cut-off마다 Queueing Model과 GA까지 돌리면 계산량이 너무 커짐. 먼저 Cross-FAB LOT이 작은 Cut-off를 LP로 고정하고, 그 다음 Route Ratio만 자세히 탐색한다.

계산은 줄지만 $\Pi$와 $R$을 Joint Optimization하지 않으므로 Global Optimum은 보장하지 않는다.

### Figure 1 — 두 단계 해법의 전체 구조

<img width="1000" alt="cutoff point를 LP로 정하고 route ratio를 GA와 queueing network로 정하는 solution framework" src="/assets/img/paper-reviews/2026-08-20/wu-fig1.svg" />

> Source: Wu et al. (2009), Figure 1. 논문 이해를 위한 일부 인용 및 크롭. [Original article](https://doi.org/10.1080/00207540802172029)

그림에서도 Module 1의 Cut-off가 Module 2로 한 방향으로 넘어간다. 빠르게 문제를 나누는 대신, 뒤에서 좋은 Route Ratio를 찾더라도 앞의 Cut-off를 다시 바꾸지는 않는 구조.

---

## 6) Experiment Setting

실제 기업의 현재 FAB Log가 아니라 Wein(1988)의 HP fab data를 변형한 수치실험.

- Fab A: Machine 93대
- Fab B: Machine 72대
- 각 FAB: Batch Workstation 4개, Series Workstation 21개
- Failure: MTBF/MTTR, Exponential Distribution
- 제품 3종
- Operation: Product 1/2는 172개, Product 3은 150개

두 FAB의 장비대수는 다르지만 세 제품을 각각 처음부터 끝까지 생산할 수 있다고 가정.

비교방법:

- **LP-GA:** LP Cut-off Search + Queueing Network + GA
- **M-GA:** Cut-off를 Route 중앙으로 고정하고 GA
- **N-GA:** Cross-FAB 생산 금지

두 Product Mix:

$$
R_A=(3:2:5),
\qquad
R_B=(5:4:1)
$$

높은 Utilization을 만드는 목표 Throughput은 $Q_A=128$, $Q_B=169$ LOT.

| Product | 총 Operation | $R_A$ Cut-off | $R_B$ Cut-off |
|---|---:|---:|---:|
| 1 | 172 | 85번째 | 84번째 |
| 2 | 172 | 85번째 | 84번째 |
| 3 | 150 | 129번째 | 78번째 |

Product 3은 Mix가 바뀌자 Cut-off가 129에서 78로 크게 이동. Route 중앙을 고정하는 방식이 항상 Capa Balance에 맞지 않는다는 예.

---

## 7) Result

### 같은 Throughput에서 Cycle Time

| Product Mix | 방법 | Throughput | 평균 CT(min) | LP-GA 대비 Gap |
|---|---|---:|---:|---:|
| $R_A$ | LP-GA | 128 | 11,080 | 0% |
| $R_A$ | M-GA | 128 | 12,175 | 9.88% |
| $R_A$ | N-GA | 128 | 12,463 | 12.48% |
| $R_B$ | LP-GA | 169 | 11,639 | 0% |
| $R_B$ | M-GA | 169 | 12,811 | 10.06% |
| $R_B$ | N-GA | 169 | 14,075 | 20.90% |

LP-GA는 Midpoint보다 약 10%, Cross-FAB 금지보다 약 12~21% 짧은 CT. 일부 LOT을 다른 FAB로 넘겨 병목부하를 나누는 효과가 추가 운송부담보다 컸던 Scenario다.

### 같은 Cycle Time에서 Throughput

$R_A$는 $CT_0=11{,}081$분, $R_B$는 $CT_0=11{,}445$분.

| Product Mix | 방법 | Throughput(LOT) | LP-GA 대비 Gap |
|---|---|---:|---:|
| $R_A$ | LP-GA | 128 | 0% |
| $R_A$ | M-GA | 125 | 2.34% |
| $R_A$ | N-GA | 124 | 3.12% |
| $R_B$ | LP-GA | 169 | 0% |
| $R_B$ | M-GA | 165 | 2.37% |
| $R_B$ | N-GA | 161 | 4.73% |

LP-GA의 Throughput은 M-GA보다 약 2.3%, N-GA보다 3.1~4.7% 높음.

### Figure 3 — Throughput 증가에 따른 Cycle Time 비교

<img width="1000" alt="LP-GA, midpoint GA, no-cross-fab GA의 throughput-cycle time curve" src="/assets/img/paper-reviews/2026-08-20/wu-fig3.svg" />

> Source: Wu et al. (2009), Figure 3. 논문 이해를 위한 일부 인용 및 크롭. [Original article](https://doi.org/10.1080/00207540802172029)

Throughput이 낮을 때는 세 방법의 차이가 작지만 고부하 구간으로 갈수록 CT 차이가 커진다. 여유 Capa가 많을 때보다 병목과 운송 Queue가 커지는 구간에서 Cross-FAB Allocation의 효과가 더 큼.

### 계산시간

| Product Mix | Module 1 | Module 2 |
|---|---:|---:|
| $R_A$ | 3.5초 | 95.578초 |
| $R_B$ | 4.2초 | 103.265초 |

Module 1은 약 64개 LP를 풀고 수 초. 대부분의 계산시간은 Queueing Evaluator를 반복 호출하는 GA에서 사용된다. 2009년 Hardware/구현 기준의 수치이므로 지금 Solver 속도와 직접 비교할 값은 아니다.

---

## 8) 이 구조에서 봐야 할 점

이 논문은 “남는 FAB로 물량을 넘긴다”를 두 변수로 나눈다.

$$
\underbrace{\Pi}_{\text{어디에서 넘길지}}
\longrightarrow
\underbrace{R}_{\text{얼마나 넘길지}}
\longrightarrow
\underbrace{(TH,CT)}_{\text{혼잡 이후 성능}}
$$

Cut-off가 바뀌면 각 FAB의 Workload Matrix $W$가 바뀌고, Route Ratio가 바뀌면 장비와 운송구간의 실제 Traffic이 바뀐다. 마지막으로 Queueing Network가 Traffic을 CT로 변환.

LP만 쓰면 Capa feasibility는 볼 수 있지만 Queueing Delay를 놓침. 반대로 모든 것을 GA/Queueing으로 풀면 탐색량이 커진다. 두 모델을 나눈 이유가 명확한 편이다. 운송구간도 Capa 1인 Resource로 넣었기 때문에 Cross-FAB LOT을 무한히 늘리는 해도 막을 수 있다.

다만 읽을 때 다음 범위를 같이 봐야 함.

- FAB 2개만 고려
- 두 FAB 모두 모든 제품을 완결생산 가능
- 제품당 FAB 전환은 한 번
- 물리경로는 고정, Vehicle Dispatch/Conflict/Shortest Path 없음
- Fleet, Merge Conflict, Blocking은 직접 모델링하지 않음
- $\Pi$와 $R$을 따로 풀기 때문에 Joint Optimum 보장 없음
- LP 실행횟수가 $2^n$에 비례해 제품 수가 많으면 확장 어려움
- HP fab 변형 Benchmark이며 실제 Dual-FAB 검증이 아님
- 수요변동, Hot LOT, 장비/운송 장애에 따른 Online Re-planning 없음

---

## 9) 정리

논문의 핵심은 Dual-FAB Capa Sharing을 다음 두 질문으로 만든 것.

1. 긴 공정 Route를 어느 Operation에서 나눌 것인가?
2. A, B, $A\rightarrow B$, $B\rightarrow A$ Route에 물량을 얼마나 배분할 것인가?

LP로 Workstation Capa를 만족하면서 Cross-FAB LOT이 적은 Cut-off를 찾고, Queueing Network+GA로 목표 Cycle Time 아래에서 Throughput이 높은 Route Ratio를 찾는다. 실험에서는 중앙 Cut-off나 FAB 독립운영보다 Cycle Time/Throughput 모두 개선.

다만 제목의 Route는 **차량 이동경로가 아니라 생산공정 Route**다. 물리적 Path 자체는 고정되어 있고, 이 논문이 최적화하는 것은 그 Path를 지나게 될 Cross-FAB 물량과 공정분할점이다.
