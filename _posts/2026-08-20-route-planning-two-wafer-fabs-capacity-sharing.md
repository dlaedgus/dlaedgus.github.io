---
title: "Route Planning for Two Wafer Fabs with Capacity-Sharing Mechanisms (IJPR 2009)"
date: 2026-08-20 12:40:00 +0900
categories: [paper_review, OR]
tags: [semiconductor-manufacturing, capacity-sharing, linear-programming, genetic-algorithm, queueing-network]
math: true
---

# Paper Review — *Route Planning for Two Wafer Fabs with Capacity-Sharing Mechanisms* (IJPR 2009)

- **1저자:** Muh-Cherng Wu  
- **공저자:** Chen-Fu Chen, Chang-Fu Shih  
- **제목:** Route Planning for Two Wafer Fabs with Capacity-Sharing Mechanisms  
- **저널:** *International Journal of Production Research*, 47(20), 5843–5856  
- **년도:** 2009  
- **DOI:** [10.1080/00207540802172029](https://doi.org/10.1080/00207540802172029)  
- **원문:** [National Yang Ming Chiao Tung University Repository](https://ir.lib.nycu.edu.tw/items/df5811e0-1f8f-43d9-b14d-d3fcfb4bdcba)

---

## 1) High-Level Summary (3–5 sentences)

본 논문은 서로 이웃한 두 개의 wafer fab이 생산능력을 공유하는 **dual-fab** 환경에서, 제품별 **공정분할점(cut-off point)**과 네 가지 **생산 route의 물량비율(route ratio)**을 결정하는 문제를 다룹니다. 여기서 `route`는 차량의 물리적 최단경로가 아니라, 제품의 공정 sequence를 어느 지점에서 나누어 Fab A와 Fab B가 각각 담당할지를 뜻합니다. 논문은 먼저 Linear Programming(LP)과 이분 탐색으로 fab 간 이동 LOT 수가 적은 cut-off point를 찾고, 이어 Queueing Network와 Genetic Algorithm(GA)을 결합해 목표 Cycle Time을 만족하는 Throughput을 최대화합니다. HP fab 문헌 데이터를 변형한 수치실험에서는 route의 중간 지점을 고정하는 방법과 cross-fab 생산을 금지하는 방법보다 Cycle Time과 Throughput이 모두 개선되었습니다. 다만 한 번의 fab 전환, 고정된 물리 운송경로, 두 fab의 완전한 공정 수행능력 등을 가정하므로 결과의 적용범위는 명확히 제한됩니다.

---

## 2) 문제 설정: 두 FAB이 Capa를 공유한다는 것의 의미

### 2.1 왜 cross-fab 생산이 필요한가

두 fab이 동일 제품을 처음부터 끝까지 생산할 수 있더라도, 장비군별 보유대수와 제품 Mix가 다르면 특정 Workstation은 혼잡하고 다른 Workstation은 유휴 상태가 될 수 있습니다. 특히 fab을 처음 설계할 때 예상한 제품 Mix와 실제 시장수요의 Mix가 달라지면, 개별 fab만 독립적으로 운영해서는 두 fab의 총 Capa를 충분히 활용하지 못합니다.

이 논문은 한 제품의 공정 sequence를 두 구간으로 나눈 뒤, 앞 구간과 뒤 구간을 서로 다른 fab이 처리하도록 허용합니다. 제품 $i$의 cut-off point를 $\pi_i$라 하면, 가능한 생산 route는 네 가지입니다.

- $A$: 모든 공정을 Fab A에서 수행
- $B$: 모든 공정을 Fab B에서 수행
- $A\rightarrow B$: cut-off 이전은 A, 이후는 B
- $B\rightarrow A$: cut-off 이전은 B, 이후는 A

각 제품의 route ratio는

$$
r_i=(a_i,b_i,c_i,d_i)
$$

로 표현합니다. $a_i,b_i,c_i,d_i$는 각각 위 네 route에 배정하는 제품 $i$ 물량의 비율입니다.

### 2.2 반드시 구분해야 하는 “Route”의 의미

논문 제목의 Route Planning은 **AGV/OHT가 어느 통로로 갈지**, 또는 **두 fab 사이의 최단경로를 어떻게 찾을지**를 결정하는 문제가 아닙니다. 결정하는 것은 다음 두 가지입니다.

1. 제품의 공정 sequence를 어디에서 나눌 것인가: $\pi_i$
2. 네 가지 생산공정 route에 물량을 어떤 비율로 배분할 것인가: $r_i$

물리적인 운송 path는 오히려 논문의 Assumption 3에서 **두 station 사이에 하나의 고정 path만 존재**한다고 가정합니다. 따라서 본 연구는 `공정 route와 cross-fab 물량배분 최적화`이지, `물리경로 선택 최적화`는 아닙니다.

### 2.3 세 가지 핵심 가정

- **Functional comprehensiveness:** 각 fab은 다른 fab의 도움 없이 모든 제품을 완성할 수 있습니다.
- **One cut-off point:** 한 제품은 공정 sequence에서 한 번만 분할되므로 cross-fab route도 fab을 한 번만 바꿉니다.
- **Unique transportation path:** Workstation 또는 Buffer 사이 물리 운송경로는 사전에 하나로 고정됩니다.

최종 목표는 모든 제품의 cut-off point 집합 $\Pi=[\pi_1,\ldots,\pi_n]$와 route ratio 집합 $R=[r_1,\ldots,r_n]$을 정해, 평균 Cycle Time이 목표값 $CT_0$ 이하인 범위에서 두 fab의 총 Throughput을 최대화하는 것입니다.

---

## 3) Module 1: Cut-off Point와 Cross-FAB 물량의 LP

Module 1은 시간에 따른 Queue와 Cycle Time을 아직 계산하지 않는 **정적 Capa 배분 문제**입니다. 특정 cut-off point 집합 $\Pi$가 주어졌다고 가정한 뒤, 목표 생산량 $Q$를 처리하면서 cross-fab 생산 LOT 수를 최소화합니다.

### 3.1 기호

- $i=1,\ldots,n$: 제품 index
- $g=1,\ldots,m_a$: Fab A의 Workstation index
- $h=1,\ldots,m_b$: Fab B의 Workstation index
- $Q$: 두 fab이 높은 가동률에서 처리할 것으로 추정한 목표 Throughput(LOT)
- $P_i$: Product Mix에서 제품 $i$의 비율, $\sum_iP_i=1$
- $C_g,C_h$: 각 Workstation의 가용 Machine Hour
- $W^a_{ig}$: 제품 $i$를 route $A$로 만들 때 Fab A의 $g$에 필요한 LOT당 총 처리시간
- $W^c_{ig},W^d_{ig}$: route $A\rightarrow B$, $B\rightarrow A$에서 Fab A의 $g$에 필요한 처리시간
- $W^b_{ih},W^c_{ih},W^d_{ih}$: 각 route가 Fab B의 $h$에 요구하는 처리시간

여기서 $W$는 단일 Operation의 시간이라기보다, 주어진 route와 cut-off point 아래에서 해당 Workstation을 재방문하는 시간을 모두 합한 **LOT당 Workload**입니다.

### 3.2 목적함수

$$
\min Z(\Pi)
=
\sum_{i=1}^{n} QP_i(c_i+d_i)
$$

$c_i+d_i$는 제품 $i$ 중 fab 경계를 넘는 비율입니다. 따라서 목적함수는 목표량 $Q$를 생산할 때 발생하는 cross-fab LOT 수를 최소화합니다. 논문의 논리는 cross-fab LOT이 늘수록 추가 운송시간과 혼잡이 커지므로, 동일한 Capa feasibility를 만족한다면 fab 간 이동량이 작은 설계가 유리하다는 것입니다.

### 3.3 Route Ratio 보존

$$
a_i+b_i+c_i+d_i=1,
\qquad i=1,\ldots,n
$$

각 제품의 전체 물량을 네 route에 빠짐없이 배분합니다. 모든 비율은 $[0,1]$ 범위에 있어야 합니다.

### 3.4 Fab A의 Workstation Capa

$$
\sum_{i=1}^{n} QP_i
\left(
a_iW^a_{ig}
+d_iW^d_{ig}
+c_iW^c_{ig}
\right)
\le C_g,
\qquad g=1,\ldots,m_a
$$

Fab A에서 전 공정을 수행하는 물량, $B\rightarrow A$의 뒤 구간, $A\rightarrow B$의 앞 구간이 Fab A 장비의 Capa를 함께 사용합니다.

### 3.5 Fab B의 Workstation Capa

$$
\sum_{i=1}^{n} QP_i
\left(
b_iW^b_{ih}
+d_iW^d_{ih}
+c_iW^c_{ih}
\right)
\le C_h,
\qquad h=1,\ldots,m_b
$$

동일한 방식으로 Fab B의 모든 Workstation 부하가 가용 Machine Hour 이하가 되게 합니다.

### 3.6 LP가 실제로 답하는 질문

이 LP는 “어떤 LOT을 지금 어느 장비에 Dispatch할 것인가”를 답하지 않습니다. 다음과 같은 상위 수준의 물량계획을 답합니다.

> 제품 Mix와 목표 생산량이 주어졌을 때, 두 fab의 Workstation Capa를 넘지 않으면서 각 제품 물량을 네 공정 route에 어떤 비율로 나눌 수 있는가? 그중 fab을 넘는 물량이 가장 작은 해는 무엇인가?

또한 $Q$가 사전에 주어지므로 Module 1 자체가 Throughput을 최대화하는 것은 아닙니다. $Q$의 feasibility를 확보하면서 cross-fab LOT을 최소화하는 **surrogate problem**입니다.

---

## 4) 전체 해법: LP–Binary Search–Queueing Network–GA

### 4.1 Cut-off Point 탐색

제품 route가 수백 Operation으로 구성되면 모든 cut-off 조합을 그대로 열거하기 어렵습니다. 논문은 각 제품의 현재 후보구간을 절반으로 나누고 1사분점과 3사분점에 두 후보 cut-off를 둡니다. 제품이 $n$개이면 한 iteration에서 가능한 조합은 $2^n$개이고, 각 조합마다 앞의 LP를 풀어 $Z(\Pi)$를 비교합니다.

한 iteration의 절차는 다음과 같습니다.

1. 각 제품의 현재 구간에 두 cut-off 후보를 생성
2. $2^n$개 조합에 대해 LP를 모두 풀이
3. $Z(\Pi)$가 가장 작은 조합 $\Pi_i^*$ 선택
4. 선택된 cut-off가 속한 절반 구간만 남김
5. 원하는 해상도까지 반복

제품 route의 Operation 수 $m$이 $2^{x-1}<m\le2^x$를 만족하면, 논문이 제시한 LP 실행횟수는 대략

$$
N_{LP}=x\cdot2^n
$$

입니다. Operation 수에는 로그로 증가하지만, cross-fab 후보 제품 수 $n$에는 지수적으로 증가합니다.

### 4.2 Queueing Network로 Cycle Time 평가

Module 1은 운송시간을 0, 운송 Capa를 무한대로 가정합니다. Module 2에서는 이 가정을 완화합니다. Connors et al.(1996)의 Queueing Network를 확장해, Workstation뿐 아니라 각 고정 운송 path도 **Capa가 1인 conveyor machine**으로 모델링합니다.

Queueing evaluator의 입출력 관계는 다음과 같이 요약됩니다.

$$
CT=f(TH,R,\Pi)
$$

- $TH$: 목표 총 Throughput
- $R$: 제품별 route ratio
- $\Pi$: 제품별 cut-off point
- $CT$: 두 fab의 평균 Cycle Time

같은 route ratio라도 $TH$가 증가하면 Workstation과 운송 path의 Traffic Intensity가 올라가고, Queueing Delay 때문에 $CT$가 증가합니다. 특정 $(R,\Pi^*)$와 목표 $CT_0$가 주어지면, $TH$에 대한 이분 탐색으로

$$
f(TH,R,\Pi^*)=CT_0
$$

를 만족하는 최대 $TH$를 찾습니다. 이 값이 GA chromosome의 Fitness입니다.

### 4.3 GA로 Route Ratio 탐색

Chromosome은 모든 제품의 route ratio를 이어 붙인

$$
R=[r_1,\ldots,r_n],
\qquad r_i=(a_i,b_i,c_i,d_i)
$$

입니다. 각 gene segment는 합이 1이고 각 원소가 $[0,1]$에 있어야 합니다.

- **초기화:** 유효한 chromosome $N_p$개를 무작위 생성
- **Crossover:** 두 parent의 gene 값을 일부 교환하고, 나머지 한 값을 조정해 합이 1이 되도록 복구
- **Mutation:** 선택한 제품의 gene 두 개를 맞바꿈
- **Selection:** 기존 개체와 새 개체를 합친 Pool에서 Fitness가 높은 $N_p$개 유지
- **종료:** Best Solution이 일정 기간 바뀌지 않거나 최대 iteration에 도달

논문이 보고한 설정은 $P_0=100$, crossover 확률 $P_{cr}=0.8$, mutation 확률 $P_m=0.1$, 그리고 $T_b=1000$, $T_u=30$입니다. 다만 본문에서 설명한 두 종료기호의 역할과 보고된 수치의 크기 순서는 직관적으로 일치하지 않아, 재현 시에는 원 코드 또는 저자 확인 없이 임의로 해석하지 않는 편이 안전합니다.

### 4.4 두 Module을 분리한 이유

- **Module 1:** 시간축을 무시한 정적 Capa Allocation, LP로 빠르게 Cut-off를 평가
- **Module 2:** 혼잡과 Cycle Time을 포함한 동적 성능평가, Queueing Network와 GA 사용

모든 cut-off 조합마다 Queueing Simulation 또는 GA까지 수행하면 계산량이 매우 커집니다. 논문은 “cross-fab 이동이 적은 cut-off”를 먼저 LP로 압축한 뒤 route ratio만 정교하게 탐색하는 Decomposition을 택했습니다. 계산효율은 높지만, $\Pi$와 $R$을 동시에 최적화하지 않으므로 전체 문제의 Global Optimum을 보장하지는 않습니다.

---

## 5) 실험 설정

### 5.1 데이터와 FAB 구성

실험 데이터는 실제 기업의 현재 운영데이터가 아니라, Wein(1988)에 제시된 HP fab 데이터를 변형해 구성했습니다.

- Fab A: Machine 93대
- Fab B: Machine 72대
- 각 fab: Batch Workstation 4개, Series Workstation 21개
- Machine Failure: MTBF와 MTTR을 사용하며 지수분포로 가정
- 제품: 3종
- 공정 수: Product 1과 2는 각각 172 Operation, Product 3은 150 Operation

두 fab은 장비 대수는 다르지만, 세 제품을 각각 독립적으로 완성할 수 있는 기능을 갖춘 것으로 설정합니다.

### 5.2 비교 방법

- **LP-GA:** 제안 방법. LP 기반 cut-off 탐색 + Queueing Network 기반 GA
- **M-GA:** 각 제품 cut-off를 공정 route의 정중앙으로 고정하고 GA만 수행
- **N-GA:** Cross-fab 생산을 허용하지 않고 GA 수행

### 5.3 Product Mix와 Cut-off 결과

두 Product Mix를 평가합니다.

$$
R_A=(3:2:5),
\qquad
R_B=(5:4:1)
$$

높은 가동률을 만드는 목표 Throughput은 각각 $Q_A=128$ LOT, $Q_B=169$ LOT입니다. LP-GA가 찾은 cut-off는 다음과 같습니다.

| Product | 총 Operation | $R_A$ Cut-off | $R_B$ Cut-off |
|---|---:|---:|---:|
| 1 | 172 | 85번째 | 84번째 |
| 2 | 172 | 85번째 | 84번째 |
| 3 | 150 | 129번째 | 78번째 |

Product 3을 보면 Mix에 따라 129번째와 78번째로 크게 달라집니다. 즉, 단순히 route의 중앙을 cut-off로 두는 것이 항상 Capa 균형에 유리하지 않음을 보여줍니다.

---

## 6) 실험 결과

### 6.1 동일 Throughput에서 Cycle Time 비교

| Product Mix | 방법 | Throughput | 평균 CT(min) | LP-GA 대비 Gap |
|---|---|---:|---:|---:|
| $R_A$ | LP-GA | 128 | 11,080 | 0% |
| $R_A$ | M-GA | 128 | 12,175 | 9.88% |
| $R_A$ | N-GA | 128 | 12,463 | 12.48% |
| $R_B$ | LP-GA | 169 | 11,639 | 0% |
| $R_B$ | M-GA | 169 | 12,811 | 10.06% |
| $R_B$ | N-GA | 169 | 14,075 | 20.90% |

동일한 생산량을 처리할 때 LP-GA는 중앙 cut-off 방식보다 약 10%, cross-fab을 금지한 방식보다 약 12–21% 짧은 평균 Cycle Time을 보입니다. 두 fab을 독립적으로 운영하는 것보다, 제한적인 cross-fab 물량을 허용해 병목부하를 나누는 편이 유리한 실험입니다.

### 6.2 동일 Cycle Time에서 Throughput 비교

논문은 $R_A$에서 $CT_0=11{,}081$분, $R_B$에서 $CT_0=11{,}445$분을 사용합니다.

| Product Mix | 방법 | Throughput(LOT) | LP-GA 대비 Gap |
|---|---|---:|---:|
| $R_A$ | LP-GA | 128 | 0% |
| $R_A$ | M-GA | 125 | 2.34% |
| $R_A$ | N-GA | 124 | 3.12% |
| $R_B$ | LP-GA | 169 | 0% |
| $R_B$ | M-GA | 165 | 2.37% |
| $R_B$ | N-GA | 161 | 4.73% |

LP-GA의 Throughput은 M-GA보다 약 2.3%, N-GA보다 약 3.1–4.7% 높습니다. 논문의 Throughput–Cycle Time Curve에서는 생산량이 높아질수록 세 방법의 차이도 커지므로, Capa가 여유로운 상황보다 고부하 상황에서 cross-fab 배분의 가치가 더 크게 나타납니다.

### 6.3 계산시간

| Product Mix | Module 1 | Module 2 |
|---|---:|---:|
| $R_A$ | 3.5초 | 95.578초 |
| $R_B$ | 4.2초 | 103.265초 |

세 제품, 최대 172 Operation인 문제에서 Module 1은 약 64회의 LP를 수행하며 수 초 안에 끝납니다. 반면 Queueing Evaluator를 반복 호출하는 GA가 대부분의 계산시간을 차지합니다. 다만 이는 2009년 논문의 특정 구현과 Hardware에서 측정된 수치이므로 현대 Solver의 속도로 직접 환산해서는 안 됩니다.

---

## 7) 논문의 핵심 기여

### 7.1 Capa Sharing을 “공정분할점 + 물량비율”로 모델링

Cross-fab 생산을 단순 허용 여부로 처리하지 않고, 제품별로 **어디에서 넘길지**와 **얼마나 넘길지**를 분리한 점이 핵심입니다. Cut-off는 각 fab이 부담하는 공정별 Workload를 바꾸고, route ratio는 실제 물량부하를 바꿉니다.

### 7.2 정적 Capa와 동적 혼잡을 단계적으로 연결

LP만 사용하면 Capa feasibility는 판단할 수 있지만 Queue와 Cycle Time을 표현하기 어렵습니다. 반대로 Queueing Network와 GA만으로 모든 cut-off를 탐색하면 계산량이 커집니다. 논문은 LP로 후보를 좁히고 Queueing Model로 시간 성능을 평가하는 계층적 접근을 제안합니다.

### 7.3 운송구간도 유한 Capa Resource로 반영

Fab 내부 Workstation만이 아니라, station 사이 고정 운송 path를 Capa 1인 Conveyor Machine으로 추가합니다. 따라서 cross-fab 물량을 과도하게 늘렸을 때 운송 Traffic이 Cycle Time을 악화시키는 효과를 모형 안에 포함합니다.

---

## 8) 강점과 한계

### 강점

- 반도체 Re-entrant Flow의 긴 공정 route와 Workstation별 Capa를 직접 고려합니다.
- Capa feasibility, cross-fab 이동량, Queueing Delay, Throughput을 한 Framework 안에서 연결합니다.
- 중앙 cut-off와 fab 독립운영이라는 이해하기 쉬운 Baseline을 함께 비교합니다.
- Cut-off 탐색을 LP로 분리해 작은 제품 수에서는 계산이 빠릅니다.
- “이동을 무조건 늘리는 것”이 아니라 필요한 Capa 공유를 달성하는 범위에서 cross-fab LOT을 최소화합니다.

### 한계

- **두 fab 한정:** 세 개 이상 fab의 Network에는 그대로 적용되지 않습니다.
- **Fab당 완결 생산 가능:** 한 fab이 특정 공정을 수행하지 못하는 비대칭 Qualification 문제는 제외합니다.
- **한 번의 fab 전환:** 한 제품의 다중 cut-off 또는 재진입식 inter-fab route는 다루지 않습니다.
- **물리경로 고정:** 최단경로, 충돌회피, Vehicle Dispatching을 최적화하지 않습니다.
- **단순화된 운송모형:** 각 Path를 Capa 1의 Conveyor로 근사하며 실제 Vehicle Fleet, Merge Conflict, Blocking은 명시적으로 모델링하지 않습니다.
- **Decomposition의 근사성:** Cut-off를 cross-fab LOT 최소화로 먼저 고정하므로 $(\Pi,R)$의 Joint Global Optimum을 보장하지 않습니다.
- **확장성:** Module 1의 LP 횟수가 $2^n$에 비례하므로 cross-fab 대상 제품이 많아지면 급격히 커집니다.
- **Benchmark 기반:** HP fab 문헌 데이터를 변형한 수치실험이며 실제 dual-fab 운영에 대한 현장 실증은 아닙니다.
- **불확실성 제한:** 수요변동, 긴급 LOT, 장비상태 변화, 운송장애를 Online으로 재계획하지 않습니다.

저자들도 향후 과제로 Multi-fab, 제품당 여러 cut-off, 다제품 대규모 문제를 제시합니다.

---

## 9) 수식을 통해 읽은 의사결정 구조

이 연구의 의사결정은 세 층으로 정리할 수 있습니다.

$$
\underbrace{\Pi}_{\text{공정을 어디에서 나눌지}}
\longrightarrow
\underbrace{R}_{\text{각 생산 route에 얼마를 배분할지}}
\longrightarrow
\underbrace{(TH,CT)}_{\text{혼잡을 반영한 시스템 성능}}
$$

첫 번째 층에서는 공정분할이 각 fab의 Workload Matrix $W$를 바꿉니다. 두 번째 층에서는 route ratio가 Workstation과 운송구간의 실제 Traffic을 바꿉니다. 마지막 층에서는 Queueing Network가 Traffic을 Cycle Time으로 변환하고, GA는 $CT\le CT_0$에서 Throughput이 큰 물량배분을 찾습니다.

이 구조의 장점은 “Capa가 남는 fab으로 물량을 보내자”라는 정성적 아이디어를 다음과 같은 정량 질문으로 바꾸는 데 있습니다.

- 어느 공정 이후에 보내야 각 장비군의 Load가 가장 잘 맞는가?
- Capa 공유효과를 얻기 위해 최소 몇 LOT이 fab 경계를 넘어야 하는가?
- 추가 이동으로 생기는 Queueing Delay까지 고려했을 때 순효과가 양수인가?
- 목표 Cycle Time을 지키며 가능한 총 Throughput은 얼마인가?

---

## 10) 결론

본 논문은 dual-fab의 Capa Sharing을 제품별 **cut-off point**와 네 가지 **공정 route ratio**의 최적화 문제로 정식화합니다. LP는 Workstation Capa를 만족하면서 cross-fab LOT이 적은 cut-off를 찾고, Queueing Network와 GA는 운송구간의 혼잡까지 반영해 목표 Cycle Time 아래의 Throughput을 높입니다. 수치실험에서 중앙 cut-off 또는 cross-fab 금지 방식보다 Cycle Time과 Throughput이 모두 개선되어, 제품 Mix 변화로 생긴 두 fab의 Capa 불균형을 공정분할과 물량배분으로 완화할 수 있음을 보입니다.

동시에 제목의 “Route”를 정확히 해석해야 합니다. 이 연구는 물리적 차량경로 탐색이 아니라 **생산공정 route 설계**이며, 물리 운송 path는 고정되어 있습니다. 따라서 논문의 가장 정확한 한 줄 요약은 다음과 같습니다.

> 두 wafer fab 사이에서 제품 공정을 어느 지점에서 나누고 얼마의 물량을 넘길지를 정해, 유한 Capa와 Queueing Delay 아래에서 전체 생산성을 높이는 OR 연구이다.
