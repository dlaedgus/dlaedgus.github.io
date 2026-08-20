---
title: "Product Mix Optimization for a Semiconductor Fab: Modeling Approaches and Decomposition Techniques (WSC 2012)"
date: 2026-08-20 12:00:00 +0900
categories: [paper_review, OR]
tags: [semiconductor-manufacturing, product-mix, linear-programming, decomposition]
math: true
---

# Paper Review — *Product Mix Optimization for a Semiconductor Fab: Modeling Approaches and Decomposition Techniques* (WSC 2012)

- **1저자:** Andreas Klemmt  
- **공저자:** Martin Romauch, Walter Laure  
- **제목:** Product Mix Optimization for a Semiconductor Fab: Modeling Approaches and Decomposition Techniques  
- **학회:** Winter Simulation Conference (WSC)  
- **년도:** 2012  
- **원문:** [WSC Proceedings PDF](https://www.informs-sim.org/wsc12papers/includes/files/inv213.pdf)  
- **DOI:** [10.1109/WSC.2012.6465270](https://doi.org/10.1109/WSC.2012.6465270)

> 이 글은 Klemmt, Romauch, Laure의 **2012년 WSC 논문**을 기준으로 정리합니다. 동일 제목의 2015년 *Computers & Operations Research* 확장 논문은 저자 구성과 세부 내용이 다르므로 구분해야 합니다.

---

## 1) High-Level Summary (3–5 sentences)

본 논문은 반도체 FAB에서 **어떤 제품을 얼마나 생산할지(Product Mix)**와 그 물량이 만드는 공정 부하를 **어떤 장비에 배분할지(Static Capacity Allocation)**를 하나의 Linear Programming 문제로 연결합니다. 제품별 이익과 최소·최대 수요를 고려해 이익을 최대화하되, 각 제품의 Route가 발생시키는 Job Class별 작업량이 실제 Qualified Equipment의 Capa를 넘지 않아야 합니다. 전체 모형은 선형이지만 제품·공정·장비·Dedication을 모두 펼치면 커지기 때문에, 저자들은 `제품 Mix를 정하는 Master Problem`과 `장비 Load Balancing Subproblem`을 번갈아 푸는 Decomposition Heuristic을 제안합니다. 1,215개 생성 Instance와 변형된 실제 FAB 데이터에서, 기준 Mix를 제품별 ±5%만 조정해도 약 3% 수준의 이익 개선 여지가 있으며 병목 Work Center의 부하를 낮출 수 있음을 보입니다. 다만 이 논문은 시간축의 LOT Scheduling이나 실시간 Dispatching이 아니라, 평균 물량 흐름을 가정한 **정적·전술적 생산계획 모델**입니다.

---

## 2) 문제 설정: Product Mix와 Capa를 왜 동시에 봐야 하나

반도체 생산에서는 제품별 생산량만 정한다고 계획이 끝나지 않습니다. 제품마다 Route가 다르고, 하나의 Route는 수백 개의 Operation으로 구성되며, 동일하거나 유사한 Operation들은 특정 장비군만 처리할 수 있습니다. 따라서 고수익 제품의 생산량을 늘리더라도 그 제품이 특정 노광·식각·증착 계열 장비에 큰 부하를 만들면 실제로는 생산할 수 없습니다.

논문은 이 구조를 다음 순서로 축약합니다.

1. 제품 $p$의 생산량이 정해집니다.
2. 제품의 Route를 통해 Job Class $j$별 필요 작업량이 계산됩니다.
3. 각 Job Class는 처리 가능한 Resource $i$에만 배분됩니다.
4. 배분된 작업의 처리시간 합이 Resource별 Capa를 넘지 않아야 합니다.

여기서 **Job Class**는 장비 소요 패턴이 동일하거나 매우 유사한 Operation을 묶은 단위입니다. 모든 제품·Operation을 개별적으로 다루는 대신, Capa 관점에서 같은 작업을 묶어 모델 규모를 줄입니다. **Activity** $k$는 특정 Job Class와 이를 처리할 수 있는 Resource의 가능한 조합입니다. Capability 또는 Dedication이 없는 조합은 Activity 자체가 존재하지 않습니다.

논문이 다루는 의사결정은 두 층위입니다.

- **상위 의사결정:** 수요 범위 안에서 제품별 생산량을 선택해 총이익을 최대화
- **하위 의사결정:** 선택한 Mix가 만든 작업량을 Qualified Resource에 배분해 Capa-feasible한 Load Profile을 생성

따라서 단순한 `제품별 수익 순위` 문제도 아니고, 주어진 물량을 장비에 나누는 `Load Balancing`만의 문제도 아닙니다. 제품 Mix가 바뀌면 Job Class별 부하가 바뀌고, 장비 대체 가능성에 따라 같은 Mix의 실제 Capa 소비도 달라지는 결합 문제입니다.

---

## 3) 기초 모형: 주어진 물량의 Load Balancing

먼저 제품 Mix가 이미 주어졌다고 가정합니다. Job Class $j$의 필요 처리량을 $\lambda_j$, Resource $i$가 Job Class $j$를 처리하는 Service Rate를 $\mu_{ji}$, 배정량을 $x_{ji}$라고 하면 최대 장비부하를 최소화하는 LP는 다음과 같습니다.

$$
\min \rho
$$

$$
\sum_{i\in I}x_{ji}=\lambda_j \qquad \forall j\in J
$$

$$
\sum_{j:\mu_{ji}>0}\frac{x_{ji}}{\mu_{ji}}\le \rho
\qquad \forall i\in I
$$

$$
x_{ji}\ge 0.
$$

기호의 의미는 다음과 같습니다.

- $x_{ji}$: Job Class $j$의 물량 중 Resource $i$에 배정한 양
- $\lambda_j$: Job Class $j$가 처리해야 하는 총 물량
- $\mu_{ji}$: Resource $i$에서 Job Class $j$를 처리하는 속도
- $x_{ji}/\mu_{ji}$: 그 배정으로 발생하는 Resource 사용시간
- $\rho$: 모든 Resource 가운데 가장 큰 Load

첫 번째 제약은 모든 Job Class 수요가 빠짐없이 배분되도록 합니다. 두 번째 제약은 Resource별 총 처리시간의 상한을 $\rho$로 묶습니다. 목적함수에서 $\rho$를 최소화하므로 결과적으로 최대부하가 가장 낮은 배분을 찾습니다.

논문의 예제에서는 네 장비의 초기 Load가 $(170,150,140,160)$시간인 배분에서 최대부하는 170시간입니다. 단순 Min-Max Load Balancing을 풀면 $(100,150,160,160)$시간으로 최대부하를 160시간까지 낮출 수 있습니다. 다만 Min-Max만으로는 최대부하가 같은 장비의 수나 하위 Load Level을 충분히 구분하지 못합니다. 저자들은 Resource Pooling 개념을 사용해 가장 높은 Load Level부터 계층적으로 균형화하고, 예제에서 $(145,138,145,160)$시간의 Profile을 제시합니다.

### Connected Component와 Resource Pool

두 Resource가 공통 Job Class를 처리할 수 있으면 직접 연결되어 있고, 이러한 연결이 연쇄적으로 이어지면 하나의 **Connected Component**, 즉 Closed Machine Set(CMS)을 이룹니다. 서로 다른 CMS는 Job Class를 공유하지 않으므로 Load Balancing 문제를 독립적으로 풀 수 있습니다. 목적도 분리 가능하다면 Component별 계산을 병렬화할 수 있습니다.

이 구조는 뒤의 Decomposition에서 중요합니다. 제품은 여러 CMS를 동시에 통과하므로 전체 Product Mix 문제는 쉽게 분리되지 않지만, Mix가 고정된 하위 Load Balancing은 CMS별로 분해할 수 있기 때문입니다.

---

## 4) 핵심 수리모형: Product Mix와 장비배분의 통합 LP

### 4.1 변수와 파라미터

- $P$: 제품 집합, 제품 인덱스 $p$
- $J$: Job Class 집합, 인덱스 $j$
- $I$: Resource 집합, 인덱스 $i$
- $K$: 가능한 Job Class–Resource Activity 집합, 인덱스 $k$
- $y_p$: 제품 $p$의 생산량
- $x_k$: Activity $k$에 배정한 작업량
- $D=(d_{jp})$: 제품 한 단위가 Job Class $j$를 몇 번 요구하는지를 나타내는 Technology Matrix
- $A=(a_{jk})$: Activity $k$가 어느 Job Class의 처리에 해당하는지 나타내는 Capability/Incidence Matrix
- $R=(r_{ik})$: Activity $k$가 Resource $i$에서 소비하는 시간의 Service-Time Matrix
- $c_p$: 제품 $p$ 한 단위의 이익
- $\Delta_p^-,\Delta_p^+$: 제품별 최소 의무물량과 최대 수요
- $\rho_i^{\max}$: Resource $i$의 허용 Capa

### 4.2 Global Product Mix LP

논문의 기본 모형은 다음과 같습니다.

$$
\max_{x,y}\;c^\top y
$$

$$
\text{s.t.}\quad Ax=Dy
$$

$$
Rx\le \rho^{\max}
$$

$$
x\ge0
$$

$$
\Delta^-\le y\le\Delta^+.
$$

각 식의 역할은 명확합니다.

- $c^\top y$: 제품별 이익과 생산량을 곱한 총이익
- $Dy$: 선택한 Product Mix가 발생시킨 Job Class별 총 필요 작업량
- $Ax$: 실제 가능한 Activity에 배정된 Job Class별 작업량
- $Ax=Dy$: 필요한 모든 공정작업이 실제 장비 Activity에 배정되어야 한다는 Flow Balance
- $Rx\le\rho^{\max}$: 장비별 총 소요시간이 허용 Capa를 초과하지 않는다는 Capacity Constraint
- $\Delta^-\le y\le\Delta^+$: 계약·의무생산 하한과 Forecast 또는 시장수요 상한

예를 들어 제품 A의 단위이익이 가장 높아도, A가 특정 Job Class를 많이 요구하고 그 Job Class의 Qualified Resource가 적다면 $Rx\le\rho^{\max}$가 A의 증산을 막습니다. 반대로 단위이익이 조금 낮더라도 여유 Capa를 활용하는 제품 B를 늘리는 편이 FAB 전체 이익에는 더 나을 수 있습니다. 이 점이 Capa를 무시한 Greedy Mix와의 근본적인 차이입니다.

### 4.3 모형 규모

Activity 수 $|K|$는 Service-Time Matrix의 Nonzero 수와 같습니다. 제품 하나가 평균 $L$개의 Job Class를 요구한다고 할 때 저자들은 기본 LP 규모를 다음처럼 추정합니다.

- Row 수: $|J|+|I|$
- Column 수: $|P|+|K|+1$
- Nonzero 수: $|P|L+2|K|$

예시로 Resource 1,000개, 제품 1,000개, Job Class 5,000개, Job Class당 평균 5개의 가능한 Resource, 제품당 평균 100개의 Job Class를 가정하면 약 6,000개 Row, 26,001개 Column, 152,000개 Nonzero가 됩니다. LP 자체는 다항시간에 풀 수 있지만, 실제 MES 수준의 모든 Route·Operation·Equipment·Dedication을 반영할수록 데이터 구축과 반복 계산이 무거워집니다.

### 4.4 추가 Capa와 Soft Bottleneck 확장

논문은 기본 Capa에 더해 다음을 표현하는 확장도 제시합니다.

- $\rho_1^+$: Bottleneck Management로 한시적으로 허용하는 Soft-Bottleneck Capa
- $\rho_2^+$: 투자 또는 Outsourcing으로 확보하는 추가 Capa
- $h^\top\rho_1^+\le I_1$: Bottleneck 대응 Budget
- $g^\top\rho_2^+\le I_2$: 추가 Capa Budget
- $\rho_1^+\le R_1^+$, $\rho_2^+\le R_2^+$: 유형별 추가 가능량

Capacity Constraint는

$$
Rx\le\rho^{\max}+\rho_1^++\rho_2^+
$$

로 확장됩니다. 다만 원문의 식 (14)는 $g$를 추가 Capa의 비용으로 설명하면서 목적함수에 $+g^\top\rho_2^+$를 적고, 동시에 식 (18)에서 $g^\top\rho_2^+\le I_2$를 사용합니다. `비용을 차감한다`는 해석이라면 목적함수 부호가 $-$여야 하므로 표기상 모호성이 있습니다. 따라서 재현 구현에서는 기본 LP를 우선 사용하고, 확장 모형은 비용계수의 의미와 부호를 다시 정의하는 것이 안전합니다.

---

## 5) 해법: Master Problem과 Load Balancing의 반복

Global LP를 직접 풀 수도 있지만, 저자들은 대규모 계산을 위해 단순한 Decomposition Heuristic을 제안합니다. 핵심은 제품별 장비 소요량을 나타내는 행렬 $T$를 현재 Load Balancing 결과로 반복 갱신하는 것입니다.

### 5.1 Master Problem

$$
\max_y\;c^\top y
$$

$$
Ty\le\rho^{\max}
$$

$$
\Delta^-\le y\le\Delta^+.
$$

$T_{ip}$는 제품 $p$ 한 단위를 생산할 때 Resource $i$가 소비하는 시간의 현재 추정치입니다. $T$가 고정되어 있으면 Master Problem은 Job Class와 Activity 변수를 포함하지 않는 작은 LP가 됩니다.

### 5.2 Load Balancing Subproblem

Master가 Mix $y$를 내면 Job Class 수요는

$$
\lambda=Dy
$$

가 됩니다. 이 수요를 입력으로 앞의 Load Balancing 문제 $LB(R,A,\lambda)$를 풀어 Activity 배분 $x$와 Resource Load를 계산합니다. CMS가 여러 개라면 이 단계는 Component별로 나눠 병렬 처리할 수 있습니다.

### 5.3 Resource Consumption Matrix 갱신

Load Balancing 결과를 사용해 제품별 Resource 소요계수를 다음 구조로 갱신합니다.

$$
T_{ip}(x)
=
\sum_{j:d_{jp}>0}
\left(\frac{x_{ji}}{\mu_{ji}}\right)
\left(\frac{d_{jp}}{\lambda_j}\right),
\qquad p:y_p>0.
$$

- $x_{ji}/\mu_{ji}$: Job Class $j$ 때문에 Resource $i$가 실제 사용한 시간
- $d_{jp}/\lambda_j$: Job Class $j$의 전체 수요 중 제품 $p$ 한 단위가 차지하는 몫
- $T_{ip}$: 이 배분을 기준으로 추정한 제품 $p$의 Resource $i$ 한 단위당 소요시간

반복과정에서는 현재 $y_p>0$인 제품만 위 식으로 갱신합니다. $y_p=0$인 제품은 관련 $\lambda_j$가 0이 되어 $d_{jp}/\lambda_j$가 정의되지 않을 수 있으므로, 이전 반복의 $T_{ip}$를 유지합니다. 초기화 단계에서는 모든 제품을 양의 물량으로 둔 Load Balancing을 한 번 수행해 $T$를 만듭니다.

### 5.4 전체 절차

1. 초기 Mix로 Load Balancing을 풀어 $T$를 초기화합니다.
2. 최소 의무물량 $\Delta^-$에서 시작해 Feasible한 기준해를 만듭니다.
3. 현재 $T$로 Master Problem을 풀어 새로운 $y$를 구합니다.
4. $\lambda=Dy$를 계산하고 Load Balancing을 다시 풉니다.
5. 새 배분 $x$로 $T$를 갱신합니다.
6. 목적값 개선 $z_i-z_{i-1}$이 허용오차 $\varepsilon$ 이하가 될 때까지 반복합니다.

저자들은 $\Delta^-$가 Feasible하고 Capa를 전혀 소비하지 않는 제품이 없다는 가정 아래, 알고리즘이 매 반복마다 Feasible한 Mix를 유지하고 목적값이 Non-decreasing이며 상한이 있으므로 종료한다고 설명합니다. 하지만 이는 **전역최적성 증명**이 아닙니다. 반복은 좋은 Feasible Solution으로 수렴하되, 일부 Instance에서는 Local Optimum에 멈출 수 있습니다.

---

## 6) 실험 설계

저자들은 구조가 다른 문제에서 최적화 잠재력과 Heuristic 품질이 어떻게 달라지는지 보기 위해 Instance Generator를 만들었습니다. 초기 Mix $y_0$에 대해 제품별 허용범위를

$$
\Delta^-=0.95y_0,\qquad \Delta^+=1.05y_0
$$

로 두고, 각 Connected Component에 적어도 하나의 Resource가 $\rho^{\max}$ 근처로 부하되도록 기준 Instance를 구성했습니다.

다섯 요인을 각각 Small·Medium·Large의 세 수준으로 바꾸었습니다.

- CMS Size
- CMS 개수
- Service-Time Matrix $R$의 분산 수준
- Capability Matrix $A$의 Density
- 제품 수

$3^5$개 설정마다 Seed 5개를 사용해 총 **1,215개 Instance**를 생성했습니다. 규모 범위는 다음과 같습니다.

- Connected Component 수: 20, 40, 60
- Resource 수: 57–2,349
- Job Class 수: 74–1,500
- 제품 수: 5–1,132
- Capability Matrix Density: 0–50%
- Service Time의 상대 변동: 0–35%
- 제품 이익의 변동계수: 11–36%

평가지표는 기준 이익 $z_0$, Global LP의 최적값 $z_{opt}$, Decomposition 값 $z_{decomp}$로 정의합니다.

$$
\mathrm{potential}=\frac{z_{opt}-z_0}{z_0}
$$

$$
\mathrm{coverage}
=\frac{z_{decomp}-z_0}{z_{opt}-z_0},
\qquad
\mathrm{relative\ shortfall}
=\frac{z_{opt}-z_{decomp}}{z_{opt}-z_0}
=1-\mathrm{coverage}.
$$

원문 Table 2는 첫 번째 비율을 `gap`이라고 인쇄합니다. 이 식은 1일 때 개선 가능량을 전부 포착하므로 실제 의미는 **개선 포착률(coverage)**입니다. 반면 Figure 2와 본문은 potential이 커질수록 `gap`이 0에 가까워진다고 설명하므로, 그 문맥의 gap은 두 번째 **상대 미포착률(relative shortfall)**과 일치합니다. 원문 안의 용어·식 불일치를 구분해 해석해야 합니다.

Global LP는 2.6GHz Quad-Core, RAM 4GB 환경에서 IBM ILOG CPLEX로 풀었습니다. Decomposition의 LP Subproblem은 같은 PC에서 GLPK와 LinProg를 사용했습니다.

---

## 7) 실험 결과와 실제 FAB 사례

### 7.1 생성 Instance 결과

제품별 생산량을 기준 Mix에서 ±5% 범위로만 조정했는데도 최대 이익개선 가능성은 **0–4.75%**였습니다. 전체적으로 저자들은 약 **3%의 이익 개선**이 가능했다고 보고하며, 이는 실제 적용 경험과도 부합한다고 설명합니다.

관찰된 구조적 관계는 다음과 같습니다.

- **제품 수 / Resource 수 비율이 높을수록** Mix 조정의 이익개선 잠재력이 커지는 경향
- Capability Matrix가 **희소할수록**, 즉 Dedication 제약이 복잡하고 장비 대체 가능성이 낮을수록 잠재력이 커지는 경향
- 제품별 이익의 변동이 클수록 고수익 제품으로 Mix를 재배분할 여지가 커지는 경향
- 개선 잠재력이 큰 문제에서는 Decomposition이 Global Optimum에 가까워지는 경향
- 개선 잠재력이 작은 일부 문제에서는 Heuristic이 Local Optimum에 멈추는 현상

논문은 Figure의 상관관계를 중심으로 설명하므로, 이 결과를 엄밀한 인과관계나 모든 FAB에 적용되는 고정 비율로 해석해서는 안 됩니다.

### 7.2 변형된 실제 FAB 데이터

저자들은 실제 FAB 데이터를 변형한 사례에서도 Product Mix 최적화를 수행했습니다. X축에는 Connected Component, 즉 Work Center를 두고 현재 Mix가 만드는 최대 장비가동률, 평균가동률, Sensitivity Interval과 병목을 비교합니다.

최적화된 Mix는 전체 **Layer Starts per Week를 유지**하면서 병목 Work Center의 Load를 유의미하게 낮췄습니다. 생산량 총합을 무조건 줄여 병목을 해소한 것이 아니라, 각 제품이 소비하는 Capa Pattern의 차이를 이용해 Mix를 바꾼 결과입니다. 저자들은 MES에 있는 모든 Route·Product·Operation·Equipment·Dedication을 가장 세밀한 수준으로 반영해도 전체 계산이 수 분 정도라고 보고합니다.

이 사례가 보여주는 핵심은 목적함수 값 하나보다도 **Sensitivity Analysis**입니다. Mix 변화가 어느 Work Center를 병목으로 만들거나 완화하는지, 어떤 제품이 특정 병목의 Driver인지, 수요변화에 장비부하가 어떻게 반응하는지를 정량화할 수 있습니다.

---

## 8) 이 모형이 실제로 표현하는 것과 표현하지 않는 것

### 표현하는 것

- 실제 반도체 FAB의 Product Route와 Re-entrant Operation이 만드는 평균 Capa 부하
- Operation/Job Class별로 가능한 Equipment가 제한되는 Dedication·Capability 구조
- 제품별 최소 의무물량, 수요상한, 단위이익의 Trade-off
- 같은 Operation을 처리할 수 있는 장비 사이의 물량배분
- Work Center 또는 Resource별 병목부하와 Product Mix Sensitivity
- 정적 Capa를 전제로 한 Lot Starts 또는 Layer Starts 수준의 전술 계획

### 표현하지 않는 것

- LOT별 시작·완료시점과 Due Date
- Setup Sequence, Batch 형성, Preventive Maintenance의 구체적 시점
- 장비고장·Yield Loss·수요확률분포 같은 불확실성
- Queue 길이와 Load에 따라 비선형적으로 증가하는 Cycle Time
- OHT·AGV·Stocker의 물리적 반송시간과 경로
- 일·시간·초 단위 Dispatching Rule

논문은 Stationary Case와 Fluid-Flow 관점을 사용합니다. 따라서 $x$는 개별 LOT를 어느 장비에 몇 시에 투입할지 정하는 정수 Scheduling 변수가 아니라, 일정 기간의 평균 작업량을 장비에 연속적으로 나눈 값입니다. Capa Feasibility와 Product Mix 설계에는 적합하지만, 실행 Schedule의 가능성을 보장하려면 이후 단계에서 Discrete-Event Simulation이나 상세 Scheduling 검증이 필요합니다.

---

## 9) 강점·한계·재현 시 주의점

### 강점

- **경제성과 Capa의 직접 결합:** 이익이 높은 제품을 선택하되 실제 Equipment Capa를 넘지 않는 Mix를 계산합니다.
- **반도체 공정구조 반영:** Route, Job Class, Equipment Capability, Dedication을 하나의 선형 구조로 연결합니다.
- **해석 가능성:** Shadow Price, Bottleneck Load, 제품별 Resource Consumption을 통해 결과의 원인을 추적할 수 있습니다.
- **확장성:** Outsourcing, 추가 Capa, Soft Bottleneck, Cold Steel, 병목 수 제한 등으로 확장 가능합니다.
- **산업 데이터와 생성 실험 병행:** 구조 요인을 통제한 1,215개 Instance와 변형된 실제 FAB 사례를 모두 제시합니다.

### 한계

- **정적·결정론적 모형:** Demand, Breakdown, Yield, Processing Time의 불확실성을 직접 다루지 않습니다.
- **Cycle Time 단순화:** Capa 상한을 지키면 Cycle Time 문제가 없다는 운영적 Threshold를 전제로 하며 Queueing 비선형성은 모델 밖에 있습니다.
- **연속 Flow:** $x$와 $y$가 연속변수이므로 Lot Granularity와 Integer Release 제약이 없습니다.
- **Heuristic의 전역최적성 부재:** Feasible과 종료는 보이지만 Global Optimum은 보장하지 않습니다.
- **실제 데이터 공개 제한:** 변형된 FAB 데이터의 상세값이 없어 산업 사례를 그대로 재현할 수 없습니다.
- **확장식 표기 모호성:** 추가 Capa 비용 $g$의 목적함수 부호는 구현 전 재정의가 필요합니다.

### 재현 구현의 최소 단위

논문의 핵심을 코드로 재현하려면 다음 세 단계를 분리하는 것이 좋습니다.

1. `Global LP`: $D,A,R,c,\Delta^-,\Delta^+,\rho^{\max}$를 입력받아 $x,y$를 동시에 최적화
2. `Load Balancing LP`: 고정 $y$에서 $\lambda=Dy$를 계산하고 Min-Max Load 배분
3. `Decomposition`: Master–Load Balancing–$T$ Update를 반복하고 Global LP와 목적값·시간·Feasibility를 비교

검증 지표는 총이익, 최대 Resource Load, Capa 위반량, 반복 횟수, Global Optimum 대비 개선 포착률이 적절합니다.

---

## 10) 결론

이 논문의 핵심은 Product Mix를 단순한 판매·수익 문제로 보지 않고, **제품의 Route가 만드는 장비별 Capa 소비까지 포함한 수리적 의사결정 문제**로 만든 데 있습니다. $Ax=Dy$가 제품물량과 공정작업을 연결하고, $Rx\le\rho^{\max}$가 실제 Equipment Capacity를 제한하며, $c^\top y$가 그 안에서 가장 수익성 높은 Mix를 선택합니다.

또한 전체 LP를 직접 푸는 방식과 함께, 제품계획과 장비 Load Balancing을 반복하는 Decomposition을 제시해 실제 규모의 데이터에 접근할 수 있는 계산 구조를 보여줍니다. 생성 실험과 변형된 FAB 사례는 작은 Mix 조정만으로도 병목을 완화하고 경제적 성과를 개선할 여지가 있음을 보여줍니다. 다만 이 결과는 상세 Scheduling의 대체물이 아니라, 이후 실행계획과 Simulation이 따라와야 하는 **전술적 Capa–Mix 최적화의 기준해**로 해석하는 것이 정확합니다.
