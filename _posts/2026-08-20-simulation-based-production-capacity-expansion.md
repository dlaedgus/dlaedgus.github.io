---
title: "Simulation-Based Optimization for Integrated Production Planning and Capacity Expansion Decisions (WSC 2016)"
date: 2026-08-20 12:20:00 +0900
categories: [paper_review, OR]
tags: [semiconductor-manufacturing, capacity-planning, simulation-optimization, clearing-function, simulated-annealing]
math: true
---

# Paper Review — *Simulation-Based Optimization for Integrated Production Planning and Capacity Expansion Decisions* (WSC 2016)

- **저자:** Timm Ziarnetzky, Lars Mönch  
- **소속:** University of Hagen  
- **학회:** Winter Simulation Conference (WSC)  
- **년도:** 2016  
- **원문:** [WSC Proceedings 공식 PDF](https://www.informs-sim.org/wsc16papers/263.pdf)

---

## 1) High-Level Summary (3–5 sentences)

본 논문은 단일 **Front-End(FE)**와 단일 **Back-End(BE)**로 이루어진 단순화된 반도체 Supply Chain에서, **생산 투입계획과 Capa 확장 의사결정**을 함께 최적화합니다. 핵심 Trade-off는 비싼 FE 병목장비를 더 높은 가동률로 쓰면 Throughput과 수익은 늘 수 있지만 FE Queue와 Cycle Time이 증가하고, 상대적으로 증설이 쉬운 BE Capa를 늘리면 BE 구간의 Cycle Time을 줄여 전체 Supply Chain 성능을 보완하는 대신 비용이 발생한다는 점입니다. 저자들은 FE에는 혼잡을 반영하는 **Allocated Clearing Function 기반 LP**, BE에는 **Fixed Lead Time 기반 LP**를 사용하고, LP가 만든 Release Schedule을 **Discrete-Event Simulation(DES)**으로 평가합니다. 바깥쪽에서는 **Simulated Annealing(SA)**이 FE 최소 병목가동률 $M$과 BE 추가 Capa 비율 $a$를 탐색하며, 실험에서 기준 계획 대비 평가 목적값을 **18.35%–37.17%** 개선했습니다. 다만 실제 기업의 운영 데이터를 적용한 연구가 아니라, MIMAC I과 공개 BE 모델을 결합한 **대규모 반도체 Benchmark Simulation 연구**입니다.

---

## 2) 문제 설정: 왜 생산계획과 Capa를 분리하면 안 되는가

논문이 다루는 Supply Chain은 다음과 같습니다.

$$
\text{Wafer Fab/Probe (FE)}
\rightarrow \text{Die Bank (DB)}
\rightarrow \text{Assembly/Test (BE)}
\rightarrow \text{Distribution Center (DC)}.
$$

여기서 FE는 Wafer Fabrication과 Probe/Sort를, BE는 Assembly와 Final Test를 포함합니다. 실제 반도체 Supply Chain은 여러 FAB·DB·Assembly·Test Site가 연결된 Network이지만, 논문은 방법론을 검증하기 위해 FE 1개와 BE 1개로 축소합니다.

생산관리 관점의 핵심 문제는 다음 세 문장으로 정리할 수 있습니다.

1. FE의 Steppers와 같은 고가 병목장비를 낮게 활용하면 투자된 Capa를 낭비하고 판매 가능한 Output도 줄어듭니다.  
2. 반대로 병목가동률을 무조건 높이면 WIP가 쌓여 FE Cycle Time이 비선형적으로 증가합니다.  
3. BE의 추가 Capa는 전체 Cycle Time을 낮출 수 있지만, 증설 자체가 비용이므로 필요한 만큼만 선택해야 합니다.

따라서 의사결정은 단순한 “가동률 최대화”가 아닙니다. 논문은 다음 두 값을 동시에 선택합니다.

- $M$: 각 기간에 확보하려는 **FE 병목 Work Center의 최소 Utilization**
- $a$: 기존 BE Capa 대비 **추가 Capa 비율**

그 뒤 제품별·기간별 Release와 WIP 흐름을 LP로 만들고, 실제 실행 시 장비고장·Batch·Setup 등으로 발생하는 KPI를 Simulation에서 확인합니다. 즉, 계획모형이 “무엇을 투입할지”를 결정하고, Simulation이 “그 계획이 Shop Floor에서 실제로 어떤 성능을 내는지”를 검증합니다.

---

## 3) 제안 방법: LP–Simulation–SA의 3계층 구조

전체 방법은 세 의사결정 층으로 나뉩니다.

### 3.1 내부 계획모형: Release Schedule 산출

주어진 $(M,a)$에 대해 LP가 제품별·기간별 FE 투입량과 FE/BE Output, WIP, DB·DC Inventory, Backlog를 결정합니다. FE에서는 Load와 Output의 비선형 관계를 Piecewise Linear **Clearing Function**으로 근사하고, BE에서는 Simulation으로 추정한 Lead Time과 유한 Capa를 적용합니다.

### 3.2 Base System: Discrete-Event Simulation

LP의 Release Schedule을 반도체 제조 특성을 포함한 Simulation에서 실행합니다. 여기서는 계획값이 아니라 실현된 Cycle Time, 병목 Utilization, WIP·Inventory·Backlog가 산출됩니다. 이 과정이 중요한 이유는 LP만으로는 Breakdown, Sequence-dependent Setup, Batch Processing, Operator와 Secondary Resource의 상호작용을 충분히 표현하기 어렵기 때문입니다.

### 3.3 외부 탐색: Simulated Annealing

SA는 $(M,a)$를 바꾸면서 매 후보마다 다음 절차를 반복합니다.

$$
(M,a)
\rightarrow \text{Production Planning LP}
\rightarrow \text{Release Schedule}
\rightarrow \text{DES}
\rightarrow \text{Profit/Cycle Time/Utilization 평가}.
$$

결국 본 연구는 단일 수리모형 하나가 아니라, **최적화로 계획을 만들고 Simulation으로 현실성을 평가한 뒤 Metaheuristic으로 계획 파라미터를 조정하는 Simulation-based Optimization**입니다.

---

## 4) 수리모형: FE 혼잡과 BE Capa를 생산계획 안에 넣기

### 4.1 주요 집합과 변수

- $G$: 제품 집합, $g\in G$
- $t=1,\ldots,T$: 계획기간
- $K^F$, $K^B$: FE·BE Work Center 집합
- $O^F(g)$, $O^B(g)$: 제품 $g$의 FE·BE 공정 Route
- $O(k)$: Work Center $k$가 처리하는 공정 집합
- $Y^F_{gtl}$, $Y^B_{gtl}$: 기간 $t$에 공정 $l$을 완료한 제품 $g$ 수량
- $Y^F_{gt}$, $Y^B_{gt}$: 각각 FE·BE Route의 마지막 공정에서 완료된 제품 $g$ 수량
- $X^F_{gtl}$: 기간 $t$에 FE 공정 $l$을 시작한 수량
- $X^B_{gt}$: 기간 $t$에 BE 첫 공정으로 Release한 수량
- $W^F_{gtl}$, $W^B_{gt}$: FE 공정별 WIP와 BE WIP
- $I^{DB}_{gt}$, $I^{DC}_{gt}$: Die Bank와 Distribution Center의 재고
- $B^{DC}_{gt}$: DC Backlog
- $Z^k_{gtl}$: FE Work Center $k$의 Output Capa 중 제품–공정 $(g,l)$에 배정한 비율

### 4.2 계획모형의 비용함수

내부 LP는 FE WIP, BE WIP, DB·DC 재고, DC Backlog 비용의 합을 최소화합니다.

$$
\min C_{\mathrm{plan}}
=
\sum_{g\in G}\sum_{t=1}^{T}
\left[
\sum_{l\in O^F(g)}\omega^F_{gt}W^F_{gtl}
+\omega^B_{gt}W^B_{gt}
+h^{DB}_{gt}I^{DB}_{gt}
+h^{DC}_{gt}I^{DC}_{gt}
+b^{DC}_{gt}B^{DC}_{gt}
\right].
$$

- $\omega^F,\omega^B$: FE·BE WIP 단위비용
- $h^{DB},h^{DC}$: DB·DC 재고보유비용
- $b^{DC}$: Backlog 단위비용

이 목적함수는 “Output만 크게”가 아니라 WIP·재고·미납을 동시에 통제합니다. 동일한 Demand를 충족하더라도 Cycle Time이 길어져 WIP가 커지거나 완제품을 너무 일찍 만들어 재고가 늘면 비용이 증가합니다.

### 4.3 FE 공정별 물량보존

각 FE 공정의 WIP는 이전 WIP와 신규 Start에서 완료량을 뺀 값입니다.

$$
W^F_{gtl}=W^F_{g,t-1,l}+X^F_{gtl}-Y^F_{gtl}.
$$

이 식은 “들어온 물량 = 처리된 물량 + 남은 물량”이라는 기본 Flow Conservation입니다. Re-entrant Route가 길더라도 공정 $l$ 단위로 Balance를 유지할 수 있습니다.

FE 마지막 공정의 Output은 DB 재고로 들어가고, BE Release가 이를 소진합니다.

$$
I^{DB}_{gt}=I^{DB}_{g,t-1}+\Lambda_gY^F_{gt}-X^B_{gt}.
$$

$\Lambda_g$는 FE Lot과 BE Lot의 Size 차이를 변환하는 계수입니다. 실험에서는 FE Lot 48 Wafer, BE Lot 16 Wafer이므로 두 제품 모두 $\Lambda_g=3$입니다.

### 4.4 Allocated Clearing Function: Load가 증가해도 Output은 선형 증가하지 않는다

고정 Lead Time 계획은 Release를 늘려도 Cycle Time이 변하지 않는다고 보기 쉽습니다. 그러나 실제 FAB은 병목 Utilization이 높아질수록 Queue가 급증합니다. 논문은 FE Work Center의 가능한 Output을 Piecewise Linear Clearing Function으로 제한합니다.

$$
\alpha_{gl}Y^F_{gtl}
\le
\mu_n^k Z^k_{gtl}
+\beta_n^k\alpha_{gl}
\left(X^F_{gtl}+W^F_{g,t-1,l}\right),
\qquad n\in C(k).
$$

- $\alpha_{gl}$: 제품 $g$의 공정 $l$ Processing Time
- $\mu_n^k$, $\beta_n^k$: Work Center $k$ Clearing Function의 $n$번째 선분의 절편과 기울기
- $C(k)$: Piecewise Linear 선분 집합
- $X^F+W^F$: 해당 기간에 처리 대상으로 존재하는 Workload
- $Z^k_{gtl}$: Work Center Output 능력의 할당 비율

또한 한 Work Center의 할당비율 합은 1입니다.

$$
\sum_{g\in G}\sum_{l\in O(k)}Z^k_{gtl}=1.
$$

Clearing Function의 의미는 단순합니다. 투입 초기에는 WIP 증가가 Output 증가로 이어지지만, 혼잡구간에서는 WIP를 더 넣어도 Output 증가폭이 둔화됩니다. 이로써 LP 안에 `WIP 증가 → Congestion 증가 → Cycle Time 악화`의 효과를 간접적으로 넣습니다. 계수는 FE Simulation Data로 Fitting합니다.

### 4.5 FE 병목 최소가동률

병목 Work Center $b$가 계획기간마다 최소한 $M$만큼 활용되도록 다음 조건을 둡니다.

$$
\sum_{g\in G}\sum_{l\in O(b)}
\alpha_{gl}Y^F_{gtl}
\ge M\widetilde C_b.
$$

$\widetilde C_b$는 FE 병목의 시간 Capa입니다. $M$을 높이면 LP는 병목에서 더 많은 Processing Time을 사용하도록 Release를 늘립니다. 그러나 실제 Simulation에서는 Breakdown과 변동성 때문에 목표 $M$을 정확히 달성하지 못할 수 있으므로, 외부 평가함수에서 Shortfall을 벌점으로 처리합니다.

### 4.6 BE의 Fixed Lead Time과 유한 Capa

BE WIP Balance는 다음과 같습니다.

$$
W^B_{gt}=W^B_{g,t-1}+X^B_{gt}-Y^B_{gt}.
$$

BE 공정 $l$의 완료량은 Release 시점과 추정 Lead Time $L(g,l)$로 연결됩니다.

$$
Y^B_{gtl}=X^B_{g,t-\lfloor L(g,l)\rfloor}.
$$

Lead Time은 제품별 Flow Factor $FF_g$와 누적 Processing Time으로 추정합니다.

$$
L(g,l)=L(g,l-1)+FF_g\alpha_{gl},
\qquad L(g,0)=0.
$$

$FF_g$는 BE에 투입된 Material이 FGI가 되기까지의 평균시간을 총 Processing Time으로 나눈 값이며, 긴 Simulation Run에서 얻습니다. 높은 병목가동률과 작은 추가 Capa에서는 Queue가 커지므로 $FF_g$도 커집니다.

Work Center별 처리시간 합은 Available Capa를 초과할 수 없습니다.

$$
\sum_{g\in G}\sum_{l\in O(k)}
\alpha_{gl}Y^B_{gtl}
\le C_k(a),
\qquad
C_k(a)=(1+a)\widehat C_k.
$$

$\widehat C_k$는 기존 BE Capa이고 $a$는 SA가 선택하는 추가비율입니다. 논문은 장비 한 대씩의 구매 여부를 고르는 Integer Model이 아니라, **BE Work Center Capa를 공통 비율로 확대하는 Aggregate Planning Model**입니다.

### 4.7 Simulation 기반 최종 평가함수

원문 목적의 의미를 유지하면서 최대화 Score로 정리하면 다음과 같습니다. 원문 본문에는 한 차례 “minimized”라고 적혀 있지만, 식의 Profit-minus-penalty 구조와 이후 “maximum $f$” Grid Search 설명은 최대화 문제임을 분명히 합니다.

$$
\max_{M,a}\;F(M,a)
=
\sum_{g,t}r_{gt}\widetilde Y^B_{gt}
-C\!\left(\widetilde W^F,\widetilde W^B,
\widetilde B^{DC},\widetilde I^{DB},\widetilde I^{DC}\right)
-\frac{\beta}{T}\sum_{g,t}(\widetilde C_{gt}-C^*_{gt})^+
-\frac{\delta}{T}\sum_t(M\widetilde C_b-\widetilde U_{bt})^+
-\lambda a.
$$

여기서 $(x)^+=\max(0,x)$입니다. $\widetilde Y^B$, $\widetilde W$, $\widetilde B$, $\widetilde I$, $\widetilde C_{gt}$, $\widetilde U_{bt}$는 실행된 Release Schedule과 Simulation에서 얻은 실현값입니다. 반면 $\widetilde C_b$는 물결표가 붙어 있어도 실현 KPI가 아니라 논문이 정의한 FE 병목의 고정 시간 Capa입니다.

- $r_{gt}\widetilde Y^B_{gt}$: 실현 Output의 매출
- $C(\widetilde W^F,\widetilde W^B,\widetilde B^{DC},\widetilde I^{DB},\widetilde I^{DC})$: 실행 결과의 WIP·재고·Backlog 비용
- $\widetilde C_{gt}-C^*_{gt}$: 허용 최대 Cycle Time 초과분
- $M\widetilde C_b-\widetilde U_{bt}$: FE 최소가동률 목표 미달분
- $\lambda a$: BE 추가 Capa 비용

따라서 $M$과 $a$는 둘 다 무한정 커질 수 없습니다. FE를 공격적으로 가동해 얻는 매출보다 Cycle Time Penalty가 커지면 $M$을 낮추고, BE Capa의 한계효과보다 증설비용이 커지면 $a$를 줄이는 해가 선택됩니다.

---

## 5) 알고리즘: Simulated Annealing으로 $(M,a)$ 탐색

후보 Grid는 110개입니다.

$$
M\in\{0.50,0.55,\ldots,0.95\},
\qquad
a\in\{0,0.05,\ldots,0.50\}.
$$

모든 점을 평가하는 Grid Search는 Simulation 비용이 크므로, SA가 인접 Grid Point를 탐색합니다. 현재까지의 Incumbent와 후보해를 각각 $(M,a)$와 $(M',a')$라 하고 $\Delta=F(M,a)-F(M',a')$로 두면, 개선해는 수용하고 악화해도 다음 확률로 수용합니다.

$$
P(\text{accept})=
\begin{cases}
1, & \Delta\le 0,\\
\exp(-\Delta/\mathcal T), & \Delta>0,
\end{cases}
$$

여기서 $\mathcal T$는 Temperature입니다. 초기에는 나쁜 이동도 받아들여 Local Optimum을 벗어나고, Geometric Cooling으로 $\mathcal T$를 낮추면서 탐색을 안정화합니다.

실행 절차는 다음과 같습니다.

1. 후보 $(M,a)$를 선택합니다.  
2. Clearing Function과 BE Capa를 반영한 LP를 풀어 Release Schedule을 만듭니다.  
3. 동일 Schedule을 DES에서 여러 번 실행해 실현 Cycle Time·Utilization·Output을 측정합니다.  
4. 매출, WIP·재고·Backlog, Cycle Time 위반, Utilization 미달, 추가 Capa 비용으로 $F(M,a)$를 계산합니다.  
5. SA 수용규칙에 따라 다음 후보로 이동합니다.  
6. 연속 5개 Temperature에서 수용된 이동이 없으면 종료합니다.

논문 설정에서는 Temperature마다 16회 Iteration을 수행합니다. 즉, SA의 한 Move조차 단순 함수계산이 아니라 LP 풀이와 반복 Simulation을 포함하는 계산집약적 절차입니다.

---

## 6) 실험: 반도체 Benchmark 규모와 결과

### 6.1 Simulation Model

- FE: **MIMAC I**, 장비 200대 이상, 69개 Work Center
- FE 계획병목: Stepper Work Center
- BE: 23개 Work Center
- 제품: 2개
- Route:
  - 제품 1: FE 211 Step, BE 25 Step
  - 제품 2: FE 246 Step, BE 31 Step
- 반영 특성: Batch Processing, Sequence-dependent Setup, Exponential Breakdown, Operator, Secondary Resource
- Dispatching: FE FIFO, BE FIFO와 Same Setup Rule
- 구현: AutoSched AP 및 일부 C++ Customization

이 모델은 실제 회사의 생산 로그를 재현한 것은 아닙니다. 다만 Re-entrant Route, Batch, Breakdown, Setup 등 Wafer FAB의 대표 복잡성을 포함한 공개·연구용 Semiconductor Simulation Model이며, 결합된 모델은 Industry Domain Expert에게 Validation을 받았다고 논문이 보고합니다.

### 6.2 실험설계

- 계획기간: 15주, 주 단위 Period
- 평가기간: 끝단효과를 피하기 위해 처음 12주
- Product Mix: 1:1
- 평균 FE 병목가동률: Low 70%, High 90%
- Demand CV: 0.10, 0.25
- 각 조건당 Demand Scenario: 5개
- 각 Grid Point당 Simulation Replication: 20회
- LP 1회 평균 계산시간: 약 3분

Grid Search는 총 **44,400회 Simulation Run**을 필요로 했고, SA는 총 **7,920회**를 사용했습니다. 논문은 이를 평균 19개 Grid Point를 조사한 결과로 서술하지만, 보고된 Run 수를 20개 Demand Scenario와 Point당 20회 Replication으로 나누면 정확히는 평균 19.8개 Point입니다. 어느 값을 기준으로 해도 SA는 Simulation Burden을 약 82% 줄였습니다.

### 6.3 결과

SA가 기준 $(M,a)=(0,0)$ 대비 얻은 평가 목적값 개선은 다음과 같습니다.

| 평균 병목가동률 | Demand CV | 목적값 개선 | 평균 Utilization Shortfall | 평균 Cycle Time 초과 |
|---|---:|---:|---:|---:|
| Low | 0.10 | 22.22% | 0.68% | 383.10분 |
| Low | 0.25 | 18.35% | 0.99% | 310.77분 |
| High | 0.10 | 37.17% | 0.19% | 284.00분 |
| High | 0.25 | 28.19% | 0.50% | 159.78분 |

두 가지를 함께 봐야 합니다.

- 개선폭은 **18.35%–37.17%**로 컸습니다.
- 그러나 Stochasticity 때문에 가장 좋은 계획도 최대 Cycle Time과 최소가동률을 소폭 위반했습니다.

SA/GS Score 비율은 한 Outlier를 제외하면 대부분 **0.98–1.00**이었고, 많은 Scenario에서 Grid Search와 동일하거나 반올림상 동일한 목적값을 얻었습니다. 다만 CV 0.25의 Low Utilization 한 Scenario에서는 0.76, High Utilization 한 Scenario에서는 0.93이므로 “항상 전역최적”이라고 말할 수는 없습니다.

Low Utilization 환경에서는 기존 계획보다 $M$을 약 10%p 올리고 BE Capa를 소폭 확장하는 조합이 자주 선택됐습니다. High Utilization에서는 오히려 $M$을 기존 계획보다 낮춰 병목가동률을 Smoothing하면서 BE Capa를 늘리는 조합도 나타났습니다. 이는 `병목은 무조건 100% 가동`이 아니라 수요변동과 Downstream Capa를 함께 보아야 한다는 결과입니다.

---

## 7) 논문의 기여

1. **생산계획과 Capa 확장을 통합**했습니다. FE Release만 결정하거나 BE 증설만 평가하지 않고, 두 선택이 Cycle Time과 Profit에 미치는 상호작용을 한 Framework에서 다룹니다.  
2. FE에 **Clearing Function**을 써서 고정 Lead Time 모형의 약점을 보완했습니다. Utilization이 높아질수록 Queue와 Cycle Time이 비선형적으로 커지는 제조현상과 생산계획을 연결합니다.  
3. 수리계획과 **High-fidelity DES**를 결합했습니다. 계획에서 누락되기 쉬운 Breakdown·Batch·Setup·Secondary Resource의 영향을 실행평가에 반영합니다.  
4. SA로 Full Grid Search 대비 Simulation 횟수를 크게 줄이면서 대부분의 Scenario에서 근접한 해를 얻었습니다.

---

## 8) 강점·한계·읽을 때 주의할 점

### 강점

- `Capa 확대 → Queue 감소 → Cycle Time 감소 → Profit 변화`를 정량적 의사결정 구조로 보여줍니다.
- 단순 Capacity Requirement 계산보다 한 단계 더 나아가 혼잡과 실행 변동성을 고려합니다.
- 반도체 FE의 장비 200대 이상, 69 Work Center와 수백 Step Route를 가진 비교적 큰 Benchmark를 사용합니다.
- 목적함수에 Output뿐 아니라 WIP·Inventory·Backlog·Cycle Time·Capa Cost가 함께 들어가므로 제조 KPI의 Trade-off가 분명합니다.

### 한계

- **실제 FAB 적용 실증이 아닙니다.** MIMAC I과 공개 BE Model을 결합한 Simulation Study입니다.
- FE 1개와 BE 1개뿐이며, 실제처럼 여러 FAB·Site·Die Bank가 연결된 Network는 다루지 않습니다.
- 수요는 결정론적 계획값을 기본으로 하고 실험에서 Normal Perturbation을 가합니다. Forecast Update가 들어오는 Rolling Horizon은 후속과제로 남습니다.
- $a$는 BE Work Center별 장비대수가 아니라 공통적인 Capa 비율입니다. 개별 Tool 구매·이설·Qualification Lead Time·인력제약을 직접 결정하지 않습니다.
- FE–DB와 BE–DC 운송은 Infinite Capacity로 두고 연속 공정 사이 Transfer도 즉시라고 가정합니다. 따라서 FAB 간 물류나 OHT 병목 연구로 해석하면 안 됩니다.
- SA는 전역최적성을 보장하지 않으며 실제로 한 Scenario에서 Grid Search 대비 0.76의 Outlier가 있습니다.
- 원문 결과의 개선률은 설정한 Revenue와 Penalty 계수에 의존합니다. 논문도 비용·Penalty 값이 개선폭의 크기에 영향을 준다고 명시합니다.

---

## 9) 결론

이 논문의 핵심은 `FE 병목을 얼마나 밀어붙일 것인가`와 `BE Capa를 얼마나 보완할 것인가`를 분리하지 않은 데 있습니다. LP는 WIP·재고·Backlog를 고려한 Release Schedule을 만들고, DES는 장비고장과 Queue로 인한 KPI를 측정하며, SA는 두 계획 파라미터를 탐색합니다. 동시에 단일 FE/BE Benchmark와 Aggregate Capacity라는 한계가 있으며, 더 일반적인 적용에는 다수 FAB Network, Tool별 Capability, Qualification, Maintenance, 물류, Rolling Horizon Data가 필요합니다.
