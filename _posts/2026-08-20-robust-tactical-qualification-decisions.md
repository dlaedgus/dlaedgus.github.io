---
title: "Robust Tactical Qualification Decisions in Flexible Manufacturing Systems (Omega 2022)"
date: 2026-08-20 12:10:00 +0900
categories: [paper_review, OR]
tags: [semiconductor-manufacturing, robust-optimization, qualification, capacity-planning]
math: true
---

# Paper Review — *Robust Tactical Qualification Decisions in Flexible Manufacturing Systems* (Omega 2022)

- **1저자:** Antoine Perraudat  
- **공저자:** Stéphane Dauzère-Pérès, Philippe Vialletelle  
- **제목:** Robust Tactical Qualification Decisions in Flexible Manufacturing Systems  
- **저널:** Omega, Vol. 106, Article 102537  
- **년도:** 2022  
- **DOI:** [10.1016/j.omega.2021.102537](https://doi.org/10.1016/j.omega.2021.102537)  
- **원문:** [HAL Open Access PDF](https://hal.science/hal-03352370/document)

---

## 1) High-Level Summary (3–5 sentences)

본 논문은 High-Mix 반도체 FAB에서 각 Operation–Machine 조합을 새로 **Qualification**할지, 그리고 그 Qualification을 언제 시작할지를 결정하는 전술적 Capa Planning 문제를 다룹니다. Qualification은 장비가 특정 Recipe를 품질·Yield 저하 없이 수행할 수 있음을 인증하는 절차이며, 수 주에서 수개월이 걸릴 수 있으므로 수요가 확정된 뒤 대응하면 늦을 수 있습니다. 저자들은 먼저 Qualification Cost와 Lead Time, 장비별 Throughput·가용시간·최대가동률을 반영한 MILP를 제안하고, 제품군 내부의 **Product Cannibalization**을 Budgeted Uncertainty Set으로 표현해 모든 허용 수요에서 Capa를 지키는 Static Robust Counterpart를 유도합니다. 프랑스 Crolles의 실제 반도체 공장 내 두 Critical Work Center 데이터로 실험한 결과, Robustness 확보에 필요한 추가 Qualification 수는 비교적 작지만 Nominal Demand만 보고 정한 Qualification은 수요변동이 커질 때 빈번한 Capa 위반을 만들 수 있었습니다. 핵심 메시지는 “Qualification을 많이 여는 것”보다 **불확실한 Product Mix에서 부하를 실제로 Pooling할 수 있는 올바른 Qualification을 선택하는 것**이 중요하다는 점입니다.

---

## 2) 산업 문제: Qualification이 왜 Capa를 결정하는가

반도체 Front-End에서는 하나의 Wafer가 완성되기까지 최대 약 1,000개의 Operation을 거칠 수 있고, 같은 Work Center를 40회 이상 재방문하는 Re-entrant Flow가 발생할 수 있습니다. 하나의 Work Center에는 수 대에서 최대 약 200대의 Parallel Machine이 있지만, 장비가 설치되어 있다고 모든 Recipe를 즉시 처리할 수 있는 것은 아닙니다.

Recipe는 Pressure, Temperature, Chemical Condition과 작업 절차를 정의합니다. 장비가 해당 Recipe를 실행해도 제품의 Quality와 Yield가 저하되지 않는다는 검증을 통과해야만 그 Operation을 처리할 수 있습니다. 논문에서 Qualification은 정확히 **(Operation, Machine) Pair**입니다.

Qualification 상태는 세 가지로 구분됩니다.

- 이미 Qualified되어 즉시 사용할 수 있는 Pair
- 기술적으로 Qualifiable하지만 아직 Qualification 절차가 필요한 Pair
- 장비 세대·Hardware·Software 제약 때문에 애초에 Qualify할 수 없는 Pair

High-Mix 공장에서는 수백 개 제품이 같은 장비 Capa를 경쟁합니다. 제품별 Route와 Re-entrant Flow Factor가 다르므로 제품 한 단위가 만드는 Workload도 다릅니다. 따라서 제품 A의 수요가 줄고 같은 Family의 제품 B 수요가 늘어 총 제품수는 같더라도, 특정 Work Center의 부하는 크게 증가할 수 있습니다.

논문의 계획 Horizon은 보통 **6–12개월의 Tactical Level**입니다. 이 기간에는 신규 장비가 들어오고, 신제품이 Ramp-up하며, 기존 제품의 수요도 변합니다. Qualification은 수 주에서 수개월이 걸리고 Test Lot·Quality Task로 Capa 손실도 발생하므로 미리 결정해야 합니다. 목표는 모든 기간의 수요를 처리할 수 있도록 필요한 Qualification만 선택하되, Qualification 수 또는 비용을 최소화하는 것입니다.

---

## 3) Deterministic Model: Minimum Cost Qualification Configuration Problem

저자들은 결정론적 문제를 **MCQCP(Minimum Cost Qualification Configuration Problem)**라고 부릅니다. 한 Work Center에 서로 다른 세대의 Unrelated Parallel Machine $M$대가 있고, $R$개 Operation을 $T$개 기간 동안 처리한다고 가정합니다.

### 3.1 주요 파라미터

- $M,R,P,T$: Machine, Operation, Product, Period의 수
- $q_{r,m}$: Qualification 상태. 1이면 기존 Qualified, 2이면 신규 Qualifiable, 0이면 불가능
- $tp_{r,m}$: Machine $m$에서 Operation $r$의 Throughput Rate
- $c_{t,m}$: Period $t$에서 Machine $m$의 Production Availability(hours)
- $u_{t,m}^{\max}$: 허용 가능한 최대 Utilization
- $rf_{p,r}$: Product $p$ 한 단위 생산 시 Operation $r$을 수행하는 횟수(Re-entrant Flow Factor)
- $d_{t,p}$: Period $t$의 Product $p$ 수요
- $l_{t,r,m}$: Period $t$에 Qualification을 시작할 때 필요한 Lead Time
- $cq_{r,m}$: Operation $r$–Machine $m$ Qualification Cost
- $\delta_t$: Period별 Discount Factor

### 3.2 결정변수

$$
OQ_{t,r,m}\in\{0,1\}
$$

Period $t$에 Operation $r$을 Machine $m$에 Qualification하는 절차를 시작하면 1입니다.

$$
WIP_{t,r,m}\in[0,1]
$$

Period $t$의 Operation $r$ 수요 중 Machine $m$이 처리하는 비율입니다. 이름은 $WIP$지만 LOT 수 자체가 아니라 **부하배분 비율**입니다.

### 3.3 목적함수

$$
\min \sum_{t,r,m}\delta_t cq_{r,m}OQ_{t,r,m}.
$$

필요한 Qualification의 Discounted Cost를 최소화합니다. 모든 비용이 동일하면 Qualification 개수를 최소화하는 문제가 됩니다. $\delta_t\ge\delta_{t+1}$로 두면 늦은 기간의 비용이 작아져 Qualification을 가능한 늦게 시작하도록 유도할 수 있습니다.

### 3.4 Machine Capacity Constraint

제품수요를 Operation별 Workload로 바꾸면 Period $t$, Operation $r$의 필요량은

$$
L_{t,r}=\sum_p rf_{p,r}d_{t,p}
$$

입니다. 이를 Machine별 처리비율과 Throughput으로 시간으로 환산하면,

$$
\sum_r
\frac{\left(\sum_p rf_{p,r}d_{t,p}\right)WIP_{t,r,m}}
{tp_{r,m}}
\le c_{t,m}u_{t,m}^{\max}
\qquad \forall t,m.
$$

좌변은 Machine $m$에 배정된 총 필요 처리시간이고, 우변은 해당 기간의 가용시간 중 실제로 사용을 허용하는 최대시간입니다. $u^{\max}$를 1보다 낮게 두는 이유는 Utilization이 1에 접근할수록 Queue와 Cycle Time이 급격히 증가하기 때문입니다.

### 3.5 Flow Constraint

수요가 존재하는 Operation은 모든 Machine 배분비율의 합이 1이어야 합니다.

$$
\sum_m WIP_{t,r,m}=1
\qquad
\forall t,r:\sum_p rf_{p,r}d_{t,p}>0.
$$

즉, Operation Load를 일부 누락해 Capa를 맞추는 해는 허용하지 않습니다.

### 3.6 Qualification과 Lead Time

기존 Qualified 또는 Qualify 불가능한 Pair에는

$$
WIP_{t,r,m}\le q_{r,m}
\qquad \forall t,r,m:q_{r,m}\ne2
$$

를 둡니다. $q=1$이면 최대 1까지 배분할 수 있고, $q=0$이면 배분이 차단됩니다.

아직 Qualify되지 않은 Pair에는 Qualification이 완료된 뒤에만 배분할 수 있도록

$$
WIP_{t,r,m}
\le
\sum_{t'=1:\;t-t'\ge l_{t',r,m}}^{t}OQ_{t',r,m}
\qquad \forall t,r,m:q_{r,m}=2
$$

를 둡니다. Period $t'$에 시작한 Qualification이 Lead Time을 지나 완료되어야 Period $t$의 생산에 쓸 수 있다는 뜻입니다.

### 3.7 모형의 성격

MCQCP는 Operation 배분 $WIP$는 연속변수, Qualification 선택 $OQ$는 이진변수인 MILP이며, 저자들은 Generalized Assignment Problem을 환원해 단일 Period에서도 NP-Hard임을 보입니다. Work Center끼리는 Operation을 공유하지 않기 때문에 공장 전체 문제를 Work Center별로 나눠도 최적성이 보존됩니다.

반면 이 모형은 상세 Scheduling이 아닙니다. Operation Precedence, LOT별 Queue, Setup Sequence를 직접 넣지 않고, Tactical Capacity 관점에서 전체 수요가 Qualified Capa 안에 배분 가능한지만 봅니다. Maintenance, Engineering, Setup, Qualification Test로 인한 Capacity Loss는 $c_{t,m}$에 Exogenous하게 반영합니다.

---

## 4) Demand Uncertainty와 Product Cannibalization

확률분포를 추정하기 어려운 이유는 신제품 데이터가 부족하고, 제품별 수요가 독립적이지 않기 때문입니다. 같은 Application을 겨냥한 제품들은 서로 대체됩니다. 한 Micro-controller의 판매가 증가하면 같은 Family의 다른 제품 판매는 감소할 수 있습니다. 저자들은 이를 Product Family별 Budget으로 표현합니다.

### Figure 2 — 한 제품의 12개월 Demand Profile

<img width="900" alt="평균수요 대비 월별 수요가 ramp-up과 ramp-down하는 제품 demand profile" src="/assets/img/paper-reviews/2026-08-20/perraudat-fig2.svg" />

> Source: Perraudat et al. (2022), Figure 2, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). 원문 그림을 크롭했습니다. [Author manuscript](https://hal.science/hal-03352370/document)

5–8개월 구간의 급격한 Ramp-up과 이후 하락은 명목수요 하나만으로 Qualification을 정할 때의 위험을 보여줍니다. 이 논문은 수요분포를 특정하기보다 가능한 수요범위와 Product Family별 총수요 Budget을 정의해 이 변동을 방어합니다.


### 4.1 Uncertainty Set

- $\bar d_{t,p}$: Nominal Demand
- $\hat d_{t,p}$: Nominal에서 허용하는 최대 편차, $0\le\hat d_{t,p}\le\bar d_{t,p}$
- $\alpha_{p,f}$: Product $p$가 Family $f$에 속하면 1
- $\Gamma_{t,f}$: Period $t$, Family $f$의 총 Demand Budget

제품별 수요는

$$
d_{t,p}\in
[\bar d_{t,p}-\hat d_{t,p},\;\bar d_{t,p}+\hat d_{t,p}]
$$

범위 안에서 움직이고, Family 총수요는

$$
\sum_{p:\alpha_{p,f}=1}d_{t,p}\le\Gamma_{t,f}
$$

를 만족합니다. 따라서 Period $t$의 Uncertainty Set은

$$
\mathcal D_t=
\left\{
d_t:\begin{array}{l}
\bar d_{t,p}-\hat d_{t,p}\le d_{t,p}\le\bar d_{t,p}+\hat d_{t,p}\quad\forall p,\\
\sum_{p:\alpha_{p,f}=1}d_{t,p}\le\Gamma_{t,f}\quad\forall f
\end{array}
\right\}.
$$

실험에서는

$$
\Gamma_{t,f}=\sum_{p:\alpha_{p,f}=1}\bar d_{t,p}
$$

로 설정합니다. Family 총수요의 **상한**을 Nominal 총량으로 두고 그 안에서 제품별 비율이 달라지는 상황입니다. 총량이 Nominal보다 작은 실현도 $\mathcal D_t$에 포함됩니다. Family 총수요가 같더라도 제품별 $rf_{p,r}$과 $tp_{r,m}$가 다르기 때문에 Operation Workload는 커질 수 있습니다.

이 설정은 Demand Distribution과 Scenario Probability를 요구하지 않습니다. 대신 모델러가 반드시 방어하고 싶은 Plausibility Region을 정의하고, 그 안의 **모든 수요실현**에서 Capa를 지키도록 합니다.

---

## 5) Robust Counterpart: Worst-Case Capacity를 선형화하기

Robust Model인 **MCRQCP(Minimum Cost Robust Qualification Configuration Problem)**는 Flow와 Qualification 제약을 유지하면서 모든 $d_t\in\mathcal D_t$에 대해 Machine Capacity를 만족해야 합니다.

### 5.1 Worst-Case Capacity Constraint

Period $t$, Machine $m$에 대해

$$
\max_{d_t\in\mathcal D_t}
\sum_p d_{t,p}
\left(
\sum_r\frac{rf_{p,r}WIP_{t,r,m}}{tp_{r,m}}
\right)
\le c_{t,m}u_{t,m}^{\max}.
$$

괄호 안은 Product $p$ 한 단위가 Machine $m$에 만드는 시간부하입니다. 최악의 Product Mix가 왔을 때에도 허용 Capa를 넘지 않게 합니다. Operation을 많이 재방문하거나 처리속도가 느린 Product의 수요가 증가하는 조합이 Worst Case가 됩니다.

### 5.2 Inner Maximization의 Dual

내부 문제는 Linear Program이므로 Strong Duality를 이용해 별도의 Max 문제를 없앨 수 있습니다. Demand Lower Bound, Upper Bound, Family Budget에 대응하는 Nonnegative Dual Variable을 각각

$$
y_{t,m,p}^{\min},\qquad
y_{t,m,p}^{\max},\qquad
y_{t,m,f}^{\gamma}
$$

로 둡니다. Robust Capacity Constraint는 다음 선형식으로 바뀝니다.

$$
\sum_p-(\bar d_{t,p}-\hat d_{t,p})y_{t,m,p}^{\min}
+\sum_f\Gamma_{t,f}y_{t,m,f}^{\gamma}
+\sum_p(\bar d_{t,p}+\hat d_{t,p})y_{t,m,p}^{\max}
\le c_{t,m}u_{t,m}^{\max}
$$

그리고 각 Product $p$에 대해

$$
-y_{t,m,p}^{\min}
+y_{t,m,p}^{\max}
+\sum_{f:\alpha_{p,f}=1}y_{t,m,f}^{\gamma}
\ge
\sum_r\frac{rf_{p,r}WIP_{t,r,m}}{tp_{r,m}}.
$$

모든 Dual Variable은 0 이상입니다. 이 변환 덕분에 “무한히 많은 Demand Scenario에 대한 Capa Constraint”를 유한한 Linear Constraint로 표현할 수 있습니다.

Flow Constraint의 활성 조건도 Nominal Demand가 아니라 가능한 최대수요를 기준으로 바뀝니다.

$$
\sum_mWIP_{t,r,m}=1
\quad
\forall t,r:
\sum_p rf_{p,r}(\bar d_{t,p}+\hat d_{t,p})>0.
$$

### 5.3 Static Robustness의 의미와 한계

이 논문은 $WIP_{t,r,m}$도 수요실현 전에 고정되는 **Static Robust Reformulation**을 사용합니다. 실제 수요가 드러난 뒤 Machine별 배분비율을 조정하는 Adjustable Robust Model보다 보수적일 수 있습니다. 저자들도 수요 불확실성이 여러 Machine Capacity Row에 동시에 나타나므로 Row-wise Uncertainty 조건이 충족되지 않으며, Adjustable Extension은 후속연구로 남깁니다.

예시 규모 $P=238$, $R=1{,}208$, $F=3$, $M=20$, $T=7$에서 Deterministic MCQCP는 Continuous Variable 169,120개, Binary Variable 169,120개, Constraint 685,076개입니다. Robust MCRQCP는 Binary Variable 수는 같지만 Continuous Variable 202,860개, Constraint 785,456개입니다. MCQCP를 분모로 계산하면 각각 약 **20.0%**, **14.7% 증가**입니다. 원문 Table 2의 16.6%, 12.8%는 증가분을 MCRQCP 총수로 나눈 비표준 비율이므로 일반적인 증가율과 구분해야 합니다. Capa도 더 Tight해져 실제 계산난도는 숫자 증가 이상으로 커질 수 있습니다.

---

## 6) 주어진 Qualification Set의 Robustness 측정

제품별 $\hat d_{t,p}$를 직접 정하기 어렵다면, 반대로 현재 Qualification Set이 수요를 몇 %까지 흡수할 수 있는지 계산할 수 있습니다.

### 6.1 Maximum Robustness Budgeted Qualification Problem

제품별 대칭 편차비율 $0\le\theta_{t,p}\le1$을 두어

$$
d_{t,p}\in
[\bar d_{t,p}(1-\theta_{t,p}),\;
 \bar d_{t,p}(1+\theta_{t,p})]
$$

로 정의하고, 가중 Utility

$$
f(\theta)=\sum_{t,p}\beta_{t,p}\theta_{t,p},
\qquad \beta_{t,p}\ge0
$$

를 최대화합니다. 단, 추가 Qualification은 허용하지 않고 주어진 Qualification Matrix에서 모든 Robust Capacity Constraint를 만족해야 합니다.

이 문제는 Robust Constraint의 Dual Variable과 $\theta$가 곱해지는 Decision-Dependent Uncertainty 문제라 Bilinear Term을 포함합니다. 일반형은 계산이 어렵습니다.

### 6.2 Binary Search

저자들은 Period별 모든 Product에 같은 $\theta_t$를 적용하고 Feasibility를 반복 확인하는 Binary Search를 제안합니다.

1. 하한 $\theta_t^{\min}=0$, 상한 $\theta_t^{\max}=\theta_t^0$를 설정합니다.
2. 중간값 $\theta_t=(\theta_t^{\min}+\theta_t^{\max})/2$를 계산합니다.
3. 그 $\theta_t$에서 Robust Capacity LP가 Feasible한지 확인합니다.
4. Feasible이면 하한을 올리고, Infeasible이면 상한을 내립니다.
5. Interval과 상대오차가 $\varepsilon$ 이하가 될 때까지 반복합니다.

Qualification을 전부 미리 열어 둔 이상적 상황에서 이 절차를 돌리면 해당 Work Center가 구조적으로 감당할 수 있는 최대 Robustness $\theta^{\max}$도 추정할 수 있습니다.

---

## 7) 산업 데이터와 실험 설정

실험은 프랑스 Crolles의 반도체 공장 내 두 Critical Work Center 데이터로 수행했습니다. 원시 Demand와 제품·Operation별 세부값은 기밀이라 공개하지 않고 분포 요약값을 제공합니다.

### Work Center A

- Machine 20대
- Product 238개, Operation 1,208개
- 허용 최대가동률 $u^{\max}=0.95$
- Re-entrant Flow Factor: 14–72, 평균 41.2
- 가능한 신규 Qualification: 2,843개
- Qualification Lead Time: 0–2개월, 평균 1.6개월
- Throughput: 11.4–527.8 Wafers/hour, 평균 221.6

### Work Center B

- Machine 30대
- Product 238개, Operation 401개
- $u^{\max}$ 평균 0.80, 범위 0.63–0.87
- Re-entrant Flow Factor: 1–28, 평균 16.0
- 가능한 신규 Qualification: 1,266개
- Qualification Lead Time: 0–2개월, 평균 1.1개월
- Throughput: 6.8–83.3 Wafers/hour, 평균 48.0

계획기간은 **7개월($T=7$)**이며 첫 달에는 Demand Uncertainty를 두지 않았습니다. Product Family는 3개이고 각각 120, 64, 54개 Product를 포함합니다. 실제 Qualification Cost를 확보하지 못했으므로 모든 $cq_{r,m}=1$로 두어 Qualification 수를 최소화했습니다.

실험한 대칭 편차비율은 Work Center A에서 $\theta=0.1$부터 0.7까지입니다. Work Center B의 최대 Robustness는 약 $\theta^{\max}=0.294$이고, 실험 Grid의 0.3 이상은 Infeasible하므로 0.1과 0.2만 보고합니다. 각 $\theta$마다 Demand Scenario 3,600개를 생성해 Nominal Qualification의 Capa 위반과 Price of Uncertainty를 평가했습니다.

구현은 Java 8, IBM ILOG CPLEX 12.9를 사용했고, Intel Xeon W3530 2.80GHz, 8 Threads, RAM 12GB 환경에서 각 Solve에 3,600초 제한을 두었습니다. Binary Search Tolerance는 $\varepsilon=0.0001$입니다.

---

## 8) 실험 결과

### 8.1 Price of Uncertainty

논문은 Robust Qualification 수에서 실제 Demand를 미리 안다고 가정한 Perfect-Hindsight Qualification 수를 뺀 값을 PoU로 측정합니다.

Work Center A의 평균 PoU는 다음과 같이 증가했습니다.

- $\theta=0.1$: 평균 1.08개, 최대 2개
- $\theta=0.2$: 평균 3.10개, 최대 4개
- $\theta=0.3$: 평균 5.05개, 최대 7개
- $\theta=0.4$: 평균 7.96개, 최대 10개
- $\theta=0.5$: 평균 12.70개, 최대 15개
- $\theta=0.6$: 평균 18.44개, 최대 21개
- $\theta=0.7$: 평균 31.99개, 최대 35개

$\theta=0.7$의 최악에도 Machine당 평균 1.75개의 추가 Qualification이며, 가능한 신규 Pair 2,843개에 비하면 작습니다. Work Center B의 최대 PoU는 $\theta=0.1$에서 5개, $\theta=0.2$에서 19개였습니다.

### Figures 3–4 — Robustness 수준과 필요한 Qualification 수

<img width="1200" alt="두 work center에서 robustness theta에 따른 robust qualification 수와 perfect hindsight 평균 비교" src="/assets/img/paper-reviews/2026-08-20/perraudat-fig3-4.svg" />

> Source: Perraudat et al. (2022), Figures 3–4, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). 두 그림을 한 이미지로 크롭했습니다. [Author manuscript](https://hal.science/hal-03352370/document)

왼쪽 열은 목표 Robustness $	heta$를 사전에 보장하기 위해 모델이 여는 Qualification 수이고, 오른쪽 열은 실제 수요를 미리 아는 Perfect Hindsight의 평균입니다. 보호수준을 높일수록 사전 Qualification 비용이 커지는 Price of Uncertainty가 시각적으로 드러납니다.


모든 신규 Pair를 열었을 때의 최대 Robustness는 A에서 $\theta^{\max}=0.77$, B에서 0.294였습니다. 같은 Robustness를 달성하는 데 A는 96개, 즉 가능한 신규 Qualification의 3.37%만 필요했습니다. B는 135개로 10.6%였으나 3,600초 후 Optimality Gap이 25%였으므로 최적값으로 단정할 수 없습니다. Gap을 이용한 저자들의 하한 계산에서는 최소 102개, 즉 8.05% 이상이 필요합니다.

### 8.2 계산시간

Nominal MCQCP는 A에서 약 3초, B에서 약 1초였습니다. A의 Robust MCRQCP는 $\theta=0.1$에서 46초, $\theta=0.7$에서 1,551초가 걸렸습니다. B는 $\theta=0.1$에서 85초, $\theta=0.2$에서 3,472초가 걸렸습니다. Robust Counterpart가 Binary Variable을 추가하지 않더라도 더 Tight한 Capa와 다수의 Dual Variable·Constraint 때문에 계산이 크게 어려워진다는 결과입니다.

### 8.3 Nominal Qualification의 Capa 위반

각 $\theta$에서 3,600개 Scenario를 만들고 Nominal Qualification만 고정한 뒤 Total Overtime을 최소화했습니다. Overtime이 양수이면 적어도 하나의 Machine–Period Capacity Constraint가 위반된 것입니다.

Work Center A에서 Capa 위반 Scenario 비율은 다음과 같습니다.

- $\theta=0.1$: 0.72%
- $\theta=0.2$: 15.64%
- $\theta=0.3$: 26.19%
- $\theta=0.4$: 30.78%
- $\theta=0.5$: 38.39%
- $\theta=0.6$: 45.31%
- $\theta=0.7$: 57.83%

Work Center B에서는 $\theta=0.1$만 되어도 15.56%, $\theta=0.2$에서는 44.28%였습니다. A의 $\theta=0.4$ 최악 Scenario에서는 140개 Machine–Period Constraint 중 13개가 위반되고 최대 초과율이 0.066이었습니다. 여기에는 원문 내부의 수치 불일치가 있습니다. Table 5의 정의대로 `violation = U_{t,m}-u_{t,m}^{\max}`이고 A의 $u^{\max}=0.95$라면 $0.95+0.066=1.016$이지만, 본문은 실제가동률을 1.003으로 서술합니다. 공개된 식과 표만으로는 이 차이를 해소할 수 없으므로, 재현 시에는 Table의 정의에 따라 직접 계산하고 원문의 1.003을 그대로 검증값으로 사용하지 않는 편이 안전합니다.

현재 Nominal Qualification Set의 Binary Search Robustness는 A에서 $\theta=0.043$, B에서 0.024에 불과했습니다. 반면 A에서는 Nominal Qualification 4개 대신 Robust Qualification 7개만 선택해도 $\theta=0.2$를 방어했고, 9개면 $\theta=0.3$을 방어했습니다. B에서도 4개 대신 7개로 $\theta=0.1$을 방어했습니다.

### 8.4 “많이 Qualify하면 된다”는 전략의 실패

각 Operation이 최소 $\alpha$대 Machine에 Qualified되도록 강제하되, 기술적으로 Qualifiable한 Machine이 $\alpha$대보다 적으면 가능한 최대수만 여는 단순 $\alpha$-Flexibility Design도 비교했습니다. A에서 $\alpha=4$이면 611개의 Qualification을 추가하지만 Robustness는 0.071에 불과했고, $\alpha=5$에서 1,119개를 추가해도 0.152였습니다. 최적화된 Robust Qualification은 훨씬 적은 Pair로 목표 $\theta$를 직접 방어합니다.

이는 Qualification의 **수**보다 Network Structure와 Workload Pooling 위치가 중요함을 보여줍니다. Demand가 크게 흔들릴 Product와 높은 Re-entrant Load를 만드는 Operation을 실제 여유 Machine에 연결하지 못하면 Qualification을 많이 추가해도 Robustness가 낮습니다.

---

## 9) 강점·한계·해석 시 주의점

### 강점

- **반도체 Qualification의 실제 제약 반영:** 기존/신규/불가능 Pair, Lead Time, Machine별 Throughput와 가용시간을 구분합니다.
- **Product Cannibalization 모델링:** 독립 Demand가 아니라 Family 내 대체관계를 Budgeted Uncertainty로 표현합니다.
- **확률분포 불필요:** 신제품과 빠른 Mix 변화로 Distribution 추정이 어려운 환경에 적합합니다.
- **명시적 Worst-Case Guarantee:** 정한 Uncertainty Set 안의 모든 Demand에서 Capa Feasibility를 요구합니다.
- **산업 규모 검증:** 238개 Product, 최대 1,208개 Operation의 실제 Work Center 데이터로 계산합니다.
- **실무적 Insight:** 적은 수의 올바른 Qualification이 많은 무차별 Qualification보다 효과적임을 수치로 보입니다.

### 한계

- **Static Robust의 보수성:** Demand가 실현된 뒤 $WIP$ 배분을 바꾸는 Adjustable Decision을 허용하지 않습니다.
- **불확실성 범위 제한:** Demand만 불확실하며 Capacity, Throughput, Qualification Cost·Lead Time은 Deterministic입니다.
- **Qualification Cost 단순화:** 실제 Cost를 확보하지 못해 모두 1로 두었습니다. PoU가 낮게 나온 이유 중 하나일 수 있습니다.
- **상세 실행 미반영:** Operation Precedence, LOT Scheduling, Setup Sequence, WIP Peak를 직접 모델링하지 않습니다.
- **Scenario 생성의 평가 한계:** 실제 Demand Distribution을 모르므로 3,600개 Scenario는 확률적 발생빈도를 의미하지 않습니다.
- **일부 Robust Instance 미최적:** Work Center B의 최대 Robustness 사례는 3,600초 후에도 25% Gap이 남습니다.
- **비공개 원시 데이터:** 산업 Instance는 요청 시 제공 가능하다고 적혀 있지만 논문에 전체 Raw Data는 없습니다.
- **Work Center별 분리:** Factory-wide 연쇄병목과 Work Center 간 동적 상호작용은 직접 모델링하지 않습니다.

### 해석할 때 피해야 할 오해

`Robust`는 모든 가능한 현실을 방어한다는 뜻이 아닙니다. 사용자가 정의한 $\mathcal D_t$ 안의 수요만 방어합니다. $\hat d$와 $\Gamma$가 너무 작으면 실제 변동을 놓치고, 너무 크면 Problem이 Infeasible하거나 Qualification이 과도해질 수 있습니다. 따라서 Dual Variable의 Sensitivity를 이용해 어떤 Product Bound 또는 Family Budget이 Qualification 수를 크게 만드는지 확인하고 Uncertainty Set을 반복 조정하는 과정이 필요합니다.

또한 $WIP_{t,r,m}$은 이름과 달리 현장의 순간 Queue가 아닙니다. 한 달 단위 Operation Demand의 Machine별 배분비율입니다. 이 결과를 실시간 Dispatch Rule로 읽으면 안 됩니다.

---

## 10) 결론

본 논문은 Qualification을 단순한 품질 인증 목록이 아니라, Product Mix 변동에 대응하는 **Tactical Process-Flexibility Design**으로 정식화합니다. Deterministic MILP는 어떤 Operation–Machine Pair를 언제 열어야 수요를 처리할 수 있는지 계산하고, Robust Counterpart는 Product Family 내 Cannibalization으로 가능한 모든 Mix에서 Machine Capa를 지키도록 Qualification Set을 선택합니다.

산업 데이터 결과는 Robustness가 공짜는 아니지만 비교적 저렴할 수 있음을 보여줍니다. Work Center A에서 $\theta=0.2$를 방어하기 위해 Nominal 4개보다 단 3개 많은 7개의 Qualification이면 충분했지만, Nominal Set은 15.64%의 평가 Scenario에서 Capa 위반을 만들었습니다. 반대로 수백 개 Pair를 일률적으로 추가하는 $\alpha$-Flexibility는 훨씬 적은 Robustness만 얻었습니다.

가장 중요한 결론은 `Qualification 수를 늘리면 Capa가 좋아진다`가 아니라, **어떤 Demand Shift가 어떤 Re-entrant Operation의 부하를 키우는지 계산하고 그 부하를 실제로 분산할 수 있는 Machine에 Qualification을 여는 것이 중요하다는 것**입니다. 다만 Static Robustness, 동일 Cost, Demand-only Uncertainty라는 가정을 기억하고, 상세 Scheduling과 Cycle-Time Simulation의 전 단계인 Tactical Capacity Configuration Model로 해석해야 합니다.
