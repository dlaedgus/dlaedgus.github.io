---
title: "Robust Tactical Qualification Decisions in Flexible Manufacturing Systems (Omega 2022)"
date: 2026-08-20 12:10:00 +0900
categories: [paper_review, OR]
tags: [semiconductor-manufacturing, robust-optimization, qualification, capacity-planning]
math: true
---

# Paper Review — *Robust Tactical Qualification Decisions in Flexible Manufacturing Systems* (Omega 2022)

- **제목:** Robust Tactical Qualification Decisions in Flexible Manufacturing Systems  
- **1저자:** Antoine Perraudat  
- **공저자:** Stéphane Dauzère-Pérès, Philippe Vialletelle  
- **저널:** Omega, Vol. 106, Article 102537  
- **년도:** 2022  
- **DOI:** [10.1016/j.omega.2021.102537](https://doi.org/10.1016/j.omega.2021.102537)  
- **원문:** [HAL Open Access PDF](https://hal.science/hal-03352370/document)

---

## 1) High-Level Summary

High-Mix 반도체 FAB에서 어떤 `(Operation, Machine)` pair를 새로 Qualification할지, 또 언제 시작할지를 정하는 tactical capacity planning 논문.

Qualification에는 lead time과 비용이 들기 때문에 demand가 확정된 뒤 시작하면 늦을 수 있다. 논문은 deterministic MILP를 먼저 만들고, 같은 Product Family 안에서 demand mix가 바뀌는 상황을 uncertainty set으로 표현한다. 이후 worst-case capacity constraint를 dualization해 하나의 robust MILP로 바꾼다.

Crolles 반도체 공장의 두 Critical Work Center를 사용한 결과, 무작정 많은 pair를 여는 것보다 demand shift 때 실제 병목 load를 분산할 수 있는 pair를 고르는 것이 훨씬 중요했다.

---

## 2) Qualification과 FAB Capa

장비가 설치되어 있다고 모든 Recipe를 바로 처리할 수 있는 것은 아니다. Pressure, Temperature, Chemical Condition 등이 정해진 Recipe를 그 장비에서 실행해도 Quality와 Yield가 유지된다는 검증이 끝나야 실제 생산에 사용 가능.

이 논문에서 Qualification의 단위는 `(Operation, Machine)` pair다.

- 이미 Qualified: 바로 배정 가능
- Qualifiable: 기술적으로 가능하지만 절차와 lead time 필요
- Not qualifiable: Hardware 또는 Software 차이로 불가능

반도체 Front-End는 re-entrant flow라 같은 Work Center를 여러 번 다시 방문한다. Product마다 방문 횟수와 route가 다르기 때문에 family 전체 수요가 같아도 product mix가 달라지면 장비 load는 달라질 수 있다.

계획 horizon은 6–12개월 수준의 tactical planning. 신규장비 도입, 신제품 ramp-up, 기존제품 demand 변화가 일어나는 동안 어떤 Qualification을 미리 열어야 하는지를 결정한다.

여기서 보는 것은 LOT scheduling이 아니라 `예상 operation load가 qualified machine capacity 안에 배분 가능한가`이다.

---

## 3) Deterministic MILP: MCQCP

논문에서는 Minimum Cost Qualification Configuration Problem, 줄여서 MCQCP라고 부른다.

주요 parameter:

- $q_{r,m}$: Qualification 상태. 1은 기존 Qualified, 2는 신규 Qualifiable, 0은 불가능
- $tp_{r,m}$: Machine $m$에서 Operation $r$의 throughput
- $c_{t,m}$: 기간 $t$의 production availability
- $u_{t,m}^{\max}$: 최대 utilization
- $rf_{p,r}$: Product $p$가 Operation $r$을 수행하는 횟수
- $d_{t,p}$: Product demand
- $l_{t,r,m}$: Qualification lead time
- $cq_{r,m}$: Qualification cost

결정변수는 두 개.

$$
OQ_{t,r,m}\in\{0,1\}
$$

기간 $t$에 Operation $r$–Machine $m$ Qualification을 시작하면 1.

$$
WIP_{t,r,m}\in[0,1]
$$

Operation $r$의 load 중 Machine $m$에 배분하는 비율. 이름은 WIP지만 실제 LOT 수나 queue length가 아니다.

목적함수는

$$
\min \sum_{t,r,m}\delta_t cq_{r,m}OQ_{t,r,m}.
$$

필요한 Qualification의 discounted cost 최소화. 논문 실험에서는 실제 cost data가 없어 모두 1로 두었으므로 Qualification 수를 최소화한 셈이다.

### Capacity와 load allocation

기간 $t$, Operation $r$의 총 load는

$$
L_{t,r}=\sum_p rf_{p,r}d_{t,p}.
$$

이를 Machine별 비율로 나누고 throughput으로 시간을 계산하면

$$
\sum_r
\frac{\left(\sum_p rf_{p,r}d_{t,p}\right)WIP_{t,r,m}}
{tp_{r,m}}
\le c_{t,m}u_{t,m}^{\max}
\qquad \forall t,m.
$$

왼쪽은 Machine $m$에 배정된 총 처리시간, 오른쪽은 실제 사용을 허용한 시간. $u^{\max}<1$로 두는 이유는 utilization이 1에 가까워질수록 queue와 cycle time이 급격히 증가하기 때문.

모든 operation load는 빠짐없이 배분해야 한다.

$$
\sum_m WIP_{t,r,m}=1
\qquad \forall t,r:\sum_p rf_{p,r}d_{t,p}>0.
$$

Qualification 상태와 lead time은 다음 제약으로 연결된다.

$$
WIP_{t,r,m}\le q_{r,m}
\qquad \forall t,r,m:q_{r,m}\ne2
$$

$q=1$이면 사용 가능, $q=0$이면 배정 불가.

아직 Qualification이 필요한 pair는

$$
WIP_{t,r,m}
\le
\sum_{t'=1:\;t-t'\ge l_{t',r,m}}^{t}OQ_{t',r,m}
\qquad \forall t,r,m:q_{r,m}=2.
$$

이전에 시작한 Qualification이 lead time을 지나 완료된 경우에만 production load를 줄 수 있다.

$OQ$는 binary, $WIP$는 continuous라 전체는 MILP. 저자들은 Generalized Assignment Problem의 reduction을 통해 single-period도 NP-hard임을 보인다. Work Center끼리 operation을 공유하지 않는 설정에서는 Work Center별로 분리해서 풀 수 있음.

---

## 4) Demand uncertainty를 어떻게 만드나

신제품은 historical data가 적고 같은 market을 겨냥한 제품은 서로 수요를 뺏는다. Product A demand가 늘 때 같은 family의 B가 줄 수 있으므로 각 demand를 독립 random variable로 두는 것도 자연스럽지 않다.

논문은 Product Family별 budget을 둔다.

- $\bar d_{t,p}$: nominal demand
- $\hat d_{t,p}$: 제품별 최대 편차
- $\alpha_{p,f}$: Product $p$가 Family $f$에 속하면 1
- $\Gamma_{t,f}$: Family 전체 demand budget

각 제품은

$$
d_{t,p}\in
[\bar d_{t,p}-\hat d_{t,p},\;\bar d_{t,p}+\hat d_{t,p}]
$$

안에서 움직이고, family 합은

$$
\sum_{p:\alpha_{p,f}=1}d_{t,p}\le\Gamma_{t,f}
$$

를 만족한다. 전체 uncertainty set은

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

로 설정. Family 전체 수요의 상한은 nominal total로 고정하면서 family 안의 product mix가 움직이는 구조다.

### Figure 2 — 한 제품의 12개월 Demand Profile

<img width="900" alt="평균수요 대비 월별 수요가 ramp-up과 ramp-down하는 제품 demand profile" src="/assets/img/paper-reviews/2026-08-20/perraudat-fig2.svg" />

> Source: Perraudat et al. (2022), Figure 2, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). 원문 그림을 크롭했습니다. [Author manuscript](https://hal.science/hal-03352370/document)

월별 demand가 고정되지 않고 ramp-up과 ramp-down을 반복한다. 이런 상황에서 nominal 한 점만 기준으로 Qualification을 정하면 mix가 조금만 바뀌어도 특정 machine Capa가 먼저 깨질 수 있다.

확률분포나 scenario probability는 필요 없다. 대신 $\mathcal D_t$ 안의 모든 demand를 방어한다. Robustness 범위는 결국 $\hat d$와 $\Gamma$를 어떻게 잡느냐에 달려 있음.

---

## 5) Worst-case capacity와 Dualization

Robust model은 모든 $d_t\in\mathcal D_t$에 대해 Machine Capa가 유지되어야 한다.

Period $t$, Machine $m$의 constraint는

$$
\max_{d_t\in\mathcal D_t}
\sum_p d_{t,p}
\left(
\sum_r\frac{rf_{p,r}WIP_{t,r,m}}{tp_{r,m}}
\right)
\le c_{t,m}u_{t,m}^{\max}.
$$

괄호 안은 Product $p$ 한 단위가 Machine $m$에 만드는 시간 load. Re-entrant 횟수가 많거나 처리속도가 느린 product demand가 커지는 조합이 worst case가 된다.

그런데 이 상태로는 가능한 demand가 무한히 많다. 내부 maximization이 LP이므로 strong duality를 사용해 하나의 finite linear system으로 바꾼다.

lower bound, upper bound, family budget에 대한 nonnegative dual variable을 각각

$$
y_{t,m,p}^{\min},\qquad
y_{t,m,p}^{\max},\qquad
y_{t,m,f}^{\gamma}
$$

로 두면 worst-case capacity는

$$
\sum_p-(\bar d_{t,p}-\hat d_{t,p})y_{t,m,p}^{\min}
+\sum_f\Gamma_{t,f}y_{t,m,f}^{\gamma}
+\sum_p(\bar d_{t,p}+\hat d_{t,p})y_{t,m,p}^{\max}
\le c_{t,m}u_{t,m}^{\max}
$$

으로 바뀐다. 각 Product에는

$$
-y_{t,m,p}^{\min}
+y_{t,m,p}^{\max}
+\sum_{f:\alpha_{p,f}=1}y_{t,m,f}^{\gamma}
\ge
\sum_r\frac{rf_{p,r}WIP_{t,r,m}}{tp_{r,m}}
$$

을 둔다.

정리하면

`모든 demand scenario에 대해 capacity 만족`

이라는 semi-infinite한 조건을 dual variable이 들어간 유한 개 linear constraint로 바꾼 것. 이 부분이 논문의 수리적 핵심이다.

단, $WIP_{t,r,m}$도 demand가 실현되기 전에 고정되는 **Static Robust** model이다. 실제 demand를 본 뒤 machine allocation을 다시 바꾸는 Adjustable Robust보다 보수적일 수 있다.

238 Products, 1,208 Operations, 20 Machines, 7 Periods인 예에서 deterministic model은 continuous와 binary variable이 각각 169,120개, constraint 685,076개. Robust model은 continuous 202,860개, binary 169,120개, constraint 785,456개다. Variable 수 증가보다 tight한 capacity constraint 때문에 solve가 더 어려워진다.

---

## 6) 현재 Qualification Set의 Robustness

$\hat d$를 먼저 정하지 않고 현재 set이 몇 % 변동까지 버티는지 역으로 계산하는 방법도 제시한다.

$$
d_{t,p}\in
[\bar d_{t,p}(1-\theta_{t,p}),
 \bar d_{t,p}(1+\theta_{t,p})]
$$

$\theta$가 클수록 더 넓은 demand mix를 방어한다. 일반형은 dual variable과 $\theta$의 bilinear term이 생기므로 어렵다.

일반 목적은 제품·기간별 중요도 $\beta_{t,p}$를 둔 다음 utility를 최대화하는 것.

$$
f(\theta)=\sum_{t,p}\beta_{t,p}\theta_{t,p},
\qquad \beta_{t,p}\ge0
$$

논문에서는 같은 period의 모든 product에 동일한 $\theta_t$를 적용하고 binary search를 사용한다.

1. $\theta$ 중간값 설정
2. Robust Capacity LP feasibility 확인
3. feasible이면 범위를 위로, infeasible이면 아래로 이동
4. tolerance 이하까지 반복

모든 가능한 Qualification을 열어 둔 상태에서 돌리면 Work Center 자체가 감당할 수 있는 최대 $\theta^{\max}$도 구할 수 있다.

---

## 7) Industrial data와 Result

프랑스 Crolles의 두 Critical Work Center data를 사용. raw demand와 operation data는 기밀이라 summary만 공개한다.

| 구분 | Work Center A | Work Center B |
|---|---:|---:|
| Machine | 20 | 30 |
| Product | 238 | 238 |
| Operation | 1,208 | 401 |
| 신규 Qualifiable Pair | 2,843 | 1,266 |
| 평균 Lead Time | 1.6개월 | 1.1개월 |
| 최대가동률 | 0.95 | 평균 0.80 |

계획기간은 7개월, Product Family는 3개. $cq_{r,m}=1$로 두고 CPLEX 12.9에서 3,600초 time limit을 사용했다. 각 $\theta$마다 3,600개 demand scenario로 nominal set의 violation을 평가.

### Figures 3–4 — Robustness 수준과 필요한 Qualification 수

<img width="1200" alt="두 work center에서 robustness theta에 따른 robust qualification 수와 perfect hindsight 평균 비교" src="/assets/img/paper-reviews/2026-08-20/perraudat-fig3-4.svg" />

> Source: Perraudat et al. (2022), Figures 3–4, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). 두 그림을 한 이미지로 크롭했습니다. [Author manuscript](https://hal.science/hal-03352370/document)

Robustness가 커질수록 필요한 Qualification 수가 늘어난다. 다만 실제 demand를 미리 아는 Perfect Hindsight 대비 추가분, 즉 Price of Uncertainty는 생각보다 크지 않았다.

Work Center A의 평균 PoU:

- $\theta=0.1$: 1.08개
- $\theta=0.3$: 5.05개
- $\theta=0.5$: 12.70개
- $\theta=0.7$: 31.99개

$\theta=0.7$에서도 가능한 신규 pair 2,843개 중 일부만 추가한다. 모든 pair를 열었을 때 최대 robustness는 A에서 0.77, B에서 0.294. A는 96개, 전체 후보의 3.37%만으로 최대 수준을 달성했다. B의 135개 결과는 time limit 후 gap 25%가 남아 최적값으로 볼 수는 없다.

Nominal Qualification만 사용했을 때 capacity violation scenario 비율은 빠르게 증가했다.

- Work Center A: 0.72%($\theta=0.1$) → 57.83%($\theta=0.7$)
- Work Center B: 15.56%($\theta=0.1$) → 44.28%($\theta=0.2$)

반면 A는 nominal 4개 대신 7개를 고르면 $\theta=0.2$, 9개면 $\theta=0.3$을 방어했다. B도 4개 대신 7개로 $\theta=0.1$ 방어 가능.

### Qualification 수만 늘리는 방법과 비교

각 Operation이 최소 $\alpha$대 Machine에서 처리 가능하도록 pair를 여는 단순 flexibility rule도 비교한다.

A에서 $\alpha=4$는 611개 Qualification을 추가하지만 robustness는 0.071. $\alpha=5$는 1,119개를 추가해도 0.152였다. Robust optimization은 훨씬 적은 pair로 목표 $\theta$를 방어한다.

Qualification의 개수보다 위치가 중요하다는 결과. demand가 흔들릴 product와 load가 큰 re-entrant operation을 실제 여유 machine에 연결해야 한다.

---

## 8) 읽으면서 주의한 부분

`Robust`는 현실의 모든 demand를 방어한다는 뜻이 아니다. 직접 정한 $\mathcal D_t$ 안에서만 guarantee가 있다. 범위를 작게 잡으면 실제 변동을 놓치고, 크게 잡으면 model이 infeasible하거나 Qualification이 지나치게 늘어날 수 있음.

또한 Static Robust라 demand가 나온 뒤 $WIP$ allocation을 조절하지 않는다. Demand 이외의 throughput, availability, cost, lead time도 deterministic. 상세 operation precedence, setup sequence, LOT queue는 포함하지 않는다.

논문 안의 숫자에도 한 가지 불일치가 있다. Table 5 정의대로 A의 violation 0.066을 $u^{\max}=0.95$에 더하면 1.016인데 본문은 actual utilization을 1.003이라고 쓴다. 재현할 때는 Table의 식으로 직접 계산하는 편이 낫다.

Raw industrial data는 논문에 포함되지 않는다. 공개 data로 구현한다면 qualification matrix와 route factor는 benchmark에서 만들고, demand family와 cost는 별도로 구성해야 함.

---

## 9) Conclusion

이 논문에서 가장 인상적인 부분은 uncertainty set 자체보다 그다음 dualization이다.

$$
\max_{d_t\in\mathcal D_t}
\text{Machine Load}(d_t,WIP)
\le \text{Available Capacity}
$$

를 dual constraint로 바꾸면서 deterministic qualification MILP를 robust MILP로 확장한다.

결과도 단순하다. Qualification을 많이 만들면 flexibility가 좋아지는 것이 아니라, product mix가 바뀔 때 커지는 load를 실제 여유 machine으로 넘길 수 있는 pair를 열어야 한다. Qualification을 품질 인증 list가 아니라 demand uncertainty에 대응하기 위한 process-flexibility design 문제로 보는 논문.
