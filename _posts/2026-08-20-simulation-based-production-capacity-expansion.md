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

## 1) High-Level Summary

단일 Front-End(FE)와 Back-End(BE)로 구성된 반도체 supply chain에서 생산 투입계획과 Capa 확장을 함께 정하는 논문.

FE 병목을 세게 가동하면 throughput은 늘지만 queue와 cycle time도 커진다. BE Capa를 늘리면 downstream 혼잡을 줄일 수 있지만 증설비용이 생김. 논문은 생산계획 LP로 release schedule을 만들고, DES로 실제 KPI를 측정한 뒤, Simulated Annealing이 FE 병목 목표가동률 $M$과 BE 추가 Capa 비율 $a$를 탐색한다.

실험에서는 기준 계획보다 목적값이 18.35–37.17% 개선됐다. 실제 기업 log가 아니라 MIMAC I과 공개 BE model을 결합한 benchmark simulation이라는 점은 구분할 필요가 있다.

---

## 2) Problem

논문이 가정한 supply chain:

$$
\text{Wafer Fab/Probe (FE)}
\rightarrow \text{Die Bank (DB)}
\rightarrow \text{Assembly/Test (BE)}
\rightarrow \text{Distribution Center (DC)}.
$$

실제 network는 여러 FAB과 Assembly/Test site가 연결되지만 이 논문은 FE 1개, BE 1개로 축소한다.

선택해야 하는 값은 두 개.

- $M$: FE bottleneck Work Center의 minimum utilization target
- $a$: 기존 BE Capa 대비 추가 Capa 비율

$M$을 높이면 expensive FE equipment를 더 활용하고 output을 늘릴 수 있다. 하지만 utilization이 포화에 가까워질수록 WIP와 cycle time은 비선형적으로 증가.

$a$를 높이면 BE queue를 줄일 수 있지만 capacity cost가 발생한다. 결국 `FE를 얼마나 밀어붙일지`와 `BE Capa를 얼마나 보완할지`를 같이 봐야 한다.

단순한 utilization maximization이 아니라 revenue, WIP, inventory, backlog, cycle time violation, capacity cost 사이의 trade-off.

---

## 3) 전체 구조: LP–DES–SA

주어진 $(M,a)$에서 한 번의 평가가 다음처럼 진행된다.

$$
(M,a)
\rightarrow \text{Production Planning LP}
\rightarrow \text{Release Schedule}
\rightarrow \text{DES}
\rightarrow \text{KPI evaluation}.
$$

### Figure 1 — LP–Simulation–SA 폐루프

<img width="1100" alt="생산계획 LP와 이산사건 시뮬레이션 및 simulated annealing을 연결한 전체 구조" src="/assets/img/paper-reviews/2026-08-20/ziarnetzky-fig1.svg" />

> Source: Ziarnetzky and Mönch (2016), Figure 1. 논문 이해를 위한 일부 인용 및 크롭. © 2016 IEEE. [Original PDF](https://www.informs-sim.org/wsc16papers/263.pdf)

LP는 제품별·기간별 release, output, WIP, DB/DC inventory와 backlog를 정한다. 하지만 LP 하나에 breakdown, batch, setup, operator까지 다 넣기 어려우므로 release schedule을 DES에 넘겨 실제 performance를 다시 측정.

Simulation 결과가 좋지 않으면 바깥의 SA가 $(M,a)$를 바꾸고 같은 과정을 반복한다.

이 논문의 핵심은 특정 LP formulation 하나보다 이 closed loop에 있다. Optimization이 plan을 만들고 Simulation이 그 plan의 실행결과를 되돌려주는 구조.

---

## 4) Production Planning LP

내부 LP는 FE/BE WIP, DB/DC inventory, DC backlog cost를 최소화한다.

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

Output만 크게 만드는 model이 아니다. 너무 일찍 생산하면 inventory cost가 생기고, release를 과도하게 넣으면 WIP cost, demand를 놓치면 backlog cost가 커진다.

FE 공정별 flow balance는

$$
W^F_{gtl}=W^F_{g,t-1,l}+X^F_{gtl}-Y^F_{gtl}.
$$

이전 WIP와 이번 period start 중 처리되지 않은 양이 다음 WIP가 된다.

FE 마지막 output은 Die Bank로 넘어가고 BE release가 이를 소진한다.

$$
I^{DB}_{gt}=I^{DB}_{g,t-1}+\Lambda_gY^F_{gt}-X^B_{gt}.
$$

$\Lambda_g$는 FE lot과 BE lot size의 변환계수. 실험은 FE 48 Wafer, BE 16 Wafer라 $\Lambda_g=3$을 사용한다.

### Allocated Clearing Function

고정 lead time model에서는 release를 늘려도 처리속도가 일정하다고 보기 쉽다. 실제 FAB에서는 load가 높아질수록 queue가 커져 같은 양을 더 오래 잡고 있게 된다.

논문은 이 관계를 piecewise linear clearing function으로 넣는다.

$$
\alpha_{gl}Y^F_{gtl}
\le
\mu_n^k Z^k_{gtl}
+\beta_n^k\alpha_{gl}
\left(X^F_{gtl}+W^F_{g,t-1,l}\right),
\qquad n\in C(k).
$$

- $X^F+W^F$: 이번 period에 처리 대상으로 쌓인 workload
- $Y^F$: 실제로 clear할 output
- $Z^k_{gtl}$: Work Center $k$의 output capacity 중 제품–공정에 준 비율
- $\mu_n^k,\beta_n^k$: simulation data로 fitting한 piecewise line

$$
\sum_{g\in G}\sum_{l\in O(k)}Z^k_{gtl}=1.
$$

WIP가 적을 때는 투입 증가가 output 증가로 이어지지만 congestion 구간에서는 증가폭이 둔화된다. Queue를 직접 계산하는 대신 `현재 workload에서 period 안에 얼마나 clear할 수 있는가`를 LP constraint로 근사한 것.

### FE utilization과 BE expansion

FE 병목 $b$가 최소한 $M$만큼 사용되도록

$$
\sum_{g\in G}\sum_{l\in O(b)}
\alpha_{gl}Y^F_{gtl}
\ge M\widetilde C_b
$$

를 둔다. $M$이 높아질수록 LP는 bottleneck output을 더 만들도록 release를 조정한다.

BE Work Center capacity는

$$
\sum_{g\in G}\sum_{l\in O(k)}
\alpha_{gl}Y^B_{gtl}
\le C_k(a),
\qquad
C_k(a)=(1+a)\widehat C_k.
$$

$a$는 장비별 구매 integer가 아니라 모든 BE Work Center Capa를 늘리는 aggregate ratio다.

BE completion은 simulation에서 추정한 flow factor와 fixed lead time으로 release에 연결한다.

$$
L(g,l)=L(g,l-1)+FF_g\alpha_{gl},
\qquad L(g,0)=0.
$$

$$
Y^B_{gtl}=X^B_{g,t-\lfloor L(g,l)\rfloor}.
$$

FE는 clearing function으로 congestion을 planning model 안에 넣지만 BE는 fixed lead time approximation을 사용한다는 차이가 있음.

---

## 5) Simulation에서 계산하는 최종 objective

LP objective와 최종 평가값은 다르다. LP가 만든 schedule을 simulation에 넣은 뒤 실현된 KPI로 다시 계산한다.

원문의 의미를 maximum score로 정리하면

$$
\max_{M,a}\;F(M,a)
=
\sum_{g,t}r_{gt}\widetilde Y^B_{gt}
-C(\widetilde W^F,\widetilde W^B,
\widetilde B^{DC},\widetilde I^{DB},\widetilde I^{DC})
-\frac{\beta}{T}\sum_{g,t}(\widetilde C_{gt}-C^*_{gt})^+
-\frac{\delta}{T}\sum_t(M\widetilde C_b-\widetilde U_{bt})^+
-\lambda a.
$$

$(x)^+=\max(0,x)$.

- 첫 항: simulation에서 실제 완료된 output의 revenue
- 두 번째: 실현된 WIP, inventory, backlog cost
- 세 번째: cycle time limit 초과 penalty
- 네 번째: FE utilization target 미달 penalty
- 마지막: BE capacity expansion cost

원문 본문에는 한 번 `minimized`라고 쓰지만 식이 profit-minus-penalty이고 뒤의 grid search도 maximum $f$를 찾는다. 실제 해석은 maximization이 맞다.

$M$을 높여도 breakdown 때문에 목표 utilization을 못 채울 수 있고, $a$를 높여 cycle time을 줄여도 expansion cost가 더 커질 수 있다. 이 실현값을 봐야 두 parameter의 적정점을 찾을 수 있음.

---

## 6) Simulated Annealing

탐색 grid:

$$
M\in\{0.50,0.55,\ldots,0.95\},
\qquad
a\in\{0,0.05,\ldots,0.50\}.
$$

총 110개 조합. 모든 point마다 LP와 반복 simulation을 돌리면 계산량이 크기 때문에 SA가 인접 point를 탐색한다.

현재와 후보 score 차이를 $\Delta=F(M,a)-F(M',a')$라 두면

$$
P(\text{accept})=
\begin{cases}
1, & \Delta\le0,\\
\exp(-\Delta/\mathcal T), & \Delta>0.
\end{cases}
$$

개선된 후보는 항상 받고, 나쁜 후보도 temperature $\mathcal T$가 높을 때 일정 확률로 받는다. 탐색 후반으로 갈수록 temperature를 낮춰 local search처럼 수렴.

한 move의 계산도 가볍지 않다.

1. $(M,a)$ 선택
2. LP solve, release schedule 생성
3. 같은 schedule을 DES에서 여러 번 실행
4. cycle time, utilization, output으로 $F(M,a)$ 계산
5. SA rule로 이동 여부 결정

Temperature마다 16회 iteration, 연속 5개 temperature에서 accepted move가 없으면 종료한다.

---

## 7) Experiment and Result

Simulation model:

- FE: MIMAC I, 200대 이상 장비, 69 Work Centers
- FE bottleneck: Stepper Work Center
- BE: 23 Work Centers
- Product 2개
- Product 1: FE 211 steps, BE 25 steps
- Product 2: FE 246 steps, BE 31 steps
- Batch, sequence-dependent setup, exponential breakdown, operator, secondary resource 반영
- AutoSched AP + 일부 C++ customization

계획기간 15주 중 처음 12주를 평가. Product Mix는 1:1. FE 평균 bottleneck utilization은 70%와 90%, Demand CV는 0.10과 0.25를 조합한다. 조건별 5개 demand scenario, grid point마다 20 simulation replications. LP 한 번은 평균 약 3분.

Full Grid Search는 44,400회 simulation, SA는 7,920회 사용. 평균적으로 약 19.8개 grid point를 본 셈이고 simulation burden을 약 82% 줄였다.

### Main Result

<img width="1100" alt="평균 utilization과 demand CV별 utilization shortfall, cycle time violation, 개선율 결과표" src="/assets/img/paper-reviews/2026-08-20/ziarnetzky-table2.svg" />

> Source: Ziarnetzky and Mönch (2016), Table 2. 논문 이해를 위한 일부 인용 및 크롭. © 2016 IEEE. [Original PDF](https://www.informs-sim.org/wsc16papers/263.pdf)

| 평균 병목가동률 | Demand CV | 목적값 개선 | Utilization shortfall | Cycle time 초과 |
|---|---:|---:|---:|---:|
| Low | 0.10 | 22.22% | 0.68% | 383.10분 |
| Low | 0.25 | 18.35% | 0.99% | 310.77분 |
| High | 0.10 | 37.17% | 0.19% | 284.00분 |
| High | 0.25 | 28.19% | 0.50% | 159.78분 |

기준 $(M,a)=(0,0)$ 대비 improvement는 18.35–37.17%. 다만 stochastic simulation이라 선택된 plan도 cycle time limit과 utilization target을 조금씩 위반한다.

SA/Grid Search score ratio는 한 outlier를 제외하면 대부분 0.98–1.00. Low utilization, CV 0.25의 한 scenario에서는 0.76, High utilization 한 scenario에서는 0.93이므로 SA가 항상 global best를 찾은 것은 아니다.

Low utilization에서는 기존보다 $M$을 약 10%p 올리고 BE를 조금 늘리는 조합이 자주 선택됐다. High utilization에서는 오히려 $M$을 낮춰 FE load를 smoothing하고 BE Capa를 늘리는 해도 나왔다.

병목장비를 무조건 100%에 가깝게 쓰는 것이 좋은 것이 아니라 downstream Capa와 cycle time penalty를 같이 봐야 한다는 결과.

---

## 8) 읽으면서 구분한 범위

이 연구는 실제 FAB 운영 data를 검증한 case study가 아니다. MIMAC I과 공개 BE benchmark를 합친 simulation study. Industry expert validation은 받았지만 결과 개선률은 설정한 revenue와 penalty coefficient에도 영향을 받는다.

$a$는 tool별 투자대수가 아니라 공통 expansion ratio다. 개별 장비 purchase, transfer, qualification lead time, staffing은 없음. FE–DB와 BE–DC transport는 infinite capacity이고 공정 간 transfer도 즉시라 FAB 간 물류나 OHT bottleneck을 다루지 않는다.

그래도 production planning과 simulation을 따로 쓰지 않고 feedback loop로 만든 점이 좋다. LP에서 다 넣기 어려운 breakdown과 setup은 DES로 넘기고, DES만으로 찾기 어려운 release plan은 LP가 만든다.

---

## 9) Conclusion

이 논문의 전체 흐름은

$$
(M,a)
\rightarrow LP
\rightarrow Release\ Schedule
\rightarrow DES
\rightarrow F(M,a)
\rightarrow SA
$$

로 정리된다.

FE bottleneck utilization과 BE expansion을 먼저 고르고, planning LP와 simulation을 모두 거친 실현 KPI로 두 값을 다시 평가한다. `병목을 더 가동하면 output이 늘어난다`는 한 방향만 보는 대신 WIP, cycle time, backlog, expansion cost까지 포함해 Capa decision을 내리는 방법.
