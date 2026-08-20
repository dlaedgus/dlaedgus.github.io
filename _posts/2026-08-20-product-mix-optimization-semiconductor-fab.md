---
title: "Product Mix Optimization for a Semiconductor Fab: Modeling Approaches and Decomposition Techniques (WSC 2012)"
date: 2026-08-20 12:00:00 +0900
categories: [paper_review, OR]
tags: [semiconductor-manufacturing, product-mix, linear-programming, decomposition]
math: true
---

# Paper Review — *Product Mix Optimization for a Semiconductor Fab: Modeling Approaches and Decomposition Techniques* (WSC 2012)

- **제목:** Product Mix Optimization for a Semiconductor Fab: Modeling Approaches and Decomposition Techniques  
- **1저자:** Andreas Klemmt  
- **공저자:** Martin Romauch, Walter Laure  
- **학회:** Winter Simulation Conference (WSC)  
- **년도:** 2012  
- **원문:** [WSC Proceedings PDF](https://www.informs-sim.org/wsc12papers/includes/files/inv213.pdf)  
- **DOI:** [10.1109/WSC.2012.6465270](https://doi.org/10.1109/WSC.2012.6465270)

동일 제목의 2015년 *Computers & Operations Research* 논문도 있는데 저자와 세부 모형이 다르다. 여기서는 2012년 WSC 논문만 정리.

---

## 1) High-Level Summary

반도체 FAB에서 어떤 제품을 얼마나 생산할지, 그리고 그 물량이 만드는 load를 어떤 장비에 나눌지를 같이 푸는 논문.

고수익 제품을 많이 생산하는 것이 목적이지만 제품마다 route와 장비 소요 pattern이 다르다. 특정 제품을 늘렸을 때 노광이나 식각 계열 병목 Capa가 먼저 차면 그 제품은 더 이상 늘릴 수 없음. 그래서 `Product Mix 결정`과 `Static Capacity Allocation`을 하나의 LP로 연결한다.

전체 LP를 직접 푸는 방법 외에

`Product Mix를 정하는 Master Problem → 장비 Load Balancing → 제품별 장비 소요량 갱신`

을 반복하는 decomposition heuristic도 제안한다. 1,215개 생성 instance와 변형된 실제 FAB data에서 기준 Mix를 제품별 ±5%만 바꿔도 대략 3% 수준의 이익 개선 가능성을 확인.

단, 이 논문은 LOT별 시간계획이나 dispatching이 아니다. 일정 기간의 평균 물량을 기준으로 제품 Mix와 장비 load를 맞추는 tactical/static model에 가깝다.

---

## 2) Product Mix와 Capa를 같이 봐야 하는 이유

제품별 생산량만 정한다고 실제 생산계획이 만들어지는 것은 아니다.

- 제품마다 Route가 다름
- 하나의 Route는 많은 Operation으로 구성
- Operation마다 처리 가능한 Equipment가 제한됨
- 동일 제품이라도 어느 장비로 배분하느냐에 따라 load profile이 달라짐

논문은 이 과정을 다음처럼 줄인다.

1. 제품 $p$의 생산량 결정
2. Route를 따라 Job Class $j$별 필요 작업량 계산
3. 각 Job Class를 처리 가능한 Resource $i$에 배분
4. Resource별 처리시간이 Capa 이내인지 확인

### Figure 1 — Product Mix가 장비 Utilization으로 변환되는 구조

<img width="1200" alt="Product mix와 route, job class, equipment assignment, utilization profile의 연결 구조" src="/assets/img/paper-reviews/2026-08-20/klemmt-fig1.svg" />

> Source: Klemmt, Romauch, and Laure (2012), Figure 1. 논문 이해를 위한 일부 인용 및 크롭. © 2012 IEEE. [Original PDF](https://www.informs-sim.org/wsc12papers/includes/files/inv213.pdf)

그림에서 Product Route가 Job Class별 load로 바뀌고, 가능한 Equipment에 배정된 뒤 최종 utilization profile이 만들어진다. 논문의 수식 전체가 이 흐름을 matrix로 옮긴 것에 가깝다.

여기서 **Job Class**는 장비 소요 pattern이 같거나 유사한 Operation을 묶은 단위. 모든 Operation을 하나씩 쓰지 않고 Capa 관점에서 같은 작업을 묶어 model size를 줄인다.

**Activity** $k$는 `(Job Class, Resource)`의 가능한 조합이다. Capability 또는 Dedication이 없는 조합은 Activity에 들어가지 않는다.

결정은 크게 두 층으로 볼 수 있다.

- Product level: 수요 범위 안에서 제품별 생산량을 정해 이익 최대화
- Resource level: 선택한 Mix가 만든 작업을 Qualified Resource에 나눠 Capa-feasible한 load 생성

제품 Mix가 달라지면 공정별 부하가 달라지고, 같은 Mix도 장비 대체 가능성에 따라 Capa 소비가 달라진다. 둘을 따로 풀기 어려운 이유.

---

## 3) 먼저 주어진 물량을 장비에 나누는 문제

Product Mix가 이미 정해졌다고 두자.

- $\lambda_j$: Job Class $j$의 총 필요량
- $\mu_{ji}$: Resource $i$가 Job Class $j$를 처리하는 service rate
- $x_{ji}$: Job Class $j$ 중 Resource $i$에 배정한 양

최대 장비부하를 최소화하는 LP는

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

$x_{ji}/\mu_{ji}$는 해당 배정이 Resource $i$에서 차지하는 처리시간. 모든 Job Class 물량을 빠짐없이 배분하면서 장비별 처리시간을 공통 상한 $\rho$ 아래에 두고, 그 $\rho$를 최소화한다.

논문의 예에서는 초기 load $(170,150,140,160)$시간이 Min-Max 이후 $(100,150,160,160)$으로 바뀌어 maximum load가 170에서 160으로 감소한다. 최대값만 같고 나머지 분포가 다른 해를 구분하기 위해 Resource Pooling으로 높은 load level부터 차례로 balance하며, 최종 profile은 $(145,138,145,160)$이 된다.

### Connected Component / Resource Pool

두 Resource가 같은 Job Class를 처리할 수 있으면 서로 연결되어 있다고 본다. 이 연결이 이어진 집합이 **Connected Component**, 논문 표현으로 Closed Machine Set(CMS).

서로 다른 CMS는 Job Class를 공유하지 않아 Mix가 고정된 Load Balancing은 CMS별로 나눠 풀 수 있다. 제품 하나는 여러 CMS를 지나므로 Product Mix 전체는 쉽게 분리되지 않는다는 차이가 있음.

---

## 4) Product Mix + Equipment Allocation 통합 LP

### 변수와 parameter

- $P,J,I$: Product, Job Class, Resource 집합
- $K$: 가능한 Job Class–Resource Activity 집합
- $y_p$: Product $p$의 생산량
- $x_k$: Activity $k$에 배정한 작업량
- $D=(d_{jp})$: 제품 한 단위가 Job Class $j$를 몇 번 요구하는지 나타내는 Technology Matrix
- $A=(a_{jk})$: Activity $k$가 어떤 Job Class 처리인지 나타내는 Incidence Matrix
- $R=(r_{ik})$: Activity $k$가 Resource $i$에서 소비하는 시간
- $c_p$: 제품 한 단위의 이익
- $\Delta_p^-,\Delta_p^+$: 최소 의무물량과 최대 수요
- $\rho_i^{\max}$: Resource $i$의 Capa

### Global LP

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

식은 짧지만 각 matrix가 하는 일이 분명하다.

- $Dy$: 현재 Product Mix가 만들어 낸 Job Class별 총 작업량
- $Ax$: 가능한 Activity를 통해 실제 장비에 배정한 작업량
- $Ax=Dy$: 필요한 공정 load를 하나도 빼지 않고 장비에 배정
- $Rx$: 배정 결과를 Resource별 처리시간으로 변환
- $Rx\le\rho^{\max}$: 장비 Capa 이내
- $c^\top y$: 그 안에서 총이익 최대화

가령 Product A의 margin이 가장 높더라도 A가 특정 Job Class를 많이 반복하고 Qualified Equipment도 적다면 Capacity Constraint에서 막힌다. 이 경우 A보다 이익은 조금 낮아도 여유 장비를 사용하는 Product B를 늘리는 편이 전체 FAB 이익에는 더 좋을 수 있다.

### 추가 Capa 표현

기본 Capa 외에 Bottleneck Management로 잠시 확보하는 $\rho_1^+$와 투자·Outsourcing으로 얻는 $\rho_2^+$도 둘 수 있다.

$$
Rx\le\rho^{\max}+\rho_1^++\rho_2^+
$$

원문 식 (14)는 $g$를 additional capacity cost라고 설명하면서 목적함수에 $+g^\top\rho_2^+$를 사용한다. 비용이라면 이익에서 빼야 하므로 부호가 모호하다. 구현할 때는 basic LP를 먼저 확인하고 확장식의 cost coefficient를 다시 정의하는 편이 안전.

---

## 5) Decomposition

Global LP를 한 번에 풀 수도 있지만 저자들은 큰 문제를 위해 작은 Master와 Load Balancing을 반복하는 heuristic을 제안한다.

### Master Problem

$$
\max_y\;c^\top y
$$

$$
Ty\le\rho^{\max}
$$

$$
\Delta^-\le y\le\Delta^+.
$$

$T_{ip}$는 Product $p$ 한 단위를 만들 때 Resource $i$가 얼마나 사용되는지 나타내는 현재 추정값. $T$가 고정되면 Job Class와 Activity 변수가 빠져 Master가 작아진다.

### Load Balancing Subproblem

Master가 $y$를 정하면

$$
\lambda=Dy
$$

로 Job Class별 수요를 만들고, 앞에서 본 $LB(R,A,\lambda)$를 풀어 Activity allocation $x$와 Resource load를 계산한다. 이 부분은 CMS별 분해 가능.

### $T$ update

Load Balancing 결과로 제품별 Resource 소요계수를 다시 계산한다.

$$
T_{ip}(x)
=
\sum_{j:d_{jp}>0}
\left(\frac{x_{ji}}{\mu_{ji}}\right)
\left(\frac{d_{jp}}{\lambda_j}\right),
\qquad p:y_p>0.
$$

식을 나눠 보면

- $x_{ji}/\mu_{ji}$: Job Class $j$ 때문에 Resource $i$가 실제 사용된 시간
- $d_{jp}/\lambda_j$: 그 Job Class 전체 수요 중 Product $p$가 차지하는 몫
- 둘을 곱해 합산: Product $p$ 한 단위의 Resource $i$ 소요시간 추정

$y_p=0$인 Product는 $\lambda_j=0$ 때문에 비율이 정의되지 않을 수 있어 이전 $T_{ip}$를 유지한다. 초기에는 모든 제품에 양의 물량을 주고 한 번 Load Balancing을 풀어 $T$를 만든다.

전체 흐름은

1. 초기 Mix로 Load Balancing, $T$ 초기화
2. 최소 의무물량 $\Delta^-$에서 feasible 기준해 생성
3. 현재 $T$로 Master를 풀어 새 $y$ 계산
4. $\lambda=Dy$로 Load Balancing
5. 새 $x$로 $T$ update
6. 이익 개선이 $\varepsilon$ 이하가 될 때까지 반복

$\Delta^-$가 feasible하고 Capa를 전혀 사용하지 않는 Product가 없다는 가정 아래 매 iteration의 Mix는 feasible하며 objective는 감소하지 않는다. 다만 이것이 global optimality를 보장하는 것은 아니다. 일부 instance에서는 local optimum에 멈춘다.

---

## 6) Experiment

초기 Mix $y_0$에서 제품별 허용범위를

$$
\Delta^-=0.95y_0,\qquad \Delta^+=1.05y_0
$$

로 설정. 각 CMS에는 적어도 한 Resource가 $\rho^{\max}$ 근처에 있도록 instance를 생성한다.

CMS Size·개수, Service-Time variance, Capability Matrix density, Product 수를 각각 세 수준으로 바꾸고 seed 5개를 사용해 총 $3^5\times5=1,215$개 instance를 만들었다. Resource 57–2,349개, Job Class 74–1,500개, Product 5–1,132개 범위.

기준 이익, Global LP, decomposition 결과를 각각 $z_0,z_{opt},z_{decomp}$라 두고 다음 값을 본다.

$$
\mathrm{potential}=
\frac{z_{opt}-z_0}{z_0}
$$

현재 Mix에서 Global LP로 얼마나 개선 가능한지.

$$
\mathrm{coverage}
=\frac{z_{decomp}-z_0}{z_{opt}-z_0}
$$

decomposition이 가능한 개선분을 얼마나 잡았는지.

$$
\mathrm{relative\ shortfall}
=\frac{z_{opt}-z_{decomp}}{z_{opt}-z_0}
=1-\mathrm{coverage}.
$$

원문 Table 2는 coverage에 해당하는 첫 비율을 `gap`이라고 표시한다. 그런데 Figure 2와 본문에서는 potential이 커질수록 gap이 0에 가까워진다고 설명한다. 그 문맥은 relative shortfall과 맞는다. 같은 `gap`이라는 말을 서로 다른 방향으로 쓴 것으로 보여 결과를 읽을 때 구분이 필요하다.

Global LP는 CPLEX로, decomposition subproblem은 GLPK와 LinProg로 계산.

---

## 7) Result

### Generated instances

제품별 물량을 ±5%만 조정했는데도 이익개선 potential은 0–4.75%. 저자들은 전체적으로 약 3%의 개선 가능성을 보고한다.

Product/Resource 비율이 높고 Capability Matrix가 sparse하며 제품별 margin 차이가 클수록 Mix 조정 여지가 커지는 경향이 있었다. Potential이 큰 문제에서는 decomposition이 global optimum에 가까웠고, 작은 일부 문제에서는 local optimum이 나타났다. Figure 기반 상관관계이므로 고정된 인과관계로 볼 수는 없음.

### Modified real FAB data

실제 FAB data를 변형한 case에서도 최적화를 수행했다. Connected Component를 Work Center처럼 두고 current Mix의 최대/평균 장비가동률, sensitivity interval, bottleneck을 비교한다.

### Figure 6 — 현재 Product Mix의 Work Center별 부하

<img width="1200" alt="현재 product mix에서 work center별 최대 및 평균 utilization과 sensitivity interval" src="/assets/img/paper-reviews/2026-08-20/klemmt-fig6.svg" />

> Source: Klemmt, Romauch, and Laure (2012), Figure 6. 논문 이해를 위한 일부 인용 및 크롭. © 2012 IEEE. [Original PDF](https://www.informs-sim.org/wsc12papers/includes/files/inv213.pdf)

현재 Mix에서는 일부 Work Center의 maximum utilization이 뚜렷하게 높다. 평균값만 보면 감춰질 수 있는 병목을 sensitivity interval과 같이 보여준다.

### Figure 7 — Product Mix 변경 후 병목 완화

<img width="1200" alt="제품 mix 최적화 후 동일 layer starts를 유지하면서 병목 utilization이 낮아진 결과" src="/assets/img/paper-reviews/2026-08-20/klemmt-fig7.svg" />

> Source: Klemmt, Romauch, and Laure (2012), Figure 7. 논문 이해를 위한 일부 인용 및 크롭. © 2012 IEEE. [Original PDF](https://www.informs-sim.org/wsc12papers/includes/files/inv213.pdf)

Figure 6과 비교하면 전체 Layer Starts는 그대로인데 주요 bottleneck load가 약 11.11%, 4.47%, 6.13% 낮아진다. 총생산량을 줄인 것이 아니라 제품마다 다른 capacity consumption pattern을 이용해 Mix를 바꾼 결과.

저자들은 MES의 Route, Product, Operation, Equipment, Dedication을 가장 세밀한 수준으로 넣어도 계산은 수 분 정도라고 보고한다.

여기서 목적값만큼 유용한 부분이 sensitivity analysis다. 어떤 Product가 특정 bottleneck의 driver인지, Mix가 변할 때 어느 Work Center가 다음 병목이 되는지를 수치로 볼 수 있다.

---

## 8) 정리하면서 본 장점과 한계

좋았던 부분은 profit과 FAB Capa를 아주 직접적으로 연결한다는 점. $D,A,R$ 세 matrix만 보면 Product Route가 만드는 작업량, Equipment Capability, 실제 장비 사용시간의 관계가 드러난다. Shadow Price나 $T_{ip}$를 이용하면 어떤 Product가 어느 bottleneck을 쓰는지도 추적 가능하다.

반면 $x$는 LOT를 몇 시에 어느 장비로 보낼지 정하는 scheduling variable이 아니라 일정 기간의 평균 작업량이다. Model도 static, deterministic, continuous-flow 가정이라 demand uncertainty, breakdown, yield, queueing effect는 밖에 있다. Decomposition은 feasible solution과 종료는 보이지만 global optimum을 보장하지 않으며 변형된 FAB raw data도 공개되지 않았다.

직접 구현한다면 세 부분으로 나누는 것이 좋다.

1. `Global LP`: $D,A,R,c,\Delta^-,\Delta^+,\rho^{\max}$를 받아 $x,y$ 동시 최적화
2. `Load Balancing LP`: 고정 $y$에서 $\lambda=Dy$를 만들고 Min-Max Load 계산
3. `Decomposition`: Master–Load Balancing–$T$ update 반복

비교할 값은 profit, max resource load, capacity violation, iteration 수, global optimum 대비 coverage.

---

## 9) Conclusion

이 논문에서 가장 중요한 두 식은

$$
Ax=Dy,
\qquad
Rx\le\rho^{\max}
$$

이다. Product Mix를 공정별 작업량으로 바꾸고, 그 작업량을 실제 가능한 Equipment에 배분한 뒤 Capa 이내인지 확인한다. 그 범위 안에서 $c^\top y$를 최대화.

Product Mix를 단순히 제품별 margin 순서로 정하지 않고 `제품 Route가 어느 장비 Capa를 얼마나 쓰는가`까지 포함한 optimization problem으로 만든 점이 핵심이다. 다만 detailed scheduling을 대체하는 model은 아니다. 이후 release planning과 simulation으로 이어지는 tactical 기준해로 보는 편이 맞다.
