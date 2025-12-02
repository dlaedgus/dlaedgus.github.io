---
title: "Interior Point Solving for LP-based Prediction+optimisation (NeurIPS 2020)"
date: 2025-07-07 12:00:00 +0900
categories: [paper_review, OR]
tags: [decision-focused-learning]
math: true
---

# Paper Review — *Interior Point Solving for LP-based Prediction+optimisation* (NeurIPS 2020)

- **1저자:** Jayanta Mandi  
- **공저자:** Tias Guns  
- **학회:** 34th NeurIPS (2020)  
- **별칭:** **IntOpt** (Interior-Point 기반 DFL 레이어)

---

## 1) High-Level Summary (3–5 sentences)

본 논문은 **predict-then-optimize** 문제에서 **연속 이완 + KKT 미분(QPTL)** 대신, **내부점법(interior point)**의 **동형 자기쌍대(Homogeneous Self-Dual; HSD)** 임베딩을 직접 **미분 가능한 레이어**로 사용하여 **end-to-end** 학습을 수행. 핵심은 **log-barrier**를 쓰는 **LP** 해법의 **Forward**와 동일한 구조를 **Backward**에 재활용해 **$ \partial x^* / \partial c $**를 안정적으로 계산하는 것. 실험적으로, 단순한 문제(**knapsack**)에서는 **SPO**가 강하지만, **day-ahead 에너지 스케줄링**·**트위터 최단경로** 같은 어려운 세팅에서는 **IntOpt**가 **QPTL/SPO** 대비 **낮은 regret**을 보임.

---

## 2) Problem — 무엇이 문제인가?

전통적 **two-stage** 파이프라인은 **예측 정확도(MSE/CE/AUC)**만을 최소화해 모델을 학습한 뒤, 그 출력을 최적화 모듈에 전달. 그러나 **의사결정 품질(decision quality)**은 예측 정확도와 일치하지 않으며, 특히 **LP/MILP**의 **argmin**은 이산/비매끄러워 **미분 불가** 혹은 **거의 모든 곳에서 0**인 문제가 있음. 이로 인해 **regret** 같은 **task-loss**를 직접 최소화하기가 어려움.

---

## 3) Proposed Solution — Interior-Point (HSD) 기반 End-to-End

- **아이디어:** **MILP → LP 이완** 후, **log-barrier**를 포함한 **HSD 임베딩**을 사용하여 **LP 레이어**를 구성하고, 이를 **내부점법**으로 풂.  
- **Forward:** 내부점법으로 **$x^*(\hat c; A, b)$** 계산.  
- **Backward:** 동일한 **HSD 선형계**를 역으로 풀어 **$$ \partial x^* / \partial \hat c $$** 계산 → **$ \partial L/\partial \theta = (\partial L/\partial x^*)(\partial x^*/\partial \hat c)(\partial \hat c/\partial \theta) $$**.  
- **안정화:** **$\lambda$-cutoff(early stopping)**, **Tikhonov damping** 등 수치 안정화 기법 적용.

---

## 4) Formal Problem — MILP 표준형 & 이완

**표준형 MILP**
$$
\min_x \; c^\top x 
\quad \text{s.t. } Ax=b,\; x\ge 0,\; \text{some } x_i \in \mathbb{Z}.
$$

여기서 $x\in\mathbb{R}^k,\; c\in\mathbb{R}^k,\; A\in\mathbb{R}^{p\times k},\; b\in\mathbb{R}^{p}$ 입니다.  
예측 모델 $g(z;\theta)$ 가 **계수** $c$를 예측($\hat c$)하고, 우리는 **regret**
$$
\operatorname{regret}(\hat c,c;A,b)
= c^\top\!\big(x^*(\hat c;A,b)-x^*(c;A,b)\big)
$$
을 줄이도록 $\theta$를 학습합니다. **뒤 항 $$c^\top x^*(c)$$는 상수**이므로, 학습 시 **$$c^\top x^*(\hat c)$$** 최소화에 집중합니다. 이때 **MILP → LP**로 이완하여 **미분 경로**를 만듦

---

## 5) 미분 가능 경로 — KKT 대신 **Log-Barrier + HSD**

- **문제점:** LP 해는 꼭짓점에 놓여 **비매끄러움**(미분 불가/불연속).  
- **QPTL(기존):** 목적에 $-\gamma\|x\|^2$ 추가 → **concave QP**, KKT 미분.  
- **본 논문:** **KKT 직접 미분** 대신, **log-barrier**가 들어간 **HSD 임베딩**을 **그대로 미분**. 내부점법의 **탐색 방향**과 **필요한 gradient** 사이의 연결을 이용해, **quadratic 정규화 없이**도 안정적 미분을 수행합니다.

---

## 6) Application Areas — 예시 과제

- **부동산 투자 최적화 (knapsack)**  
- **에너지 비용 고려 Day-Ahead 스케줄링**  
- **소셜 그래프(트위터) 최단 경로 (shortest path)**  
→ 모두 예측값이 **downstream 최적화**에 직접 영향을 주는 **predict-and-optimize** 문제입니다.

---

## 7) IntOpt 레이어 — Forward/Backward 개요

**Forward (Interior-Point on HSD):**
- **HSD 제약(개념):**
$$
Ax-b\tau=0,\quad 
A^\top y+t-c\tau=0,\quad 
-c^\top x+b^\top y-\kappa=0,
$$
$$
t=\lambda X^{-1}\mathbf e,\qquad 
\kappa=\lambda/\tau,\qquad 
x,t,\tau,\kappa\ge 0.
$$
$\lambda$를 점차 낮추며 **중심경로**를 따라가 **해**를 근사. 초기 **feasible** 필요 없이 항상 시작 가능.

**Backward (Gradient via HSD Linear System):**
- 동일한 **뉴턴 선형계**를 이용해 $ \partial x^*/\partial \hat c $ 계산.  
- 수치 이슈를 줄이기 위해 **Tikhonov damping**($M \leftarrow M+\alpha I$) 사용.  
- **$\lambda$-cutoff**를 조정해 불안정 구간 진입 전에 정지.

---

## 8) Algorithm 1 — End-to-end training of an LP (relaxed MILP)

**Input** : $A, b$, training data $\mathcal D=\{(z_i, c_i)\}_{i=1}^n$  
1 **initialize** $\theta$  
2 **for** epochs **do**  
3 &nbsp;&nbsp; **for** batches **do**  
4 &nbsp;&nbsp;&nbsp;&nbsp; **sample** batch $(z, c)\sim \mathcal D$  
5 &nbsp;&nbsp;&nbsp;&nbsp; $\hat c \leftarrow g(z;\theta)$  
6 &nbsp;&nbsp;&nbsp;&nbsp; $x^* \leftarrow$ **Neural LP layer Forward Pass** to compute $x^*(\hat c;A,b)$  
7 &nbsp;&nbsp;&nbsp;&nbsp; $\partial x^*/\partial \hat c \leftarrow$ **Neural LP layer Backward Pass**  
8 &nbsp;&nbsp;&nbsp;&nbsp; **Compute** $\partial L/\partial \theta = (\partial L/\partial x^*) (\partial x^*/\partial \hat c) (\partial \hat c/\partial \theta)$ and update $\theta$  
9 &nbsp;&nbsp; **end**  
10 **end**

---

## 9) Method Details — 핵심 식 모음(개념)

**Chain Rule**
$$
\frac{\partial L}{\partial \theta}
= \frac{\partial L}{\partial x^*}\;
  \frac{\partial x^*}{\partial \hat c}\;
  \frac{\partial \hat c}{\partial \theta},
\qquad
\text{(regret의 경우 } \partial L/\partial x^*=c\text{)}.
$$

**LP 이완의 Primal/Dual**
$$
\min_x c^\top x\;\text{s.t. }Ax=b,\;x\ge0
\quad\Longleftrightarrow\quad
\max_{y,t} b^\top y\;\text{s.t. }A^\top y+t=c,\;t\ge0.
$$

**KKT 미분(개념)과 Log-Barrier**
$$
\mathcal L(x,y;c)=c^\top x+y^\top(b-Ax),
\qquad
\partial_x\mathcal L=c-A^\top y,\quad Ax-b=0.
$$
미분하면
$$
\begin{bmatrix}
f_{cx}(c,x)\\[2pt] 0
\end{bmatrix}
+
\begin{bmatrix}
f_{xx}(c,x) & -A^\top\\[2pt]
A & 0
\end{bmatrix}
\begin{bmatrix}
\partial x/\partial c\\[2pt]
\partial y/\partial c
\end{bmatrix}
=0.
$$
비매끄러움을 피하려 **log-barrier** 도입:
$$
f(c,x)=c^\top x-\lambda\sum_{i=1}^k \ln x_i,\quad
c-t-A^\top y=0,\;\; Ax-b=0,\;\; t=\lambda X^{-1}\mathbf e.
$$

**HSD 임베딩 (개념)**
$$
Ax-b\tau=0,\qquad
A^\top y+t-c\tau=0,\qquad
-c^\top x+b^\top y-\kappa=0,\qquad
t=\lambda X^{-1}\mathbf e,\quad
\kappa=\lambda/\tau.
$$

---

## 10) Experiments — 데이터·과제·핵심 결과

### 10.1 Knapsack (부동산 투자)
- **설정:** 실거래/경제 변수로 **주택 분양가** 예측 → 예측값으로 **0-1 knapsack**(예산 제약) 최적화.  
- **요지:** **two-stage**가 **MSE**는 최소. **SPO**가 최종 **regret**에서 강함. **IntOpt ≳ QPTL**, 단 이 간단 문제에서는 두 방법 모두 two-stage보다 불리할 수 있음.

### 10.2 Energy-Cost Aware Day-Ahead Scheduling
- **설정:** 시계열 전력요금 예측과 자원 제약을 함께 고려하는 스케줄링.  
- **요지:** **IntOpt**가 **최저 regret**. **HSD+log-barrier** 변형이 가장 안정적이며, **$\lambda$-cutoff**가 너무 낮으면 수치 이슈.

### 10.3 Shortest Path on Twitter Ego Network
- **설정:** 노드/엣지 특징으로 생성한 가중치를 예측 후 **최단경로** 의사결정.  
- **요지:** **IntOpt**가 **regret** 최소(근소 우세). 선형목적의 **스케일 불변성** 때문에 MSE가 높아도 의사결정 품질은 우수할 수 있음을 강조.

---

## 11) Contributions — 기여 요약

1) **내부점법(HSD) 임베딩을 직접 미분**하는 **LP 레이어(IntOpt)** 제안  
2) **log-barrier** 채택으로 **별도 QP 정규화 불필요**  
3) **$\lambda$-cutoff, central-path 기반 종단** 등 실용적 안정화 절차 제시  
4) **Knapsack / Day-Ahead / Shortest Path** 3과제에서 **QPTL/SPO**와 비교 평가 (과제 난이도에 따라 우열 교차)

---

## 12) Strengths — 좋은 점

- **해법-미분**을 **같은 수학적 골격(HSD)**로 연결 → 구현/이해 일관성  
- **정규화 항 없이** log-barrier로 **미분 가능성** 확보  
- **어려운** predict-and-optimize 과제에서 **낮은 regret** 달성  
- **산업 솔버 의존↓** (자체 interior-point 구현으로 연구 재현 용이)

---

## 13) Limitations — 한계

- **연산량 큼:** **HSD interior-point** 방식은 계산량이 많아 **epoch당 시간이 길다 (최대 약 15분)**  
- **쉬운 문제:** (예: 간단 **knapsack**)에서는 **SPO**가 더 좋은 결과를 내는 경우 존재  
- **솔버 구현:** 상용 **solver(Gurobi 등)**을 쓰지 않고 **자체 interior-point solver** 사용

---

## 14) Improvement Ideas — 개선 아이디어

- **산업용 solver 통합**(e.g., Gurobi)으로 계산 시간/안정성 향상  
- **LP relaxation 강화**(컷/강화) 적용으로 **MILP** 성능 개선  
- **$\lambda$ 스케줄/stop 규칙 자동화**(adaptive cutoff)와 **배치-공유 선형계 재활용**으로 역전파 비용 절감

---

### 간단 정리(한 줄)

**IntOpt =** “**LP 내부점법(HSD)**를 **그대로 미분**하는 **예측-최적화 레이어**” — **정규화 없이** log-barrier로 안정적 gradient를 만들고, **어려운 과제**에서 **낮은 regret**을 실증합니다.
