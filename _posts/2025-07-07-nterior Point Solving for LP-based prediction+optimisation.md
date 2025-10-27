---
title: "Interior Point Solving for LP-based Prediction+optimisation (NeurIPS)"
date: 2025-07-07 12:00:00 +0900
categories: [paper_review, decision_focused_learning]
tags: [interior-point, differentiable-optimization, lp, spo, kkt-free, neural-optimisation]
math: true
---

# Paper Review — *Interior Point Solving for LP-based Prediction+optimisation* (NeurIPS)

---

## 1) High-Level Summary (3–5 sentences)

본 논문은 기존 SPO(Smart Predict-then-Optimise) 프레임워크에서 등장하는 **QPTL + KKT 미분 기반 접근의 구조적 한계** — 즉, (i) LP는 non-smooth여서 직접 미분이 어렵고 (ii) 이를 피하기 위해 추가한 quadratic term이 다시 KKT 미분 문제를 야기한다 — 를 해결하기 위해, **Interior-Point 기반의 differentiable solver 자체를 활용하는 새로운 경로**를 제안한다.  
핵심 아이디어는 **log-barrier interior-point 경로를 그대로 미분 가능한 연산자로 포함**시켜 end-to-end 학습이 가능하도록 하는 것이다.  
이를 통해 **KKT 시스템을 명시적으로 미분하지 않고도 gradient backpropagation 가능**해지며, SPO 학습의 안정성과 일관성이 크게 개선된다.  
실험적으로 기존 QPTL 방식 대비 학습 안정성 및 decision quality 모두에서 우월함을 입증한다.

---

## 2) Problem — 무엇이 문제인가?

- 기존 predict-then-optimize(SPO) 파이프라인은  
  `ML 학습(loss minimization)` → `예측값으로 LP 최적화` → `Decision` 의 **two-stage 분리 구조**
- 그러나 LP 기반 의사결정 품질을 학습 단계에서 직접 반영하려 할 경우:  
  - 단순 loss로는 decision quality alignment 실패
  - decision-aware gradient를 구하려면 LP 최적화 layer를 **미분 가능하게 해야 함**
- 기존 해결책: *(QPTL)* LP 목적에 $-\gamma\lVert x\rVert^2$ 추가 → QP로 변형 → **KKT 시스템 미분**
- 근본 문제:
  1) LP는 해가 vertex에 있어 non-smooth → gradient 불연속
  2) QP로 바꿔도 결국 KKT → **KKT 미분 복잡/불안정**
  3) gradient 경로가 solver 외부에서 재구성됨 → numerical instability

> 요약: **“LP-based SPO에서 KKT 미분을 전제로 하는 기존 QPTL은 구조적으로 불안정”**  
> 이 한계를 제거할 필요가 있음.

---

## 3) Proposed Solution — Interior Point as a Differentiable Layer

핵심 접근:

1) QP로 변형하거나 KKT를 직접 미분하지 않는다  
2) 대신 **Primal-dual interior-point(log-barrier) 경로 자체를 differentiable operator로 사용**  
3) 즉, solver 내부 경로 $x^\*(\theta)$ 를 그대로 computational graph로 포함  
4) backprop은 solver trajectory를 따라 자동 미분(autograd)을 통해 수행

이로써:

- **KKT explicit differentiation 불필요**
- **LP의 non-smooth vertex pathology 회피**
- **stable & end-to-end 미분 가능 pipeline 확보**

---

## 4) Formal Reformulation (핵심관점만 유지)

일반 LP 기반 decision problem:

$$
\max_x \;\; \theta^{\mathsf T} x
\quad \text{s.t.}\quad
Ax = b,\;\; Gx \le h
$$

기존 QPTL 방식은 다음을 쓴다:

$$
\max_x \left( \theta^{\mathsf T} x - \gamma \lVert x \rVert^2 \right)
\quad \text{s.t.}\quad
Ax = b,\;\; Gx \le h
$$

그러나 이때도 해는 KKT 시스템:

$$
\nabla_x L(x,\lambda,\mu) = 0,\quad
Ax=b,\quad
\mu\odot(Gx-h)=0,\;\mu\ge0
$$

→ 따라서 다시 **KKT 미분을 해야 하는 구조적 루프**에서 벗어나지 못함.

---

## 5) 핵심 아이디어 — Barrier Interior-Point Differentiation

Interior-point의 log-barrier formulation 예시는:

$$
\mathcal{L}(x,\lambda;\mu)
=
\theta^{\mathsf T}x
+
\lambda^{\mathsf T}(Ax-b)
+
\mu \sum_i \log(h_i - G_i x)
$$

여기서 $\mu\to 0$로 barrier를 줄여가는 경로 자체가 $x^\*(\theta)$ 에 대한 **smooth differentiable trajectory**를 형성  
이 전체 최적화 경로를 computational graph로 삽입 → **autograd로 미분 가능**

> 즉 “KKT를 풀고 그것을 미분하는 것이 아니라”,  
> **“풀이 과정 전체를 미분 가능한 알고리즘으로 embedding”**

---

## 6) Experiments — 핵심 해석요약

- 기존 QPTL(KKT differentiation) 기반 SPO와 직비교
- Interior-point 기반 differentiable solving은:
  - 학습 안정성 ↑ (gradient oscillation 감소)
  - decision quality ↑ (end-to-end alignment 강화)
  - 수치적 robustness ↑ (KKT sensitivity 제거 효과)

---

## 7) Contributions — 정리

- **KKT 미분 기반 QPTL의 구조적 한계를 실질적으로 제거**
- **Interior-point log-barrier trajectory를 그대로 differentiable layer로 사용**
- **Stability + smoothness + decision-aware alignment를 동시 만족**
- **LP-based SPO 학습에서 새로운 class of differentiable solver를 제시**

---

## 8) Limitations — 저자 스스로의 인정/기술적 논점

- Interior-point 기반 경로 미분도 computational burden 존재
- Barrier schedule tuning 및 stopping policy가 gradient 품질에 영향
- 모든 LP family에 즉시 적용 가능한 plug-and-play 보장은 아님  
  (특히 sparse massive LP or combinatorial facets)

---

## 9) One-line takeaway

> **“KKT를 미분하는 대신 — solver를 미분한다.”**  
> 이 관점 전환이 SPO-learning을 안정하게 만든다.

---

