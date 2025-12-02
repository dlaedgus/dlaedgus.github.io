---
title: "Melding the Data-Decisions Pipeline (AAAI 2019)"
date: 2025-07-01 12:00:00 +0900
categories: [paper_review,OR]
tags: [decision-focused-learning]
math: true
---

# Paper Review — *Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization* (AAAI 2019)

- **1저자:** Bryan Wilder  
- **제목:** Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization  
- **저널/학회:** The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI 2019)  
- **년도:** 2019

---

## 1) High-Level Summary (3–5 sentences)

본 논문은 예측(predictive modeling)과 최적화(decision making)를 **별개(two-stage)**로 수행하던 기존 방식의 한계를 지적하고, **Decision-Focused Learning**이라는 프레임워크를 제안하여 **학습(ML)**과 **조합 최적화(combinatorial optimization)**를 **end-to-end**로 통합합니다. 목표 함수는 **예측 정확도**가 아니라 **최종 의사결정의 품질(decision quality)**입니다. 이를 위해 미분이 어려운 이산 최적화를 **continuous relaxation**으로 변환하고, **미분 가능한 surrogate objective**를 통해 **gradient-based** 학습을 가능하게 합니다. 제안법은 **linear programming** 및 **submodular maximization**에 적용되며, 실험에서 **전통적 two-stage** 접근보다 **일관되게 더 나은 decision quality**를 보입니다.

---

## 2) Problem — 무엇이 문제인가?

- 현실의 의사결정 파이프라인은 일반적으로 `Data → Predictive Models → Decision Making`입니다.  
- 보통은 **ML 모델을 예측 정확도(loss)**로만 학습하고, 그 산출값을 **최적화 알고리즘**에 넘겨 최종 결정을 만듭니다.  
- 그러나 **예측 정확도가 높다고 해서 좋은 의사결정이 보장되지는 않습니다.** 즉, prediction과 decision 사이에 **disconnect**가 존재합니다.  
- 또한 loss를 실제 **optimization objective**에 맞추어 **hand-tuning**하는 일은 어렵고, 실무에서 종종 **생략**됩니다.  

> 요점: **예측-최적화 분리(two-stage)**의 근본 한계를 해결할 필요가 있습니다.

---

## 3) Proposed Solution — Decision-Focused Learning

핵심 아이디어:

1. **ML 모델 내부에 최적화(optimization)를 삽입**하여 **end-to-end 학습**을 수행합니다.  
2. **Discrete decision problem → Continuous relaxation**으로 변환해 **미분 가능**하게 만듭니다.  
3. **Surrogate objective**를 설계해 **backpropagation**을 가능하게 합니다.  
4. 학습 목표를 **예측 정확도**가 아니라 **의사결정 품질(decision quality)**로 둡니다.

**적용 대상 영역(예시):**  
- **Linear Programming**으로 모델링 가능한 문제들: shortest path, maximum flow, bipartite matching 등  
- **Submodular maximization**: budget allocation, sensor placement, facility location, diverse recommendation 등

---

## 4) Formal Problem Description

다음 형태의 **조합적 의사결정 문제**를 고려합니다:

$$
\max_{x \in \chi} f(x,\theta)
$$

- $\chi$: feasible decisions의 **이산 집합**  
- $\theta$: **알 수 없는(unknown)** 파라미터  
- 의사결정자는 $\theta$를 직접 알 수 없으나 관련된 정보 $y \in \Upsilon$를 관측  
- 따라서 최적화를 풀기 전에 **learning problem($\theta$ 예측)**을 먼저 풀어야 합니다.

데이터 생성 가정:  
- $(y, \theta)$는 분포 $P$로부터 표본화  
- 훈련 데이터: $(y_1, \theta_1), \ldots, (y_N, \theta_N) \sim \text{i.i.d. } P$

모델:  
- $m: \Upsilon \to \Theta$ 가 관측 feature $y$를 파라미터 추정치 $\theta$로 매핑  
- 목표는 **expected performance**를 최대화하는 $m$을 찾는 것

핵심 정리:  
1. 실제 최적화 문제는 $\max_{x \in \chi} f(x,\theta)$  
2. $\theta$는 미지수이며 **대신** 관련된 $y$를 관측  
3. $y$로부터 $\hat{\theta} = m(y)$ 를 예측  
4. $\hat{\theta}$ 로 최적화 문제를 풀어 $x^*$ 를 얻음  
5. 최종 utility는 **진짜 $\theta$ 기준** 성능 $f(x^*, \theta)$  
6. 따라서 학습 시 단순 예측 정확도보다 **좋은 결정을 만들어주는 예측**을 목표로 해야 함

---

## 5) Making It Differentiable — 연속 이완과 KKT

문제점:  
- $\arg\max$ 는 이산/비매끄러워 **미분 불가**  
- $x^*$ 가 binary set에서 나오거나, 연속이어도 $\arg\max$ 자체를 미분해야 하는 이슈 발생

해결책:  
- **Continuous relaxation**을 사용해 **미분 가능한 경로**를 만듭니다.  
- 특히 **LP**의 경우, 해가 꼭짓점(vertex)에 위치해 **non-smooth** → 작은 $\theta$ 변화에 $x^*$ 가 **불연속적**으로 변할 수 있어 chain rule의 중간항이 붕괴합니다.  
- 이를 막기 위해 **정규화 항** $-\gamma \lVert x \rVert^2$ 를 추가하여 **concave QP** 로 변환 → 해의 **유일성/연속성/미분가능성**을 확보합니다.  
- 이후 **KKT 조건**을 활용해 **gradient**를 추출합니다.

연쇄법칙(개념식):

$$
\frac{d\, f\!\big(x(\hat{\theta}), \theta\big)}{d\omega}
=
\frac{d f}{dx}\cdot
\frac{d x}{d \hat{\theta}}\cdot
\frac{d \hat{\theta}}{d \omega}
$$

KKT에서 유도되는 블록 시스템(개념식):

$$
\begin{bmatrix}
\nabla_x^2 f(x,\theta) & A^{\mathsf T} \\
\operatorname{diag}(\lambda)\,A & \operatorname{diag}(Ax-b)
\end{bmatrix}
\!
\begin{bmatrix}
\dfrac{d x}{d \hat{\theta}} \\
\dfrac{d \lambda}{d \hat{\theta}}
\end{bmatrix}
=
\begin{bmatrix}
\dfrac{d}{d\hat{\theta}}\, \nabla_x f(x,\theta) \\
\mathbf{0}
\end{bmatrix}
$$

> 한계 관찰(아래 §10과 연결): QP 최적해도 **KKT**로 정의되므로 **KKT 시스템을 직접 미분**해야 하고, 이는 구현/수치 안정성 측면에서 까다롭습니다.

---

## 6) Application Details — LP와 Submodular로의 적용

### 6.1 Linear Programming (LP)
- shortest path, maximum flow, bipartite matching 등 다양한 조합 최적화 문제가 **LP**로 모델링 가능  
- 그러나 **LP 최적해의 non-smooth성** 때문에 직접 미분이 어렵습니다.  
- **Quadratic term**(예: $-\gamma \lVert x \rVert^2$)을 목적함수에 추가해 **concave QP**로 바꾸고, **유일/연속 해**를 유도하여 **미분 경로**를 만듭니다.  
- 이후 **KKT 조건**을 통해 gradient를 계산합니다.

### 6.2 Submodular Maximization
- budget allocation, sensor placement, facility location, **diverse recommendation** 등의 핵심 문제들을 **submodular maximization**으로 다룰 수 있습니다.  
- 논문은 **continuous surrogate**를 설계해 **gradient-based 학습**과 **의사결정 품질 최적화**를 연결합니다.

---

## 7) Experiments — 결과 요약 및 핵심 포인트

### 7.1 Table 1: End-to-End Pipeline의 **결정 품질** 비교
- **NN1-Decision, NN2-Decision**: 대부분의 경우 **최고/상위** 성능  
- 특히 **Budget Allocation**에서 **NN1-Decision**이 최고 성능  
- **Two-stage**(NN1-2Stage, NN2-2Stage)는 예측 정확도가 높아도 **결정 품질은 낮을 수 있음**  
- **Random**, **RF-2Stage**는 baseline

> 결론: **Prediction metrics**가 좋아도 **Decision quality**가 좋다는 보장은 없다.

### 7.2 Table 2: 표준 **예측 정확도(MSE/CE/AUC 등)** 비교
- **Two-stage**가 **예측 정확도**(MSE, CE, AUC 등)에서는 더 좋게 나오는 경우가 많음  
- 그러나 **Table 1**과 대조하면, **예측이 잘 되어도 좋은 결정으로 이어지지 않을 수 있음**을 명확히 보여줌

### 7.3 Figure 1: Heatmaps — $\theta$ 행렬 시각화 비교
- (a) **Ground Truth**  
- (b) **NN1-2Stage의 예측**: GT 구조는 잘 따르지만 **최적화 관점**에서는 비효율적일 수 있음  
- (c) **NN1-Decision의 예측**: 구조가 달라도 **최적화 목적**에 더 부합하는 형태로 학습됨

### 7.4 Figure 2: Scatter Plots — 예측 총 out-weight vs 실제값 상관
- **Decision 방식 (NN1-Decision)**: $r^2 = 0.94$  
- **Two-stage (NN1-2Stage)**: $r^2 = 0.64$  
→ **Decision-aware** 학습이 더 **안정적 구조**로 $\theta$를 예측

---

## 8) Contributions — 이 논문의 기여

- **조합 최적화 문제**에 대해 **differentiable surrogate** 기반 학습 구조를 제안  
- **LP** 및 **submodular maximization** 각각에 적용 가능한 **continuous relaxation** 기법 제시  
- 다양한 실험에서 **decision-focused learning**이 **two-stage**보다 **일관되게 더 나은 decision quality**를 산출함을 입증

---

## 9) Strengths — 좋은 점

- 기존 **two-stage** 방식의 문제를 명확히 설명하고, 해결 방향을 **end-to-end 통합**으로 제시  
- **Optimization + Deep Learning**을 실제로 묶어 학습하는 **체계적 프레임워크** 제공  
- **Combinatorial optimization**의 **미분 가능 학습 경로**를 구체적으로 구성  
- **실험**으로 제안법의 **우월성**을 일관되게 보여줌

---

## 10) Limitations — 한계

- 본질적으로 **LP는 꼭짓점 해**라 미분이 어렵고, 이를 피하려면 **이차항 추가(QP화)**가 필요  
- 그래도 **QP 최적해**는 결국 **KKT 조건**을 따름 → **KKT 시스템을 직접 미분**해야 gradient 획득  
- **KKT**는 연립방정식 + **보완성 조건**을 포함 → 수학적 처리/구현이 어렵고, **해석적으로 불안정**할 수 있음  
- **예측 → decision → loss**의 경로가 **간접적**이라 **해석성**이 떨어질 수 있음

---

## 11) Improvement Ideas — 후속 개선 아이디어

- **Interior-point/log-barrier 기반 differentiable solver**를 사용해, 기존 **QPTL + KKT 미분**의 복잡성을 피하고 **solver trajectory를 직접 미분**하는 방식 제안(후속 연구 경향).  
- 목적: **KKT 미분의 복잡성/불안정성**을 완화하고, **안정적 학습 경로** 제공.

---

## 12) Extended Notes — 정리/배경

### 12.1 전체 파이프라인의 맥락
- 현실 AI 응용의 목표: **data → predictive models → decision making** 파이프라인 구축  
- 이 결합으로 **evidence-based decision**이 가능, 현실 분야에서 **transformative potential**  
- predictive models는 **ML 모델**, decision making은 **optimization algorithm**  
- ML 모델은 data로부터 **unknown 값**을 예측, optimization은 예측값으로 **목적함수 극대화 결정** 산출

### 12.2 왜 분리하면 문제가 생기는가?
- 일반적으로 **separate**: ML은 **predictive accuracy**로 학습 → 예측값을 **optimization**에 투입  
- 하지만 ML에서 쓰는 **loss**와 실제 **decision objective**가 **다를 수 있음**  
- 예: **Loss를 낮췄는데** 실제로는 **잘못된 decision**을 낼 수 있음  
- **Loss를 optimization 목표와 일치**시키기 위한 hand-tuning은 어려워 **대부분 생략**

### 12.3 논문의 해결책
- **Decision-Focused Learning**: prediction + optimization을 하나의 **end-to-end** 시스템으로 통합  
- 과거: `[ML 따로 학습] → [결과를 optimization에 넣음]`  
- 제안: `[ML 모델 안에서 optimization까지 같이 수행]` → **하나의 큰 end-to-end 모델**  
- 초점: **combinatorial optimization**  
- 접근: **discrete → continuous relaxation**, 학습 시 **backprop** 가능하게, **테스트 시** 연속 해를 **이산 해로 반올림**

---

## 13) Integration into Gradient Loop — 학습 루프 내 통합의 기술적 포인트

- 예측 모델 $m(y, \omega)$ 의 파라미터 $\omega$ 가 변하면 $\hat{\theta} = m(y, \omega)$ 가 변하고, 그에 따라 $x^*$ 도 변함  
- 결국 **gradient가 $\arg\max$ 에 의존**하는 구조  
- **문제 1:** $x^*$ 는 binary 등 **이산적** → **미분 불가**  
- **문제 2:** 연속이더라도 **$\arg\max$ 자체를 미분**해야 함  
- **해결:** **continuous relaxation** + **KKT 활용**으로 미분 경로 구성

---

## 14) LP Case — QP로의 변형과 안정화

- 많은 조합 최적화 문제가 **LP** 로 표현 가능  
- 그러나 **LP 해의 non-smooth성** 때문에 작은 $\theta$ 변화에 $x^*$ 가 **불연속** → chain rule 붕괴  
- **정규화 항** $-\gamma \lVert x \rVert^2$ 추가 → 목적을 **concave QP** 로 변환  
- 결과: 해가 **유일/연속/미분 가능**  
- **단점:** 여전히 **KKT 시스템 미분**이 필요 → 구현/수치 난이도 ↑

---

## 15) Submodular Case — Surrogate를 통한 미분 경로

- **Submodular maximization** 에 대해 **continuous surrogate** 를 설계하여  
  **gradient-based 학습** 과 **의사결정 품질** 을 연결  
- 대표 응용: **budget allocation**, **sensor placement**, **facility location**, **diverse recommendation** 등

---

---

## 16) Experiments — Table & Figures Summary

### Table 1 — Solution quality of each method for the full data-decisions pipeline
**목적:** 최종적인 의사결정(최적화 문제)에서 각 방법이 얼마나 좋은 해를 찾았는지를 비교  
**포인트:**  
- NN1-Decision, NN2-Decision: 대부분의 경우에서 성능이 우수함 (특히 Budget Allocation에서 NN1-Decision 최고 성능)  
- Two-stage 방법들 (NN1-2Stage, NN2-2Stage)은 예측 모델 정확도는 높을 수 있어도 의사결정 품질은 낮음  
- Random이나 RF-2Stage는 baseline 역할  

---

### Table 2 — Accuracy of each method according to standard measures
**목적:** 예측 정확도 (MSE, CE, AUC 등)를 기준으로 각 모델을 평가  
**포인트:**  
- Two-stage 방식(NN1-2Stage, NN2-2Stage)이 예측 정확도는 더 높음  
- 특히 Matching, Diverse Recommendation에서 CE, AUC 기준으로 Decision 방식보다 낫지만  
- **Table 1과 대조하면 예측이 잘 된다고 해서 좋은 결정이 나오는 것은 아님**을 보여줌  

---

### Figure 1 — Heatmaps
**목적:** 예측된 $\theta$ 행렬을 시각적으로 비교  
(a) 실제 ground truth  
(b) NN1-2Stage의 예측  
(c) NN1-Decision의 예측  

**포인트:**  
- (b)는 ground truth 구조를 따르지만 실제 최적화 성능은 비효율적  
- (c)는 구조가 달라도 **최적화 목적에 더 부합하는 형태로 학습됨**

---

### Figure 2 — Scatter plots
**목적:** 각 아이템의 예측된 총 out-weight와 실제값의 상관관계(정확도) 비교  

- 왼쪽: Decision 방식 (NN1-Decision) → $r^2 = 0.94$  
- 오른쪽: Two-stage 방식 (NN1-2Stage) → $r^2 = 0.64$

<img width="1421" height="304" alt="image" src="https://github.com/user-attachments/assets/5a37f4e9-9fa7-4130-8b60-3e805ac5f606" />
<img width="1221" height="238" alt="image" src="https://github.com/user-attachments/assets/3da4bd4e-a269-4099-a899-5a3364c2896d" />

<img width="604" height="433" alt="image" src="https://github.com/user-attachments/assets/58ef464f-a12a-4bf5-bf4d-94e98ccd9eda" />
<img width="609" height="279" alt="image" src="https://github.com/user-attachments/assets/24d744b7-943a-444e-be6e-93973e92b4f4" />

