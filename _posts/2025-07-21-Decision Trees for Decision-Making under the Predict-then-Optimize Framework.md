---
title: "Decision Trees for Decision-Making under the Predict-then-Optimize Framework (ICML 2020)"
date: 2025-10-28 12:00:00 +0900
categories: [paper_review, decision_focused_learning]
tags: [predict-then-optimize, SPO-loss, decision-trees, SPO-tree, SPO-forest, decision-focused-learning, icml2020]
math: true
---

# Paper Review — *Decision Trees for Decision-Making under the Predict-then-Optimize Framework* (ICML 2020)

- **1저자:** Adam N. Elmachtoub  
- **제목:** Decision Trees for Decision-Making under the  
Predict-then-Optimize Framework  
- **저널명:** ICML  
- **년도:** 2020

---

## 1) High-Level Summary (3–5 sentences)

본 논문은 predictive 모델과 optimization 모델을 통합하는 Predict-then-Optimize 문제를  
다룬다. LP 기반의 결정 문제에서 기존의 MSE 기반 예측 중심 학습 방식이 실제 (decision  
quality를 보장하지 못한다는 점을 지적하며, 이를 극복하기 위해 SPO (Smart Predict then  
Optimize) Loss에 기반한 decision-focused learning방법을 제안. 기존 SPO 논문은 SPO  
loss가 미분이 불가능하다는 문제를 해결하기 위해 surrogate loss(SPO+)를 사용했지만, 본  
논문에서는 그러한 근사(surrogate) 없이 SPO loss 자체를 직접 최적화한다는 점에서 큰  
차이점이 있음. 제안된 SPO Trees는 decision quality에 직접 영향을 미치는 SPO loss를  
기준으로 split을 선택하며, greedy 방식과 MILP 기반 방식 모두를 제시하고, 이를 SPO  
Forest로 확장. 또한, shortest path 문제와 뉴스 추천 문제에 대한 실험을 통해 제안된 방법의  
의사결정 품질 우수성을 보여줌

---

## 2) 논문이 풀고자 하는 문제

기존의 예측-후-최적화 프레임워크에서는 예측 모델이 비용이나 수요 등의 파라미터를  
예측하고, 그 예측값으로 최적화 문제를 푸는 방식. 하지만 예측 모델이 MSE 기준으로  
학습되면, 실제 의사결정 결과에서 손실이 커질 수 있다는 문제가 있음. 즉, 예측 정확도가  
높아도 실제 결정 품질은 낮을 수 있음.

---

## 3) 어떤 방법으로 이 문제를 해결

SPO Loss: 예측된 파라미터가 실제 최적화 결과에 미치는 영향 자체를 손실 함수로 설정하여,  
예측 모델을 결정 품질에 직접 기반하여 학습.  
SPO Tree: 의사결정 트리 구조에서 SPO Loss를 기준으로 split을 선택하는 그리디 알고리즘  
기반 트리 학습법.  
SPO MILP: SPO Loss 최적화를 **정수계획(MILP)**으로 정식화하여 최적의 트리 구조를 학습.  
SPO Forest: 여러 개의 SPO Tree를 앙상블로 구성한 랜덤 포레스트 형태 확장.

---

## 4) application area

predicting travel times for shortest path problems, predicting demand for inventory  
management problems, predicting returns for portfolio optimization

---

## 5) 기여(contribution)

기존 예측 중심 학습의 한계를 지적하고, 결정 품질 중심(SPO Loss) 학습 프레임워크 제안  
SPO Loss를 이용한 의사결정 트리(SPO Tree) 학습 방법 제시 (그리디 + MILP)  
SPO Tree의 랜덤 포레스트 확장인 SPO Forest 제안  
MILP 기반 정확한 학습 vs 그리디 근사 학습의 비교 분석

---

## 6) 논문의 좋은점 / 한계 / 개선 아이디어

**논문의 좋은점:**  
해석 가능성(interpretable decision tree 구조)을 유지하면서도 높은 품질의 결정 성능  

**한계:**  
MILP 학습 방식은 시간 소모가 크고, 작은 depth에만 적용 가능  

**개선 아이디어:**  
MILP 대신 differentiable surrogate SPO loss 도입 (예: SPO+와 같은 근사 loss)

---

## 7) 정리

현실에서의 많은 decision-making problem들을 해결하기 위한 편리하고 널리 사용되는  
framework가 predict-then-optimize구조  
이 구조는 (1) 과거 data로 학습된 ML model을 사용해 uncertain input parameters를 예측하고  
(2) 예측된 parameters를 사용해 optimization problem을 풀어 decision를 내림  
이때의 ML model은 MSE를 측정하는 Loss function 기준으로 학습되며 prediction이 후속  
optimization 문제에 미치는 영향은 고려하지 않음  
따라서 본 논문에서는 예측 오류가 아닌 decision error를 최소화하도록 하는 decision tree를  
학습하는 방법론 제안  
SPO 논문에서는 prediction task와 optimization task를 통합하여, 예측된 input parameter로  
인해 유도된 sub-optimality of the decisions를 직접 측정하는 Loss function으로 ML model을  
학습시켰음  
이때 저자들은 SPO loss function이 non-convex이고 discontinuous이기 때문에 이를 사용해 ML  
model을 학습하는 것은 실질적으로 불가능하다고 말함  
따라서 저자들은 convex한 surrogate loss function인 SPO+ loss를 제안 - 이를 통해 특정 조건  
하에서 SPO loss에 대해 Fisher consistency(surrogate loss function으로 학습하더라도 충분히  
많은 데이터가 있으면 진짜 최적 손실(SPO)을 근사할 수 있다는 개념)을 만족함을 보임  
그러나 이러한 surrogate loss function들은 SPO loss에 대해 optimal decision을 보장하지  
못하며 approximation을 제공할 뿐임  
SPO loss를 직접 사용하여 ML model을 학습할 수 있는 방법론은 아직 제안 되지 않음  
본 논문에서는 SPO loss를 minimize하는 DT 알고리즘을 제안 - SPO trees (SPOTs)  
SPO loss function이 nonconvex이고 discontinuous 여도 DT의 구조적 특성을 활용하면 이  
loss를 기반으로 하는 optimization 문제를 크게 단순화할 수 있음  
즉, deep learning처럼 gradient를 쓸 필요 없이, tree 구조의 split 특성을 이용해 SPO loss  
최적화 가능  
따라서 본 논문은 SPO loss를 직접 사용하는 ML model 학습 방법을 처음으로 구현한 논문

---

## 8) 수식/정의 (원문 내용 그대로, 수식만 LaTeX 표기)

- 의사결정 가능한 해의 영역: $S \subseteq \mathbb{R}^{d}$.  
- 의사결정 문제의 최적값:
  $$
  z^*(c) \;=\; \min_{w \in S} \; c^\top w.
  $$
- 비용 벡터 $c \in \mathbb{R}^{d}$, 결정변수 벡터 $w \in \mathbb{R}^{d}$.  
- 최적해 집합:
  $$
  W^*(c) \;=\; \arg\min_{w\in S}\{\,c^\top w\,\}, \qquad w^*(c)\in W^*(c).
  $$

Predict-then-Optimize framework에서는 optimal decision을 내리기 위한 $w^*(\cdot)$를 풀 때  
진짜 목적함수 계수 $c$를 알 수 없기 때문에 ML model이 예측한 $\hat{c}$를 사용함.  
dataset $\{(x_1,c_1),(x_2,c_2),\ldots,(x_n,c_n)\}$ 을 기반으로 ML model을 학습함으로써 예측을 수행.  
$x \in \mathbb{R}^{p}$ 는 cost vector $c$를 예측하기 위해 사용 가능한 입력 feature들.  
$\mathcal{H}$ : $x$로부터 $c$를 예측하는 ML model들의 hypothesis class.  
$\ell(\cdot,\cdot): \mathbb{R}^{d}\times \mathbb{R}^{d} \to \mathbb{R}_{+}$ : ML model 학습할 때 사용하는 loss function(본 논문에서는 SPO).  
ERM:
$$
f^* \;=\; \arg\min_{f\in\mathcal{H}} \; \frac{1}{n}\sum_{i=1}^{n} \ell\big(f(x_i),\, c_i\big).
$$

- $\ell_{\mathrm{MSE}}(\hat c, c)$ : 흔히 사용하는 MSE loss, 예측의 정확도만을 평가함.  
- SPO loss:
  $$
  \ell_{\mathrm{SPO}}(\hat c, c) \;=\; c^\top w^*(\hat c)\;-\; z^*(c),
  $$
  즉, 예측 $\hat c$로 최적화했을 때 나온 결정의 실제 비용과 진짜 비용으로 얻을 수 있는 최적 비용의 차이.  
  최적해가 다수 존재할 수 있으므로 보수적으로
  $$
  \ell_{\mathrm{SPO}}(\hat c, c)
  \;=\;
  \max_{w\in W^*(\hat c)} c^\top w \;-\; \min_{w\in S} c^\top w,
  $$
  이 항이 non-convex, discontinuous한 것이 문제.

---

## 9) Decision Trees for Decision-Making

본 논문에서는 predict-then-optimize framework 하에서 decision tree 활용  

<img width="899" height="358" alt="image" src="https://github.com/user-attachments/assets/60dd3f02-45cf-4f8d-9147-f65202175d46" />

두 개의 node와 그 사이에 두 개의 후보 도로가 있는 간단한 최단 경로 문제 가정  
각각의 도로에는 알려지지 않은 간선 비용 $c_1, c_2$ 존재  
간선 비용을 예측하기 위한 3개의 feature가 있다고 가정  
$x_1$: 평일 여부를 나타내는 이진형 변수, $x_2$: 현재 시간, $x_3$: 눈이 오는지 여부(이진형)  
관측된 특징으로 $c$를 예측하고 그 예측값으로 최적의 decision을 내림  
SPO loss와 MSE loss를 사용하여 학습된 DT (SPOTs vs CART)의 동작을 설명하기 위한  
간단한 예시  
feature는 $x$ 한 개, 10000개의 feature–cost 쌍으로 이루어진 data 생성. $x \sim \mathrm{Uniform}(0,1)$에서  
샘플링하고 그 $x$로부터 두 엣지의 cost를
$$
c_1 = 5x + 1.9, \qquad c_2 = (5x + 0.4)^2
$$
로 계산, 이를 SPO loss, MSE loss 로 학습시켜 비교

---

## 10) 실험결과

<img width="1134" height="459" alt="image" src="https://github.com/user-attachments/assets/cb652df3-544d-4fb3-8ec9-442fa0e8dfc2" />


SPOTs는 매우 높은 품질의 decision을 내리면서도 CART보다 훨씬 간단한 구조 유지

---

## 11) SPOTs 설명 (트리 목적함수와 분할)

모든 DT의 목표는 train data를 $L$개의 leaf인 $R_1,\ldots,R_L := R_{1:L}$로 분할하는 것,  
이 리프들이 만드는 예측이 전체 loss function을 최소화하도록 하는 것:
$$
\min_{R_{1:L}\in\mathcal{T}} \;\frac{1}{n}\sum_{\ell=1}^{L}
\Bigg(\min_{c_\ell}\;\sum_{i\in R_\ell}\ell(c_\ell, c_i)\Bigg).
$$

**Theorem 1.** 리프 $\ell$ 내 모든 관측값들의 평균 cost vector를
$$
c_\ell \;=\; \frac{1}{|R_\ell|}\sum_{i\in R_\ell} c_i
$$
라고 할 때, 이에 대응되는 decision problem이 **고유한 최소해**를 가지면 ($|W^*(c_\ell)|=1$)  
이는 리프 안에서의 SPO loss를 minimize함, 즉
$$
c_\ell \in \arg\min_{u}\sum_{i\in R_\ell} \ell_{\mathrm{SPO}}(u, c_i).
$$

따라서
$$
\min_{R_{1:L}\in\mathcal{T}} \frac{1}{n}\sum_{\ell=1}^{L}
\left(\min_{c_\ell}\sum_{i\in R_\ell}\ell(c_\ell, c_i)\right)
\;=\;
\min_{R_{1:L}\in\mathcal{T}} \frac{1}{n}\sum_{\ell=1}^{L}\sum_{i\in R_\ell}
\big(c_i^\top w^*(c_\ell) - z^*(c_i)\big).
$$

이제 이 식을 어떻게 풀 것인가??  
위의 목적 함수에 대해 SPO트리를 학습시키는 방법으로 **recursive partitioning** 기법을 제안.  
$x_{i,j}$: $i$번째 학습 sample에서의 $j$번째 feature 값, 트리 분할은 feature $j$를 기준으로 임계값 $s$보다  
작거나 같은지 여부에 따라 데이터를 두 group으로 나눔:
$$
R_1(j,s)=\{\,i: x_{i,j}\le s\,\},\qquad
R_2(j,s)=\{\,i: x_{i,j}> s\,\}.
$$

Tree의 첫 분할은 아래 최적화 문제를 minimize하는 쌍 $(j,s)$를 계산하여 선택:
$$
\min_{j,s}\;\frac{1}{n}\!\left(
\sum_{i\in R_1(j,s)} \big[c_i^\top w^*(c_1)-z^*(c_i)\big]
\;+\;
\sum_{i\in R_2(j,s)} \big[c_i^\top w^*(c_2)-z^*(c_i)\big]
\right),
$$
두 그룹 $R_1,R_2$ 각각에 대해 평균 cost $c_1, c_2$를 기준으로 최적 결정 $w$를 내린 후  
그것이 실제 각 $c_i$의 최적 결정과 얼마나 차이나는지를 계산한 다음 합산한 값을 최소화하는 식.  
Theorem 1을 활용하면 목적함수 값은 (1) 분할기준에 따라 학습 데이터를 나누고, (2) 각 leaf에서 평균 cost 벡터와 이에 대한 최적 결정 벡터를 구하고, (3) 해당 결정으로 인해 생기는 SPO loss를 계산하고 SPO loss를 모두 합산한 뒤 $n$으로 나누어 평균 손실을 구함.  
이는 빠르게 수렴하지만 최적은 보장하지 않으므로 **MILP**를 활용.  
SPO forests — SPO tree 를 여러 개 앙상블 하여 만듦

---

## 12) Experiments

Noisy Shortest Path  

<img width="1107" height="781" alt="image" src="https://github.com/user-attachments/assets/8eb888ae-0ba7-4e03-a39c-769893d3805b" />

News Article Recommendation

<img width="940" height="650" alt="image" src="https://github.com/user-attachments/assets/2e00e14d-edff-4cba-af07-a266bce5b2a5" />
