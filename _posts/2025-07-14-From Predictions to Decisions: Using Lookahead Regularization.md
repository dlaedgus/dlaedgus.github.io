---
title: "From Predictions to Decisions: Using Lookahead Regularization (NeurIPS 2020)"
date: 2025-07-14 12:00:00 +0900
categories: [paper_review, decision_focused_learning]
tags: [lookahead-regularization, predict-then-optimize, decision-quality, uncertainty, covariate-shift, neurips2020]
math: true
---

# Paper Review — *From Predictions to Decisions: Using Lookahead Regularization* (NeurIPS 2020)

- **1저자:** Nir Rosenfeld  
- **제목:** From Predictions to Decisions: Using Lookahead Regularization  
- **학회:** NeurIPS 2020

---

## High-Level Summary (3–5 sentences)

본 논문은 prediction model이 실제 사용자 decision에 영향을 미치는 상황을 고려하여, 단순한 정확도(MSE)뿐 아니라 decision quality까지 함께 고려하는 machine learning framework를 제안. 이를 위해 lookahead regularization이라는 새로운 regularization 기법을 도입하여, 모델이 미래 decision의 결과를 미리 고려하여 더 나은 방향으로 학습되도록 유도. 이 방법은 기존 statistical learning의 틀 안에 그대로 포함되며, synthetic data와 real-world data (wine, diabetes)를 활용한 세 가지 experiment를 통해 그 성능을 입증. 예측 정확도를 크게 해치지 않으면서도 사용자의 decision improvement를 의미 있게 높일 수 있음을 보여줌.

---

## 논문이 풀고자 하는 문제

기존의 supervised learning에서는 prediction accuracy만을 최적화하지만, 실제 환경에서는 그 prediction이 user decision에 영향을 미쳐 결과적으로 잘못된 decision을 초래할 수 있음. 이 논문은 prediction model이 사용자의 행동을 유도하는 상황에서 decision quality까지 고려하지 않으면 부적절하거나 위험한 결과가 나올 수 있다는 문제를 해결하고자 함.

---

## 어떤 방법으로 이 문제를 해결

Lookahead regularization: 예측 모델 학습 시, 단순히 MSE 최소화뿐 아니라 그 예측이 초래할 decision의 expected improvement를 함께 고려하는 새로운 loss를 도입. 이때 decision은 counterfactual outcome을 기반으로 가정되며, uncertainty-aware model $g$와 propensity model $h$을 활용하여 미래의 변화 방향을 구성. 학습 과정에서는 $x \to x'$로의 decision shift를 고려하여, 실제로 개선이 가능한 방향으로 예측이 이루어지도록 함.

---

## application area

의료 area, 마케팅 area, 정책 결정 지원 시스템

---

## 기여(contribution)

사용자 decision까지 고려하는 새로운 regularization 기법: lookahead regularization 제안. 이 방법이 기존 RMSE 기반 모델보다 decision improvement에 효과적임을 보이는 정량적 실험 분석 제공.

---

## 논문의 좋은점

기존 prediction accuracy 중심의 학습에서 벗어나, 실제 user decision까지 고려한 학습 구조 설계

---

## 한계

Uncertainty model $g$이나 propensity model $h$ 등 여러 보조 모델이 필요해 복잡도 및 모델링 비용이 큼.

---

## 개선 아이디어

이전 논문들에서 다루었던 deep neural network나 ML model과의 조합으로 확장성 확보.

---

## 정리

ML model은 의료, 금융, 제조업 등 사람들의 삶에 큰 영향을 미치는 분야로 사용↑.  
Model의 transparency의 요구 ↑ 로 인하여 model이 단순한 prediction을 넘어서 행동 변화까지 유도하는 dual role 가짐.  
즉, prediction에서 끝나는 것이 아니라 **prediction → behavior 유도 → 결과 변화**까지 연결.

따라서 본 논문에서는 predictive accuracy뿐만 아니라 그 예측이 유도하는 decision이 결과를 improve할 거라는 높은 확신을 갖춘 모델을 추구, 단순히 잘 예측 하는 게 아니라, 그 예측이 행동도 결과를 개선하는 지가 핵심.

그렇다면 good decision을 내리기 위해 어떤 degree of freedom을 사용해야하나? accuracy, decision quality 간의 trade-off가 존재한다면 이를 problem of model selection 로 변환할 수 있음, 예: prediction accuracy는 비슷하지만 coefficient가 달라서 유도하는 행동이 완전히 다른 모델들이 있음.

이러한 tradeoff를 해결하기 위해 lookahead regularization을 제안, 이 방식은 accuracy와 decision결과 개선을 함께 고려. 이 방식은 사용자가 prediction을 보고 어떻게 행동할지를 lookahead하고 그 행동이 outcome을 개선할 것이라는 high confidence가 없으면 모델에 penalty 부과.

decisions, which depend on the predictive model 은 initial distribution $p$와 다른 특성 distribution $p'$에서 작동.  
$p$: 데이터가 원래 따르던 분포 (훈련 데이터에서의 분포).  
$p'$: 예측 결과를 보고 사용자가 행동한 뒤, 변화된 특성들의 분포.

예: 원래는 심장병 위험이 높은 사람이 많았지만, 예측 결과에 따라 많은 사람이 운동을 시작하면 전체 분포는 바뀌고 위험군 ↓.  
→ 즉, 모델이 행동을 유도하고, 그 결과 데이터 분포 자체가 바뀌는 것 $p'$.

covariates $x$를 가진 개인은 model에 의해 유도된 decision을 통해 new covariates $x'$으로 바뀜.  
예: 원래 $x(\text{운동량}=0,\ \text{혈압}=\text{높음})$ → 운동 권유 → 운동 시작 → 결과: $x'(\text{운동량}=\text{많음},\ \text{혈압}=\text{낮음})$.  
즉, 행동이 특성 자체를 바꾸는 것이므로, 학습할 때와는 **다른 분포** 위에서 결과가 나옴.

사전에 지정된 신뢰 수준 $\tau$에 대해 인구 중 적어도 $\tau$만큼은 행동 이후 분포 $p'$에서의 결과가 이전 분포 $p$보다 개선되길 원함.

**technical challenge:** $p'$과 $p$의 차이가 크면 decision의 효과를 추정하는 데에 큰 uncertainty가 생김. 즉, 우리가 알고 있는 것은 $p$에서의 모델 성능뿐인데, 사용자가 행동을 바꾸면 완전히 다른 상황인 $p'$에서 결과가 나타남. 그런데 우리는 $p'$에 대해 알 수 없거나, 샘플이 적거나, 직접 관측하지 못함. → 그래서 모델이 유도한 행동이 진짜로 도움이 되는지 추정이 어려움.

**해결 방법:** lookahead regularization은 의사결정 결과 주변에 대한 **confidence interval**을 제공하는 uncertainty model 사용.  
예: “운동을 하세요”라는 유도 행동에 대해, 그 행동이 얼마나 효과적일지를 신뢰 구간으로 추정.  
이 uncertainty model은 **importance weighting** 기법을 사용해 입력 특성의 분포 변화 보정, 변화된 $p'$에 대해 CI를 정확히 추정하도록 학습.

입력 data의 density $p(x)$는 그래프의 왼쪽에 집중, $y$는 $x$에 대해 $f'(x)$로 주어짐.

(A) 결과 $y$를 더 좋게 만들고 싶은 User는 $x$에서 predictive 모델을 참고하여 어떻게 decision 내릴지 결정, 예측 모델 $f(x)$의 gradient를 따라 $x \to x'$로 이동 — “어떻게 해야 더 나은 결과를 얻을 수 있을까?” 모델을 통해 배움.

(B) 사용자가 모델을 믿고 행동했더니, $x'$가 모델이 train 되어있지 않은 영역(훈련데이터 부족)으로 위치. 이 영역에서는 $f$가 훈련 데이터의 제약을 받지 않으므로 잘못된 예측을 할 수 있음.

(C) 이런 decision 이후의 결과에 대한 uncertainty를 고려하기 위해, $x'$에 대해 CI를 예측하는 interval 모델 $g(x') = [\ell', u']$를 학습. $f$와 독립적으로 변화된 분포 $p'$를 목표로 하며, $y'$가 이 구간 안에 있을 확률이 최소한 $\tau$ 이상이 되도록 보장.

(D) 결과 $y$가 CI 하한 $\ell'$보다 높으면 model은 이를 penalty로 간주하고 이로 인해 predictive model은 이전 상태보다 결과가 더 나쁘지 않도록 보장되게 학습.

---

## Method

$x \in \mathcal{X} = \mathbb{R}^d$: 특성 vector(환자, 고객, 와인 빈티지).  
$y \in \mathbb{R}$: label(결과의 품질, 클수록 좋은 결과).

관측 dataset $S = \{(x_i, y_i)\}_{i=1}^m$, 특성 $x$와 결과 $y$에 대한 공동 분포 $p(x,y)$에서 추출된 iid 샘플.  
특성 $x$에 대한 marginal distribution은 $p(x)$.

$f:\mathcal{X}\to\mathbb{R}$ 은 dataset $S$에 대해 학습된 model, 이 model $f$는 두 가지 방식으로 사용:  
1. **Prediction:** $p(x)$로부터 추출된 객체 $x$에 대해, 그 결과 $y$를 예측하는 것.  
2. **Decision:** 더 나은 결과를 얻기 위해, 특성 $x$를 변경함으로써 행동을 취하는 것.

사용자 행동이 각 $x$를 새로운 $x' \in \mathcal{X}$로 매핑한다고 가정.  
$x'$는 사용자의 decision 또는 action으로 간주, 그 결과로 발생하는 출력값은 $y' \in \mathbb{R}$로 표현.  
$x' = d(x)$로 정의, $d:\mathcal{X}\to\mathcal{X}$ 를 decision function이라고 부름.

사용자가 decision을 내릴 때 model $f$를 참고한다고 가정:  
1) 사용자가 예측값 자체에만 관심이 있음.  
2) 모델의 결정이 결과에 미치는 영향을 대리적으로 보여주는 것이라 생각하기 때문.

사용자가 모델 $f$의 기울기 방향으로 한걸음 이동한다고 가정.  
mutable, immutable features를 $\Gamma:\mathcal{X}\to\{0,1\}$ 를 통해 구분.

**Assumption 1 (user decision model).** 사용자가 gradient 방향으로 이동할 때,  
$$
x' = x + \eta\,\Gamma\!\big(\nabla f(x)\big).
$$

Assumption 1에 의하여 사용자 decision은 특정한 decision function을 유도, 이로부터 특성 공간 $\mathcal{X}$ 위의 target distribution을 얻음. 이 분포: $p'(x)$. 이로 인해 decision이 새로운 결과를 유도하면서 $(x', y') \sim p'(x,y)$ 생성.

**Assumption 2.** Decision function $d$가 무엇이든 **조건부 분포** $p(y\mid x)$는 고정되어 있고, 새로운 결합 분포는  
$$
p'(x',y) \;=\; p(y\mid x')\,p'(x').
$$

---

### Learning objective

Predictive와 decision 간의 균형.  
학습 결과 생성되는 model이 $x\sim p(x)$로부터 온 입력에 대해 예측값 $y = f(x)$가 실제 정답 $y$와 매우 잘 일치하기를 원함 **+** 학습된 model이 파생된 decision $x'$을 유도하되 이 $x'$이 속하는 **counterfactual 분포** $p'(x)$에서의 결과 $y'$가 원래 결과 $y$보다 향상되길 원함.

이 두 objective 간의 균형을 맞추기 위해 predictive loss function에 **좋은 decision을 장려하는 정규화 항** 추가.  
이때 사용자가 $f$를 보고 $x'$를 선택하고 그로 인해 $y'$가 발생하므로, $y'$의 실험값은 학습 시점에는 관찰 불가(예측이 아니라 미래 행동의 결과). 이로 인해 학습시 $y' \ge y$ 형태의 단순한 제약조건을 거는 것이 어려움. 대신 $x'$가 어떤 분포 $p'(x)$를 따른다고 보고 그로부터 나올 결과 $y'$의 조건부 분포 $p(y\mid x')$를 활용해 **기대값으로 정규화**.

**기대 향상치:** $\mu = \mathbb{E}_{y'\sim p(\,\cdot\mid x')}[y']$.  
$\mu - y$가 **작거나 음수**일 경우 이를 **패널티**로 부여.

이 식의 2가지 문제:  
1) $f$가 overfit → 이를 해결하기 위해 **두 개의 분리된 모델** 사용.  
2) 실제 응용에서 개인별로 결과가 **평균적으로만** 향상된다는 보장은 부족.

기존 방식은 단순히 **기대값** 기준으로 improvement가 있는지 봤다면, 본 논문은 **일정 수준 이상의 개선이 이루어질 확률이 충분히 높은가?** 를 기준으로 판단. MSE를 최소화하면서 **개선 확률이 $\tau$보다 낮은 경우**에 **패널티** 부여. 이때 $y'$는 $x'$에 대한 결과이므로 확률을 직접 계산할 수 없음 → **확률 추정** 필요.

---

### Estimating uncertainty

lookahead regularization은 미래를 내다보며 결정이 안전한지 판단하는 규제 항이므로 이때 쓰이는 **불확실성 추정**이 핵심 역할. 그런데 train data 분포 밖에 있는 $x'$에 대해서는 추정이 어려움.

다행히, 주어진 $f$에 대해 $p'(x)$는 **Assumption 1**에 의해 알려져 있으며, **covariate transform**을 사용해 **샘플 집합 $S'$**를 만들 수 있음. $S'$에 대해 label이 없더라도 $g$를 추정하는 문제는 **covariate shift** 하에서 학습하는 문제가 됨. Covariate shift 하에서의 학습 방법은 많지만, **importance weighting**으로 요약 가능.

---

## Algorithm

(개념적 단계 요약: User shift 생성 → Uncertainty/Propensity 추정 → Lookahead 패널티 계산 → 파라미터 업데이트)

---

## Experiments

(논문 본문의 실험: synthetic + real-world(wine, diabetes); 예측 정확도 유지하면서 decision 지표 개선)
