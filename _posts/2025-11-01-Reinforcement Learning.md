---
title: "[RL] 강화학습 정리"
date: 2025-11-01
categories: [textbook-summary,Reinforcement-learning]
tags: [RL, bandit, MDP, Bellman]
math: true
permalink: /rl/multi-armed-bandit-mdp/
---


## 1. 다중 선택 문제 (k-armed bandit problem)

**정의**  
k개의 서로 다른 option(action) 중 하나를 반복적으로 선택하여 일정 기간 동안 얻는 **보상의 총합의 기대값을 최대화**하는 문제.

$$
q_*(a)\mathrel{\dot{=}} \mathbb{E}[R_t|A_t=a]
$$

- $A_t$: 시간 단계 $t$에서 선택된 action  
- $R_t$: $t$에서 선택된 action에 대한 보상  
- $q_*(a)$: 임의의 action $a$가 선택되었을 때 얻는 평균 보상(value)  
- $Q_t(a)$: 시간 $t$에서 추정된 action $a$의 value  
  → $Q_t(a)$가 $q_*(a)$에 가까울수록 정확한 추정

---

### Greedy / $\epsilon$-Greedy 전략

- **Greedy action**: 각 $t$마다 추정 value가 가장 높은 action 선택  
  → *exploiting* (현재까지의 정보 활용)  
- 다른 action을 시도하는 것은 *exploring* (정보 탐색)  
- → 두 전략의 **균형(exploration vs. exploitation trade-off)** 이 중요

**행동 가치 방법(action-value method)**  
action의 value를 추정하고, 그 추정값으로부터 action을 선택하는 방법.

$$
Q_t(a) \mathrel{\dot{=}} 
\frac{
\text{시각 }t \text{ 이전에 취해진 action }a\text{에 대한 보상의 합}
}{
\text{시각 }t\text{ 이전에 action }a\text{를 취한 횟수}
}
= 
\frac{
\sum_{i=1}^{t-1} R_i\cdot\mathbb{1}_{A_i=a}
}{
\sum_{i=1}^{t-1} \mathbb{1}_{A_i=a}
}
$$

즉, **sample-average 방법**이다.

---

### Action 선택 규칙

1. **Greedy 방식**
   $$
   A_t = \underset{a}{\arg\max}\; Q_t(a)
   $$
2. **$\epsilon$-greedy 방식**  
   - 확률 $\epsilon$로 무작위 action 선택  
   - 확률 $1-\epsilon$로 greedy action 선택

---

### 10중 선택 테스트

- 행동: 10개 $(a=1,\dots,10)$  
- 각 $q_*(a) \sim \mathcal{N}(0,1^2)$  
- 실제 보상값: $R_t \sim \mathcal{N}(q_*(a),1^2)$

---

### 점증적 구현 (Incremental Implementation)

$$
Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n]
$$

새로운 추정값 = 이전 추정값 + **step-size(학습률)** × [목표값 − 이전 추정값]

---

### 비정상 환경 (Nonstationary Problem)

Stationary 문제는 보상 분포가 시간이 지나도 변하지 않지만,  
Nonstationary 환경에서는 최신 보상에 더 큰 가중치를 줘야 함.

$$
Q_{n+1} = Q_n + \alpha [R_n - Q_n]
$$

이를 전개하면:

$$
Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n} \alpha(1-\alpha)^{n-i}R_i
$$

즉, **기하급수적 최신 가중 평균 (Exponential Recency-Weighted Average)**

---

### 긍정적 초기값 (Optimistic Initial Values)

모든 방법은 초기 추정값 $Q_1(a)$에 영향을 받음 → **bias 존재**

---

### 신뢰 상한 탐색 (Upper Confidence Bound, UCB)

$$
A_t = \underset{a}{\arg\max}\; \Big[ Q_t(a) + c\sqrt{\frac{\ln{t}}{N_t(a)}} \Big]
$$

추정값의 불확실성(탐색 필요성)을 반영하여 **탐욕적 행동 + 신뢰 상한**을 함께 고려.

---

### 경사도 다중 선택 (Gradient Bandit Algorithm)

소프트맥스 분포 기반으로 행동 선택:

$$
\text{Pr}\{A_t=a\} = 
\frac{e^{H_t(a)}}{\sum_{b=1}^{k} e^{H_t(b)}} 
\mathrel{\dot{=}} \pi_t(a)
$$

업데이트 규칙:

$$
H_{t+1}(A_t) = H_t(A_t) + \alpha(R_t-\bar{R_t})(1-\pi_t(A_t))
$$

$$
H_{t+1}(a) = H_t(a) - \alpha(R_t-\bar{R_t})\pi_t(a) \quad \forall a \neq A_t
$$

---

## 2. 유한 마르코프 결정 과정 (Finite Markov Decision Process, MDP)

**정의**  
어떤 action이 즉각적인 reward뿐 아니라 **미래의 state**에도 영향을 주어  
결국 장기적인 reward에 영향을 미치는 **연속적 의사결정 문제**.

즉, “지연된 보상(delayed reward)”을 포함.

---

### Agent–Environment Interface

- $S_t \in \mathcal{S}$ : 현재 상태  
- $A_t \in \mathcal{A}(s)$ : 선택한 행동  
- $R_{t+1} \in \mathcal{R}$ : 행동의 결과로 받은 보상  
- 다음 상태: $S_{t+1} \in \mathcal{S}$  

Trajectory:  
$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots$$

환경의 확률적 전이 모델:

$$
p(s',r|s,a) \mathrel{\dot{=}} \text{Pr}\{S_t=s', R_t=r | S_{t-1}=s, A_{t-1}=a\}
$$

---

### 보상 정의

1. **상태 전이 확률**
   $$
   p(s'|s,a) = \sum_{r\in\mathcal{R}} p(s',r|s,a)
   $$

2. **보상의 기댓값**
   $$
   r(s,a) = \sum_{r\in\mathcal{R}} r \sum_{s'\in\mathcal{S}} p(s',r|s,a)
   $$

3. **특정 전이에 대한 보상**
   $$
   r(s,a,s') = \sum_{r\in\mathcal{R}} r \frac{p(s',r|s,a)}{p(s'|s,a)}
   $$

---

### 목표와 보상

**보상 가설 (Reward Hypothesis)**  
에이전트의 목표는 **받는 보상의 총합을 최대화**하는 것.

---

### 보상과 에피소드

- **Episodic task**  
  $$
  G_t = R_{t+1} + R_{t+2} + \dots + R_T
  $$  
  (종단 상태 T 존재)

- **Continuing task**  
  $$
  G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
  $$  
  (할인율 $\gamma$ 적용)

---

### 통합 표기법

$$
G_t = \sum_{k=t+1}^T \gamma^{k-t-1} R_k
$$

---

### 정책과 가치함수

- **정책(policy)**:  
  $$
  \pi(a|s) = \text{Pr}(A_t=a|S_t=s)
  $$

- **상태가치함수(state-value function)**:  
  $$
  v_\pi(s) = \mathbb{E}_\pi [\, G_t | S_t = s \,]
  = \mathbb{E}_\pi [\, R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s \,]
  $$

- **행동가치함수(action-value function)**:  
  $$
  q_\pi(s,a) = \mathbb{E}_\pi [\, G_t | S_t = s, A_t = a \,]
  = \mathbb{E}_\pi [\, R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) | S_t = s, A_t = a \,]
  $$

---

### 벨만 방정식 (Bellman Equation)

$$
v_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[\,r + \gamma v_\pi(s')\,]
$$

---

### 최적 정책과 최적 가치함수

**강화학습의 목표**  
장기적으로 보상을 최대화하는 정책 $\pi_*$를 찾는 것.

$$
v_*(s) = \max_\pi v_\pi(s), \qquad
q_*(s,a) = \max_\pi q_\pi(s,a)
$$

---

### 최적 벨만 방정식

**상태 가치 형태**

$$
v_*(s) = \max_a \sum_{s',r} p(s',r|s,a)[\,r + \gamma v_*(s')\,]
$$

**행동 가치 형태**

$$
q_*(s,a) = \sum_{s',r} p(s',r|s,a)[\,r + \gamma \max_{a'} q_*(s',a')\,]
$$

---

