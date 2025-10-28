---
title: "TWO-STAGE PREDICT+OPTIMIZE FOR MIXED INTEGER LINEAR PROGRAMS WITH UNKNOWN PARAMETERS IN CONSTRAINTS (NeurIPS 2023)"
date: 2025-10-28 12:00:00 +0900
categories: [paper_review, decision_focused_learning]
tags: [two-stage, predict-then-optimize, MILP, constraints, regret, interior-point, neurips2023]
math: true
---

# Paper Review — *TWO-STAGE PREDICT+OPTIMIZE FOR MIXED INTEGER LINEAR PROGRAMS WITH UNKNOWN PARAMETERS IN CONSTRAINTS* (NeurIPS 2023)

- **1저자:** Xinyi Hㅊ  
- **제목:** TWO-STAGE PREDICT+OPTIMIZE FOR MIXED INTEGER LINEAR PROGRAMS WITH  
UNKNOWN PARAMETERS IN CONSTRAINTS  
- **저널명:** NeurIPS  
- **년도:** 2023

---

## 1) High-Level Summary (3–5 sentences)

본 논문은 제약조건에 unknown parameter가 포함된 MILP 문제를 다루기 위해, Two-Stage  
framework 제안 앞선 논문은 packing/covering 구조의 LP 문제에 한정되었으며, 예측된  
파라미터를 이용해 풀고, 실제 파라미터가 나중에 드러나도 수정 없이 그대로 사용한다는  
점에서 단일 단계 최적화 후 regret 기반 학습 구조. 반면, 본 논문은 Stage 1에서 soft하게 해를  
선택한 후, Stage 2에서 실제 파라미터가 드러난 뒤 그 해를 수정할 수 있도록 penalty를  
부과하며 다시 MILP를 최적화.  
이로 인해 기존 논문에서 어려웠던 미분 불가능, MILP 처리 불가 문제, 제약 조건 내 파라미터  
예측의 한계를 해결하며, LP를 넘어 MILP까지 적용할 수 있는 범용적이고 end-to-end 학습  
가능한 구조를 실현

---

## 2) 논문이 풀고자 하는 문제

제약 조건의 일부 파라미터가 사전에 알 수 없고, 예측에 기반해 결정을 내려야 하는 MILP  
문제에서  
기존 SPO 방식은 단일 단계 + correction function만을 활용하여 한계 존재재  
(1) 미분 불가능  
(2) MILP에서는 작동하지 않음,  
(3) correction function이 설계 자체가 어려움

---

## 3) 어떤 방법으로 이 문제를 해결

Two-Stage Optimization을 도입:  
- **Stage 1:** 예측된 파라미터 $\hat{\theta}$으로 먼저 문제를 풀어 soft commitment $x_1$ 결정  
- **Stage 2:** 실제 파라미터 $\theta$가 관측되면, $x_1$에서 크게 벗어나지 않도록 하는 penalty를 포함하여  
  문제를 다시 최적화해 $x_2$ 결정  

학습 시에는 post-hoc regret을 손실함수로 사용  
MILP의 미분 불가한 문제를 해결하기 위해 Mandi방식의 interior-point 완화 기법 사용  
를 통해 surrogate loss를 정의하고, 미분 가능하게 만들어 end-to-end 학습 가능

---

## 4) application area

MILP 상황

---

## 5) 기여(contribution)

새로운 Two-Stage Predict+Optimize 구조 제안 — soft commitment + penalty 기반 최적화

---

## 6) 논문의 좋은점 / 한계 / 개선 아이디어

- **논문의 좋은점:**  
  기존 SPO의 구조적 한계를 근본적으로 보완하며 MILP까지 확장
- **한계:**  
  두 단계 모두 MILP로 정의되어야 하므로, LP 이외의 복잡한 구조에는 여전히 적용 어려움
- **개선 아이디어:**  

——————————————————————————————————————————  
—————

## 7) 정리

이전의 제약조건에 unknown parameter가 있는 상황의 연장  
앞서 논문에서는 제약 조건 내의 unknown parameter를 다루기 위해 예측된 해를 나주에  
틀려도 soft commitment로 간주함(나중에 틀려도 고칠 수 있음)  
이를 위해 infeasible한 해를 feasible하게 바꾸는 correction function이 필요 → 하지만 이러한  
방법은 비정형적이여서 일반성이 떨어짐  
또한 여러가지 방법이 있을 수 있는데 하나의 방법을 선택해야함 → 다양한 최적화 문제에 적용  
가능한 일반적인 알고리즘 framework 만들기 어려움  
따라서 본 논문은 제약조건 속의 unknown parameter가 있는 상황에서 다양한 MILP에 쓸 수  
있는 일반적인 framework 제시  
기여: Two-Stage Predict+Optimize 라는 새로운 framework 제시, correction function 불필요,  
only penalty function만 사용 - 수정 절차 자체가 아니라 수정의 비용만 정의하면 됨  
Penalty function만으로 correction process를 정의하는 데 충분함 이 과정을 2stage optimization  
problem으로 수식화하여 최초 예측된 해와 penalty function 모두 고려  
모든 MILP에 적용 가능한 일반적인 end-to-end 학습 알고리즘 추가 제안

---

## 8) 수식/정의 (원문 내용 그대로, 수식만 LaTeX 표기)

parameterized Optimization problem $P(\theta)$:
$$
x^*(\theta) \;=\; \arg\min_{x}\ \mathrm{obj}(x,\theta)
\quad \text{s.t. }\ C(x,\theta).
$$

각 파라미터는 $m$개의 feature로부터 예측됨  
$n$개의 data를 기반으로 학습된 ML model에 의해 수행되며 $\theta$ 예측, 이를 통해 예측 해 $x^*(\theta)$ 얻음  
이때 unknown parameter가 s.t.에 나타나는 경우 prediction solution이 feasible region에  
존재하지 않을 수 있음

이전 논문은 Post-hoc regret을 loss function으로 설정하고 이를통해 model $f$를 학습하였음  
이때의 단점:  
문제 상황에 맞는 penalty, correction function을 수학적으로 명시해야하고 그 함수가 미분가능  
해야만 사용 가능  
하나의 미분 가능한 correction function이 필요하다는 이유로 일반화 가능한 correction  
function을 만들기 위해 packing , covering LP 문제에밖에 사용 불가, MILP같은 현실 문제에  
적용 어려움  
예측 해가 실행 불가능한 경우에만 수정을 고려하지만, 실제로는 실행은 가능하지만 성능이  
떨어지는 경우에도 수정이 유효한 문제들이 존재 - 이를 다루지 않음

본 논문은 MILP에도 적용 가능한 framework 제안  
Correction function 사용하지 않음 - correction process 자체를 새로운 optimization problem으로  
간주함  
예측 해와 실제 파라미터가 주어졌을 때 기존 목적 함수에 penalty를 추가하여 새로운 최적화  
문제를 다시 풂

세 가지 주요 장점 요약:  
1. Practitioner가 correction function을 설계할 필요가 없다.  
   → 더 이상 사람이 수정 규칙을 일일이 작성할 필요가 없으며, 프레임워크의 일반성이 높아짐.  
2. feasible 해조차 수정 가능하다 (비효율적이면 penalty 내고 바꿈).  
   → 단순히 실행 가능하다는 이유만으로 해를 고정할 필요 없음.  
3. 같은 penalty 함수 하에서, 본 방식의 성능은 correction function 방식보다 항상 좋거나 같다.  
   → 이론적으로도 성능이 보장됨

---

## 9) Two-Stage Optimization (수식 표기)

**Stage 1** — feature들로부터 $\theta$ 예측, 이를 통해 stage 1 optimization을 풀고 solution $x_1^*$을 얻음.  
이는 **soft commitment**로 간주되며 stage 2에서 penalty를 내고 수정 가능.

**Stage 2** — 실제 파라미터 $\theta$가 드러나며, 기존 목적함수에 penalty 항이 추가된 문제를 다시 풂:
$$
x_2^*(\theta)
\;=\;
\arg\min_{x}\ \mathrm{obj}(x,\theta)\;+\;\mathrm{Pen}\!\big(x_1^* \to x,\ \theta\big)
\quad \text{s.t. }\ C(x,\theta).
$$

Stage 2 해 $x_2^*$는 regret 함수를 기준으로 평가:
$$
\mathrm{PReg}(\theta,\theta)
\;=\;
\mathrm{obj}(x_2^*,\theta)\;+\;\mathrm{Pen}\!\big(x_1^* \to x_2^*,\ \theta\big)\;-\;\mathrm{obj}\!\big(x^*(\theta),\theta\big).
$$
