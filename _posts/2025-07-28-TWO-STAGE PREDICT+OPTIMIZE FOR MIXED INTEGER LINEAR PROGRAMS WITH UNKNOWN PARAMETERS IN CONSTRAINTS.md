---
title: "Two-Stage Predict+Optimize for MILPs with Unknown Parameters in Constraints (NeurIPS 2023)"
date: 2025-10-28 12:00:00 +0900
categories: [paper_review, decision_focused_learning]
tags: [predict-then-optimize, two-stage, post-hoc-regret, interior-point, MILP, constraints, decision-focused-learning, neurips2023]
math: true
---

# Paper Review — *Two-Stage Predict+Optimize for Mixed Integer Linear Programs with Unknown Parameters in Constraints* (NeurIPS 2023)

- **1저자:** Xinyi Hu  
- **공저자:** Jasper C. H. Lee, Jimmy H. M. Lee  
- **제목:** Two-Stage Predict+Optimize for Mixed Integer Linear Programs with Unknown Parameters in Constraints  
- **학회:** NeurIPS  
- **년도:** 2023

---

## 1) High-Level Summary (3–5 sentences)

본 논문은 **제약식에 미지 파라미터가 등장**하는 상황에서, 예측(ML)과 최적화(OP)를 **두 단계**로 통합하는 **Two-Stage Predict+Optimize** 프레임워크를 제안합니다. 핵심은 1단계(Stage 1)에서 예측 파라미터로 문제를 풀어 **soft commitment 해** $x_1^*$을 만들고, 2단계(Stage 2)에서 **진짜 파라미터가 공개된 뒤** 원문제에 **패널티 함수**를 더해 **수정 최적화**로 $x_2^*$를 산출하는 것입니다. 이로써 **수정 규칙(correction function)을 설계할 필요가 없고**, LP에 한정되지 않으며 **모든 MILP**로 일반화됩니다. 학습은 **post-hoc regret**(2단계 목적 + 패널티 − 진최적 목적)을 손실로 사용하고, **내부점법(로그 장벽)**을 활용해 **미분 가능한 대리 손실**을 구성, **엔드투엔드**로 신경망을 학습합니다. 합금 생산(covering LP), 0–1 배낭(MILP), 간호사 스케줄링(MILP)에서 SOTA 및 고전 기법 대비 **일관된 regret 개선**을 보입니다. :contentReference[oaicite:0]{index=0}

---

## 2) 문제 설정: 왜 ‘제약의 미지수’가 어렵나

전통적 Predict-then-Optimize는 **목적함수 계수만** 미지일 때(예: 비용/수요) 잘 작동합니다. 그러나 **제약식에 미지수**가 있으면, 예측 파라미터로 얻은 해 $\hat{x}$가 **진짜 파라미터** $\theta$ 하에서는 **실행 불가능**할 수 있습니다. 기존의 선행연구(예: **Hu et al., 2022**)는 **수정 규칙 + 패널티**를 요구했지만,  
- (i) 수정 규칙 설계가 **임의적**이고 **미분 가능성** 가정이 필요하며,  
- (ii) **packing/covering LP**로 범위를 제한,  
- (iii) **실행 가능하지만 나쁜 해**도 수정하고 싶어도 다루기 어렵다는 한계가 있었습니다. :contentReference[oaicite:1]{index=1}

---

## 3) 제안 방법: Two-Stage Predict+Optimize (개념)

**핵심 아이디어:** 수정 규칙 자체를 **또 하나의 최적화**로 본다.

- **Stage 1 (soft commitment):**  
  예측 $\hat{\theta}$로 파라미터화된 Para-OP을 풀어
  $$
  x_1^* \;=\; \arg\min_x\ \mathrm{obj}(x,\hat{\theta})\quad\text{s.t. } C(x,\hat{\theta}).
  $$
  $x_1^*$는 **변경 가능**한 ‘약한 약속’입니다.

- **Stage 2 (정정 최적화 with penalty):**  
  진짜 $\theta$ 공개 후, 원문제에 **패널티** $\mathrm{Pen}(x_1^*\!\to x,\theta)$를 더해
  $$
  x_2^*(\theta) \;=\; \arg\min_x\ \mathrm{obj}(x,\theta)\;+\;\mathrm{Pen}(x_1^*\!\to x,\theta)
  \quad\text{s.t. } C(x,\theta).
  $$
  패널티가 유한하면 **실행 가능해도** 더 좋은 해로 **수정 허용**됩니다.

- **평가 (post-hoc regret):**
  $$
  \mathrm{PReg}(\hat{\theta},\theta)\;=\;
  \mathrm{obj}(x_2^*,\theta)\;+\;\mathrm{Pen}(x_1^*\!\to x_2^*,\theta)\;-\;\mathrm{obj}(x^*(\theta),\theta).
  $$
  여기서 $x^*(\theta)$는 진짜 파라미터의 진최적 해. **동일 패널티**를 썼을 때, 본 프레임워크는 어떤 수정 규칙 접근보다 **항상 손해가 크지 않다**(Proposition A.1). :contentReference[oaicite:2]{index=2}

---

## 4) 학습: 미분 가능한 대리 손실과 그래디언트

학습 목표는 평균 post-hoc regret을 최소화하는 예측함수 $f$ (NN 등)입니다:
$$
f^* \;=\; \arg\min_{f}\ \frac{1}{n}\sum_{i=1}^n \mathrm{PReg}\big(f(A_i),\theta_i\big),
$$
여기서 $A_i$는 각 $\theta_i$의 feature. **체인룰**에 따라 가중치 $w_e$에 대한 미분은
$$
\frac{d\,\mathrm{PReg}(\hat{\theta},\theta)}{dw_e}=
\underbrace{\frac{\partial \mathrm{PReg}}{\partial x_2^*}}_{\text{2단계 목적+패널티의 $x_2^*$ 민감도}}
\underbrace{\frac{\partial x_2^*}{\partial x_1^*}}_{\text{2단계 해의 1단계 해 민감도}}
\underbrace{\frac{\partial x_1^*}{\partial \hat{\theta}}}_{\text{1단계 해의 예측 민감도}}
\underbrace{\frac{\partial \hat{\theta}}{\partial w_e}}_{\text{표준 역전파}}
\;+\;
\underbrace{\frac{\partial \mathrm{PReg}}{\partial x_1^*}}_{\text{패널티를 통한 $x_1^*$ 영향}}
\underbrace{\frac{\partial x_1^*}{\partial \hat{\theta}}}_{}
\underbrace{\frac{\partial \hat{\theta}}{\partial w_e}}_{}.
$$

문제: $x_1^*,x_2^*$는 **MILP 최적해** → 작은 섭동에 **불연속/영미분** 가능.  
해결: Mandi & Guns(2020)식 **내부점(로그장벽) 완화**로 **LP 완화문**을 풀어 **KKT**를 미분(암시함수정리) → **유용한 그래디언트**를 얻는 **대리 손실** $\tilde{\mathrm{PReg}}$를 구성합니다:
- 표준 MILP
  $$
  \min_x\ c^\top x\quad \text{s.t.}\ Ax=b,\ Gx\ge h,\ x\ge0,\ x_S\in\mathbb{Z}
  $$
  를 다음 **장벽 완화**로 대체:
  $$
  \min_{x,s}\; c^\top x\;-\;\mu\sum_i\ln x_i\;-\;\mu\sum_i\ln s_i\quad
  \text{s.t.}\ Ax=b,\ Gx-s=h,
  $$
  여기서 $s\ge0$은 slack, $\mu\!\ge\!0$은 장벽 파라미터(감소 경로의 종료값 사용).  
- 이 완화의 최적해 $(x^*,s^*)$는 **슬레이터 조건**을 만족 → **KKT 미분**으로  
  $\partial x/\partial c,\partial A,\partial b,\partial G,\partial h$를 계산 가능.  
- Stage 1/2 모두에 적용하여 $\tilde{x}_1,\tilde{x}_2$로 **미분 가능한** $\tilde{\mathrm{PReg}}$ 학습. :contentReference[oaicite:3]{index=3}

> **장점 요약**  
> (i) **수정 규칙이 불필요** → 임의성과 미분가능성 제약 해소.  
> (ii) **실행 가능해도 수정 가능**(패널티가 유한한 한).  
> (iii) **MILP 전반**으로 확장(목적·제약의 임의 위치에 미지수 가능).  
> (iv) SOTA 대비 **regret 우수** + **학습 속도**도 우수한 경우 보고. :contentReference[oaicite:4]{index=4}

---

## 5) 모델링 레시피: Soft/Hard/Recourse

논문 부록 A.1은 Two-Stage 관점에서 다양한 의사결정 변수를 **깨끗하게** 모델링하는 방법을 제공합니다. :contentReference[oaicite:5]{index=5}

- **Soft commitment 변수**: 1단계에서 정하지만 2단계에서 **유한 패널티**로 변경 가능  
  → 패널티 $\mathrm{Pen}(x_1\!\to x_2,\theta)$로 비용을 모델링.
- **Hard commitment 변수**: 2단계에서 절대 변경 불가  
  → 패널티에 $\infty\cdot\mathbf{1}[x_{1}\ne x_{2}]$ 항을 추가.
- **Recourse 변수**: 2단계에서만 등장/자유롭게 조정  
  → 목적함수에 해당 비용을 **직접 산입**, 패널티는 0.

> **예시 1 (Soft): 재고주문/창고용량 불확실)**  
> 1단계: 예측 용량으로 주문 $x_1^*$, 2단계: 실제 용량 공개 후 **추가/감액**에 **수수료 패널티** 반영하여 $x_2^*$.  
> **예시 2 (Hard+Recourse): 설비선정/초과근무)**  
> 설비 가동 여부 $x$는 hard, 초과근무량 $\sigma$는 recourse. 2단계에서 $x$는 불변(무한 패널티), $\sigma$만 조정.

> **교정 불가(진짜 hard world)?**  
> 패널티를 **$\infty$에 가깝게** 잡으면 학습이 **보수적 예측**을 학습하여 1단계 해의 **2단계 실행 가능성**을 높입니다(부록 A.2, Table 6 경향). :contentReference[oaicite:6]{index=6}

---

## 6) 수식 모음 (블로그용 핵심만)

- **파라미터화된 OP** $P(\theta)$:
  $$
  x^*(\theta)\;=\;\arg\min_x\ \mathrm{obj}(x,\theta)\quad\text{s.t. } C(x,\theta).
  $$

- **Two-Stage 최적화**  
  Stage 1:
  $$
  x_1^*=\arg\min_x\ \mathrm{obj}(x,\hat{\theta})\quad\text{s.t. } C(x,\hat{\theta})
  $$
  Stage 2:
  $$
  x_2^*=\arg\min_x\ \mathrm{obj}(x,\theta)+\mathrm{Pen}(x_1^*\!\to x,\theta)\quad\text{s.t. } C(x,\theta)
  $$

- **Post-hoc regret(평가/학습 손실)**:
  $$
  \mathrm{PReg}(\hat{\theta},\theta)=\mathrm{obj}(x_2^*,\theta)+\mathrm{Pen}(x_1^*\!\to x_2^*,\theta)-\mathrm{obj}(x^*(\theta),\theta).
  $$

- **내부점 완화(대리 손실용)**:
  $$
  \min_{x,s}\ c^\top x-\mu\!\!\sum_i\ln x_i-\mu\!\!\sum_i\ln s_i\quad\text{s.t. }Ax=b,\ Gx-s=h.
  $$

---

## 7) 알고리즘(학습 절차 요약)

1. **예측**: NN이 feature $A$로부터 $\hat{\theta}$ 산출.  
2. **Stage 1 완화**: 내부점(로그장벽)으로 완화문을 풀어 $\tilde{x}_1$ 획득.  
3. **Stage 2 완화**: 진짜 $\theta$ 공개 후, 패널티 포함 완화문으로 $\tilde{x}_2$ 획득.  
4. **대리 손실** $\tilde{\mathrm{PReg}}$ 계산(2단계 목적+패널티 − 진최적 목적).  
5. **KKT 미분 + 역전파**로 $\partial \tilde{\mathrm{PReg}}/\partial w$ 계산, 파라미터 업데이트.  
6. 테스트 시엔 **진짜 Two-Stage**를 그대로 수행해 **post-hoc regret**으로 평가. :contentReference[oaicite:7]{index=7}

---

## 8) 실험: 벤치마크/설정/결과

**벤치마크 세 가지**  
- **합금 생산**(covering LP, brass/titanium) — 미지: 성분 농도(파라미터당 4096 feature), 패널티 계수 $\sigma$는 공급사별로 샘플.  
- **0–1 배낭**(MILP) — 미지: 아이템 가격·크기(각각 4096 feature), 용량 100/150/200/250, 패널티 스케일 다변.  
- **간호사 스케줄링**(MILP) — 미지: 환자 로드(shift별), feature 8개, 15명×7일×3교대, 선호도 기반 목적. :contentReference[oaicite:8]{index=8}

**비교 기법**  
- 제안법 **2S**(우리 방법), **IntOpt-C**(Hu et al., correction 기반), **Ridge/k-NN/CART/RF/NN**(고전), **CombOptNet**(제약 학습, 0–1 배낭에서만 코드 사용).  
- 모든 고전 기법은 **평가 시** Two-Stage의 **Stage 2 수정 최적화**를 수행해 공정 비교. :contentReference[oaicite:9]{index=9}

**요지**  
- **합금 생산**: 우리 프레임워크(2S+Two-Stage)는 Hu et al. 프레임워크(IntOpt-C+correction) 대비 **6–36% 작은 mean regret**(brass), **7–30%**(titanium)를 달성. 동일 Two-Stage 평가에서도 **2S가 일관된 1등**(IntOpt-C보다 1–5%p 더 작음), 고전 대비 격차는 훨씬 큼.  
- **0–1 배낭**: IntOpt-C 적용 불가(제약·MILP). **2S가 CombOptNet 및 고전 전부 압도**, 용량이 클수록 모든 방법의 regret이 줄지만 **2S의 상대 우위는 오히려 커짐**.  
- **NSP**: IntOpt-C 부적용(MILP). **2S가 모든 패널티 스케일에서 최저 regret**, 고전 대비 **7–62%+ 개선**(패널티가 커질수록 격차 확대).  
- **시간(부록 H)**: 일부 고전이 더 빠르나 regret 열세. 합금에선 IntOpt-C가 약간 빠르지만 비슷한 오더, 배낭에선 **2S가 더 빠름**. :contentReference[oaicite:10]{index=10}

---

## 9) 강점·한계·실전 팁

**강점**  
- 수정 규칙 **불필요**, **MILP 전반** 적용, feasible라도 **합리적 수정** 가능, **이론적 우월성**(고정 패널티 하 correction-based 이상).  

**한계**  
- 두 단계 모두 **MILP로 표현**되어야 함(선형 목적/제약). 비선형 패널티는 도입이 제한적이나, **절대값** 등은 표준 선형화로 처리가능. 내부점 미분을 **일반 미분 가능 목적**으로 확장 가능성 언급. :contentReference[oaicite:11]{index=11}

**실전 팁**  
- **패널티 설계**가 핵심: 실제 비즈니스의 수정 비용(추가주문/감액/폐기/초과근무 등)과 일치시키기.  
- **수정 불가 환경**은 패널티를 **매우 크게** → 보수적 예측 유도(1단계 실행 가능성↑).  
- **학습/평가 일관성**: 학습은 대리 손실(완화), 평가는 실제 Two-Stage로.  
- **엔지니어링**: NN은 4~5층 FC(512 등), 내부점 종료 $\mu$는 솔버의 **cutoff 값** 사용, 하이퍼파라미터는 **CV**로. :contentReference[oaicite:12]{index=12}

---

## 10) 결론

Two-Stage Predict+Optimize는 **제약의 미지수**를 포함한 광범위한 의사결정 문제를 **단순·강력**하게 다루는 정석적 프레임워크로 제안됩니다. 수정 규칙의 임의성과 미분가능성 제약을 제거하고, **모든 MILP**로 일반화하며, 실험적으로 **일관된 regret 개선**을 입증합니다. 구현 관점에서도 내부점 기반 대리 손실로 **안정적 그래디언트**를 제공, **엔드투엔드 학습**을 가능케 합니다. :contentReference[oaicite:13]{index=13}

