---
title: "Learning to Optimize Capacity Planning in Semiconductor Manufacturing (AsiaSim 2025)"
date: 2026-08-20 12:30:00 +0900
categories: [paper_review, OR]
tags: [semiconductor-manufacturing, capacity-planning, reinforcement-learning, heterogeneous-graph-neural-network, PPO]
math: true
---

# Paper Review — *Learning to Optimize Capacity Planning in Semiconductor Manufacturing* (AsiaSim 2025)

- **저자:** Philipp Andelfinger, Jieyi Bi, Qiuyu Zhu, Jianan Zhou, Bo Zhang, Fei Fei Zhang, Chew Wye Chan, Boon Ping Gan, Wentong Cai, Jie Zhang  
- **소속:** Nanyang Technological University, D-SIMLAB Technologies  
- **학회:** 24th Asia Simulation Conference (AsiaSim)  
- **년도:** 2025  
- **출판:** Springer, *Communications in Computer and Information Science*, pp. 67–79  
- **원문:** [저자 제공 PDF](https://philipp-andelfinger.net/pdfs/andelfinger2025learning.pdf) · [arXiv](https://arxiv.org/abs/2509.15767) · [Springer 공식 Chapter](https://link.springer.com/chapter/10.1007/978-981-95-4472-1_6)

---

## 1) High-Level Summary (3–5 sentences)

본 논문은 반도체 FAB의 WIP·Queue·Machine 상태를 보고, **어느 장비의 Uptime 또는 Processing Efficiency를 개선하고 어떤 Machine–Operation Dedication을 바꿀지**를 Deep Reinforcement Learning으로 선택합니다. 공정 Route와 장비 Capability의 복잡한 관계를 표현하기 위해 Machine Node와 Operation Node로 구성된 **Heterogeneous Graph Neural Network(HGNN)**를 Policy Encoder로 사용하고, Decoder가 Machine-level 개선확률과 Machine–Operation Pair-level Dedication 확률을 만듭니다. Reward는 동일한 Simulation State를 “행동 적용”과 “No Action” 두 Branch로 나누어 실행한 뒤 **Daily Going Rate(DGR)의 차이**로 계산하며, Policy는 n-step PPO로 학습합니다. Intel Minifab과 SMT2020 Benchmark Simulation에서 WIP 기반 Heuristic보다 높은 성능을 보였고, 큰 SMT2020 Scenario에서 완료 LOT **1.87% 증가**, 평균 Cycle Time **1.80% 감소**, DGR **1.74% 증가**를 보고합니다. 그러나 실제 FAB 배포나 실데이터 검증이 아니라 Benchmark Simulation 결과이며, Action Cost가 목적에 포함되지 않고 장기학습 중 Policy 성능이 다시 하락하는 불안정성도 확인됩니다.

---

## 2) 문제 설정: Capacity Planning을 Machine-level 순차의사결정으로 보기

반도체 Wafer FAB은 수백~수천 대 장비와 Re-entrant Route를 가집니다. 한 Operation은 Compatible Machine 일부에서만 처리되고, 한 Machine은 여러 제품·공정이 공유하므로 장비개선의 영향이 Downstream WIP와 다음 병목으로 전파됩니다.

전통적 Capacity Planning은 Expert Knowledge와 Heuristic으로 Machine–Operation Dedication, Uptime, Processing Efficiency의 “Future Change List”를 만듭니다.

Heuristic은 해석하기 쉽지만 Product Mix와 WIP가 바뀔 때 이동하는 Bottleneck을 놓칠 수 있습니다. 이 논문은 Capacity Planning을 다음 순차문제로 재정의합니다.

$$
\text{FAB State 관측}
\rightarrow \text{제한된 개선 Action 선택}
\rightarrow \text{Simulation 진행}
\rightarrow \text{KPI 변화 관측}
\rightarrow \text{다음 Action 선택}.
$$

이는 LOT의 다음 장비를 고르는 **Dispatching**이 아니라, 제한된 개선자원을 어느 Machine과 Dedication에 투입할지 고르는 Planning-level 문제입니다.

---

## 3) Formal Model: FAB를 MDP로 정의하기

### 3.1 제품·공정·장비

- $P$: 제품 집합
- $M$: Machine 집합
- $O$: Operation 집합
- $O_{p,j}$: 제품 $p$의 $j$번째 Operation
- $M_{p,j}\subseteq M$: $O_{p,j}$를 처리할 수 있는 Compatible Machine 집합

제품 $p$의 Route는 선후관계를 가집니다.

$$
O_{p,1}\rightarrow O_{p,2}\rightarrow\cdots\rightarrow O_{p,n_p}.
$$

Operation 종류에는 Lithography, Diffusion, Dry/Wet Etching, Implantation 등이 포함됩니다. 동일 장비군을 Route에서 여러 번 방문하는 Re-entrant Flow와 Operation-dependent Setup도 Simulation이 반영합니다.

### 3.2 세 종류의 Action

논문은 세 가지 의사결정 유형을 네 가지 Action Category—Dedication 추가·삭제, Uptime 개선, Efficiency 개선—로 표현합니다. 다만 Dedication 삭제확률은 별도의 학습 Head가 아니라 추가확률의 Complement입니다.

1. **Dedication 추가**

$$
d^+(m_k,O_{p,j}): M\rightarrow O
$$

Machine $m_k$가 Operation $O_{p,j}$를 처리할 수 있도록 연결을 추가합니다.

2. **Dedication 삭제**

$$
d^-(m_k,O_{p,j}): M\rightarrow O
$$

기존 Machine–Operation 연결을 제거합니다.

3. **Uptime 개선**

$$
u(m_k)\leftarrow \min\{u(m_k)+3\%\text{p},100\%\}.
$$

선택한 장비의 Uptime을 3%p 높입니다.

4. **Efficiency 개선**

$$
PT(m_k)\leftarrow0.9\,PT(m_k).
$$

Machine의 LOT당 Processing Time을 기존의 90%로 줄입니다. 이는 10% 처리시간 개선을 의미합니다.

Uptime +3%p와 Processing Time 10% 감소는 실험을 위해 정한 **가상적 Action 크기**이며, 실제 개선공사의 비용·Lead Time 추정치가 아닙니다.

### 3.3 MDP

장기 Capacity Planning을 다음 MDP로 둡니다.

$$
(\mathcal S,\mathcal A,\mathcal T,\mathcal R,\gamma),\qquad \gamma<1.
$$

- $s_t\in\mathcal S$: 시점 $t$의 FAB 상태
- $a_t\in\mathcal A$: 해당 시점에 선택한 개선 Action 묶음
- $\mathcal T$: Simulation을 통한 상태전이
- $r_t=\mathcal R(s_t,a_t)$: Action으로 인한 KPI 개선
- $\gamma$: 미래 Reward Discount Factor

Minifab에서는 매 Decision Step마다 Uptime Action 1개와 Efficiency Action 1개를 선택합니다.

$$
a_t=\{u(m_i)\}_{i=1}^{\sigma_u}\cup\{r(m_i)\}_{i=1}^{\sigma_r},
\qquad \sigma_u=\sigma_r=1.
$$

SMT2020에서는 Uptime·Efficiency·Dedication 삭제·Dedication 추가를 각각 5개씩 선택합니다.

$$
\sigma_u=\sigma_r=\sigma_{d^-}=\sigma_{d^+}=5.
$$

Machine 1,314대와 훨씬 많은 Machine–Operation Pair를 Flat Action Space로 만들면 지나치게 커지므로 HGNN과 Action별 Decoder를 사용합니다.

### 3.4 Counterfactual Branch Reward

Simulation Noise 속에서 Action의 효과를 더 잘 구분하고 차이 추정의 분산을 줄이기 위해 같은 상태를 두 Branch로 복제합니다.

- Branch 1: 선택한 Action을 적용해 얻은 $KPI_1$
- Branch 0: 아무 Action도 하지 않고 얻은 $KPI_0$

Reward는 두 결과의 차이입니다.

$$
r_t=KPI_1-KPI_0.
$$

이 연구에서 KPI는 FAB-level **Daily Going Rate(DGR)**입니다.

$$
DGR=
\sum_{i=1}^{|P|}
\frac{\rho_i}{L_i}
\sum_{l=1}^{L_i}DGR_{i,l}.
$$

- $DGR_{i,l}$: 제품 $i$의 Operation $l$에서 나온 Daily Output
- $L_i$: 제품 $i$의 총 Operation 수
- $\rho_i$: FAB WIP 중 제품 $i$가 차지하는 비율

Operation 수가 긴 제품의 과대반영을 $L_i$로 보정하고 Product Mix를 $\rho_i$로 반영하므로, Route 전반의 생산진행률을 요약합니다.

State-copying은 Variance를 줄이지만 매 Action마다 두 Branch를 Simulation해야 합니다.

---

## 4) 상태표현: Machine–Operation Heterogeneous Graph

### 4.1 Graph 구성

Graph $H$에는 두 종류의 Node가 있습니다.

$$
H=(M\cup O,E_{MO}\cup E_{OO}).
$$

- **Machine Node**: 개별 장비
- **Operation Node**: 제품 Route의 개별 공정 Step
- **Operation–Machine Edge**: 어떤 장비가 어떤 공정을 처리하는지 나타내는 Assignment/Dedication
- **Operation–Operation Edge**: Route에서 선행·후속 공정관계

Graph는 단순 KPI Vector와 달리 Route와 Capability 구조를 보존합니다. Queue 길이가 같아도 Critical Operation 연결 여부에 따라 다른 Embedding을 만들 수 있습니다.

### 4.2 Feature

Machine Node Feature $m_k\in\mathbb R^{d_1}$에는 Batch Size, Waiting·Completed LOT/Wafer, 평균 Cycle·Queue·Processing Time, Productive·Down·Idle·Setup Time, Dispatch Queue, Period-end WIP가 들어갑니다. Operation Node Feature $o_{ij}\in\mathbb R^{d_2}$에는 완료 Wafer·LOT·Layer, 평균 WIP, Cycle·Queue·Processing Time, Remaining Due/Processing Time, DGR와 Dynamic Cycle Time이 포함됩니다.

Operation–Machine Edge $\epsilon_{ij,k}\in\mathbb R^{d_3}$는 Processing Time과 Setup Cost 같은 Assignment Context를, Operation–Operation Edge $\epsilon_{ij,i(j+1)}\in\mathbb R^{d_4}$는 Process Flow 관계를 나타냅니다.

---

## 5) HGNN Policy와 Critic

### 5.1 Message Passing

Encoder는 두 종류 Update를 반복합니다.

- **Machine Embedding:** 연결된 Operation Node의 정보와 Edge Feature를 Edge-aware Attention으로 Aggregation합니다. 동일 Machine에 연결된 Operation마다 병목영향이 다를 수 있으므로 Attention Weight가 다르게 주어집니다.
- **Operation Embedding:** 선행·후속 Operation과 연결된 Machine Embedding을 MLP로 결합합니다.

$L$개 Message-passing Layer를 통과한 뒤 각 Layer 표현을 평균해 최종 Embedding을 만듭니다.

$$
m_i^{\mathrm{final}}
=\frac{1}{L}\sum_{\ell=1}^{L}m_i^{(\ell)},
\qquad
o_{ij}^{\mathrm{final}}
=\frac{1}{L}\sum_{\ell=1}^{L}o_{ij}^{(\ell)}.
$$

$m_i^{(\ell)}$와 $o_{ij}^{(\ell)}$는 $\ell$번째 Layer의 Machine·Operation Embedding입니다. 마지막 Layer만 쓰지 않고 평균함으로써 Local 정보와 더 먼 Neighborhood 정보를 함께 유지합니다.

### 5.2 Uptime·Efficiency Decoder

각 Machine Embedding에 별도 MLP와 Sigmoid를 적용합니다.

$$
\pi_i^u
=\sigma\left(MLP_u(m_i^{\mathrm{final}})\right),
\qquad
\pi_i^r
=\sigma\left(MLP_r(m_i^{\mathrm{final}})\right).
$$

$\pi_i^u$는 Machine $i$의 Uptime 개선선택 확률, $\pi_i^r$은 Efficiency 개선선택 확률입니다. Machine별 독립 Score를 만들고 제한된 개수의 Target을 Sampling합니다.

### 5.3 Dedication Decoder

Dedication은 Machine 하나만 보면 결정할 수 없고 Machine–Operation Pair의 Compatibility를 봐야 합니다. 논문은 두 Embedding의 Pairwise Interaction을 사용합니다.

$$
\pi_{ij}^{d^+}
=
\sigma\left(
-C\tanh\left(
\frac{m_i^{\mathrm{final}}\cdot o_{ij}^{\mathrm{final}}}{\sqrt d}
\right)
\right),
\qquad C=10.
$$

$d$는 Embedding Dimension이고 $C$는 Exploration을 유도하는 Scaling Factor입니다. Dedication 제거확률은 논문에서 다음 Complement로 정의합니다.

$$
\pi_{ij}^{d^-}=1-\pi_{ij}^{d^+}.
$$

추가에 적합한 Pair는 제거에는 덜 적합하다는 대칭적 Inductive Bias입니다. 실제 FAB에서는 추가와 제거의 비용·Risk가 비대칭일 수 있으므로, 이는 연구모형의 단순화입니다.

### 5.4 Critic

Critic은 Policy와 HGNN Encoder를 공유합니다. 모든 Machine과 Operation Embedding을 각각 Mean Pooling한 후 이어 붙입니다.

$$
h_t=
\left(
\frac{1}{|M|}\sum_{m_i\in M}m_i^{\mathrm{final}}
\right)
\Vert
\left(
\frac{1}{|O|}\sum_{O_{p,j}\in O}o_{p,j}^{\mathrm{final}}
\right).
$$

$\Vert$는 Concatenation이며 Linear Layer가 $h_t$를 $V_\phi(s_t)$로 변환합니다. Node 수와 무관한 고정길이 Graph 표현입니다.

---

## 6) 학습: n-step PPO

논문은 표준 **Proximal Policy Optimization(PPO)**을 n-step Online Training에 사용합니다. $n$ Step 뒤 Critic Value로 Bootstrap한 Return은 역방향으로 계산됩니다.

$$
\widehat R_{t'}
=r_{t'}+\gamma\widehat R_{t'+1},
\qquad
\widehat A_{t'}
=\widehat R_{t'}-V_\phi(s_{t'}).
$$

- $\widehat R_{t'}$: n-step Bootstrap Return
- $\widehat A_{t'}$: 선택 Action이 Critic 기대보다 얼마나 좋았는지 나타내는 Advantage

논문은 Return을 Z-score로 Normalize한 뒤 PPO Loss와 Critic Loss를 계산합니다. 사용한 표준 PPO의 핵심 Clipped Objective는 다음과 같이 쓸 수 있습니다.

$$
L^{\mathrm{CLIP}}(\theta)
=
\mathbb E_t\left[
\min\left(
q_t(\theta)\widehat A_t,
\operatorname{clip}(q_t(\theta),1-\epsilon,1+\epsilon)\widehat A_t
\right)
\right],
$$

$$
q_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

$\epsilon$은 한 번의 Update에서 Policy가 지나치게 크게 바뀌는 것을 막는 Clipping Threshold입니다. Critic은 Return과 Value 예측의 제곱오차를 줄입니다.

$$
L^{\mathrm{Critic}}(\phi)
=\mathbb E_t\left[
(\widehat R_t-V_\phi(s_t))^2
\right].
$$

학습에서는 서로 다른 Seed의 $B$개 Simulation을 병렬 실행하고 HGNN Action과 No-action Branch의 DGR 차이를 Reward로 모읍니다. $n$ Step Experience에서 Bootstrap Return $\widehat R$을 계산해 Z-score Normalize한 다음, 그 Return과 Critic 값으로 Advantage $\widehat A$를 계산하여 Policy와 Critic을 $K$회 Update합니다.
### Figure 2 — Counterfactual Branch를 포함한 학습 구조

<img width="1100" alt="simulation state feature extraction, HGNN policy, action and no-action branches, PPO update flow" src="/assets/img/paper-reviews/2026-08-20/andelfinger-fig2.svg" />

> Source: Andelfinger et al. (2025), Figure 2. 논문 이해를 위한 일부 인용 및 크롭. [arXiv manuscript](https://arxiv.org/abs/2509.15767)

같은 Simulation State에서 Action Branch와 No-action Branch를 동시에 실행한 뒤 KPI 차이를 Reward로 계산합니다. 따라서 Policy는 단순히 이후 KPI가 좋아졌는지가 아니라, 아무 조치도 하지 않았을 때보다 선택한 Capa 개선조치가 얼마나 추가 효과를 냈는지를 학습합니다.


---

## 7) Simulation Environment와 실험설정

### 7.1 두 Benchmark

| Model | 제품 | Machine | Operation | 제품별 Route Step |
|---|---:|---:|---:|---:|
| Intel Minifab | 3 | 5 | 18 | 6 |
| SMT2020 | 10 | 1,314 | 4,014 | 242–583 |

**Minifab**은 Diffusion 2대, Ion Implantation 2대, Lithography 1대를 가진 소형 Testbed입니다. Batch Processing, Re-entrant Operation, Operation-dependent Setup을 포함하지만 실제 FAB보다 매우 작습니다.

**SMT2020**은 1,314대 Machine과 제품당 최대 583 Step을 가진 공개 Semiconductor Manufacturing Testbed입니다. 논문은 더 다양한 Low-volume/High-mix Dataset을 사용하고, Bottleneck을 강조하기 위해 Wafer Arrival을 25% 높입니다. 실제 FAB 특성과 규모에 접근하는 Benchmark이지만, 기업의 실제 Fab Log나 Proprietary Model은 아닙니다.

두 Model은 D-SIMCON에서 실행됩니다. 이 Discrete-Event Engine은 Process Flow, Dedication, Wafer Start, Downtime을 표현하고 Python Interface 및 State Copy를 지원합니다.

### 7.2 Training Hyperparameter

- Adam: Policy $3\times10^{-4}$, Critic $10^{-4}$
- PPO: $n=5$, $K=20$, $\epsilon=0.2$, $\gamma=0.99$, $B=16$
- HGNN: $d=64$, Minifab $L=1$, SMT2020 $L=2$
- Decision Horizon: Minifab은 매일·5일, SMT2020은 매주·25주

Baseline은 **No Action**, **Random**, WIP 수준으로 Target을 Sampling하는 **Heuristic**입니다. Exact Optimization이나 상용 Capacity Planning System과의 비교는 아닙니다.

---

## 8) 실험결과

### 8.1 Minifab

16개 Simulation Instance를 사용하고 Strategy 간 비교에는 같은 16개 Seed Sequence를 재사용했을 때 $d=64$, $L=1$이 가장 좋았습니다.

| Strategy | Completed Lots | Cycle Time | DGR |
|---|---:|---:|---:|
| HGNN Policy | 55.50 | 1.08일 | 266.56 |
| Heuristic | 54.63 | 1.11일 | 265.42 |
| Random | 54.63 | 1.11일 | 265.00 |
| No Action | 48.75 | 1.27일 | 251.09 |

HGNN은 Heuristic 대비 완료 LOT **1.59% 증가**, Cycle Time **2.70% 감소**, DGR **0.43% 증가**를 기록했습니다. No Action 대비 개선폭은 각각 13.85%, 14.96%, 6.16%입니다.

### 8.2 SMT2020

| Strategy | Completed Lots | Cycle Time | DGR |
|---|---:|---:|---:|
| HGNN Policy | 6,019.75 | 59.100일 | 749.79 |
| Heuristic | 5,909.44 | 60.181일 | 736.99 |
| Random | 5,904.13 | 60.211일 | 736.65 |
| No Action | 5,872.94 | 60.507일 | 732.58 |

HGNN은 Heuristic 대비 다음 개선을 보였습니다.

- Completed Lots: **1.87% 증가** $(+110.3)$
- Average Cycle Time: **1.80% 감소** $(-1.08\text{일})$
- DGR: **1.74% 증가** $(+12.80)$

No Action 대비로는 완료 LOT 2.50% 증가, Cycle Time 2.33% 감소, DGR 2.35% 증가입니다. Training 초반 30–40 Epoch 동안 KPI가 좋아질 때 Policy는 Lithography와 Diffusion Machine에 Efficiency Action을 더 자주 배정했습니다. 이는 Bottleneck 후보를 학습했다는 정성적 증거이지만, Causal Bottleneck Analysis나 설명가능성 검증까지 제공한 것은 아닙니다.
### Figure 3 — SMT2020 학습 중 FAB-level KPI 변화

<img width="1100" alt="epoch별 daily going rate, completed lots, average cycle time 학습 곡선" src="/assets/img/paper-reviews/2026-08-20/andelfinger-fig3.svg" />

> Source: Andelfinger et al. (2025), Figure 3. 논문 이해를 위한 일부 인용 및 크롭. [arXiv manuscript](https://arxiv.org/abs/2509.15767)

DGR과 Completed Lots는 약 30–40 Epoch에서 가장 좋아지고 Average Cycle Time은 같은 구간에서 낮아집니다. 이후 다시 악화되는 곡선 때문에 최고 Epoch의 성능과 안정적으로 수렴한 최종 Policy의 성능을 구분해야 합니다.


가장 중요한 결과는 **Epoch 40 이후 KPI가 초기 수준 쪽으로 다시 하락**했다는 점입니다. 저자들은 5주로 줄인 Scenario, 고정 Seed, Dedication Action 제거, EMA Reward Baseline도 시험했지만 같은 하락경향이 남았다고 보고합니다. 따라서 표의 SMT2020 성능은 안정적으로 수렴한 Final Policy가 아니라 성능이 좋았던 Epoch 40의 Evaluation입니다.

---

## 9) 기여·강점·한계

### 기여

1. Semiconductor Capacity Planning을 Machine-level Action의 Long-horizon MDP로 구성했습니다.  
2. Operation Route와 Machine Dedication을 보존하는 HGNN을 대규모 SMT2020 Simulation에 적용했습니다.  
3. 동일 State의 Action/No-action Branch 차이를 Reward로 써 Simulation Variance를 줄이는 구조를 제시했습니다.  
4. Uptime·Efficiency·Dedication을 한 Policy가 함께 선택하도록 Action Space를 확장했습니다.

### 강점

- FAB의 핵심 구조인 Re-entrant Route와 Machine–Operation Compatibility를 Graph로 직접 표현합니다.
- 단일 Machine의 Queue뿐 아니라 선후공정과 대체장비 관계를 Message Passing으로 반영합니다.
- Minifab뿐 아니라 Machine 1,314대, Operation 4,014개의 SMT2020까지 Scaling을 시도했습니다.
- KPI가 작은 비율로 변해도 생산량 절대값과 Cycle Time 변화량을 함께 보고해 효과크기를 확인할 수 있습니다.

### 한계

- **실제 FAB Deployment가 아닙니다.** Minifab과 SMT2020이라는 공개 Benchmark를 D-SIMCON에서 Simulation한 연구입니다.
- SMT2020 실험은 논문 스스로 “Preliminary” 성격이며, 장기학습에서 Policy Collapse와 유사한 성능하락을 보입니다.
- Uptime +3%p, Processing Time 10% 단축은 가정된 Action입니다. 실제 Engineering Project의 Cost, Duration, Feasibility, Risk가 Reward에 없습니다.
- Dedication 추가·삭제도 Qualification Lead Time, Recipe Validation, Yield/Quality Risk를 직접 모델링하지 않습니다.
- Reward가 DGR 하나이므로 Action Cost, Yield, Delivery Tardiness, Energy, Maintenance, Quality와 다목적 Trade-off를 다루지 않습니다.
- Heuristic·Random·No Action과만 비교하므로 다른 RL, Mathematical Programming, Simulation Optimization 대비 우월성은 검증되지 않았습니다.
- State Feature가 매우 많아 현업에서 동일 수준의 Data Quality와 실시간 Availability를 확보할 수 있는지는 논의하지 않습니다.
- Counterfactual 두 Branch를 매 Step 실행하므로 Training Simulation Cost가 큽니다. 논문은 정확한 Wall-clock Training Cost를 제시하지 않습니다.
- 미래 Wafer Start 정보가 State에 없으며, 저자들도 이를 Future Work로 둡니다.

---

## 10) 결론

본 논문의 핵심은 Capacity Planning을 정적인 Capa Table 작성이 아니라, **동적으로 변하는 WIP와 Machine 상태에서 제한된 개선 Action을 어디에 배분할지 학습하는 순차의사결정**으로 만든 데 있습니다. HGNN은 Machine–Operation–Route 관계를 보존하고, PPO는 Simulation Interaction을 통해 Uptime·Efficiency·Dedication Action의 장기효과를 학습합니다. SMT2020에서 Heuristic 대비 완료 LOT 1.87% 증가와 Cycle Time 1.80% 감소를 보인 점은 연구가능성을 보여주지만, 이는 공개 Benchmark Simulation의 **Epoch-40 Evaluation** 결과입니다. 실제 제조 의사결정으로 확장하려면 Policy 안정성, Action Cost와 Qualification Risk, 다목적 KPI, Forecasted Wafer Starts, 실제 Industry-scale Model 검증이 추가되어야 합니다.
