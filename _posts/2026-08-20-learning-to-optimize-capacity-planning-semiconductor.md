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

FAB의 WIP·Queue·Machine 상태를 보고 어느 장비의 Uptime/Processing Time을 개선할지, 어떤 Machine–Operation Dedication을 바꿀지를 RL로 선택하는 연구. Route와 장비 Capability를 Heterogeneous Graph로 만들고 HGNN Policy를 PPO로 학습함. 같은 State에서 Action과 No-action Simulation을 각각 돌려 DGR 차이를 Reward로 쓰는 것이 핵심. SMT2020에서 WIP Heuristic 대비 Completed Lots 1.87% 증가, Cycle Time 1.80% 감소를 보였지만 실제 FAB가 아닌 Benchmark Simulation이며 학습 후반에는 성능이 다시 내려가는 문제가 남음.

---

## 2) 논문에서 풀려는 문제

반도체 FAB에는 수백~수천 대의 Machine이 있고 제품별 Route도 다름. 같은 Machine을 여러 Operation이 공유하며, 특정 장비를 개선했을 때 효과가 바로 Fab-out으로 나타나는 것도 아니다. 앞 공정 Queue가 줄면 뒤쪽에 새로운 Bottleneck이 생길 수 있기 때문.

기존 Capacity Planning에서는 Expert가 WIP, Utilization, Dedication 등을 보고 Future Change List를 만든다. 이 논문은 그 과정을 다음과 같은 Sequential Decision으로 봄.

$$
\text{FAB State}
\rightarrow \text{Capa 개선 Action 선택}
\rightarrow \text{Simulation}
\rightarrow \text{KPI 변화}
\rightarrow \text{다음 Action}
$$

여기서 고르는 것은 LOT의 다음 장비가 아니다. 어느 Machine의 Uptime/Processing Time을 개선할지, 어떤 Machine–Operation Dedication을 추가하거나 없앨지를 고른다. Dispatching이 아니라 Planning-level Action에 가까움.

논문의 전체 구성을 짧게 쓰면 다음과 같다.

> Machine과 Operation 관계를 Heterogeneous Graph로 표현 → HGNN Policy가 개선대상 선택 → D-SIMCON으로 Action/No-action을 각각 Simulation → DGR 차이를 Reward로 PPO 학습

Intel Minifab과 SMT2020에서 WIP Heuristic보다 성능이 높았고, SMT2020에서는 Completed Lots 1.87% 증가, Average Cycle Time 1.80% 감소, DGR 1.74% 증가. 다만 실제 FAB 적용 결과가 아니라 Benchmark Simulation 결과이며, 뒤에서 보듯 학습이 끝까지 안정적으로 수렴한 것도 아니다.

---

## 3) FAB와 Action 정의

**제품 Route와 Compatible Machine**

- $P$: 제품 집합
- $M$: Machine 집합
- $O$: Operation 집합
- $O_{p,j}$: 제품 $p$의 $j$번째 Operation
- $M_{p,j}\subseteq M$: $O_{p,j}$를 처리할 수 있는 Machine 집합

제품 $p$의 Route:

$$
O_{p,1}\rightarrow O_{p,2}\rightarrow\cdots\rightarrow O_{p,n_p}
$$

Lithography, Diffusion, Dry/Wet Etching, Implantation 등이 들어가고 동일 장비군을 여러 번 방문하는 Re-entrant Flow도 포함. Operation-dependent Setup, Batch Processing도 Simulation에 반영한다.

**Action**

1. **Dedication 추가**

$$
d^+(m_k,O_{p,j}):M\rightarrow O
$$

Machine $m_k$에서 Operation $O_{p,j}$를 처리할 수 있도록 연결을 추가.

2. **Dedication 삭제**

$$
d^-(m_k,O_{p,j}):M\rightarrow O
$$

기존 Machine–Operation 연결 제거.

3. **Uptime 개선**

$$
u(m_k)\leftarrow\min\{u(m_k)+3\%\text{p},100\%\}
$$

4. **Efficiency 개선**

$$
PT(m_k)\leftarrow0.9\,PT(m_k)
$$

Processing Time을 10% 줄이는 Action. Uptime +3%p와 Processing Time 10% 감소는 실제 개선공사의 효과를 추정한 값이 아니라 실험을 위해 정한 Action 크기다. 이 차이는 뒤의 결과를 볼 때 중요함.

**MDP**

$$
(\mathcal S,\mathcal A,\mathcal T,\mathcal R,\gamma),\qquad \gamma<1
$$

- $s_t$: 현재 FAB State
- $a_t$: 선택한 개선 Action 묶음
- $\mathcal T$: Simulation으로 구현되는 상태전이
- $r_t$: Action 이후 KPI 변화
- $\gamma$: 미래 Reward Discount

Minifab은 매 Step Uptime 1개, Efficiency 1개를 고른다.

$$
a_t=\{u(m_i)\}_{i=1}^{\sigma_u}\cup\{r(m_i)\}_{i=1}^{\sigma_r},
\qquad \sigma_u=\sigma_r=1
$$

SMT2020은 Uptime, Efficiency, Dedication 삭제, Dedication 추가를 각각 5개씩 선택.

$$
\sigma_u=\sigma_r=\sigma_{d^-}=\sigma_{d^+}=5
$$

Machine 1,314대에 Machine–Operation Pair까지 전부 Flat Action Space로 만들면 너무 커짐. HGNN을 쓰는 이유가 여기 있다.

---

## 4) Reward: 같은 State에서 Action과 No-action 비교

Simulation 결과에는 Random Failure 등의 Noise가 섞임. Action 이후 DGR만 보면 그 변화가 Action 때문인지 Randomness 때문인지 구분하기 어렵다. 그래서 같은 Simulation State를 복사해 두 Branch를 실행한다.

- Branch 1: 선택 Action 적용 → $KPI_1$
- Branch 0: 아무것도 하지 않음 → $KPI_0$

$$
r_t=KPI_1-KPI_0
$$

KPI는 FAB-level **Daily Going Rate(DGR)**.

$$
DGR=
\sum_{i=1}^{|P|}
\frac{\rho_i}{L_i}
\sum_{l=1}^{L_i}DGR_{i,l}
$$

- $DGR_{i,l}$: 제품 $i$, Operation $l$의 Daily Output
- $L_i$: 제품 $i$의 전체 Operation 수
- $\rho_i$: 전체 WIP에서 제품 $i$가 차지하는 비율

$L_i$로 나누어 Route가 긴 제품이 과도하게 반영되는 것을 막고, $\rho_i$로 Product Mix를 반영. 완제품 Output 하나만 보는 것이 아니라 Route 전체의 진행률을 요약한 KPI라고 이해할 수 있다.

같은 State를 출발점으로 두기 때문에 Action의 Incremental Effect를 비교하기는 좋아지지만, 매 Step마다 Simulation을 두 번 돌려야 함.

---

## 5) State를 Heterogeneous Graph로 표현

$$
H=(M\cup O,E_{MO}\cup E_{OO})
$$

- **Machine Node**: 개별 장비
- **Operation Node**: 제품 Route의 각 공정 Step
- **Operation–Machine Edge**: 어떤 장비가 어떤 Operation을 처리할 수 있는지
- **Operation–Operation Edge**: Route의 선후관계

단순 KPI Vector가 아니라 Route와 Capability 구조를 그대로 남긴다. Queue가 똑같이 10 LOT이어도 Critical Operation과 연결된 장비인지, 대체 가능한 Machine이 있는지에 따라 다른 State가 됨.

Machine Node에는 Batch Size, Waiting/Completed LOT·Wafer, 평균 Cycle/Queue/Processing Time, Productive/Down/Idle/Setup Time, Dispatch Queue, Period-end WIP 등이 들어간다. Operation Node에는 완료 Wafer·LOT·Layer, WIP, Cycle/Queue/Processing Time, Remaining Due/Processing Time, DGR, Dynamic Cycle Time이 사용됨.

Operation–Machine Edge는 Processing Time과 Setup Cost, Operation–Operation Edge는 Process Flow를 표현한다.

### Message Passing

Machine Embedding은 연결된 Operation과 Edge 정보를 Attention으로 모은다. Operation Embedding은 선행·후속 Operation과 연결된 Machine 정보를 MLP로 합친다. $L$개 Layer를 지난 뒤 마지막 Layer만 쓰지 않고 전체 표현을 평균.

$$
m_i^{\mathrm{final}}
=\frac1L\sum_{\ell=1}^{L}m_i^{(\ell)},
\qquad
o_{ij}^{\mathrm{final}}
=\frac1L\sum_{\ell=1}^{L}o_{ij}^{(\ell)}
$$

가까운 Neighborhood 정보와 여러 Hop을 지난 정보를 같이 남기려는 구조.

### Machine-level Decoder

$$
\pi_i^u
=\sigma\left(MLP_u(m_i^{\mathrm{final}})\right),
\qquad
\pi_i^r
=\sigma\left(MLP_r(m_i^{\mathrm{final}})\right)
$$

$\pi_i^u$는 Uptime 개선, $\pi_i^r$은 Efficiency 개선 Score. 제한된 수만큼 Machine을 Sampling한다.

### Machine–Operation Pair Decoder

Dedication은 Machine만 보고 고를 수 없기 때문에 두 Embedding의 Pairwise Interaction을 사용.

$$
\pi_{ij}^{d^+}
=
\sigma\left(
-C\tanh\left(
\frac{m_i^{\mathrm{final}}\cdot o_{ij}^{\mathrm{final}}}{\sqrt d}
\right)
\right),
\qquad C=10
$$

$$
\pi_{ij}^{d^-}=1-\pi_{ij}^{d^+}
$$

추가 Score가 높으면 삭제 Score는 낮게 두는 단순한 대칭구조. 실제 Qualification에서는 추가와 삭제의 Cost/Risk가 같지 않으므로 강한 가정에 해당한다.

### Critic

모든 Machine과 Operation Embedding을 각각 Mean Pooling하고 Concatenate.

$$
h_t=
\left(
\frac1{|M|}\sum_{m_i\in M}m_i^{\mathrm{final}}
\right)
\Vert
\left(
\frac1{|O|}\sum_{O_{p,j}\in O}o_{p,j}^{\mathrm{final}}
\right)
$$

Linear Layer가 이를 $V_\phi(s_t)$로 변환. Node 수가 달라도 고정길이 Graph 표현을 얻을 수 있다.

---

## 6) n-step PPO 학습

$$
\widehat R_{t'}
=r_{t'}+\gamma\widehat R_{t'+1},
\qquad
\widehat A_{t'}
=\widehat R_{t'}-V_\phi(s_{t'})
$$

$n$ Step을 모은 뒤 Critic Value로 Bootstrap. Return은 Z-score Normalize하고 PPO의 Clipped Objective로 Policy를 Update한다.

$$
L^{\mathrm{CLIP}}(\theta)
=
\mathbb E_t\left[
\min\left(
q_t(\theta)\widehat A_t,
\operatorname{clip}(q_t(\theta),1-\epsilon,1+\epsilon)\widehat A_t
\right)
\right]
$$

$$
q_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
$$

$\epsilon$은 한 번에 Policy가 너무 크게 바뀌지 않게 제한. Critic은 다음 MSE를 줄인다.

$$
L^{\mathrm{Critic}}(\phi)
=\mathbb E_t\left[
(\widehat R_t-V_\phi(s_t))^2
\right]
$$

서로 다른 Seed의 $B$개 Simulation을 병렬 실행 → $n$ Step Experience 수집 → Bootstrap Return 계산 → Policy/Critic을 $K$회 Update.

### Figure 2 — Counterfactual Branch를 포함한 학습 구조

<img width="1100" alt="simulation state feature extraction, HGNN policy, action and no-action branches, PPO update flow" src="/assets/img/paper-reviews/2026-08-20/andelfinger-fig2.svg" />

> Source: Andelfinger et al. (2025), Figure 2. 논문 이해를 위한 일부 인용 및 크롭. [arXiv manuscript](https://arxiv.org/abs/2509.15767)

그림에서 중요한 부분은 오른쪽의 두 Simulation Branch. 같은 State에서 한쪽에만 Action을 넣고 KPI 차이를 Reward로 사용한다. 단순히 “다음 KPI가 좋아졌는가”보다 “아무것도 하지 않았을 때보다 이 Action이 얼마나 더 좋았는가”를 학습하는 방식이다.

---

## 7) Experiment

### Benchmark와 설정

| Model | 제품 | Machine | Operation | 제품별 Route Step |
|---|---:|---:|---:|---:|
| Intel Minifab | 3 | 5 | 18 | 6 |
| SMT2020 | 10 | 1,314 | 4,014 | 242–583 |

Minifab은 Diffusion 2대, Ion Implantation 2대, Lithography 1대의 작은 Testbed. SMT2020은 공개 Semiconductor Manufacturing Testbed이고 논문에서는 Low-volume/High-mix Data를 사용한다. Bottleneck을 더 분명히 만들기 위해 Wafer Arrival을 25% 높였음.

두 Model 모두 D-SIMCON에서 실행. Process Flow, Dedication, Wafer Start, Downtime을 표현하고 Python Interface와 State Copy를 제공한다.

- Adam: Policy $3\times10^{-4}$, Critic $10^{-4}$
- PPO: $n=5$, $K=20$, $\epsilon=0.2$, $\gamma=0.99$, $B=16$
- HGNN: $d=64$, Minifab $L=1$, SMT2020 $L=2$
- Horizon: Minifab 매일/5일, SMT2020 매주/25주
- Baseline: No Action, Random, WIP 기반 Heuristic

Exact Optimization이나 다른 RL과 비교한 실험은 아님.

### Minifab 결과

| Strategy | Completed Lots | Cycle Time | DGR |
|---|---:|---:|---:|
| HGNN Policy | 55.50 | 1.08일 | 266.56 |
| Heuristic | 54.63 | 1.11일 | 265.42 |
| Random | 54.63 | 1.11일 | 265.00 |
| No Action | 48.75 | 1.27일 | 251.09 |

HGNN vs Heuristic:

- Completed Lots $+1.59\%$
- Cycle Time $-2.70\%$
- DGR $+0.43\%$

16개 Simulation Instance를 사용했고 Strategy 비교에는 같은 16개 Seed Sequence를 재사용.

### SMT2020 결과

| Strategy | Completed Lots | Cycle Time | DGR |
|---|---:|---:|---:|
| HGNN Policy | 6,019.75 | 59.100일 | 749.79 |
| Heuristic | 5,909.44 | 60.181일 | 736.99 |
| Random | 5,904.13 | 60.211일 | 736.65 |
| No Action | 5,872.94 | 60.507일 | 732.58 |

HGNN vs Heuristic:

- Completed Lots: **1.87% 증가** $(+110.3)$
- Average Cycle Time: **1.80% 감소** $(-1.08\text{일})$
- DGR: **1.74% 증가** $(+12.80)$

No Action 대비로는 Completed Lots +2.50%, Cycle Time -2.33%, DGR +2.35%. Training 초반에는 Lithography와 Diffusion Machine에 Efficiency Action을 자주 배정한다. Bottleneck 후보를 어느 정도 학습한 것으로 볼 수 있지만, 별도의 Causal Bottleneck Analysis까지 한 것은 아니다.

### Figure 3 — SMT2020 학습 중 FAB-level KPI 변화

<img width="1100" alt="epoch별 daily going rate, completed lots, average cycle time 학습 곡선" src="/assets/img/paper-reviews/2026-08-20/andelfinger-fig3.svg" />

> Source: Andelfinger et al. (2025), Figure 3. 논문 이해를 위한 일부 인용 및 크롭. [arXiv manuscript](https://arxiv.org/abs/2509.15767)

DGR과 Completed Lots는 30~40 Epoch에서 가장 좋아지고 Cycle Time도 같은 구간에서 가장 낮아진다. 그런데 그 뒤에는 KPI가 다시 나빠짐. 표에 사용된 SMT2020 결과는 마지막 Policy가 아니라 **성능이 좋았던 Epoch 40 Policy**의 Evaluation이다.

저자들은 Horizon을 5주로 축소하거나, Seed를 고정하거나, Dedication Action을 없애거나, EMA Reward Baseline을 넣는 실험도 했지만 이 하락이 남았다고 보고한다.

---

## 8) 읽으면서 중요했던 점

이 논문의 재미있는 부분은 “현재 WIP가 많은 장비를 개선한다”는 Heuristic을 Graph State와 Long-term Reward로 확장했다는 점. Re-entrant Route, 대체 Machine, Downstream Operation을 함께 보고 개선대상을 고를 수 있다. 1,314대 Machine과 4,014개 Operation까지 확장한 것도 의미가 있음.

다만 결과를 그대로 실제 Capacity Planning 성능으로 읽기는 어렵다.

- Minifab/SMT2020을 D-SIMCON에서 실행한 Simulation Study. 실제 FAB Deployment가 아님.
- Uptime +3%p, Processing Time -10%는 가상의 Action이고 Cost, Duration, Feasibility가 없음.
- Dedication 변경에도 Qualification Lead Time, Recipe Validation, Yield Risk가 들어가지 않음.
- Reward가 DGR 하나라 Delivery, Yield, Energy, Maintenance, Action Cost 간 Trade-off를 다루지 않음.
- 비교대상이 WIP Heuristic, Random, No Action뿐.
- SMT2020에서 Training 후반 성능하락이 남아 있고 Best Epoch를 결과로 사용.
- 매 Step Action/No-action 두 Branch가 필요하지만 Wall-clock Training Cost는 제시하지 않음.
- 미래 Wafer Start 정보도 State에 포함되지 않음.

특히 Action Cost가 없는 상태에서는 Policy가 “효과가 있으면 일단 개선”하는 방향으로 학습될 수 있다. 실제 문제라면 다음과 같은 목적이 더 자연스러울 것 같다.

$$
\max\;
\text{DGR improvement}
-\lambda_1\text{Action Cost}
-\lambda_2\text{Qualification Risk}
-\lambda_3\text{Delivery Penalty}
$$

---

## 9) 정리

이 논문은 Capacity Planning을 정적인 Capa Table 문제가 아니라, **변하는 WIP와 Machine 상태를 보고 제한된 개선 Action을 어디에 먼저 배분할지** 학습하는 문제로 바꾼다.

$$
\text{Machine–Operation Graph}
\rightarrow \text{HGNN Policy}
\rightarrow \text{Counterfactual Simulation Reward}
\rightarrow \text{PPO}
$$

SMT2020에서 WIP Heuristic보다 Completed Lots와 DGR은 증가하고 Cycle Time은 감소. 다만 수치는 Epoch 40의 Best Policy에 해당하며, 실제 적용 전에는 Policy 안정성, Action Cost, Qualification/Yield Risk, 미래 Wafer Start, Multi-objective KPI가 추가로 필요하다.
