---
title: "Financial Data Analysis for Time Series Forecasting"
date: 2024-11-19 12:00:00 +0900
categories: [iSME, seminar]
tags: [FTS-Diffusion, scale-invariance, irregularity, time-series, DTW, SISC, KMeans++, diffusion, DDPM, AE, Markov, augmentation]
math: true
permalink: /notes/financial-time-series/fts-diffusion/
---

# Financial Data Analysis for Time Series Forecasting

## Target Paper
[1] Huang, Hongbin, Minghua Chen, and Xiao Qiao. "Generative Learning for Financial Time Series with Irregular and Scale-Invariant Patterns." The Twelfth International Conference on Learning Representations. 2024.  
[2] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.  
[3] Sohl-Dickstein, Jascha, et al. "Deep unsupervised learning using nonequilibrium thermodynamics." International conference on machine learning. PMLR, 2015.

---

## Introduction

금융 시장 예측 모델의 학습 과정에서 데이터 부족은 주요한 문제이다. 당장 2024 OIBC Challenge (제주 전력 하루전시장 가격 예측 경진대회)에서도 주어진 데이터는 2024년 03월부터 현재로 8개월에 불과하며, 데이터가 충분해진 시점에서는 가격 또한 안정화되기에 ML/DL 기법 기반의 기대 가치가 크지 않다. 또한, 금융 시계열 데이터는 과거 기록으로 인위적인 실험 기반의 데이터 수집이 어렵기에, 시계열 모형 학습에 어려움을 겪고 있는 실정이다.

데이터 부족 문제를 해결하기 위해, 주어진 금융 시장 데이터를 증강하는 접근 방식이 제안되었다. 하지만, 금융 시계열의 도메인 지식 중 “scale-invariance”와 “irregularity”로 인해 기존 증강 기법으로는 합리적인 결과를 얻기 어렵다.

여기서 “Scale-Invariance”란, 패턴(연속된 데이터 값들의 그룹, segment라고도 불림)끼리 기간 또는 값의 스케일의 차이를 고려하지 않은 개형이 유사함을 뜻한다. 패턴들이 기간과 크기는 다양하지만 유한 개의 개형으로 표현될 수 있다는 것이다. 예를 들어, 특정 개형이 짧은 기간과 변동 폭을 갖기도 하고, 더 큰 스케일로 확장되기도 한다. 아래 그림은 투자 심리를 기반으로 가격 차트 해석 방법 중 하나로 소개된다. 마루와 골이 반복하는 패턴 뒤에 등장하는 반등의 끝자락은 가격 등락폭과 상승/하강 기간의 차이는 날 수 있으나 특정 개형들로 설명될 수 있다는 도메인 지식을 나타낸 대표적인 예시이다.

이는 기존의 시계열 데이터의 특성 중 하나인 “Scale-Dependence”와는 다른 금융 데이터만의 특성이다. 기존의 시계열 데이터는 제조에서의 신호 데이터 등 값의 크기인 진폭 자체가 핵심적인 정보로 작용한다. 진폭과 패턴의 길이인 파장 또는 패턴의 개형인 진동수는 매우 높은 상호 연관성을 보이기 때문이다. 아래의 그림은 금융 데이터의 특성인 “Scale-Invariance”와 기존 시계열 데이터의 특성인 “Scale-Dependence”를 시각화한 그림이다.

<img width="988" height="303" alt="image" src="https://github.com/user-attachments/assets/92514d7e-2e29-4944-87c4-2faf4e8d181c" />

다음으로, “Irregularity”란 하나의 특정한 패턴이 발생한 후 또 언제 발생할 지를 모르는 특성을 의미한다. 기존 시계열 데이터는 이와 반대 특성인 “Regularity”를 보이며, 이는 값과 개형이 주기 또는 규칙을 갖고 반복하는 특성이다. 해당 특성은 진동수 푸리에 변환과 같은 주파수 분해 전처리의 근거가 된다. 반대로, 일정하지 않은 간격으로 패턴이 등장하는 금융 시계열 데이터의 특성을 “Irregularity”라고 한다. (상식적으로 금융 시계열 데이터에 Irregularity가 없다면 모두가 가격을 쉽게 예측하고, 투자의 실패가 발생하기 어려울 것이다) 아래의 그림은 “Irregularity”에 대한 시각 자료이다.

<img width="1021" height="325" alt="image" src="https://github.com/user-attachments/assets/df32cd70-0ff8-4bd5-b316-91a102f9264f" />

아래의 그림은 “Scale-Invariance”와 “Irregularity”를 동시에 확인할 수 있는 예시 자료이다. 작은 코끼리 다음에는 중간 코끼리로 코끼리 모양의 개형은 비슷하지만 그 크기와 기간이 다르다는 점에서 “Scale-Invariance”하고, 이 이후에는 더 큰 코끼리 모양의 개형이 올 가능성이 불확실하다는 점에서 “Irregularity”를 보인다.

<img width="658" height="422" alt="image" src="https://github.com/user-attachments/assets/a9d74788-60fa-44f5-ac1c-dd1482b03d8b" />

본 논문에서는 이러한 두 가지 특성을 효과적으로 파악하고, 이를 반영하여 현실적인 금융 데이터를 생성하기 위해 총 세 가지 모듈로 구성된 FTS-Diffusion 프레임워크를 제안한다.

첫 번째 모듈(Pattern Recognition Module)은 시계열 데이터에서 주요 패턴의 개형과 크기를 추출하는 패턴 인지 과정으로써, 금융 시계열 데이터의 scale-invariance와 irregularity를 반영하여 정보가 잘 표현될 수 있도록 데이터를 전처리한다.

두 번째 모듈(Pattern Generation Module)은 전처리된 데이터를 바탕으로 시계열 패턴의 분포를 추정하고 생성하는 과정으로, $\{ \text{패턴의 개형 } p,\ \text{기간 } \alpha,\ \text{값 크기 } \beta \}$ 하에서의 실현된 패턴의 조건부 분포를 활용하여 다양한 기간 및 값 크기와 개형이 주어졌을 때, 합리적인 패턴을 생성한다.

세 번째 모듈(Pattern Evolution Module)은 $\{p,\alpha,\beta\}$를 사상으로 한 ‘개구리 점프 모델링’(마코프 체인 모델) 방식의 순차적인 패턴 과정의 추정 과정으로써, $\{p,\alpha,\beta\}$를 순차적으로 샘플링한다. 샘플링된 $\{p,\alpha,\beta\}$는 직전 시점 패턴 사상에만 종속적인 관계를 갖으며, 패턴 전이 확률 $Q(\{p_j,\alpha_j,\beta_j\}\mid \{p_i,\alpha_i,\beta_i\})$를 추정하여 연속적인 패턴 간의 동적 관계를 모델링하고, 순차적인 샘플링을 통해 시계열의 연속성과 일관성을 보존한다.

---

## Pattern Recognition Module

첫 번째 모듈인 Pattern Recognition Module은 패턴 인지 과정(SISC 알고리즘)으로 주어진 전체 시계열 데이터를 $m$개의 패턴 개형으로 분할하고, 이를 $K$개의 고유한 대표 패턴 개형에 할당하는 군집화 과정이다. 이를 그림으로 표현하면 아래와 같다. 그림에서는 $x_1, x_2, x_3$와 같이 notation을 사용했으나, 본 문서에서는 이를 $s_1, s_2, s_3$로 칭하겠다.

<img width="710" height="330" alt="image" src="https://github.com/user-attachments/assets/6c8af07c-0c2c-4067-9818-5435f371a8c3" />

### 클러스터 중심 초기화(K-Means++ 유사)

첫 번째는 클러스터 중심의 초기화 방식으로 K-Means++ 방식과 유사한 방법을 택한다. 먼저 전체 길이가 $T$인 시계열 데이터에서 지정된 길이 $\ell_{\max}$(하이퍼파라미터)로 중심 후보군을 $T-\ell_{\max}+1$개로 분할한다. 이때, $t$시점을 시작점으로 하는 중심 후보군 $X_{t:t+\ell_{\max}}$는 아래와 같은 그림으로 표현할 수 있다.

<img width="1020" height="400" alt="image" src="https://github.com/user-attachments/assets/3906a5a3-1e7a-49aa-b86b-dd670351752b" />

여기서, 무작위로 선택하여 첫 번째 중심 $p_1$을 선택한다. 이후 남은 패턴들 중에서 현재 선택된 중심들과의 거리가 가장 먼 패턴을 다음 중심으로 선택할 확률을 높게 하여 샘플링하여 두 번째 중심 $p_2$를 선택한다. 해당 방법을 반복적으로 진행하여 $p_K$까지 총 $K$(하이퍼파라미터)개의 중심이 선택될 때, K-Means를 진행한다.

### 최적 길이 결정 및 분할

이후 주어진 데이터로부터 패턴의 최적 길이를 결정하여 $m$개의 segment로 분할한다. 현재 패턴의 시작 시점 $t$가 segment $s_m$일 때, 패턴의 길이
$$
\ell_m^* \;=\; \arg\min_{\substack{p\in \mathcal{P} \\ \ell \in [\ell_{\min},\,\ell_{\max}]}} d\!\big(X_{t:t+\ell},\,p\big)
$$
이다.

이때, $d(X_{t:t+\ell},p)$는 주어진 구간 $[t, t+\ell]$과 대표 패턴 개형 $p$ 사이의 유사도를 측정하는 거리 함수로, 여기서 **DTW** 방법을 사용하여 거리를 측정한다.

단, DTW는 기간 또는 값 크기 차이를 자동으로 보정하지 않기 때문에, SISC 알고리즘에서는 DTW 거리 측정 이전에 각 패턴의 기간(duration scale)을 $\alpha_m=\ell_m^*$, 패턴의 값 크기(magnitude scale)를 $\beta_m=\max(s_m)-\min(s_m)$로 정의한 뒤에, $s_m$를 $\alpha_m$와 $\beta_m$로 **정규화**하여 패턴의 개형을 추출하고 계산한다. 이는 아래 그림과 같다.

<img width="942" height="420" alt="image" src="https://github.com/user-attachments/assets/97352922-3515-4e56-b01f-34165d2140fe" />

**DTW 요약.** DTW는 ‘속도 또는 길이에 따라 움직임이 다른 두 시계열 간의 거리를 측정하는 알고리즘’이다. 아래의 그림과 같이 DTW는 유클리디안 거리와 달리, 시간을 뒤틀어서 두 시계열의 거리가 최소화되는 방향으로 매칭시키는 warping(뒤틀림) 경로를 찾는다.

<img width="1039" height="244" alt="image" src="https://github.com/user-attachments/assets/ceb599f2-6daa-493e-9da4-f3ac9435113d" />

$X$와 $Y$를 DTW 거리를 측정할 두 시계열 데이터라고 가정하자. 두 시계열이 $X=(x_1, x_2, x_3, \dots, x_m)$, $Y=(y_1, y_2, y_3, \dots, y_m)$이라고 할 때 $X, Y$를 나열하여 각 instance 사이의 유클리디안 거리를 나타낸 $m\times m$ 행렬을 만든다. 행렬의 $(i,j)$번째 요소는 두 점 $x_i, y_j$ 간의 유클리디안 거리로 표현되는 것이다.

이후, 각 셀 $(i,j)$에서 두 시계열 간 최소 누적 거리를 계산하는 누적 거리 행렬 $D$를 만든다. $D(i,j)$는 $(i,j)$ 위치에서 $X$와 $Y$를 정렬할 때의 총 최소 거리를 나타내면 아래와 같이 계산된다.

$$
D(i,j)=d(q_i,c_j)+\min\!\left\{\,D(i-1,j-1),\; D(i-1,j),\; D(i,j-1)\,\right\}
$$

이후, 행렬에서 이동 가능한 경로 중 거리 차이 합의 최소를 찾는 것이 DTW이다. 예시 그림은 아래와 같으며, 이를 알고리즘적으로 구현한 것이다.

<img width="923" height="359" alt="image" src="https://github.com/user-attachments/assets/4236b030-1f3d-4370-9c5b-8ce3417a7962" />

이 과정을 통해 후보 길이 $\ell$에 대하여 모든 $K$개의 대표 개형 중심 $p_k$와의 DTW 거리 $d(s_m, p_k)$을 구하고, 가장 작은 거리를 제공하는 $\ell_m^*$를 선택하고, $$X_{t:t+\ell_m^*}$$을 패턴(segment)로 추출한다. 그 결과 패턴
$$
s_m \;=\; X_{t:t+\ell_m^*} \;=\; \{\,p_k,\ \alpha_m,\ \beta_m,\ \mathrm{normalized}_{s_m}\,\}
$$
로 분해된다.

이후 K-Means 클러스터링 과정을 통하여 각각의 패턴들을 $K$개의 중심 중 가장 가까운 중심 $p_k^*$로 할당하고, 반복적으로 중심과의 거리를 업데이트한다. 이 과정을 수렴할 때까지 반복한다.

그 결과, 주어진 전체 시계열 데이터 $X$는
$$
[\{p_i,\ \alpha_1,\ \beta_1,\ \mathrm{normalized}(s_1)\},\ \{p_j,\ \alpha_2,\ \beta_2,\ \mathrm{normalized}(s_2)\},\ \dots]
$$
형태로 $\{\text{대표 패턴 중심},\ \text{패턴},\ \text{기간 크기},\ \text{값 크기}\}$가 도출된다.

---

## Pattern Generation Module

SISC 알고리즘을 기반으로 추출된 centroid를 통해 학습 시계열 데이터를 구성하는 패턴의 개형과 크기들을 모두 추출하고, 패턴의 개형을 기준으로 군집화 했다고 가정하자. 우리의 가정은 추출된 패턴의 개형이 학습 데이터가 내포하는 금융 정보(인사이트) 중 핵심이라는 것이고, 이를 포함하여 유사한 학습 데이터를 생성하는 것이 본론 2에서 다루는 문제이다. 따라서, 생성할 패턴 개형 군집이 결정되었을 때, 해당 군집을 구성하는 패턴 개형의 분포로부터 샘플링하는 생성형 모델로 해당 문제를 해결하고자 한다. 아래의 그림에서 $x_m$은 전부 본 문서에서 $s_m$이라고 생각하면 된다.

<img width="607" height="346" alt="image" src="https://github.com/user-attachments/assets/63329770-eef1-4126-b989-0f7ad7646471" />

해당 파트에서는 위의 그림과 같이 **DDPM**(Denoising Diffusion Probabilistic Model, cf. GAN 또는 VAE와 같이 이외의 생성형 모델도 가능할 것으로 보이나 DDPM을 사용하였 다)로 고정된 길이의 패턴 계형 분포를 학습시키고, **AE**(AutoEncoder)를 통해 패턴의 크기 조절을 학습하도록 했다.

**AE의 Encoder**는 주어진 원본 데이터인 segment $s_m$를 입력으로 받아, 해당 segment의 기간과 값이 Normalized된 결과인 represent $s^0_m$을 출력 학습된다. 이때, segment의 기간은 입력 차원에 대응하므로, 결국 Encoder는 정해진 출력 차원에 Normalized된 값을 도출하도록 학습된다.

**Encoder의 출력값**인 represent $s^0_m$을 Diffusion Model은 입력으로 받아 noise를 점진적으로 첨가하여 Normal 분포를 따르는 차원에 임베딩하고, 해당 차원에서 noise를 점진적으로 제거하여 $s^0_m$과 동일한 패턴의 계형 분포로부터 샘플링된 패턴의 계형을 도출하게 된다. 이를 통해 결국 주어진 패턴의 개형과 유사한 패턴의 개형을 생성하는 **pattern-conditioned Diffusion Model**을 사용하는 것이다.

**Diffusion Model**로부터 도출된 패턴의 계형과 $\alpha_m$ 및 $\beta_m$를 AE의 **Decoder**는 입력으로 받아 주어진 원본 데이터인 segment와 동일한 기간 및 값의 스케일로 변환하는 역할을 하게 된다. 따라서, $\alpha_m$와 $\beta_m$, $\mathrm{Normalized}(s_m)$을 반영하여 원본과 유사한 패턴을 생성하는 **scale-conditioned Decoder**를 사용하는 것이다.

아래는 이를 학습하기 위한 **손실함수**이다. 여기서 $x_m$은 본 문서에서 $s_m$에 해당한다. 앞의 term은 AE의 Decoder를 집중적으로 학습하는 부분으로 본래 패턴 $s_m$의 스케일 복구 오차가 크게 작용한다. 뒤의 term은 DDPM에 집중적으로 학습하는 부분으로, Encoder의 출력인 $s^0_m$을 입력으로 받아 $i$시점에서의 noise 복구 오차로 도출된다.

$$
\mathcal{L}(\theta)
=
\mathbb{E}_{x_m}\!\left[\;\|x_m-\hat{x}_m\|_2^2\;\right]
+
\mathbb{E}_{x_m^{0},\, i,\, \epsilon}\!\left[\;\|\epsilon^{i}-\epsilon_{\theta}(x_m^{i},\, i,\, \mathbf{p})\|_2^{2}\;\right]
$$

---

## DDPM (Denoising Diffusion Probabilistic Models)

Patten Generation Module에서 갑자기 등장한 손실함수는 도대체 왜 저런 식이며, DDPM과 Diffusion Model은 무엇인가라는 의문이 들 것이다. 간단하게 소개하면 **DDPM**(Denoising Diffusion Probabilistic Models, NIPS 2020)은 Diffusion Model의 시초인 **DPM**(Deep Unsupervised Learning using Nonequilibrium Thermodynamics, ICML 2015)의 성능을 높인 방법론이다. 변경점은 크게 2가지(1. 학습 전략 단순화 2. U-Net으로 모형 구조 변경)가 있으나, U-Net (U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015) 구조를 제외하고 “학습 전략”을 중점에 두고 DDPM에 대해 소개하고자 한다. (*여기서 사용되는 $\beta$는 전체적인 흐름 상 사용되던 magnitude factor $\beta$와 다름, 추후 발표에서는 표기를 바꿀 예정)

Diffusion Model은 물리적인 현상인 “확산 (Diffusion)”에서 아이디어가 착안되었다. 투명한 물이 채워진 유리컵에 빨간 잉크를 SME 모양으로 흩뿌렸다고 생각하자. 빨간 잉크가 총 100개의 잉크 입자들로 이루어져 있을 때, 각 입자들은 시간이 지남에 따라 위치가 서서히 변화할 것이며 SME 모양은 서서히 일그러질 것이다. 결국 충분한 시간 이후에는 입자가 고르게 분포할 것이다. 통계적으로 말하면 입자가 간단한 분포(e.g. Unif, Normal)를 따를 것이다. 우리가 물리학을 통달하여 입자의 위치가 짧은 시간 직전에 어떻게 분포하는 지 안다는 가정하에 거꾸로 되돌리는 과정을 생각해보자. 현 시점 입자들 위치만으로 직전 시점 입자들의 위치 분포를 추론하고, 이를 반복하여 처음 시점이 되면 원래의 입자들의 위치 분포를 추정할 수 있을 것이다. 또한 각 시점 간격에서 위치 분포는 매우 간단하다고 해도, 이와 같은 과정이 누적되면 다양한 경우의 입자 위치 분포를 표현할 수 있을 것이다.

DDPM은 이러한 매커니즘을 데이터에 그대로 적용한다. 복잡한 분포를 따르는 원본 데이터는 사전에 정의한 간단하고 편차가 작은 노이즈를 시점에 따라 더하면, 최종적으로 간단한 분포를 따르는 노이즈 덩어리가 된다는 것이다. 그리고, 이를 역으로 복원하면 간단한 분포로부터 복잡한 원본 데이터의 분포를 표현할 수 있다. 시점에 따라 노이즈가 더해지는 것은 아래의 개구리 점프 모델 (마코프 체인)로 표현할 수 있다.

<img width="1013" height="205" alt="image" src="https://github.com/user-attachments/assets/00444481-543a-4dc0-ab90-3559f377252e" />

지금부터는 **사상의 정의**에 유의하여 보자. 그러면 현 시점인 $t$시점에서의 사상 $X_t$는 직전 시점의 사상 $X_{t-1}$에 사전에 정의된 작은 노이즈 사상 $\mathcal{N}(0, \beta_{t-1}\mathbf{I})$이 더해진 조건부 사상
$$
X_t\mid X_{t-1} \;=\; \sqrt{1-\beta_{t-1}}\, X_{t-1} \;+\; \sqrt{\beta_{t-1}}\, \mathcal{N}(0,\mathbf{I})
$$
로 표현된다. 이때, 사상 $X_{t-1}$을 $\sqrt{1-\beta_{t-1}}$로 normalize하는 부분에 집중해보자. 이는 최종 시점인 $T$시점에서의 노이즈 덩어리 사상 $X_T \sim \mathcal{N}(0,\mathbf{I})$일 때, 사상 $X_{T-1}$은 아주 작은 노이즈 사상이 제외된 사상이기에 $\mathcal{N}(0,\mathbf{I})$와 거의 유사한 사상이라 할 수 있다. $\mathrm{Var}(X_{T-1})$ 역시 Identity matrix와 매우 유사할 것이며 $T-1$시점의 노이즈 사상의 분산 $\mathrm{Var}(\mathcal{N}(0, \beta_{T-1}\mathbf{I}))=\beta_{T-1}$일 것이다. $T$시점의 사상
$$
X_T \;=\; \sqrt{1-\beta_{T-1}}\, X_{T-1} \;+\; \sqrt{\beta_{T-1}}\, \mathcal{N}(0,\mathbf{I})
$$
라 하고, 이와 같은 형식으로 $0$시점까지 사상을 확장하면 모든 시점에서의 사상 $X_t$들의 분산을 $\mathbf{I}$로 고정시킬 수 있다. 아래 수식은 이를 나타낸 것이다.

$$
q(\mathbf{x}_{1:T}\mid \mathbf{x}_0)
:= \prod_{t=1}^{T} q(\mathbf{x}_t\mid \mathbf{x}_{t-1}),
\qquad
q(\mathbf{x}_t\mid \mathbf{x}_{t-1})
:= \mathcal{N}\!\left(\mathbf{x}_t;\, \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\, \beta_t \mathbf{I}\right)
$$

$q$는 결국 시점 $t$별로 사전에 정의한 노이즈 사상을 더해준 **조건부 사상**으로, 사실상 해당 사상의 분포는 우리가 알 수 있다. 따라서 사전에 정의된 조건부 사상 $q$와 원본 데이터 사상의 실현치 $x'_0$ ($X_0$에서 샘플링 결과)를 이용하여 시점별 실현치 $[x'_t]_{t=1}^T$들을 도출하고, 사상 $X_t$의 분포로부터 사상 $X_{t-1}$의 분포인 **조건부 사상** $X_{t-1}\mid X_t$의 분포를 추정하는 것이 우리의 최종 목표이다.

앞으로 자주 사용될 상수로 $\alpha_t = 1-\beta_t$라 하고, $\bar\alpha_t = \prod_{i=1}^t \alpha_i$라 하겠다. 이를 이용하여 상수 $\alpha_t,\ \bar\alpha_t,\ X_0$로 $X_t$를 표현하면,
$$
X_t\mid X_{t-1} \;=\; \sqrt{\alpha_t}\, X_{t-1} \;+\; \sqrt{1-\alpha_t}\, \mathcal{N}(0,\mathbf{I})
$$
이므로

$$
\begin{aligned}
X_t\mid X_{t-2}
&= \sqrt{\alpha_t}\,\big[\sqrt{\alpha_{t-1}}\,X_{t-2} + \sqrt{1-\alpha_{t-1}}\,\mathcal{N}(0,\mathbf{I})\big]
+ \sqrt{1-\alpha_t}\,\mathcal{N}(0,\mathbf{I}) \\
&= \sqrt{\alpha_t\alpha_{t-1}}\,X_{t-2}
+ \big[\sqrt{\alpha_t(1-\alpha_{t-1})} + \sqrt{1-\alpha_t}\big]\mathcal{N}(0,\mathbf{I}) \\
&= \sqrt{\alpha_t\alpha_{t-1}}\,X_{t-2}
+ \sqrt{1-\alpha_t\alpha_{t-1}}\,\mathcal{N}(0,\mathbf{I})
\quad\text{[\,}\mathcal{N}(0,a)+\mathcal{N}(0,b)=\mathcal{N}(0,a+b)\text{\,]}.
\end{aligned}
$$

따라서

$$
X_t\mid X_0 \;=\; \sqrt{\bar\alpha_t}\,X_0 \;+\; \sqrt{1-\bar\alpha_t}\,\mathcal{N}(0,\mathbf{I})
$$

[계속해서 확장하여 일반화하면]임을 알 수 있다.

여기서 문제는 사전에 정의한 $q$로 도출된 사상 $X_t$의 분포는 사실상 **조건부 사상** $X_t\mid X_0$의 분포이며, 실현치 $x'_0 \sim X_0$에 종속적이다. 따라서, 조건부 사상 $X_{t-1}\mid X_t$의 분포는 알기 힘들다. 하지만 조건부 사상 $X_{t-1}\mid X_0$의 분포는 항상 알고 있기에, 조건부 사상 $X_{t-1}\mid(X_t, X_0)$의 분포 또한 알 수 있다. 위의 유도식에서 통해 $X_{t-1}\mid X_0,\ X_t\mid X_0,\ X_t\mid X_{t-1}$의 분포를 이용하여,
$$
X_{t-1}\mid(X_t, X_0)
~\propto~
f(X_{t-1}\mid X_0)\cdot f(X_t\mid X_{t-1}, X_0)
~=~
f(X_{t-1}\mid X_0)\cdot f(X_t\mid X_{t-1})
$$
[ $X_t\mid X_{t-1}$과 $X_0$은 독립 ] 임을 쓸 수 있다.

따라서, 조건부 사상들이 따르는 가우시안 분포를 활용하여 계산하면, 이 또한 가우시안 분포를 따르며 **기댓값**은 아래와 같다.
$$
\begin{aligned}
\mathbb{E}\big[X_{t-1}\mid(X_t, X_0)\big]
&=
\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{\beta_t(1-\bar\alpha_t)}\,(X_t\mid X_0)
+
\frac{\sqrt{\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,X_0 \\
&=
\frac{1}{\sqrt{\alpha_t}}
\left(
X_t\mid X_0 - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\mathcal{N}(0,\mathbf{I})
\right).
\end{aligned}
$$
[ $X_t\mid X_0 = \sqrt{\bar\alpha_t}\, X_0 + \sqrt{1-\bar\alpha_t}\,\mathcal{N}(0,\mathbf{I})$ 이용 ]

여기서, 뒤쪽 term 중에 $\sqrt{1-\bar\alpha_t}\,\mathcal{N}(0,\mathbf{I})$는 $X_t\mid X_0$ 분포에서 출발한 것으로 $X_0$의 분포와는 독립적으로 시점 1부터 $t$까지 가해진 노이즈 분포로 표현된다. 따라서, $\mathcal{N}(0,\mathbf{I})$의 실현치를 알면, 시점 $t$에서의 노이즈가 들어간 실현치인 $x'_t\mid x'_0=\sqrt{\bar\alpha_t}\,x'_0+\sqrt{1-\bar\alpha_t}\,\epsilon$로 직전 시점 $X_{t-1}$의 **MLE**를 알 수 있다. ($X_{t-1}$이 가우시안 분포를 따르므로, 기댓값이 MLE)

결과적으로 DDPM의 모델 $\epsilon_\theta$은 아래의 Algorithm 1과 같이, 시점 $t$와 해당 시점에서 실현치 $x'_t\mid x'_0 = \sqrt{\bar\alpha_t}\,x'_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$가 입력으로 들어가면, 해당 시점까지의 노이즈 $\epsilon$을 추론하도록 학습한다.

<img width="764" height="354" alt="image" src="https://github.com/user-attachments/assets/fceb1a94-d6bf-4d0e-bf3e-94c8929810de" />

**추론**의 경우, $T$시점에서의 실현치 $x'_T$부터 입력으로 들어가 추론된 노이즈 $\epsilon$을 사용한다. 직전 시점의 사상 $X_{T-1}\sim \mathcal{N}(\mathbb{E}[X_{T-1}\mid (x'_0, x'_T)], \sigma_t^2\mathbf{I})$에서 실현치 $x'_{T-1}$을 구하고, 이를 $x'_0$의 실현치를 도출할 때까지 반복한다. (이때, $\sigma_t$는 $\beta_t$처럼 사전에 정의된 각 시점별 사상의 분산으로 하이퍼파라미터이다). 이는 아래의 Algorithm 2에 나타나있다.

<img width="748" height="370" alt="image" src="https://github.com/user-attachments/assets/aedcf2a0-702b-4529-a421-8ead02d25fbc" />

---

## Pattern Evolution Module

다시, 본 논문으로 돌아오자. Pattern Generation Module은 결국 좋은 생성형 모형을 사용하면서, scale invariance 특성을 고려하여 패턴의 개형 분포를 추정하도록 했다. 이제 는 이러한 패턴들을 연속적인 과정을 모델링해야 한다. 여기선 irregularity의 특성을 고려하여, 금융 데이터의 각 패턴 개형이 갑작스럽게 등장하는 현실을 반영해야한다. 현실 세계에서 패턴은 다양한 요인에 종속적으로 결정되지만, 제안 방법론은 직후 패턴이 직전 패턴과만 종속적인 관계를 갖는다고 가정한다. 결국 모든 직후 패턴은 직전 패턴 정보인 $\{p,\alpha,\beta\}$가 주어질 때, 실현될 이산 확률을 추론할 수 있다는 것이다. 이를 아래와 같이 직전 패턴의 인덱스 $m$에 대하여 직후 패턴의 인덱스 $m+1$를 활용하여 수식으로 나타냈다.

$$
(\hat{p}_{m+1},\, \hat{\alpha}_{m+1},\, \hat{\beta}_{m+1})
= \phi(p_m,\, \alpha_m,\, \beta_m).
$$

또한 $p$의 경우 학습 데이터에 있는 패턴 중 하나로 분류하는 관점으로 보고, $\alpha$와 $\beta$는 각각의 scale을 회귀하는 관점으로 접근하여, “개구리 점프 모델”의 각 사상이 다음 사상으로 점프할 확률을 추정하는 $\phi$를 학습하기 위한 손실함수를 아래와 같이 설계하였다.
$$
\mathcal{L}(\phi)
=
\mathbb{E}_{x_m}\!\left[
\ell_{\mathrm{CE}}\!\left(p_{m+1},\, \hat{p}_{m+1}\right)
+ \left\|\alpha_{m+1}-\hat{\alpha}_{m+1}\right\|_2^{2}
+ \left\|\beta_{m+1}-\hat{\beta}_{m+1}\right\|_2^{2}
\right].
$$

---

## Conclusion

최종적으로 위의 3개 모듈을 활용하면 전체적인 아키텍처가 아래와 같다. 주어진 학습 데이터 $X$로부터 SISC 알고리즘으로 패턴의 개형과 크기로 분해한다. 이를 활용하여 $\{\text{패턴의 개형}, \text{ 크기}\}$ 하의 패턴을 생성하는 Pattern Generation Module을 학습 하고, $X$의 패턴이 진행되는 과정을 “개구리 점프 모델링”하여 패턴이 나오는 이산 확률 과정 Pattern Evolution Module을 학습한다. 이후 Pattern Evolution Module로부터 다음 시점의 $\{\text{패턴의 개형}, \text{ 크기}\}$를 샘플링하고, 해당 샘플링 값을 Pattern Generation Module의 입력으로 사용하여 패턴을 생성한다.

<img width="1008" height="296" alt="image" src="https://github.com/user-attachments/assets/cba73180-15ee-45b4-884b-4c25aa76d5b7" />

전체적으로 금융 시계열 데이터의 특성을 정의하여 이를 해결하기 위한 시도로 보인다. 하지만, 과업이 현실 데이터와 유사한 학습 데이터 증강이기에 해당 증강 기법이 효과적인지는 의문이다. 실험에서는 해당 증강 기법과 여타 생성형 모델 기반의 증강 기법을 비교하였으며, 증강된 데이터로 학습된 예측 모델의 성능을 평가했다. 다만, 다른 증강 기법은 증강 데이터가 많아질 수록 과적합을 유발하는 반면, 제안 방법론은 과적합을 유발하지 않았지만 성능 향상이 보이지도 않았다. 이는 증강된 데이터가 현실 금융 데이터 분포를 잘 추정한 것 보다는 증강된 데이터가 불규칙한 패턴들로 구성되어 있어, 이를 기반으로 학습된 모델이 노이즈에 강건하게 되었다고 판단된다. . 이를 서론에서 언급 했듯 예측 정확도 향상을 위한 용도로는 적합할 지 의문이 든다.
