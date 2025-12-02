---
title: "SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling (NeurIPS 2023)"
date: 2025-05-28 12:00:00 +0900
categories: [iSME, seminar]
math: true
permalink: /notes/isme/simmtm/
---

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/604c5239-9138-4142-a1d2-de5b202898b9" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/681f7ff9-a118-4df9-88ab-ed19535ad90c" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/7b368665-42c3-4f99-accb-71f600f8c56d" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/685daf7a-a29d-4987-95b7-116792e2d26c" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/a694d1e7-7c8e-4fa4-a7f2-0bcec9d717c0" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/aa4e23b1-10d2-420e-851d-7e995487a9fc" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/f6faab4a-3795-4c70-a4d1-ef610b4573fb" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/851b1a34-70f2-41fd-a5df-91484667b04b" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/cf8c5c66-b770-4c7b-9e6e-c89285d943d4" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/319bada1-ecd3-47c2-a999-5258d8638988" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/69589c3f-bf5d-41f0-a3b0-d039e6f3581a" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/34212574-97eb-4c4d-abb6-20f44b698e65" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/3854ed4d-3fca-4e14-a086-d3a5fd6d5d80" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/22aa4535-0ce5-4b61-a406-0c1ef81907fa" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/90827e89-3a06-41dd-b1e9-a402104ce9b5" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/548525fa-13b7-447d-966b-bfcc67524ff3" />


# SimMTM: 시계열 마스킹 기반 사전학습

## 1) 문제의식 & 동기

contrastive learning은 시계열의 **시간적 연속성**과 **구조적 정보**를 충분히 반영하지 못해 복원 성능/표현 학습에서 한계를 보이는 경우가 많다. 무작위 마스킹 후 **단일 시계열만**으로 복원하면 **국소 정보**에 의존해 전체 흐름(temporal variation)과 **구조 전환점**을 놓치기 쉽다.  
이를 해결하기 위해 **SimMTM**(NeurIPS 2023)은 **여러 마스킹 시계열을 함께 활용**해 **구조적으로 유사한 이웃**들의 정보를 가중 평균으로 끌어와 복원한다. 핵심은 값 그 자체보다 **시계열이 공유하는 잠재 구조(Manifold)** 를 학습·활용하는 것.

---

## 2) Manifold 가정

- 시계열들은 고차원에 무작위로 흩어진 게 아니라 **저차원 manifold** 위에 놓여 있음.  
- 원본 시계열(파란 점)에서 **일부 마스킹(빨간 점)** 되면 정보 손실로 manifold에서 이탈.  
- 단일 시계열 복원은 이 왜곡에서 출발해 **오류가 누적**될 수 있음.  
- **여러 시계열의 부분 정보**를 결합하면 manifold 위의 **정합된 위치**로 복원 가능.

---

## 3) 전체 파이프라인(요약)

1) 원본과 마스킹 시계열을 **인코더**에 넣어 **point-wise(시점별)** 표현 획득 → **프로젝터**로 **series-wise(전체)** 표현 생성  
2) **series-wise 유사도**로 구조 이웃을 찾고, **point-wise 가중 평균**으로 시점별 표현 복원  
3) **디코더**로 최종 시계열 복원  
4) **복원 손실 + 제약 손실(InfoNCE 스타일)** 로 joint 학습

---

## 4) 표기/정의

$$
N:\ \text{시계열 개수},\quad
L:\ \text{길이},\quad
D:\ \text{point 임베딩 차원},\quad
d_s:\ \text{series-wise 임베딩 차원},\quad
M:\ \text{마스킹 버전 수}.
$$

마스킹 데이터 집합:

$$
\{\bar{x}_{ij}\}_{j=1}^{M}\ \text{는 시계열 } x_i \text{의 } M \text{개 마스킹 버전}.
$$

인코더 \(f_\theta\), 프로젝터 \(g\), 디코더 \(d\):

$$
z_i[t] = f_\theta(x_i)[t]\ \in \mathbb{R}^D,\qquad
s_i = g\!\big(f_\theta(x_i)\big)\ \in \mathbb{R}^{d_s}.
$$

$$
\bar{z}_{ij}[t] = f_\theta(\bar{x}_{ij})[t],\qquad
\bar{s}_{ij} = g\!\big(f_\theta(\bar{x}_{ij})\big).
$$

---

## 5) 유사도 행렬과 이웃 가중치(softmax)

코사인 유사도:

$$
R_{a,b}
=\frac{\langle s_a,\, s_b\rangle}{\|s_a\|\,\|s_b\|}.
$$

온도 하이퍼파라미터:

$$
\tau>0.
$$

복원 대상 \(\bar{x}_{ij}\)의 이웃 가중치:

$$
\alpha_{ij\to k}
=
\frac{\exp\!\big(R_{\bar{s}_{ij},\, s_k}/\tau\big)}
{\sum_{\ell}\exp\!\big(R_{\bar{s}_{ij},\, s_\ell}/\tau\big)}.
$$

---

## 6) Point-wise 가중 평균 복원 → 최종 복원

시점 \(t\)별 가중합:

$$
\hat{z}_{ij}[t]
=\sum_{k}\alpha_{ij\to k}\; z_{k}[t].
$$

스택하여 시계열 임베딩:

$$
\hat{Z}_{ij}
=\mathrm{stack}\!\big(\hat{z}_{ij}[1{:}L]\big)\ \in \mathbb{R}^{L\times D}.
$$

디코더 복원:

$$
\hat{x}_{ij}=d\!\big(\hat{Z}_{ij}\big).
$$

---

## 7) 학습 목표: Reconstruction + Constraint

복원 손실:

$$
\mathcal{L}_{\mathrm{rec}}
=
\frac{1}{N}\sum_{i=1}^{N}\frac{1}{M}\sum_{j=1}^{M}
\big\|x_i-\hat{x}_{ij}\big\|_2^2.
$$

제약 손실(InfoNCE 스타일). 전체 표현 집합과 양의 쌍:

$$
S=\{\,s_i,\ \bar{s}_{ij}\,\},\qquad
S^{+}(s)=\{\text{동일 원본에서 유도된 positive 표현들}\}.
$$

$$
\mathcal{L}_{\mathrm{con}}
=
-\sum_{s\in S}
\sum_{s'\in S^{+}(s)}
\log
\frac{\exp\!\big(R_{s,s'}/\tau\big)}
{\sum_{u\in S\setminus\{s\}}\exp\!\big(R_{s,u}/\tau\big)}.
$$

최종 목적함수:

$$
\min_{\Theta}\ \mathcal{L}
=
\mathcal{L}_{\mathrm{rec}}
+
\lambda\,\mathcal{L}_{\mathrm{con}},
\qquad
\Theta=\{\theta,\, g,\, d\}.
$$

---

## 8) 단계 요약

- **동기**: 단일 시계열 복원은 국소 정보만 보고 **흐름이 어긋난 복원**을 낳을 수 있음.  
- **핵심 아이디어**: 시계열 간 **잠재 구조(manifold)** 를 학습하고, **유사 이웃의 가중 평균**으로 자연스럽고 정확한 복원 수행.  
- **표현 학습**: point-wise와 series-wise **두 수준 표현**을 함께 학습 → 구조적 유사도를 적극 활용.  
- **복원 파이프라인**:  
  1) **여러 마스킹** 생성  
  2) **인코더/프로젝터**로 표현 추출  
  3) **유사도 행렬** 기반 **point-wise aggregation**  
  4) **디코더**로 시계열 복원  
  5) **복원+제약 손실**로 joint 학습

---

## 9) 실험 결과

- **In-domain/Cross-domain** 모두에서 **최상위 성능**:  
  $$
  \text{MSE}\ \downarrow,\qquad \text{Accuracy}\ \uparrow.
  $$
- **마스킹 비율↑** 상황에서도 **강건**: contrastive 기반 대비 구조 손실 적음.  
- **소량 파인튜닝**에서도 우수: 데이터 제한 환경에서 **현실 적용성** 높음.  
- **마스킹 설정 가이드** 제공: 마스킹 개수/비율 조합에 대한 **실전 튜닝 기준** 제시.

---

## 10) 왜 잘 작동하나?

- 단일 복원의 **국소성 한계**를, **series-wise 구조 이웃의 앙상블**로 보완.  
- **복원(값 일치)** 과 **표현(구조 정돈)** 을 **동시에 최적화** → 다운스트림 작업에 유리.  
- **Manifold 관점**에서 **흐름/전환점**까지 보존하는 복원.


