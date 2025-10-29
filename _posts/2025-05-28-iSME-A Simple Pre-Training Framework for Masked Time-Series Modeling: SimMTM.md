---
title: "SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling (NeurIPS 2023)"
date: 2025-05-28 12:00:00 +0900
categories: [iSME, seminar]
tags: [SimMTM, self-supervised, time-series, masking, manifold, contrastive, cosine, reconstruction]
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

contrastive learning은 시계열의 **시간적 연속성**과 **구조적 정보**를 충분히 반영하지 못해, 복원 성능과 표현 학습에서 한계를 자주 보인다. 무작위 마스킹 후 **단일 시계열만**으로 복원하면 **국소 정보**에 과적합되어 전체 흐름(temporal variation)과 **구조 전환점**을 놓친다.  
이를 해결하기 위해 **SimMTM**(NeurIPS 2023)은 **여러 마스킹 시계열을 함께 활용**해 **구조적으로 유사한 이웃**들의 정보를 가중 평균으로 끌어와 복원한다. 핵심은 값 자체보다 **시계열이 공유하는 잠재 구조(Manifold)** 를 학습·활용한다는 점.

---

## 2) Manifold 가정

- 시계열들은 고차원에 랜덤하게 흩어진 게 아니라, **구조를 가진 저차원 manifold 위**에 존재.  
- 원본 시계열(파란 점) → 일부 마스킹(빨간 점)으로 **manifold에서 이탈**.  
- 단일 시계열 복원은 이 왜곡된 지점에서 **오류 누적**.  
- **여러 시계열의 부분 정보를 결합**하면 manifold 위의 **정합된 위치**로 복원 가능.

---

## 3) 전체 프레임워크 개요

SimMTM은 마스킹된 시계열을 복원하기 위해 **두 수준의 표현**을 학습한다.

1) **Point-wise representation**: 시점별 임베딩(로컬 패턴)  
2) **Series-wise representation**: 전체 시계열 임베딩(글로벌 구조)

흐름은 다음과 같다.

1. 여러 마스킹 버전 생성  
2. 인코더(예: CNN)로 point-wise 임베딩 → 프로젝터(MLP)로 series-wise 임베딩  
3. series-wise **유사도 행렬**로 **구조 이웃**을 찾고, 이를 이용해 **point-wise 가중 평균 복원**  
4. 디코더(MLP)로 최종 시계열 복원  
5. **복원 손실 + 제약 손실(InfoNCE 스타일)** 로 joint 학습

---

## 4) 데이터/표기 설정

- 시계열 개수: \(N\), 길이: \(L\), point 임베딩 차원: \(D\)  
- 각 시계열 \(x_i\)에 대해 **서로 다른 마스킹** \(M\)개 생성: \(\{\bar{x}_{ij}\}_{j=1}^{M}\)  
- 인코더 \(f_\theta\), 프로젝터 \(g\), 디코더 \(d\)

**Point-wise/Series-wise 표현**
- \(z_i[t] = f_\theta(x_i)[t] \in \mathbb{R}^D\)  
- \(s_i = g\!\big(f_\theta(x_i)\big) \in \mathbb{R}^{d_s}\)  
- \(\bar{z}_{ij}[t] = f_\theta(\bar{x}_{ij})[t]\), \(\; \bar{s}_{ij} = g\!\big(f_\theta(\bar{x}_{ij})\big)\)

---

## 5) 유사도 행렬과 이웃 가중치

모든 series-wise 벡터 쌍의 **코사인 유사도**:
$$
R_{a,b}
=\frac{\langle s_a,\, s_b\rangle}{\|s_a\|\,\|s_b\|}.
$$

전체 \(N(M{+}1)\)개 series-wise 표현으로 **유사도 행렬** \(R\in\mathbb{R}^{N(M+1)\times N(M+1)}\) 구성.  
복원 대상 \(\bar{x}_{ij}\)의 이웃 가중치는 softmax로 산출:
$$
\alpha_{ij\to k}
=\frac{\exp\!\big(R_{\bar{s}_{ij},\, s_k}/\tau\big)}
{\sum_{\ell}\exp\!\big(R_{\bar{s}_{ij},\, s_\ell}/\tau\big)},
$$
여기서 \(\tau\)는 temperature.

---

## 6) Point-wise 가중 평균 복원 → 최종 복원

각 시점 \(t\)에서 이웃들의 point-wise 임베딩을 가중합:
$$
\hat{z}_{ij}[t]
=\sum_{k}\alpha_{ij\to k}\; z_{k}[t],
\qquad
\hat{Z}_{ij}
=\mathrm{stack}\!\big(\hat{z}_{ij}[1{:}L]\big).
$$

디코더를 통해 **최종 시계열 복원**:
$$
\hat{x}_{ij}=d\!\big(\hat{Z}_{ij}\big).
$$

---

## 7) 학습 목표: Reconstruction + Constraint

**Reconstruction Loss** — 원본과 복원 간 L2 제곱 손실의 평균:
$$
\mathcal{L}_{\mathrm{rec}}
=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{M}\sum_{j=1}^{M}
\big\|x_i-\hat{x}_{ij}\big\|_2^2.
$$

**Constraint Loss (InfoNCE-style)** — 같은 시계열의 표현은 가깝게, 다른 시계열은 멀게:
$$
\mathcal{L}_{\mathrm{con}}
=-\sum_{s\in S}
\sum_{s'\in S^{+}(s)}
\log
\frac{\exp\!\big(R_{s,s'}/\tau\big)}
{\sum_{u\in S\setminus\{s\}}\exp\!\big(R_{s,u}/\tau\big)}.
$$

- \(S\): 전체 series-wise 표현 집합(원본+마스크)  
- \(S^{+}(s)\): 같은 원본에서 나온 positive set

**최종 목적함수**
$$
\min_{\Theta}\;\mathcal{L}
=\mathcal{L}_{\mathrm{rec}}+\lambda\,\mathcal{L}_{\mathrm{con}},
\qquad
\Theta=\{\theta,\, g,\, d\}.
$$

---

## 8) 단계별 파이프라인

1. **Masking**: 각 \(x_i\)에서 무작위 마스킹 \(M\)개 생성 → \(\{\bar{x}_{ij}\}\)  
2. **Encoding/Projection**: \(f_\theta\)로 point-wise, \(g\)로 series-wise 임베딩 산출  
3. **Similarity & Aggregation**: 유사도 행렬 \(R\)로 이웃 가중치 \(\alpha\) 계산 → 시점별 가중 평균  
4. **Decoding**: \(d\)를 통해 \(\hat{x}_{ij}\) 복원  
5. **Training**: \(\mathcal{L}_{\mathrm{rec}}+\lambda\mathcal{L}_{\mathrm{con}}\) 최소화

---

## 9) 실험 결과

- **In-Domain & Cross-Domain** 모두에서 SimMTM이 **최상위 성능**.  
  - \(x\)-축: Forecasting MSE↓, \(y\)-축: Classification Accuracy↑ — **붉은 별**(SimMTM)이 우상향/좌하향 목표를 동시에 달성.
- **마스킹 비율 변화**에 강건.  
  - TST 등 contrastive 기반은 비율↑ 시 구조 손실로 성능 급락, SimMTM은 **성능 유지**.
- **저데이터 파인튜닝**에서도 우수.  
  - 제한된 표본에서도 복원/표현 품질 유지 → **현실 적용성** 높음.
- **마스킹 설정 가이드** 제공.  
  - 마스킹 개수·비율 조합 실험으로 **실전 튜닝 기준** 제시.

---

## 10) 왜 잘 작동하나?

- 단일 시계열 복원 방식의 **국소성 한계**를, **series-wise 유사도**로 찾은 **구조 이웃의 앙상블**로 보완.  
- 복원(값 일치)과 표현(구조 정돈)을 **동시에** 최적화 → Downstream에 유리.  
- “**manifold에서의 복원**” 관점으로 이해하면, 왜 **흐름(transition, turning point)** 까지 잘 살리는지 설명 가능.

---

