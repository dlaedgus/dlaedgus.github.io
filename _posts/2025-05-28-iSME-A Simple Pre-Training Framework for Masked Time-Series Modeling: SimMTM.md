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



# SimMTM — 시계열 마스킹 기반 사전학습 (발표용 정리본)

> 발표 대본 내용을 그대로 살리되, 논문 정리 형식 + 수식 보강.

---

## 1) 배경 & 문제의식
- contrastive는 시간적 연속/구조 반영이 약해 복원·표현 품질 한계.
- 무작위 마스킹 후 **단일 시계열**만 보고 복원 → 전역 맥락 상실, 오프셋/왜곡.
- 핵심은 **변화의 패턴**. 전환점 가려지면 단일 복원으로는 맥락 회복이 어려움.
- **SimMTM(NeurIPS 2023)**: 단순하지만 효과적인 self-supervised 프레임워크.

---

## 2) 핵심 아이디어 — Multiple Masked Modeling
- 대상 + **여러 마스킹 시계열**을 함께 사용.
- **series-wise** 유사도로 이웃을 고르고, **point-wise** 표현을 **softmax 가중합**해 복원.
- 수치보다 **공유된 저차원 manifold** 구조를 학습·활용.

---

## 3) Manifold 가정
- 시계열은 고차원 공간에 무작위가 아니라 **구조적 저차원 manifold** 위에 존재.
- 마스킹 → 정보 손실로 manifold에서 이탈.
- 여러 이웃의 정보를 가중 결합해 manifold 근방으로 **복원**.

---

## 4) 파이프라인 요약
1. **마스킹**: 각 시계열에서 서로 다른 구간을 가린 **M개 버전** 생성.  
2. **표현**: Encoder → **point-wise** \(z\), Projector → **series-wise** \(s\).  
3. **유사도 행렬** \(R\): 모든 \(s\) 쌍 코사인 유사도.  
4. **가중 집계 복원**: 이웃 point-wise를 softmax 가중합 → \(\hat{z}\) → Decoder → \(\hat{x}\).  
5. **학습**: \(\mathcal{L}_{\mathrm{rec}}+\lambda \mathcal{L}_{\mathrm{con}}\) 최소화.

---

## 5) 단계별 디테일 (대본 → 수식화)

### 5.1 마스킹(입력 확장)
배치 \(\{x_i\}_{i=1}^{N},\ x_i\in\mathbb{R}^{L\times C}\).  
각 \(x_i\)에서 서로 다른 구간을 가리는 **M개** 마스크 생성:
$$
\{x_i^{\,j}\}_{j=1}^{M}=\mathrm{Mask}_r(x_i),
\qquad
X=\bigcup_{i=1}^{N}\Big(\{x_i\}\cup\{x_i^{\,j}\}_{j=1}^{M}\Big).
$$

### 5.2 인코더/프로젝터(표현)
$$
Z=\mathrm{Enc}(X)\in\mathbb{R}^{L\times d},
\qquad
S=\mathrm{Proj}(Z)\in\mathbb{R}^{d}.
$$

### 5.3 시리즈 간 유사도 행렬 \(R\)
$$
R_{u,v}=\frac{u^\top v}{\lVert u\rVert\,\lVert v\rVert},
\qquad
R\in\mathbb{R}^{D\times D},\ \ D=N(M+1).
$$

### 5.4 시점별 가중 집계(핵심 복원)
복원 대상 \(x_i\)의 시점 \(t\):
$$
\hat{z}_i(t)
=\sum_{k\neq i}
\underbrace{\frac{\exp\!\big(R_{s_i,s_k}/\tau\big)}
{\sum\limits_{v\neq i}\exp\!\big(R_{s_i,s_v}/\tau\big)}}_{\text{softmax 가중치}}
\, z_k(t),
\qquad
\hat{Z}_i=[\hat{z}_i(1),\dots,\hat{z}_i(L)].
$$
Decoder로 최종 복원:
$$
\hat{x}_i=\mathrm{Dec}(\hat{Z}_i).
$$

---

## 6) 학습 목표(손실)

### 6.1 Reconstruction Loss
$$
\mathcal{L}_{\mathrm{rec}}=\sum_{i=1}^{N}\big\lVert \hat{x}_i-x_i\big\rVert_2^2.
$$

### 6.2 Constraint(Contrastive) Loss
- positive: 같은 시계열(원본–마스크)  
- negative: 다른 시계열(및 그 마스크)
$$
\mathcal{L}_{\mathrm{con}}
=-\sum_{s\in S}\ \sum_{s^{+}\in S_i^{+}}
\log\frac{\exp\!\big(R_{s,s^{+}}/\tau\big)}
{\sum\limits_{s'\in S\setminus\{s\}}\exp\!\big(R_{s,s'}/\tau\big)}.
$$

### 6.3 최종 목적함수
$$
\min_{\Theta}\ \ \mathcal{L}_{\mathrm{rec}}+\lambda\,\mathcal{L}_{\mathrm{con}}.
$$

---

## 7) 왜 단일 복원은 한계인가?
- 국소 단서만 보고 복원 → 전역 흐름을 잘못 해석(오프셋, 왜곡).
- 전환점/핵심 구간이 가려지면 전체 구조 파악 실패.
- 이웃 기반 가중 집계로 전역 맥락 **회복**.

---

## 8) 실험 포인트
- **In-/Cross-Domain**: 예측(MSE↓)·분류(Acc↑) 모두 상위권. 붉은 별(SimMTM) 우세.  
- **마스킹 비율↑**: contrastive 계열은 급락, SimMTM은 안정(이웃 집계 보완).  
- **Few-shot**: 소량 데이터 전이 성능 강함.  
- **마스킹 아블레이션**: 개수/비율 가이드 제시.

---

## 9) 장단점
**장점**: 단순 블록(Enc/Proj/Dec), 전역 구조 반영, 복원+표현 동시 개선, 데이터 부족에도 강건.  
**한계**: 배치 다양성/이웃 질에 민감, \(r,M,\tau,\lambda\) 튜닝 필요.

---

## 10) 실무 체크리스트
- Encoder: 1D-CNN/Transformer, Projector: 소형 MLP, Decoder: MLP  
- 초기값: \(r\in[0.25,0.5],\ M\in\{2,3\},\ \tau\in[0.05,0.2],\ \lambda\in[0.1,1.0]\)  
- 이웃: 자기+타 시계열 모두 포함(퇴화 방지)  
- 긴 시계열: 패치 분할 후 결합  
- 평가: 복원(MSE/MAE) + 다운스트림(예측/분류)

---

## 11) 빠른 수식 모음(복붙)
**Cosine**
$$
\mathrm{cos}(u,v)=\frac{u^\top v}{\lVert u\rVert\,\lVert v\rVert}
$$

**Aggregation**
$$
\hat{z}_i(t)=\sum_{k\neq i}\frac{\exp\!\big(R_{s_i,s_k}/\tau\big)}{\sum\limits_{v\neq i}\exp\!\big(R_{s_i,s_v}/\tau\big)}\,z_k(t)
$$

**Reconstruction**
$$
\mathcal{L}_{\mathrm{rec}}=\sum_{i}\lVert \hat{x}_i-x_i\rVert_2^2
$$

**Contrastive(InfoNCE)**
$$
\mathcal{L}_{\mathrm{con}}
=-\sum_{s}\sum_{s^+}
\log\frac{\exp\!\big(R_{s,s^+}/\tau\big)}{\sum\limits_{s'\neq s}\exp\!\big(R_{s,s'}/\tau\big)}
$$

**Final**
$$
\min_{\Theta}\ \mathcal{L}_{\mathrm{rec}}+\lambda\,\mathcal{L}_{\mathrm{con}}
$$

---

## 참고
- **SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling**, NeurIPS 2023.
