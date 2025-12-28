---
title: "Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient (NeurIPS 2021)"
date: 2025-12-01 12:00:00 +0900
categories: [paper_review,OR]
math: true
---

# Paper Review — *Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient* (NeurIPS 2021)

- **1저자:** David Applegate
- **제목:** Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient  
- **저널명:** NeurIPS 
- **년도:** 2021

---

### **Introduction**

First-order methods(use gradient, not Hessian information) - standard practice in many areas of optimization

FOMs의 단점 : *tailing-off* - moderately accurate solution은 빠르게 찾지만 optimal solution에 가까워질수록 개선 속도가 느려짐 → LP를 풀 때의 문제 → properly enhanced하면 됨

기존의 developed FOMs의 linear rates depend on potentially loose and hard-to-compute constants → *tailling-off*  발생 가능 → combine both theoretical enhancements with practical heuristics → PDLP

matrix factorization에 의존하는 simplex method, interior-point methods는 large scale LP를 풀기 어렵지만 PDLP는 factorization이 없음, matrix-vector operation 연산 사용 → GPU환경 가능

cuPDLP.jl: A GPU Implementation of Restarted Primal-Dual Hybrid Gradient for Linear Programming in Julia 논문에 의거한 보충 설명..

### LP 문제 구조 전체 정리

LP를 풀 때 유도되는 constraint matrix는 대부분 sparse matrix  
즉, variable 수와 constraint 수는 매우 크지만, 각 제약식은 소수의 variable만 포함하므로 matrix의 대부분 원소는 0

이러한 sparse matrix는

- 행렬–벡터 곱  $\(Kx,\; K^\top y\)$ 은 효율적으로 계산 가능
- 메모리 사용량이 작고 구조적 pattern를 유지할 수 있음
- GPU 및 병렬 연산에 이론적으로는 매우 적합

그러나 LP를 푸는 일반적 방법, 즉

- Simplex method
- Interior-point methods

은 반복 과정에서 반드시 다음과 같은 sparse linear system을 풀어야 함

$$
\(Ax = b \; (A \text{ is sparse})\)
$$

이를 해결하기 위해:

- Simplex는 주로 LU factorization
- Interior-point는 주로 Cholesky factorization
를 사용

이 matrix factorization 과정에서 문제 발생

1. Fill-in 현상

- 원래는 sparse였던 행렬이
- 분해 과정에서 0이 아닌 원소가 대량으로 생성됨
- 결과적으로 $\(L, U\)$ 또는 $\(L\)$이 dense에 가까워짐

2. 메모리 폭증

- factorization 결과를 저장하는 데 막대한 메모리 필요
- 실제로는 문제 자체는 메모리에 들어가는데, factorization 결과가 안 들어가는 상황 발생것- 이것이 large scale LP에서 out-of-memory 오류의 주된 원인

3. 병렬화 불가
- LU / Cholesky는 계산 순서가 강하게 의존적
- elimination 과정이 본질적으로 sequential
- 불규칙한 메모리 접근과 branching 발생

따라서

Sparse matrix 자체는 GPU에 잘 맞지만, Sparse matrix factorization은 GPU에 구조적으로 맞지 않음

따라서 simplex나 interior-point 기반 LP 솔버는:

- GPU 활용이 극히 어렵고
- 대부분 CPU 기반 단일 공유 메모리 구조에 머무르게 된다.

이 한계를 극복하기 위해 등장한 접근이 바로 first-order methods

- 선형 시스템 $\(Ax = b\)$을 직접 풀지 않음
- LU / Cholesky factorization을 완전히 제거
- 반복마다 수행하는 연산은 오직  
  $\(Kx,\; K^\top y\)$  
  같은 sparse matrix–vector multiplication

그 결과:

- sparse 구조가 끝까지 유지되고
- 메모리 사용량이 선형적으로 증가하며
- 연산 패턴이 단순하고 규칙적이어서
- GPU 및 대규모 병렬화에 매우 적합

<img width="907" height="434" alt="image" src="https://github.com/user-attachments/assets/8031574f-bf33-44df-a75b-8f0e9b57ac29" />


---

### **Literature reivew**

PDHG - proximal point/ADMM 계열에 뿌리를 둔 primal-dual 1차 방법으로, matrix-factorization 없이 matrix-vector 연산만 사용하기 때문에 large-scale·parallel·distributed computation에 적합

FOM-based solvers - 기존의 1차 기반 솔버들은 PDHG, Nesterov 가속, ADMM 등 다양한 계열로 발전해 왔으며, 일부는 matrix-free 구현을 제공하지만 여전히 선형 시스템 해결이나 2차적 기법에 의존한다. PDLP는 이러한 접근들보다 더 빠르고 강건한 대안을 제시한다.

FOMs for LP - PDLP는 범용 솔버가 아니라 LP에 specialized 되어 있음

---

### **Preliminaries**

Linear Programming - solve primal-dual LP problems of the form: 

$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & c^\top x \\
\text{s.t.} \quad
& Gx \ge h, \\
& Ax = b, \\
& l \le x \le u
\end{aligned}
\qquad
\begin{aligned}
\max_{y \in \mathbb{R}^{m_1+m_2},\; \lambda \in \mathbb{R}^n} \quad
& q^\top y + l^\top \lambda^+ - u^\top \lambda^- \\
\text{s.t.} \quad
& c - K^\top y = \lambda, \\
& y_{1:m_1} \ge 0, \\
& \lambda \in \Lambda
\end{aligned}
$$

$$
\text{where }\;
G \in \mathbb{R}^{m_1 \times n},\;
A \in \mathbb{R}^{m_2 \times n},\;
c \in \mathbb{R}^n,\;
h \in \mathbb{R}^{m_1},\;
b \in \mathbb{R}^{m_2},\;
l \in (\mathbb{R} \cup \{-\infty\})^n,\;
u \in (\mathbb{R} \cup \{\infty\})^n,
$$

$$
K^\top = (G^\top, A^\top), \qquad
q^\top = (h^\top, b^\top)
$$

$$
\Lambda_i :=
\begin{cases}
\{0\}, & l_i = -\infty,\; u_i = \infty \\
\mathbb{R}_-, & l_i = -\infty,\; u_i \in \mathbb{R} \\
\mathbb{R}_+, & l_i \in \mathbb{R},\; u_i = \infty \\
\mathbb{R}, & \text{otherwise}
\end{cases}
$$

primal의 제약식  
$h - Gx \le 0,\; b - Ax = 0,\; l - x \le 0,\; x - u \le 0$ 으로 고침

라그랑주 승수  
$y_G \in \mathbb{R}^{m_1},\; y_A \in \mathbb{R}^{m_2},\; \lambda^+,\; \lambda^-$

$y = [y_G\; y_A] \in \mathbb{R}^{m_1+m_2}$

$y_G^\top (h-Gx)$에서 $h - Gx>0$ 인 경우 즉 제약을 위반하는 경우  
$y_G < 0$이면  $y_G^\top (h-Gx) < 0$ 이 됨. 즉, 페널티가 작아짐 따라서 ≥ 0이여야함 

$y_A$는 부호제약 x  종합하면 → $y_{1:m_1} \ge 0$

$L(x,y,\lambda^+,\lambda^-) = c^\top x + y_G^\top (h-Gx) + y_A^\top (b-Ax) + (\lambda^+)^\top (l-x) + (\lambda^-)^\top (x-u)$

$L = (c - G^\top y_G - A^\top y_A - \lambda^+ + \lambda^-)^\top x + (h^\top y_G + b^\top y_A + l^\top \lambda^+ - u^\top \lambda^-)$

듀얼 함수는  
$g(y,\lambda^+,\lambda^-) := \inf_{x \in \mathbb{R}^n} L(x,y,\lambda^+,\lambda^-)$

이때 x계수가 0이 아니면 식이 $-\infty$이므로 x계수가 0 이여야 함

$c - G^\top y_G - A^\top y_A - \lambda^+ + \lambda^- = 0$  
→  $c-K^\top y -\lambda=0$

이제 상수항을 최대화 해야하므로

$$
\max_{y \in \mathbb{R}^{m_1+m_2},\; \lambda \in \mathbb{R}^n}
h^\top y_G + b^\top y_A + l^\top \lambda^+ - u^\top \lambda^-
= q^\top y + l^\top \lambda^+ - u^\top \lambda^-
$$

이때 각 $x_i$ 마다 제약이 다름 , 즉 $l_i , u_i$가 다름 따라서 

이 primal-dual problems를 saddle-point problem으로 바꾸면

$$
\min_{x \in X} \; \max_{y \in Y} \; \mathcal{L}(x,y):= c^\top x - y^\top K x + q^\top y
\\
X := \{ x \in \mathbb{R}^n : l \le x \le u \}, 
\qquad
Y := \{ y \in \mathbb{R}^{m_1+m_2} : y_{1:m_1} \ge 0 \}
$$

saddle-point를 찾기 위해  
x - gradient descent, y - gradient ascent를 해야함

$$
\begin{aligned}
x^{k+1}&= \operatorname{proj}_X \bigl(x^k - \tau (c - K^\top y^k)\bigr), \\
y^{k+1}&= \operatorname{proj}_Y \bigl(y^k + \sigma (q - K(2x^{k+1} - x^k))\bigr)
\end{aligned}
$$

즉,  
$x \leftarrow x - \tau \nabla_x \mathcal L$  
$y \leftarrow y + \sigma \nabla_y \mathcal L$

이때의 문제, x,y의 범위에 관한 제약이 깨질 수 있음

gradient step을 하면 x, y 가  
$l < x < u,\; y > 0$ 범위 밖으로 나갈 수 있음

해결 방법 : projection  
$\text{proj}_X(z) =$ X안에서 z와 가장 가까운 점

즉,  
$x^{k+1} = \text{proj}_X(\text{gradient step})$

이때 왜 $2x^{k+1}-x^{k}$인지?

saddle-point operator

$$
F(x,y) =
\begin{pmatrix}
\nabla_x \mathcal L(x,y) \\
-\nabla_y \mathcal L(x,y)
\end{pmatrix}
=
\begin{pmatrix}
c - K^\top y \\
Kx - q
\end{pmatrix}
=
\begin{pmatrix}
0 & -K^\top \\
K & 0
\end{pmatrix}
\begin{pmatrix}
x \\
y
\end{pmatrix}
+
\begin{pmatrix}
c \\
-q
\end{pmatrix}
$$

에서  
$M = \begin{pmatrix} 0 & -K^\top \\ K & 0 \end{pmatrix} = -M^\top$  
즉, skew-symmetric이기 때문에 rotational한 특성이 있음

따라서 extragradient를 사용해서  
k+1 step으로 한 번 미리 가 보고 그 위치에서 방향을 다시 계산

$\tau , \sigma > 0$ 은 각각 primal, dual의 step size이고 둘이 독립적

$\tau\sigma\|K\|_2^{2}\le1$ 일 때 optimal 로 수렴

step size를 재파라미터화 →  
$\tau = \eta/\omega,\; \sigma=\omega\eta$

$\eta\in(0,\infty)$ 는 step size,  
$\omega \in(0,\infty)$ 는 primal weight임

따라서 수렴 조건이  
$\eta\le1/\|K\|_2$

weighted Euclidean norm

$$
\|z\|_\omega := \sqrt{\,\omega\,\|x\|_2^2 + \frac{\|y\|_2^2}{\omega}\,},
\quad z=(x,y)
$$

---

### **Practical algorithmic improvements**

<img width="1177" height="535" alt="image" src="https://github.com/user-attachments/assets/491ace8a-64b3-4c03-9cc7-a3816302f593" />
<img width="1087" height="416" alt="image" src="https://github.com/user-attachments/assets/94a0dccf-be16-49ed-96bc-1b5232ef3bfa" />
$t$ : 한 restart 안에서 몇 번 PDHG 했는지  

$k$ : global, 지금까지 총 PDHG 스텝 수, k가 커질수록 $\eta' \approx \min(\bar\eta,\eta)$ 보수적 조정  

$n$ : 몇 번째 restart인지  

$z^{n,t}$ : n번째 restart에서 t번째 PDHG의 결과  

$\omega^n$ : n번째 restart 에서 쓰이는 primal weight  

Algorithm 2는 현재 iterate $z=(x,y)$와 현재 primal weight $\omega$, step size candidate $\hat\eta$를 입력으로 받아서  
먼저 $\eta = \hat\eta$로 두고 현재 $x,y$에서 이 $\eta, \omega$를 사용해 PDHG를 한 step을 시험적으로 수행하여 $x',y'$를 얻음.  

이 $x',y'$ 에 대해 허용되는 최대 step size인 $\bar\eta$를 계산하고, 이를 이용해 다음에 사용할 step size candidate $\eta'$를 정함.  

만약 이번에 사용한 step size $\eta$가 $\eta\le\bar\eta$ 면 방금 계산한 $x',y'$를 다음 iterate로 확정하고  
$x',y'$와 $\eta, \bar\eta$를 반환함.  

반대로 $\eta>\bar\eta$면 방금 시험 계산한 $x',y'$는 버리고 새로운 step size $\eta'$로 줄인 뒤 같은 x,y 위치에서 다시 같은 과정을 반복.

PDHG를 restart할 조건 - $z$에서의 normalized duality gap

$$
\rho_r^n(z)
:=
\frac{1}{r}
\max_{\;(\hat x,\hat y)\in Z:\ \|z-\hat z\|_{\omega^n}\le r}
\left\{
\mathcal L(x,\hat y)-\mathcal L(\hat x,y)
\right\}
$$

현재 얼마나 saddle point에서 벗어났는가? 를 측정

standard duality gap의 문제?  
LP에서는 feasible set $Z$가 unbounded 여서 무한대가 나올 수 있음  

→ global이 아니라 local을 보기  

현재 $z$근처 $\|z-\hat z\|_{\omega^n}\le r$ 안에서만 duality gap을 계산  

이때 $r$이 너무 커지면 gap이 커지고 $r$이 작으면 gap도 작아짐  

→ normalized

즉, $\rho_r^n(z)$는 현재 $z$ 주변 반지름 $r$ 안에서 max saddle violation을 보고 그 값을 $r$로 나눈 값

$$
\mu_n(z,z_{\mathrm{ref}})
:=
\rho_{\|z-z_{\mathrm{ref}}\|_{\omega^n}}^n(z)
$$

reference point 기준으로 현재 얼마나 saddle point에서 벗어났는지?

고정된 n에서 PDHG를 여러 번 시행하면 $z^{n,t+1}$, $\bar z^{n,t+1}$ 이 생김  

이 둘 중 어느쪽이 saddle-point에 가까운지는 모르는 상태  

따라서 normalized duality gap이 작은 쪽으로 선택

$$
\text{GetRestartCandidate}(z^{n,t+1},\bar z^{n,t+1},z^{n,0})
=
\begin{cases}
z^{n,t+1}, & \mu_n(z^{n,t+1},z^{n,0}) < \mu_n(\bar z^{n,t+1},z^{n,0}), \\
\bar z^{n,t+1}, & \text{otherwise}
\end{cases}
$$

restart를 하면 어디서 할지를 정하는 함수

$z^{n,0}$ : n번째 outer loop 시작점 - reference  

$z^{n,t}_c$ : n번째 outer loop 시점에서 t번째 inner loop까지 봤을 때 본 restart candidate  

Restart란 PDHG를 멈추고  
$z^{n+1,0} := z_c^{n,t}$ 로 reset, primal weight 업데이트 하는 과정

---

**Restart criteria**

1. **Sufficient decay in normalized duality gap**
   
   $$
   \mu_n(z_c^{n,t+1}, z^{n,0})
   \le
   \beta_{\text{sufficient}}\,
   \mu_n(z^{n,0}, z^{n-1,0})
   $$
   이번 outer iteration 전체 결과가 이전 outer iteration 대비 필요한 만큼 $\beta$(0.9만큼)
   개선되었는지 ? + 최근 inner step에서 saddle-point 이탈 정도가 오히려 증가했는지?

2. **Necessary decay + no local progress in normalized duality gap**
   
   $$
   \mu_n(z_c^{n,t+1}, z^{n,0})
   \le
   \beta_{\text{necessary}}\,
   \mu_n(z^{n,0}, z^{n-1,0})
   $$
   and
   $$
   \mu_n(z_c^{n,t+1}, z^{n,0}) > \mu_n(z_c^{n,t}, z^{n,0})
   $$
   이번 outer iteration 전체 결과가 이전 outer iteration 대비 필요한 만큼 $\beta$(0.9만큼
   개선되었는지 ? + 최근 inner step에서 saddle-point 이탈 정도가 오히려 증가했는지?

3. **Long inner loop**
   
   $$
   t \ge \beta_{\text{artificial}}\, k
   $$
   현재 outer iteration 안에서 inner PDHG를 너무 오래 돌렸는지?

---

### **Primal weight updates**

<img width="831" height="96" alt="image" src="https://github.com/user-attachments/assets/cc86f54e-d901-4246-a8e1-df84655171b8" />

<img width="1283" height="330" alt="image" src="https://github.com/user-attachments/assets/078e1aee-f1f7-4ddb-ae92-c696a3d5fc0a" />

scale invariance 보장

Algorithm 3 aims to choose the primal weight $\omega^{n}$ such that distance to optimality in the primal and dual is the same

$x,y$ 의 균형을 맞추는 설계

$$
\|x^{n,t}-x^*\|_{\omega^n} = \omega^n\|x^{n,t}-x^*\|_{2}
$$

$$
\|y^{n,t}-y^*\|_{\omega^n} = \frac{1}{\omega^n}\|y^{n,t}-y^*\|_{2}
$$

$\omega$가 커지면 primal 거리가 커지고 dual은 작아짐  
$\omega$가 작아지면 dual 거리가 커지고 primal 거리는 작아짐

목표 →  
$$
\|x^{n,t}-x^*\|_{\omega^n} \approx \|y^{n,t}-y^*\|_{\omega^n}
$$

$$
\omega^n = \frac{\|y^{n,t}-y^*\|_{2}}{\|x^{n,t}-x^*\|_{2}}
$$

하지만 $x^*,y^*$는 모르는 값이므로 proxy 사용 →  
직전 restart 대비 이동량  

$$
Δ^n_x = \|x^{n,0}-x^{n-1,0}\|_2
$$

$$
Δ^n_y = \|y^{n,0}-y^{n-1,0}\|_2
$$

→ $$Δ^n_y/Δ^n_x$$ 사용

그런데 log smoothing 하는 이유는  
restart마다 튀는 값이 생겨 oscillate 하기 때문

최종적으로

$$
\omega^n=
\exp\big(
\theta \log(Δ^n_y/Δ^n_x)
+
(1-\theta)\log(\omega^{n-1})
\big)
$$

---

### **Presolve**

PaPILO 를 사용해서 Presolve

---

### **Diagonal Preconditioning**

constraint matrix $K = (G,A)$ 를 positive diagonal matrices $D_1,D_2$ 를 이용해

$$
\tilde K = (\tilde G,\tilde A)=D_1KD_2
$$

$\tilde K$는 well balanced

이 과정에서

$A, G, c, b, h, u, l$

은

$\tilde G, \tilde A, \hat x = D_2^{-1} x, \tilde c = D_2 c,$  

$(\tilde b, \tilde h) = D_1 (b, h),$  

$\tilde u = D_2^{-1} u, \tilde l = D_2^{-1} l$

로 바뀜

$D_1,D_2$ 를 고르는 3가지 방법

1. No scaling  
   $D_1=D_2=I$

2. Pock–Chambolle  
   
   $$
  (D_1)_{jj}=\sqrt{\lVert K_{j,:}\rVert_{2-\alpha}},
  \quad j=1,\ldots,m_1+m_2,
  \qquad
  (D_2)_{ii}=\sqrt{\lVert K_{:,i}\rVert_{\alpha}},
  \quad i=1,\ldots,n,
  \qquad
  \alpha=1
  $$


3. Ruiz  
   
   $$
   (D_1)_{jj}=\sqrt{\lVert K_{j,:}\rVert_{\infty}},
   \qquad
   (D_2)_{ii}=\sqrt{\lVert K_{:,i}\rVert_{\infty}}
   $$


PDLP 세팅에서 Ruiz 와 Pock–Chambolle 섞어서 사용

---

### **Feasibility polishing**


<img width="1004" height="435" alt="image" src="https://github.com/user-attachments/assets/c345dc7e-7b0d-47fe-831a-c74fe641e849" />
PDLP를 끝까지 최적화 하지 않고  
어느 정도 풀린 시점에서 objective는 거의 유지한 채 feasibility만 polish 해주는 algorithm

왜 필요?  
미세한 objective를 줄이려고 algorithm을 끝까지 돌리면 시간이 너무 오래 걸림

Integer programming의 branch-and-bound에서 optimality gap이 미리 정해둔 threshold 아래로 떨어지면 algorithm을 종료하는 것처럼  
FOMs도 extremely small feasibility violations and moderate (e.g., 1%) duality gaps를 가지는 휴리스틱을 개발해야함

1. Restarted PDHG and feasibility problems
   
   feasibility problem - objective function이 없는 문제 , 수렴이 빠름
   
   $$
   \min_{x \in \mathbb{R}^n} 0
   \quad \text{s.t.} \quad
   \ell_c \le Ax \le u_c,\;
   \ell_v \le x \le u_v
   $$

2. Given a starting solution, restarted PDHG will converge to a nearby optimal solution
   
   PDHG는 반복이 진행될수록 어떤 optimal solution에 대해서도 반복점과의 PDHG노름 거리가 증가하지 않음  
   → 원래 문제에 PDHG를 적용하여 얻은 primal solution을 warm start로 삼아서 위의 feasibility problem을 풀면  
   목적함수를 무시함에도 불구하고 objective값이 나빠지지 않음
   
   따라서 PDHG로 이미 거의 최적인 해를 가지고 있다면,  
   위의 feasibility problem을 풀어 얻은 해는 높은 objective quality을 유지하면서도  
   tight numerical tolerances 내에서 primal feasibility를 만족할 수 있음

   이 논리를 dual에도 적용하면,
   
   $$
   \max_{y \in \mathbb{R}^m,\, r \in \mathbb{R}^n} 0
   \quad \text{s.t.} \quad
   c - A^T y = r,\;
   y \in \mathcal{Y},\;
   r \in \mathcal{R}
   $$

   primal, dual 둘 다 polishing

이 두가지를 바탕으로  
Algorithm 4는 primal, dual 모두 10⁻⁸ maximum constraint violation 이하인 해를 찾으면서  
$\varepsilon_{\text{rel-gap}}>0$ 보다 작은 relative duality gap을 만족하는 것을 목표로 함

$$
\frac{|c^T x + p(y;\ell_c,u_c) + p(r;\ell_v,u_v)|}
{\max\{|c^T x|,\; |p(y;\ell_c,u_c) + p(r;\ell_v,u_v)|\}}
\le \varepsilon_{\text{rel-gap}}
$$

---

## Experiments

### Optimality Termination Criteria

1. Duality gap

$$
|q^\top y + l^\top \lambda^+ - u^\top \lambda^- - c^\top x|
\le
\epsilon \big(
1 + |q^\top y + l^\top \lambda^+ - u^\top \lambda^-|
+ |c^\top x|
\big)
$$

2. Primal feasibility

$$
\left\|
\begin{pmatrix}
Ax - b \\
(h - Gx)^+
\end{pmatrix}
\right\|_2
\le
\epsilon (1 + \|q\|_2)
$$

3. Dual feasibility

$$
\|c - K^\top y - \lambda\|_2
\le
\epsilon (1 + \|c\|_2)
$$

<img width="1323" height="379" alt="image" src="https://github.com/user-attachments/assets/23f1df04-0012-4d30-8178-0fa09c491738" />

KKT passes는 $K, K^\top$에 대한 matrix multiplication 비용을 의미.

<img width="1311" height="906" alt="image" src="https://github.com/user-attachments/assets/f591cf1b-f39e-4d3d-bf29-8f2ff6e0d46a" />

<img width="1326" height="818" alt="image" src="https://github.com/user-attachments/assets/eeb0aac6-861c-4ae9-a83f-43cd7c2da953" />

<img width="1271" height="695" alt="image" src="https://github.com/user-attachments/assets/87f9c6da-d38a-4e54-826f-3cf769598a70" />

