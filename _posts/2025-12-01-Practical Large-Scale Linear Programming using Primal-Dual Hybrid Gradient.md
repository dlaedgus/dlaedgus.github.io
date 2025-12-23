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

## Introduction

First-order methods (use gradient, not Hessian information) are standard practice in many areas of optimization.

FOMs의 단점: *tailing-off*  
→ moderately accurate solution은 빠르게 찾지만 optimal solution에 가까워질수록 개선 속도가 느려짐  
→ LP를 풀 때의 문제  
→ properly enhanced하면 해결 가능

기존에 개발된 FOMs의 linear rates는 계산하기 어렵고 느슨한 상수들에 의존  
→ *tailing-off* 발생 가능  
→ theoretical enhancements와 practical heuristics를 결합  
→ PDLP

Matrix factorization에 의존하는 simplex method, interior-point methods는 large-scale LP를 풀기 어려움.  
반면 PDLP는 factorization이 없고 matrix-vector operation만 사용  
→ GPU 환경에서도 사용 가능

---

## Literature Review

**PDHG**  
Proximal point / ADMM 계열에 뿌리를 둔 primal-dual 1차 방법.  
Matrix-factorization 없이 matrix-vector 연산만 사용하기 때문에 large-scale, parallel, distributed computation에 적합.

**FOM-based solvers**  
기존 1차 기반 솔버들은 PDHG, Nesterov 가속, ADMM 등으로 발전.  
일부는 matrix-free 구현을 제공하지만 여전히 선형 시스템 해결이나 2차적 기법에 의존.  
PDLP는 이러한 접근들보다 더 빠르고 강건한 대안을 제시.

**FOMs for LP**  
PDLP는 범용 솔버가 아니라 LP에 특화된(specialized) 방법.

---

## Preliminaries

### Linear Programming

다음과 같은 primal-dual LP 문제를 고려:

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

where

$$
\begin{aligned}
& G \in \mathbb{R}^{m_1 \times n}, \quad
A \in \mathbb{R}^{m_2 \times n}, \quad
c \in \mathbb{R}^n, \\
& h \in \mathbb{R}^{m_1}, \quad
b \in \mathbb{R}^{m_2}, \\
& l \in (\mathbb{R} \cup \{-\infty\})^n, \quad
u \in (\mathbb{R} \cup \{\infty\})^n, \\
& K^\top = (G^\top, A^\top), \quad
q^\top = (h^\top, b^\top)
\end{aligned}
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

Primal 제약을 다음과 같이 다시 씀:

$$
h - Gx \le 0,\quad
b - Ax = 0,\quad
l - x \le 0,\quad
x - u \le 0
$$

라그랑주 승수:

$$
y_G \in \mathbb{R}^{m_1},\quad
y_A \in \mathbb{R}^{m_2},\quad
\lambda^+,\quad
\lambda^-
$$

$$
y = [y_G\; y_A] \in \mathbb{R}^{m_1+m_2}
$$

$y_G^\top (h - Gx)$에서 $h - Gx > 0$ (제약 위반)일 때  
$y_G < 0$이면 $y_G^\top (h - Gx) < 0$이 되어 페널티가 작아짐.  
따라서 $y_G \ge 0$이어야 함.

$y_A$는 equality constraint에 대응 → 부호 제약 없음.  
종합하면:

$$
y_{1:m_1} \ge 0
$$

라그랑지안:

$$
\begin{aligned}
L(x,y,\lambda^+,\lambda^-)
&= c^\top x
+ y_G^\top (h - Gx)
+ y_A^\top (b - Ax) \\
&\quad + (\lambda^+)^\top (l - x)
+ (\lambda^-)^\top (x - u)
\end{aligned}
$$

정리하면:

$$
L = (c - G^\top y_G - A^\top y_A - \lambda^+ + \lambda^-)^\top x
+ (h^\top y_G + b^\top y_A + l^\top \lambda^+ - u^\top \lambda^-)
$$

Dual function:

$$
g(y,\lambda^+,\lambda^-) := \inf_{x \in \mathbb{R}^n} L(x,y,\lambda^+,\lambda^-)
$$

$x$의 계수가 0이 아니면 $-\infty$가 되므로:

$$
c - G^\top y_G - A^\top y_A - \lambda^+ + \lambda^- = 0
\quad\Leftrightarrow\quad
c - K^\top y - \lambda = 0
$$

상수항을 최대화:

$$
\max_{y,\lambda} \;
h^\top y_G + b^\top y_A + l^\top \lambda^+ - u^\top \lambda^-
= q^\top y + l^\top \lambda^+ - u^\top \lambda^-
$$

각 $x_i$마다 $(l_i, u_i)$가 다르므로 $\lambda \in \Lambda$.

---

### Saddle-Point Formulation

Primal-dual 문제를 saddle-point problem으로 표현:

$$
\min_{x \in X} \max_{y \in Y}
\mathcal{L}(x,y)
:= c^\top x - y^\top K x + q^\top y
$$

$$
X := \{ x \in \mathbb{R}^n : l \le x \le u \}, \qquad
Y := \{ y \in \mathbb{R}^{m_1+m_2} : y_{1:m_1} \ge 0 \}
$$

Saddle-point를 찾기 위해:

- $x$: gradient descent  
- $y$: gradient ascent

$$
\begin{aligned}
x^{k+1} &= \operatorname{proj}_X \bigl(x^k - \tau (c - K^\top y^k)\bigr), \\
y^{k+1} &= \operatorname{proj}_Y \bigl(y^k + \sigma (q - K(2x^{k+1} - x^k))\bigr)
\end{aligned}
$$

즉,

$$
x \leftarrow x - \tau \nabla_x \mathcal{L}, \qquad
y \leftarrow y + \sigma \nabla_y \mathcal{L}
$$

Gradient step 이후 제약 위반 가능 → projection 필요.

Projection 정의:

$$
\operatorname{proj}_X(z) = \arg\min_{x \in X} \|x - z\|_2
$$

---

### Extragradient 해석

Saddle-point operator:




$$
F(x,y) =
\begin{pmatrix}
\nabla_x \mathcal{L}(x,y) \\
-\nabla_y \mathcal{L}(x,y)
\end{pmatrix}=
\begin{pmatrix}
c - K^\top y \\
Kx - q
\end{pmatrix}
$$


$$
\begin{pmatrix}
0 & -K^\top \\
K & 0
\end{pmatrix}
\begin{pmatrix}
x \\ y
\end{pmatrix}
+
\begin{pmatrix}
c \\ -q
\end{pmatrix}
$$

여기서

$$
M =
\begin{pmatrix}
0 & -K^\top \\
K & 0
\end{pmatrix}
= -M^\top
$$

→ skew-symmetric  
→ rotational 특성  
→ extragradient 사용

---

### Step Size와 Weighted Norm

$\tau, \sigma > 0$는 각각 primal, dual step size.

수렴 조건:

$$
\tau \sigma \|K\|_2^2 \le 1
$$

재파라미터화:

$$
\tau = \eta / \omega, \qquad
\sigma = \omega \eta
$$

$$
\eta \le \frac{1}{\|K\|_2}
$$

Weighted norm:

$$
\|z\|_\omega :=
\sqrt{ \omega \|x\|_2^2 + \frac{\|y\|_2^2}{\omega} },
\quad z=(x,y)
$$

---

## Practical Algorithmic Improvements

<img width="1177" height="535" alt="image" src="https://github.com/user-attachments/assets/491ace8a-64b3-4c03-9cc7-a3816302f593" />
<img width="1087" height="416" alt="image" src="https://github.com/user-attachments/assets/94a0dccf-be16-49ed-96bc-1b5232ef3bfa" />

- $t$: 한 restart 안에서 PDHG 수행 횟수  
- $k$: global PDHG step 수  
- $n$: restart index  
- $z^{n,t}$: $n$번째 restart의 $t$번째 iterate  
- $\omega^n$: $n$번째 restart에서의 primal weight  

Algorithm 2는 $(z, \omega, \hat\eta)$를 입력으로 받아  
trial PDHG step을 수행하여 허용 가능한 최대 step size $\bar\eta$를 계산.

- $\eta \le \bar\eta$: update accept  
- $\eta > \bar\eta$: step size 감소 후 재시도

---

### Normalized Duality Gap

$$
\rho_r^n(z)
:= \frac{1}{r}
\max_{\hat z \in Z: \|z-\hat z\|_{\omega^n} \le r}
\left\{ \mathcal{L}(x,\hat y) - \mathcal{L}(\hat x,y) \right\}
$$

LP에서는 feasible set이 unbounded → global gap은 무한대 가능  
→ local duality gap 사용

Reference-based measure:

$$
\mu_n(z,z_{\mathrm{ref}})
:= \rho_{\|z-z_{\mathrm{ref}}\|_{\omega^n}}^n(z)
$$

Restart candidate 선택:

$$
\text{GetRestartCandidate}(z^{n,t+1}, \bar z^{n,t+1}, z^{n,0})
=
\begin{cases}
z^{n,t+1}, &
\mu_n(z^{n,t+1}, z^{n,0})
<
\mu_n(\bar z^{n,t+1}, z^{n,0}) \\
\bar z^{n,t+1}, & \text{otherwise}
\end{cases}
$$

---

### Restart Criteria

1. **Sufficient decay**

$$
\mu_n(z_c^{n,t+1}, z^{n,0})
\le
\beta_{\text{sufficient}}
\mu_n(z^{n,0}, z^{n-1,0})
$$

2. **Necessary decay + no local progress**

$$
\mu_n(z_c^{n,t+1}, z^{n,0})
\le
\beta_{\text{necessary}}
\mu_n(z^{n,0}, z^{n-1,0})
$$

and

$$
\mu_n(z_c^{n,t+1}, z^{n,0})
>
\mu_n(z_c^{n,t}, z^{n,0})
$$

3. **Long inner loop**

$$
t \ge \beta_{\text{artificial}} k
$$

---

## Primal Weight Updates

<img width="1283" height="330" alt="image" src="https://github.com/user-attachments/assets/078e1aee-f1f7-4ddb-ae92-c696a3d5fc0a" />

목표: primal/dual distance 균형

$$
\|x^{n,t} - x^*\|_{\omega^n}
\approx
\|y^{n,t} - y^*\|_{\omega^n}
$$

Proxy 사용:

$$
\Delta_x^n = \|x^{n,0} - x^{n-1,0}\|_2, \quad
\Delta_y^n = \|y^{n,0} - y^{n-1,0}\|_2
$$

Log smoothing:

$$
\omega^n
=
\exp\big(
\theta \log(\Delta_y^n / \Delta_x^n)
+ (1-\theta)\log(\omega^{n-1})
\big)
$$

---

## Presolve

PaPILO 사용.

---

## Diagonal Preconditioning

Constraint matrix $K=(G,A)$에 대해:

$$
\tilde K = D_1 K D_2
$$

변환:

$$
\hat x = D_2^{-1} x, \quad
\tilde c = D_2 c, \quad
(\tilde b, \tilde h) = D_1(b,h)
$$

Scaling 방법:

1. No scaling: $D_1=D_2=I$
2. Pock–Chambolle
3. Ruiz

PDLP에서는 Ruiz와 Pock–Chambolle 혼합 사용.

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
