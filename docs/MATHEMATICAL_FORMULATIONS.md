# Mathematical Formulations

This document provides detailed mathematical formulations for all fractal systems, neural architectures, and analysis methods used in the Koopman Fractal Spectral Learning framework.

## Table of Contents

1. [Koopman Operator Theory](#koopman-operator-theory)
2. [Fractal Dynamical Systems](#fractal-dynamical-systems)
3. [Neural Network Architectures](#neural-network-architectures)
4. [Spectral Analysis Methods](#spectral-analysis-methods)
5. [Loss Functions and Optimization](#loss-functions-and-optimization)
6. [Performance Metrics](#performance-metrics)

## Koopman Operator Theory

### Definition

The Koopman operator **K** is a linear operator that acts on observables (functions) of a dynamical system rather than on the state space directly.

For a dynamical system with flow map **F**: **X** → **X**, the Koopman operator is defined as:

```
(Kg)(x) = g(F(x))
```

where:
- **g**: **X** → ℂ is an observable function
- **x** ∈ **X** is a state in the state space
- **F**(x) represents the evolution of state x under the dynamics

### Spectral Properties

The eigenvalue equation for the Koopman operator is:

```
Kφⱼ = λⱼφⱼ
```

where:
- **φⱼ** are the Koopman eigenfunctions
- **λⱼ** are the corresponding eigenvalues
- **j** indexes the eigenfunction/eigenvalue pairs

### Finite-Dimensional Approximation

For computational purposes, we approximate the infinite-dimensional Koopman operator using a finite-dimensional matrix **A** ∈ ℝⁿˣⁿ such that:

```
x_{k+1} ≈ Ax_k
```

where **x_k** ∈ ℝⁿ represents the state at time step k.

## Fractal Dynamical Systems

### Sierpinski Gasket (Deterministic IFS)

The Sierpinski gasket is generated using three contractive affine transformations:

**Transformation 1:**
```
f₁(x, y) = (0.5x, 0.5y)
```

**Transformation 2:**
```
f₂(x, y) = (0.5x + 0.5, 0.5y)
```

**Transformation 3:**
```
f₃(x, y) = (0.5x + 0.25, 0.5y + √3/4)
```

**Matrix Form:**
```
f₁: [x'] = [0.5  0 ] [x] + [0  ]
    [y']   [0   0.5] [y]   [0  ]

f₂: [x'] = [0.5  0 ] [x] + [0.5]
    [y']   [0   0.5] [y]   [0  ]

f₃: [x'] = [0.5  0 ] [x] + [0.25     ]
    [y']   [0   0.5] [y]   [√3/4 ≈ 0.433]
```

**Iteration Algorithm:**
1. Start with initial point (x₀, y₀)
2. At each step, randomly select one of {f₁, f₂, f₃} with equal probability
3. Apply selected transformation: (x_{n+1}, y_{n+1}) = f_i(x_n, y_n)
4. Repeat for desired number of iterations

**Fractal Dimension:**
The Hausdorff dimension of the Sierpinski gasket is:
```
D = log(3) / log(2) ≈ 1.585
```

### Barnsley Fern (Probabilistic IFS)

The Barnsley fern uses four affine transformations with different probabilities:

**Transformation 1 (Stem, p = 0.01):**
```
f₁(x, y) = (0, 0.16y)
```

**Transformation 2 (Main leaflet, p = 0.85):**
```
f₂(x, y) = (0.85x + 0.04y, -0.04x + 0.85y + 1.6)
```

**Transformation 3 (Left leaflet, p = 0.07):**
```
f₃(x, y) = (0.2x - 0.26y, 0.23x + 0.22y + 1.6)
```

**Transformation 4 (Right leaflet, p = 0.07):**
```
f₄(x, y) = (-0.15x + 0.28y, 0.26x + 0.24y + 0.44)
```

**Matrix Form:**
```
f₁: [x'] = [0    0  ] [x] + [0  ]
    [y']   [0   0.16] [y]   [0  ]

f₂: [x'] = [0.85  0.04] [x] + [0  ]
    [y']   [-0.04 0.85] [y]   [1.6]

f₃: [x'] = [0.2  -0.26] [x] + [0  ]
    [y']   [0.23  0.22] [y]   [1.6]

f₄: [x'] = [-0.15  0.28] [x] + [0   ]
    [y']   [0.26   0.24] [y]   [0.44]
```

**Probabilistic Selection:**
At each iteration, select transformation f_i with probability p_i:
- P(f₁) = 0.01
- P(f₂) = 0.85  
- P(f₃) = 0.07
- P(f₄) = 0.07

**Fractal Dimension:**
The Hausdorff dimension is approximately:
```
D ≈ 1.67
```

### Julia Sets (Complex Dynamical Systems)

Julia sets are defined by the iteration of complex quadratic polynomials:

**Iteration Formula:**
```
z_{n+1} = z_n² + c
```

where:
- **z_n** ∈ ℂ is the complex state at iteration n
- **c** ∈ ℂ is a complex parameter
- **z₀** is the initial complex number

**Real-Imaginary Decomposition:**
For z_n = x_n + iy_n and c = a + ib:

```
x_{n+1} = x_n² - y_n² + a
y_{n+1} = 2x_n y_n + b
```

**Escape Condition:**
A point z₀ is considered to escape to infinity if:
```
|z_n| > R
```
for some iteration n, where R is the escape radius (typically R = 2).

**Julia Set Definition:**
The Julia set J_c is the boundary between points that remain bounded and those that escape:
```
J_c = {z₀ ∈ ℂ : |z_n| ≤ R for all n ∈ ℕ}
```

**Common Parameters:**
- **Dragon**: c = -0.7269 + 0.1889i
- **Rabbit**: c = -0.123 + 0.745i  
- **Airplane**: c = -0.7 + 0.27015i
- **Spiral**: c = -0.75 + 0.11i

**Fractal Dimension:**
Julia set dimensions vary with parameter c, typically in range:
```
1.0 ≤ D ≤ 2.0
```

## Neural Network Architectures

### Multi-Layer Perceptron (MLP)

**Architecture:**
```
Input Layer: x ∈ ℝⁿ
Hidden Layers: h₁, h₂, ..., h_L ∈ ℝᵈⁱ
Output Layer: y ∈ ℝᵐ
```

**Forward Pass:**
```
h₁ = σ(W₁x + b₁)
h₂ = σ(W₂h₁ + b₂)
⋮
h_L = σ(W_L h_{L-1} + b_L)
y = W_{out} h_L + b_{out}
```

where:
- **W_i** ∈ ℝᵈⁱˣᵈⁱ⁻¹ are weight matrices
- **b_i** ∈ ℝᵈⁱ are bias vectors
- **σ** is the activation function (ReLU, tanh, etc.)

**Activation Functions:**

*ReLU:*
```
σ(x) = max(0, x)
```

*Tanh:*
```
σ(x) = tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)
```

*Sigmoid:*
```
σ(x) = 1/(1 + e⁻ˣ)
```

**Dropout Regularization:**
During training, randomly set neurons to zero with probability p:
```
h_i^{dropout} = h_i ⊙ m_i / (1-p)
```
where m_i ~ Bernoulli(1-p) is a binary mask.

**Batch Normalization:**
```
BN(x) = γ((x - μ_B)/√(σ_B² + ε)) + β
```
where μ_B and σ_B² are batch mean and variance.

### Deep Operator Network (DeepONet)

**Architecture Components:**

*Branch Network:* Processes input functions u(x)
```
B(u) = σ(W_B^{(L)} σ(W_B^{(L-1)} ... σ(W_B^{(1)} u + b_B^{(1)}) ... + b_B^{(L-1)}) + b_B^{(L)})
```

*Trunk Network:* Processes evaluation coordinates y
```
T(y) = σ(W_T^{(L)} σ(W_T^{(L-1)} ... σ(W_T^{(1)} y + b_T^{(1)}) ... + b_T^{(L-1)}) + b_T^{(L)})
```

**Operator Approximation:**
```
G(u)(y) ≈ ∑_{i=1}^p B_i(u) T_i(y) + b₀
```

where:
- **B_i(u)** is the i-th output of the branch network
- **T_i(y)** is the i-th output of the trunk network  
- **p** is the latent dimension
- **b₀** is a bias term

**For Koopman Learning:**
- Input function u represents trajectory snapshots
- Evaluation coordinates y represent spatial locations
- Output G(u)(y) represents evolved observables

### Long Short-Term Memory (LSTM)

**Cell State Equations:**

*Forget Gate:*
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

*Input Gate:*
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```

*Cell State Update:*
```
C_t = f_t * C_{t-1} + i_t * C̃_t
```

*Output Gate:*
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)
```

**For Sequential Koopman Learning:**
```
x_t ∈ ℝⁿ: state at time t
h_t ∈ ℝᵈ: hidden state encoding temporal dependencies
y_t = W_{out} h_t + b_{out}: predicted next state
```

## Spectral Analysis Methods

### Dynamic Mode Decomposition (DMD)

**Data Matrices:**
```
X = [x₁, x₂, ..., x_{m-1}] ∈ ℝⁿˣ⁽ᵐ⁻¹⁾
Y = [x₂, x₃, ..., x_m] ∈ ℝⁿˣ⁽ᵐ⁻¹⁾
```

**Linear Approximation:**
```
Y ≈ AX
```

**SVD of X:**
```
X = UΣV*
```

**DMD Matrix:**
```
Ã = U*YVΣ⁻¹
```

**Eigendecomposition:**
```
ÃW = WΛ
```

**DMD Eigenvalues:**
```
λᵢ = Λᵢᵢ (diagonal elements of Λ)
```

**DMD Modes:**
```
Φ = YVΣ⁻¹W
```

**Exact DMD Algorithm:**
1. Compute SVD: X = UΣV*
2. Compute Ã = U*YVΣ⁻¹  
3. Eigendecompose: ÃW = WΛ
4. Compute modes: Φ = YVΣ⁻¹W

### Eigenvalue Extraction from Neural Networks

**Linear Approximation Method:**
For MLP networks, approximate the Koopman operator using the Jacobian:
```
A ≈ ∇f(x₀)|_{x₀=0}
```

**Finite Difference Method:**
```
A_{ij} ≈ (f_i(x₀ + εeⱼ) - f_i(x₀))/ε
```

**Least Squares Method:**
Solve for A in the overdetermined system:
```
Y = AX + E
```
using:
```
A = YX^T(XX^T)⁻¹
```

### Spectral Error Metrics

**Eigenvalue Distance:**
```
d(λ₁, λ₂) = |λ₁ - λ₂|
```

**Hausdorff Distance:**
```
d_H(Λ₁, Λ₂) = max{max_{λ₁∈Λ₁} min_{λ₂∈Λ₂} d(λ₁, λ₂), max_{λ₂∈Λ₂} min_{λ₁∈Λ₁} d(λ₁, λ₂)}
```

**Spectral Radius Error:**
```
ε_ρ = |ρ(A₁) - ρ(A₂)|
```
where ρ(A) = max{|λᵢ| : λᵢ ∈ σ(A)} is the spectral radius.

## Loss Functions and Optimization

### Prediction Loss

**Mean Squared Error (MSE):**
```
L_MSE = (1/N) ∑_{i=1}^N ||ŷᵢ - yᵢ||²
```

**Mean Absolute Error (MAE):**
```
L_MAE = (1/N) ∑_{i=1}^N ||ŷᵢ - yᵢ||₁
```

**Huber Loss:**
```
L_Huber(δ) = {
  (1/2)(ŷ - y)²           if |ŷ - y| ≤ δ
  δ|ŷ - y| - (1/2)δ²      otherwise
}
```

### Operator Learning Loss (DeepONet)

**Prediction Component:**
```
L_pred = (1/N) ∑_{i=1}^N ||G(uᵢ)(yᵢ) - s(uᵢ)(yᵢ)||²
```

**Consistency Component:**
```
L_cons = (1/N) ∑_{i=1}^N ||G(F(uᵢ))(yᵢ) - G(uᵢ)(F(yᵢ))||²
```

**Combined Loss:**
```
L_total = αL_pred + βL_cons
```

### Regularization Terms

**L2 Weight Decay:**
```
L_reg = λ ∑_{l=1}^L ||W_l||_F²
```

**Spectral Regularization:**
```
L_spec = γ|ρ(A) - ρ_target|²
```

### Optimization Algorithms

**Adam Optimizer:**
```
m_t = β₁m_{t-1} + (1-β₁)g_t
v_t = β₂v_{t-1} + (1-β₂)g_t²
m̂_t = m_t/(1-β₁^t)
v̂_t = v_t/(1-β₂^t)
θ_{t+1} = θ_t - α m̂_t/(√v̂_t + ε)
```

**Learning Rate Scheduling:**

*Exponential Decay:*
```
lr_t = lr₀ × γ^{t/T}
```

*Cosine Annealing:*
```
lr_t = lr_min + (lr_max - lr_min)(1 + cos(πt/T))/2
```

## Performance Metrics

### Prediction Accuracy

**Root Mean Square Error (RMSE):**
```
RMSE = √((1/N) ∑_{i=1}^N (ŷᵢ - yᵢ)²)
```

**Normalized RMSE:**
```
NRMSE = RMSE / (y_max - y_min)
```

**Coefficient of Determination (R²):**
```
R² = 1 - (SS_res/SS_tot)
```
where:
```
SS_res = ∑ᵢ(yᵢ - ŷᵢ)²
SS_tot = ∑ᵢ(yᵢ - ȳ)²
```

### Spectral Accuracy

**Eigenvalue Recovery Rate:**
```
Recovery Rate = |{λᵢ ∈ Λ_learned : min_{μⱼ ∈ Λ_true} |λᵢ - μⱼ| < ε}| / |Λ_true|
```

**Spectral Gap Preservation:**
```
Gap Error = |gap(Λ_learned) - gap(Λ_true)|
```
where gap(Λ) = min{|λᵢ - λⱼ| : λᵢ, λⱼ ∈ Λ, i ≠ j}

### Computational Efficiency

**Training Time Complexity:**
- MLP: O(NLd²E) where N=samples, L=layers, d=width, E=epochs
- DeepONet: O(N(p²L_B + q²L_T)E) where p,q=network widths, L_B,L_T=depths
- LSTM: O(N4d²SE) where d=hidden size, S=sequence length

**Memory Complexity:**
- Parameters: O(Ld²) for MLP, O(p²L_B + q²L_T) for DeepONet
- Activations: O(Bd) where B=batch size

**Inference Speed:**
```
FLOPs = ∑_{l=1}^L (2d_{l-1}d_l - d_l)
```

### Statistical Significance

**Confidence Intervals:**
For metric μ with standard error σ:
```
CI = μ ± t_{α/2,df} × σ/√n
```

**Hypothesis Testing:**
Paired t-test for comparing models:
```
t = (μ₁ - μ₂)/(s_d/√n)
```
where s_d is the standard deviation of differences.

**Effect Size (Cohen's d):**
```
d = (μ₁ - μ₂)/σ_pooled
```

This mathematical framework provides the theoretical foundation for all computations and analyses performed in the Koopman Fractal Spectral Learning project.