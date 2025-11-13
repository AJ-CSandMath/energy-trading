# Energy Trading & Portfolio Optimization System - Algorithm Documentation

This document provides comprehensive mathematical foundations and implementation details for all algorithms used in the system.

---

## Table of Contents

1. [Forecasting Models](#forecasting-models)
   - [ARIMA (AutoRegressive Integrated Moving Average)](#arima-autoregressive-integrated-moving-average)
   - [XGBoost (Extreme Gradient Boosting)](#xgboost-extreme-gradient-boosting)
   - [LSTM (Long Short-Term Memory Networks)](#lstm-long-short-term-memory-networks)
2. [Renewable Energy Models](#renewable-energy-models)
   - [Wind Power Curves](#wind-power-curves)
   - [Solar Irradiance Models](#solar-irradiance-models)
3. [Trading Strategies](#trading-strategies)
   - [Mean Reversion](#mean-reversion)
   - [Momentum](#momentum)
   - [Spread Trading](#spread-trading)
   - [Renewable Arbitrage](#renewable-arbitrage)
4. [Portfolio Optimization](#portfolio-optimization)
   - [Markowitz Mean-Variance Optimization](#markowitz-mean-variance-optimization)
   - [Risk Parity](#risk-parity)
   - [Black-Litterman Model](#black-litterman-model)
   - [CVaR Optimization](#cvar-optimization)
5. [Risk Analytics](#risk-analytics)
   - [Value at Risk (VaR)](#value-at-risk-var)
   - [Conditional Value at Risk (CVaR)](#conditional-value-at-risk-cvar)
   - [Risk Decomposition](#risk-decomposition)
6. [Performance Metrics](#performance-metrics)
7. [Transaction Cost Models](#transaction-cost-models)

---

## Forecasting Models

### ARIMA (AutoRegressive Integrated Moving Average)

#### Mathematical Formulation

An ARIMA(p, d, q) model combines:
- **AR(p)**: Autoregressive component of order p
- **I(d)**: Differencing of order d
- **MA(q)**: Moving average component of order q

**Model Equation:**

```
ŷₜ = c + φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + φₚyₜ₋ₚ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θ_qεₜ₋q + εₜ
```

Where:
- `ŷₜ` = forecast at time t
- `c` = constant term
- `φᵢ` = autoregressive coefficients (i = 1, ..., p)
- `θⱼ` = moving average coefficients (j = 1, ..., q)
- `εₜ` = white noise error term at time t
- `yₜ₋ᵢ` = observed value at time t-i

**Differencing Operation:**

For non-stationary data, differencing transforms the series:

```
∇ʸyₜ = yₜ - yₜ₋₁  (first difference, d=1)
∇²yₜ = ∇yₜ - ∇yₜ₋₁  (second difference, d=2)
```

#### Model Selection

**Akaike Information Criterion (AIC):**

```
AIC = 2k - 2ln(L)
```

Where:
- `k` = number of parameters (p + q + 1)
- `L` = maximum likelihood

**Bayesian Information Criterion (BIC):**

```
BIC = k·ln(n) - 2ln(L)
```

Where:
- `n` = number of observations

**Implementation:**
- Uses `statsmodels.tsa.arima.model.ARIMA`
- Automatic parameter selection via grid search minimizing AIC/BIC
- Stationarity testing with Augmented Dickey-Fuller test

---

### XGBoost (Extreme Gradient Boosting)

#### Mathematical Formulation

XGBoost builds an ensemble of decision trees sequentially, where each tree corrects errors from the previous trees.

**Objective Function:**

```
L(φ) = Σᵢ l(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)
```

Where:
- `l` = differentiable loss function (e.g., squared error, log loss)
- `yᵢ` = true value for sample i
- `ŷᵢ` = predicted value for sample i
- `Ω(fₖ)` = regularization term for tree k
- `fₖ` = prediction function of tree k

**Regularization Term:**

```
Ω(f) = γT + (1/2)λ Σⱼ wⱼ²
```

Where:
- `T` = number of leaves
- `wⱼ` = weight of leaf j
- `γ` = complexity penalty for number of leaves
- `λ` = L2 regularization on leaf weights

**Additive Training:**

At iteration t, the model prediction is:

```
ŷᵢ⁽ᵗ⁾ = ŷᵢ⁽ᵗ⁻¹⁾ + ηfₜ(xᵢ)
```

Where:
- `η` = learning rate (shrinkage)
- `fₜ` = new tree added at iteration t

**Second-Order Approximation:**

XGBoost uses a second-order Taylor expansion of the loss function:

```
L⁽ᵗ⁾ ≈ Σᵢ [l(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾) + gᵢfₜ(xᵢ) + (1/2)hᵢfₜ²(xᵢ)] + Ω(fₜ)
```

Where:
- `gᵢ = ∂l/∂ŷᵢ⁽ᵗ⁻¹⁾` = first-order gradient
- `hᵢ = ∂²l/∂ŷᵢ⁽ᵗ⁻¹⁾²` = second-order gradient (Hessian)

#### Feature Engineering for Energy Markets

**Temporal Features:**
- Hour of day: `h ∈ [0, 23]`
- Day of week: `d ∈ [0, 6]`
- Month: `m ∈ [1, 12]`
- Is weekend: binary `{0, 1}`
- Is peak hour: binary `{0, 1}` (typically 7-22)

**Lag Features:**
- Previous hour: `pₜ₋₁`
- Previous day same hour: `pₜ₋₂₄`
- Previous week same hour: `pₜ₋₁₆₈`

**Rolling Statistics:**
- 24-hour moving average: `MA₂₄(p) = (1/24) Σᵢ₌₀²³ pₜ₋ᵢ`
- 24-hour standard deviation: `σ₂₄(p)`
- 7-day moving average: `MA₁₆₈(p)`

**Implementation:**
- Uses `xgboost.XGBRegressor`
- Hyperparameter tuning via cross-validation
- Feature importance analysis for interpretability

---

### LSTM (Long Short-Term Memory Networks)

#### Mathematical Formulation

LSTM networks address the vanishing gradient problem in traditional RNNs by introducing gating mechanisms and a cell state.

**LSTM Cell Architecture:**

At each time step t, an LSTM cell processes input `xₜ` and previous hidden state `hₜ₋₁`:

**1. Forget Gate:**

Controls what information to discard from cell state:

```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
```

**2. Input Gate:**

Controls what new information to store:

```
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)
C̃ₜ = tanh(WC · [hₜ₋₁, xₜ] + bC)
```

**3. Cell State Update:**

```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
```

**4. Output Gate:**

Controls what information to output:

```
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
hₜ = oₜ ⊙ tanh(Cₜ)
```

Where:
- `σ` = sigmoid activation function: `σ(x) = 1/(1 + e⁻ˣ)`
- `⊙` = element-wise multiplication (Hadamard product)
- `W*` = weight matrices
- `b*` = bias vectors
- `hₜ` = hidden state at time t
- `Cₜ` = cell state at time t

**Output Layer:**

For regression (price forecasting):

```
ŷₜ = Wy·hₜ + by
```

**Loss Function:**

Mean Squared Error (MSE):

```
L = (1/n) Σᵢ (yᵢ - ŷᵢ)²
```

**Training:**

- Backpropagation Through Time (BPTT)
- Adam optimizer with adaptive learning rate
- Gradient clipping to prevent exploding gradients

#### Sequence Preprocessing

**Input Shape:** `(batch_size, sequence_length, n_features)`

For price forecasting:
- `sequence_length` = 24 to 168 hours (1 day to 1 week)
- `n_features` = prices + weather + demand + calendar features

**Normalization:**

Min-max scaling to [0, 1]:

```
x_normalized = (x - xₘᵢₙ) / (xₘₐₓ - xₘᵢₙ)
```

**Implementation:**
- Uses `tensorflow.keras.layers.LSTM`
- Bidirectional LSTM for capturing past and future context
- Dropout layers for regularization
- Early stopping based on validation loss

---

## Renewable Energy Models

### Wind Power Curves

#### Mathematical Formulation

Wind turbine power output depends on wind speed through a non-linear power curve:

**Piecewise Power Function:**

```
P(v) = {
    0,                           if v < vᶜᵘᵗ⁻ⁱⁿ
    Pᵣₐₜₑₐ(v³ - vᶜᵘᵗ⁻ⁱⁿ³)/(vᵣₐₜₑₐ³ - vᶜᵘᵗ⁻ⁱⁿ³),  if vᶜᵘᵗ⁻ⁱⁿ ≤ v < vᵣₐₜₑₐ
    Pᵣₐₜₑₐ,                      if vᵣₐₜₑₐ ≤ v < vᶜᵘᵗ⁻ᵒᵘᵗ
    0,                           if v ≥ vᶜᵘᵗ⁻ᵒᵘᵗ
}
```

Where:
- `P(v)` = power output (MW) at wind speed v
- `v` = wind speed (m/s)
- `vᶜᵘᵗ⁻ⁱⁿ` = cut-in speed (typically 3-4 m/s)
- `vᵣₐₜₑₐ` = rated speed (typically 12-15 m/s)
- `vᶜᵘᵗ⁻ᵒᵘᵗ` = cut-out speed (typically 25 m/s)
- `Pᵣₐₜₑₐ` = rated power capacity (MW)

**Theoretical Power (Betz Limit):**

Maximum extractable power from wind:

```
Pₜₕₑₒᵣᵧ = (1/2) ρ A v³ Cₚ
```

Where:
- `ρ` = air density (typically 1.225 kg/m³ at sea level)
- `A` = rotor swept area (m²) = πr², r = rotor radius
- `v` = wind speed (m/s)
- `Cₚ` = power coefficient (max 0.593 by Betz limit, practical ~0.4-0.5)

**Capacity Factor:**

```
CF = (Actual Energy Output) / (Rated Capacity × Time Period)
```

Typical onshore wind: 25-40%
Typical offshore wind: 35-50%

#### Wind Speed Distributions

**Weibull Distribution:**

Commonly used to model wind speed:

```
f(v; k, λ) = (k/λ)(v/λ)^(k-1) exp(-(v/λ)^k)
```

Where:
- `k` = shape parameter (typically 1.5-2.5)
- `λ` = scale parameter (related to mean wind speed)

**Implementation:**
- Power curve lookup with interpolation
- Height adjustment using power law: `v₂ = v₁(h₂/h₁)^α`, α ≈ 0.14-0.2
- Turbulence intensity modeling
- Wake effects for wind farm aggregation

---

### Solar Irradiance Models

#### Mathematical Formulation

Solar PV power output depends on irradiance, temperature, and panel characteristics:

**DC Power Output:**

```
Pᴰᶜ = η · A · G · [1 - β(Tᶜₑₗₗ - Tᵣₑᶠ)]
```

Where:
- `η` = panel efficiency (typically 15-22%)
- `A` = panel area (m²)
- `G` = global horizontal irradiance (W/m²)
- `β` = temperature coefficient (typically -0.004 to -0.005 /°C)
- `Tᶜₑₗₗ` = cell temperature (°C)
- `Tᵣₑᶠ` = reference temperature (typically 25°C)

**Cell Temperature:**

```
Tᶜₑₗₗ = Tₐₘᵦ + (NOCT - 20) · (G/800)
```

Where:
- `Tₐₘᵦ` = ambient temperature (°C)
- `NOCT` = Nominal Operating Cell Temperature (typically 45°C)

**AC Power Output:**

```
Pᴬᶜ = Pᴰᶜ · ηᵢₙᵥ · ηₘᵢₛᶜ
```

Where:
- `ηᵢₙᵥ` = inverter efficiency (typically 95-98%)
- `ηₘᵢₛᶜ` = miscellaneous losses (soiling, wiring, etc., typically 0.85-0.95)

#### Solar Position

**Solar Zenith Angle:**

```
cos(θz) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(ω)
```

Where:
- `θz` = solar zenith angle
- `φ` = latitude
- `δ` = solar declination angle
- `ω` = hour angle

**Solar Declination:**

```
δ = 23.45° · sin[360°(284 + n)/365]
```

Where:
- `n` = day of year (1-365)

**Clear-Sky Irradiance (Simplified):**

```
Gᶜₗₑₐᵣ = 1367 · cos(θz) · τᴬᴹ
```

Where:
- `1367 W/m²` = solar constant
- `τᴬᴹ` = atmospheric transmittance as function of air mass

**Capacity Factor:**

Typical utility-scale solar PV: 15-30% (higher in desert regions)

#### Cloud Cover Effects

**Clearness Index:**

```
kₜ = G / Gᶜₗₑₐᵣ
```

Where:
- `kₜ ∈ [0, 1]` indicates cloud cover (0 = overcast, 1 = clear sky)

**Implementation:**
- Irradiance-to-power mapping with temperature derating
- Sun position calculations (azimuth, elevation)
- Plane-of-array irradiance from GHI/DNI/DHI
- Array configuration and shading analysis

---

## Trading Strategies

### Mean Reversion

#### Mathematical Formulation

Mean reversion assumes prices oscillate around a long-term equilibrium level.

**Ornstein-Uhlenbeck Process:**

```
dpₜ = θ(μ - pₜ)dt + σdWₜ
```

Where:
- `pₜ` = price at time t
- `θ` = speed of mean reversion (θ > 0)
- `μ` = long-term mean
- `σ` = volatility
- `dWₜ` = Wiener process (Brownian motion)

**Half-Life of Mean Reversion:**

```
t₁/₂ = ln(2) / θ
```

**Z-Score:**

Standardized deviation from mean:

```
z = (pₜ - MA(p)) / σ(p)
```

Where:
- `MA(p)` = moving average (typically 20-50 periods)
- `σ(p)` = rolling standard deviation

**Trading Signals:**

```
Signal = {
    BUY,    if z < -zₜₕᵣₑₛₕ  (price below mean)
    SELL,   if z > +zₜₕᵣₑₛₕ  (price above mean)
    HOLD,   otherwise
}
```

Typical threshold: `zₜₕᵣₑₛₕ = 2.0` (2 standard deviations)

**Position Sizing:**

```
Qₜ = -K · zₜ
```

Where:
- `Qₜ` = position quantity at time t (negative z → long position)
- `K` = scaling constant

**Implementation:**
- Rolling window for MA and σ estimation
- Entry/exit thresholds based on z-score
- Stop-loss at extreme deviations (|z| > 3)
- Profit taking when price reverts (|z| < 0.5)

---

### Momentum

#### Mathematical Formulation

Momentum strategies exploit the tendency of prices to continue moving in the same direction.

**Rate of Change (ROC):**

```
ROC(t, n) = (pₜ - pₜ₋ₙ) / pₜ₋ₙ × 100%
```

**Relative Strength Index (RSI):**

```
RSI = 100 - 100/(1 + RS)

RS = (Average Gain over n periods) / (Average Loss over n periods)
```

**Moving Average Convergence Divergence (MACD):**

```
MACD = EMA₁₂(p) - EMA₂₆(p)
Signal Line = EMA₉(MACD)
Histogram = MACD - Signal Line
```

Where:
- `EMAₙ` = exponential moving average with period n
- `EMAₜ = α·pₜ + (1-α)·EMAₜ₋₁`, α = 2/(n+1)

**Trading Signals:**

```
Signal = {
    BUY,    if MACD > Signal Line AND MACD > 0
    SELL,   if MACD < Signal Line AND MACD < 0
    HOLD,   otherwise
}
```

**Trend Strength:**

```
ADX (Average Directional Index) = 100 × EMA₁₄(|+DI - -DI|) / (+DI + -DI)
```

Strong trend: ADX > 25

**Risk Management:**

```
Position Size = (Account Risk) / (Entry Price - Stop Loss)
```

Typical stop-loss: 2-5% below entry for long, above entry for short

**Implementation:**
- Multiple timeframe analysis (hourly, daily, weekly)
- Confirmation with volume indicators
- Trailing stops to lock in profits
- Avoid trading during low-volume periods

---

### Spread Trading

#### Mathematical Formulation

Spread trading exploits price relationships between correlated energy products.

**Spread Definition:**

```
Sₜ = p₁ₜ - β·p₂ₜ
```

Where:
- `p₁ₜ`, `p₂ₜ` = prices of asset 1 and asset 2
- `β` = hedge ratio

**Hedge Ratio (Ordinary Least Squares):**

```
β = Cov(p₁, p₂) / Var(p₂)
```

Or equivalently:

```
β = ρ(p₁, p₂) · σ(p₁) / σ(p₂)
```

**Cointegration Test:**

Engle-Granger two-step method:

1. Regress: `p₁ₜ = α + β·p₂ₜ + εₜ`
2. Test residuals for stationarity: `ΔSₜ = γ·Sₜ₋₁ + ηₜ`

If γ < 0 and significant, series are cointegrated.

**Z-Score of Spread:**

```
zₛ = (Sₜ - MA(S)) / σ(S)
```

**Trading Signals:**

```
Signal = {
    LONG SPREAD (buy p₁, sell p₂),   if zₛ < -zₜₕᵣₑₛₕ
    SHORT SPREAD (sell p₁, buy p₂),  if zₛ > +zₜₕᵣₑₛₕ
    CLOSE,                           if |zₛ| < zₑₓᵢₜ
}
```

Typical: `zₜₕᵣₑₛₕ = 2.0`, `zₑₓᵢₜ = 0.5`

**Profit and Loss:**

```
PnL = Q₁·Δp₁ - β·Q₂·Δp₂ - Transaction Costs
```

**Common Energy Spreads:**

- **Spark Spread:** `Electricity Price - (Heat Rate × Natural Gas Price) - VOM`
- **Dark Spread:** `Electricity Price - (Heat Rate × Coal Price) - VOM`
- **Crack Spread:** `Gasoline + Diesel - Crude Oil` (3:2:1 ratio)
- **Location Spread:** `Price(Hub A) - Price(Hub B)` (transmission congestion)

**Implementation:**
- Rolling regression for dynamic β estimation
- Johansen test for multi-asset cointegration
- Half-life of mean reversion for spread
- Calendar spread opportunities (near-month vs. far-month futures)

---

### Renewable Arbitrage

#### Mathematical Formulation

Renewable arbitrage exploits price differentials caused by renewable generation patterns.

**Duck Curve Arbitrage:**

Capitalize on solar-induced price drops midday and ramps at sunset:

```
Profit = Q·(Pₑᵥₑₙᵢₙ_ᵍ - Pₘᵢₐₐₐᵧ) - Cₛₜₒᵣₐ_ᵍₑ
```

Where:
- `Q` = energy quantity traded (MWh)
- `Pₑᵥₑₙᵢₙ_ᵍ` = evening peak price
- `Pₘᵢₐₐₐᵧ` = midday off-peak price
- `Cₛₜₒᵣₐ_ᵍₑ` = storage costs (if using batteries)

**Curtailment Opportunity:**

When renewable generation exceeds demand + transmission capacity:

```
Pᶜᵘᵣₜₐᵢₗₘₑₙₜ ≤ 0  (potentially negative prices)
```

**Complementarity Score:**

Measure of wind-solar portfolio diversification:

```
Comp(W, S) = 1 - Cor(Gᵂ, Gˢ)
```

Where:
- `Gᵂ`, `Gˢ` = wind and solar generation profiles
- `Cor` = Pearson correlation coefficient

High complementarity (low correlation) → stable combined output

**Renewable Energy Credit (REC) Arbitrage:**

```
Profit = Pᵣₑᶜ + Pₑₙₑᵣ_ᵍᵧ - Cᵍₑₙₑᵣₐₜᵢₒₙ
```

Where:
- `Pᵣₑᶜ` = REC market price ($/MWh)
- `Pₑₙₑᵣ_ᵍᵧ` = wholesale energy price ($/MWh)
- `Cᵍₑₙₑᵣₐₜᵢₒₙ` = generation cost including LCOE ($/MWh)

**Grid Services Value:**

Renewables + storage can provide ancillary services:

```
Vₜₒₜₐₗ = Pₑₙₑᵣ_ᵍᵧ + Pᶠᵣₑ_ᵧᵣₑ_ᵍ + Pᵣₑ_ᵍᵤₗₐₜᵢₒₙ + Pᵣₑₛₑᵣᵥₑ
```

**Implementation:**
- Forecast renewable generation profiles
- Identify price volatility patterns correlated with renewables
- Optimal storage dispatch (if available)
- REC market dynamics and state RPS compliance demand

---

## Portfolio Optimization

### Markowitz Mean-Variance Optimization

#### Mathematical Formulation

Modern Portfolio Theory seeks to maximize expected return for a given level of risk.

**Objective:**

```
maximize:  μₚ = wᵀμ
subject to: σₚ² = wᵀΣw ≤ σₜₐᵣ_ᵍₑₜ²
            Σwᵢ = 1
            wᵢ ≥ 0  (for long-only)
```

Where:
- `w` = vector of portfolio weights (n × 1)
- `μ` = vector of expected returns (n × 1)
- `Σ` = covariance matrix (n × n)
- `μₚ` = portfolio expected return
- `σₚ²` = portfolio variance

**Efficient Frontier:**

For a target return μ*, solve:

```
minimize:  σₚ² = wᵀΣw
subject to: wᵀμ = μ*
            wᵀ1 = 1
```

**Lagrangian:**

```
L(w, λ₁, λ₂) = (1/2)wᵀΣw - λ₁(wᵀμ - μ*) - λ₂(wᵀ1 - 1)
```

**First-Order Conditions:**

```
∂L/∂w = Σw - λ₁μ - λ₂1 = 0
```

**Closed-Form Solution:**

```
w* = Σ⁻¹(λ₁μ + λ₂1)
```

Where λ₁ and λ₂ are solved from constraints.

**Sharpe Ratio:**

```
SR = (μₚ - rᶠ) / σₚ
```

Where:
- `rᶠ` = risk-free rate

**Maximum Sharpe Ratio Portfolio:**

```
w* ∝ Σ⁻¹(μ - rᶠ1)
```

Normalized so that `wᵀ1 = 1`.

**Implementation:**
- `scipy.optimize.minimize` with constraints
- Covariance matrix regularization (shrinkage estimators)
- Rolling window estimation of μ and Σ
- Transaction cost penalty in objective function

---

### Risk Parity

#### Mathematical Formulation

Risk parity allocates capital so that each asset contributes equally to portfolio risk.

**Risk Contribution:**

The marginal contribution of asset i to portfolio variance:

```
RCᵢ = wᵢ · (∂σₚ/∂wᵢ) = wᵢ · (Σw)ᵢ / σₚ
```

**Equal Risk Contribution:**

```
RCᵢ = (1/n) · σₚ²  for all i
```

**Optimization Problem:**

```
minimize:  Σᵢ (RCᵢ - (1/n)·σₚ²)²
subject to: Σwᵢ = 1
            wᵢ ≥ 0
```

**Alternative Formulation (Leverage):**

Naive risk parity (inverse volatility weighting):

```
wᵢ = (1/σᵢ) / Σⱼ(1/σⱼ)
```

**Risk Budgeting:**

Generalization allowing different risk budgets:

```
RCᵢ = bᵢ · σₚ²
```

Where `Σbᵢ = 1` and `bᵢ` is the risk budget for asset i.

**Implementation:**
- Convex optimization with CVXPY or scipy
- Iterative algorithms (cyclical coordinate descent)
- Handles leverage via target volatility
- Works well for diversified multi-asset portfolios

---

### Black-Litterman Model

#### Mathematical Formulation

Black-Litterman combines market equilibrium with investor views to produce expected returns.

**Equilibrium Returns (Reverse Optimization):**

```
Π = λΣwₘₖₜ
```

Where:
- `Π` = implied equilibrium excess returns (n × 1)
- `λ` = risk aversion coefficient
- `Σ` = covariance matrix
- `wₘₖₜ` = market capitalization weights

**Investor Views:**

Expressed as linear combinations:

```
P·μ = Q + ε,  ε ~ N(0, Ω)
```

Where:
- `P` = pick matrix (k × n), k views on n assets
- `Q` = view returns (k × 1)
- `Ω` = diagonal uncertainty matrix for views (k × k)

**Example View:**

"Asset 1 will outperform asset 2 by 2%"

```
P = [1, -1, 0, ..., 0]
Q = [0.02]
```

**Posterior Distribution:**

Bayesian update combining equilibrium and views:

```
μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]
```

```
Σ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹
```

Where:
- `τ` = scalar uncertainty in prior (typically 0.01 - 0.05)
- `μ_BL` = Black-Litterman expected returns
- `Σ_BL` = Black-Litterman covariance

**Portfolio Optimization:**

Use `μ_BL` and `Σ_BL` in mean-variance optimization:

```
w* = (λΣ_BL)⁻¹ · μ_BL
```

**Implementation:**
- Market equilibrium from historical data or CAPM
- View specification from analyst forecasts
- Confidence in views controls Ω (higher confidence → lower Ω values)
- Handles partial views (not all assets need views)

---

### CVaR Optimization

#### Mathematical Formulation

Conditional Value at Risk (CVaR) optimization minimizes expected losses in the worst α% of cases.

**Value at Risk (VaR):**

```
VaRₐ(X) = inf{x : P(X ≤ x) ≥ α}
```

**Conditional Value at Risk (CVaR):**

Also known as Expected Shortfall (ES):

```
CVaRₐ(X) = E[X | X ≤ VaRₐ(X)]
```

**Portfolio CVaR:**

For portfolio returns `Rₚ = wᵀR`:

```
CVaRₐ(Rₚ) = E[Rₚ | Rₚ ≤ VaRₐ(Rₚ)]
```

**Optimization Problem:**

```
minimize:  CVaRₐ(-Rₚ)  (negative returns = losses)
subject to: wᵀμ ≥ μₜₐᵣ_ᵍₑₜ
            Σwᵢ = 1
            wᵢ ≥ 0
```

**Rockafellar-Uryasev Formulation:**

CVaR can be computed as:

```
CVaRₐ = min_ζ {ζ + (1/α)·E[(−Rₚ − ζ)⁺]}
```

Where:
- `ζ` = VaR threshold
- `(x)⁺ = max(x, 0)` = positive part

**Linear Programming Formulation (Scenario-Based):**

For M historical scenarios:

```
minimize:  ζ + (1/(α·M))·Σₘ uₘ
subject to: uₘ ≥ -Σᵢ wᵢrᵢₘ - ζ  for all m
            uₘ ≥ 0
            Σwᵢ = 1
            wᵢ ≥ 0
```

Where:
- `rᵢₘ` = return of asset i in scenario m
- `uₘ` = auxiliary variables representing losses beyond VaR

**Implementation:**
- Scenario generation from historical data or Monte Carlo
- Linear programming solver (CVXPY, Gurobi)
- Typical confidence level: α = 0.05 (5% CVaR)
- Robust to fat tails and extreme events

---

## Risk Analytics

### Value at Risk (VaR)

#### Mathematical Formulation

VaR quantifies the maximum expected loss over a time horizon at a given confidence level.

**Definition:**

```
P(Loss > VaRₐ) = α
```

Or equivalently:

```
P(Loss ≤ VaRₐ) = 1 - α
```

Typical confidence levels: α = 0.01 (99% VaR) or α = 0.05 (95% VaR)

**Parametric VaR (Variance-Covariance Method):**

Assumes returns are normally distributed:

```
VaRₐ = μₚ - zₐ·σₚ
```

Where:
- `μₚ` = expected portfolio return
- `σₚ` = portfolio standard deviation
- `zₐ` = z-score for confidence level α (e.g., z₀.₀₅ = 1.645)

For daily VaR with mean ≈ 0:

```
VaRₐ = zₐ·σₚ·√Δt
```

Where `Δt` = time horizon in days.

**Historical Simulation VaR:**

Non-parametric method using historical returns:

1. Compute historical portfolio returns: `Rₜ = wᵀrₜ` for t = 1, ..., T
2. Sort returns in ascending order
3. VaR = -R₍ₐₙ₎ where n = ⌈α·T⌉

**Monte Carlo VaR:**

Simulate future scenarios:

1. Generate M scenarios of returns: `r̃₁, ..., r̃ₘ`
2. Compute portfolio returns: `R̃ₘ = wᵀr̃ₘ`
3. VaR = -R̃₍ₐₘ₎

**Scaling VaR Across Time Horizons:**

For normal returns:

```
VaRₜ = VaR₁·√t
```

Where VaR₁ is 1-day VaR and t is the time horizon in days.

**Implementation:**
- Historical simulation is most common for energy portfolios
- Exponential weighting to emphasize recent data
- Monte Carlo with GARCH volatility modeling
- Backtesting: Kupiec test for VaR model validation

---

### Conditional Value at Risk (CVaR)

#### Mathematical Formulation

CVaR measures expected loss given that loss exceeds VaR.

**Definition:**

```
CVaRₐ = E[Loss | Loss > VaRₐ]
```

**For Continuous Distribution:**

```
CVaRₐ = (1/α) ∫₀ᵅ VaRᵤ du
```

**For Normal Distribution:**

```
CVaRₐ = μₚ - σₚ·φ(zₐ)/α
```

Where:
- `φ(z)` = standard normal PDF = (1/√(2π))·exp(-z²/2)

**Historical Simulation CVaR:**

```
CVaRₐ = (1/⌈α·T⌉)·Σᵢ₌₁^⌈ᵅᵀ⌉ (-R₍ᵢ₎)
```

Average of worst α% of historical returns.

**Properties:**

- **Coherent Risk Measure:** Satisfies subadditivity, monotonicity, positive homogeneity, translation invariance
- **Convex:** Allows efficient optimization
- **Tail Risk:** Captures severity of extreme losses (not just probability)

**CVaR vs VaR:**

```
CVaRₐ ≥ VaRₐ  always
```

Equality holds only if distribution has no tail beyond VaR.

**Implementation:**
- Preferred over VaR for optimization (convex)
- Used in portfolio optimization and risk budgeting
- Backtesting with Expected Shortfall tests
- Energy portfolios: captures extreme weather/outage events

---

### Risk Decomposition

#### Marginal Contribution to Risk (MCR)

**Definition:**

```
MCRᵢ = ∂σₚ/∂wᵢ = (Σw)ᵢ / σₚ
```

**Component Contribution to Risk (CCR):**

```
CCRᵢ = wᵢ·MCRᵢ
```

**Verification:**

```
Σᵢ CCRᵢ = σₚ  (Euler's theorem)
```

**Percentage Contribution:**

```
%RCᵢ = CCRᵢ / σₚ × 100%
```

#### Factor Risk Decomposition

Decompose risk into systematic factors (Barra-style):

**Factor Model:**

```
rᵢ = Σⱼ βᵢⱼ·fⱼ + εᵢ
```

Where:
- `rᵢ` = return of asset i
- `βᵢⱼ` = factor loading of asset i on factor j
- `fⱼ` = factor return j
- `εᵢ` = idiosyncratic return

**Portfolio Risk:**

```
σₚ² = wᵀ(BFBᵀ + D)w
```

Where:
- `B` = factor loading matrix (n × k)
- `F` = factor covariance matrix (k × k)
- `D` = diagonal matrix of idiosyncratic variances

**Factor Risk Contribution:**

```
FRCⱼ = (wᵀB)ⱼ · (FBᵀw)ⱼ / σₚ²
```

**Common Energy Factors:**

- **Market Risk:** Overall energy market movements
- **Fuel Risk:** Natural gas, coal, nuclear prices
- **Weather Risk:** Temperature, wind, solar
- **Demand Risk:** Economic activity, seasonality
- **Regulatory Risk:** Policy changes, carbon pricing

**Implementation:**
- Principal Component Analysis (PCA) for factor extraction
- Factor models from external providers (e.g., Barra)
- Custom factors for energy-specific risks
- Time-varying factor exposures

---

## Performance Metrics

### Returns

**Simple Return:**

```
Rₜ = (Vₜ - Vₜ₋₁) / Vₜ₋₁
```

**Log Return:**

```
rₜ = ln(Vₜ / Vₜ₋₁)
```

**Cumulative Return:**

```
Rcum = (Vₜ - V₀) / V₀
```

**Annualized Return:**

```
Rₐₙₙ = (1 + Rcum)^(252/T) - 1
```

For daily data, T = number of trading days.

### Risk Metrics

**Volatility (Standard Deviation):**

```
σ = √[(1/(n-1))·Σ(Rₜ - R̄)²]
```

**Annualized Volatility:**

```
σₐₙₙ = σₐₐᵢₗᵧ·√252
```

**Downside Deviation:**

Only considers negative returns:

```
σₐ = √[(1/(n-1))·Σ min(Rₜ - R̄, 0)²]
```

**Maximum Drawdown:**

```
MDD = max_t [(max_s≤t Vₛ - Vₜ) / max_s≤t Vₛ]
```

Largest peak-to-trough decline.

### Risk-Adjusted Returns

**Sharpe Ratio:**

```
SR = (R̄ₚ - rᶠ) / σₚ
```

**Sortino Ratio:**

Uses downside deviation instead of total volatility:

```
Sortino = (R̄ₚ - rᶠ) / σₐ
```

**Calmar Ratio:**

```
Calmar = R̄ₐₙₙ / |MDD|
```

**Information Ratio:**

Measures excess return relative to benchmark:

```
IR = (R̄ₚ - R̄ᵦ) / TE
```

Where:
- `R̄ᵦ` = benchmark return
- `TE` = tracking error = σ(Rₚ - Rᵦ)

### Alpha and Beta

**Capital Asset Pricing Model (CAPM):**

```
Rₚₜ = α + β·Rₘₜ + εₜ
```

**Beta:**

```
β = Cov(Rₚ, Rₘ) / Var(Rₘ)
```

**Alpha:**

```
α = R̄ₚ - rᶠ - β·(R̄ₘ - rᶠ)
```

Positive alpha indicates outperformance.

### Trade-Specific Metrics

**Win Rate:**

```
Win Rate = (# Winning Trades) / (# Total Trades)
```

**Profit Factor:**

```
PF = (Σ Winning Trade PnLs) / |Σ Losing Trade PnLs|
```

**Average Win/Loss Ratio:**

```
W/L Ratio = (Average Winning Trade) / |Average Losing Trade|
```

**Expectancy:**

```
E = (Win Rate × Average Win) - (Loss Rate × |Average Loss|)
```

**Implementation:**
- Rolling window metrics for time-varying analysis
- Benchmark comparison (e.g., SPY for equities, energy ETFs)
- Attribution analysis (returns from different strategies)
- Statistical significance testing (t-tests for Sharpe ratio)

---

## Transaction Cost Models

### Slippage

**Market Impact:**

Price moves against the order due to size:

```
Slippage = k·(Q / ADV)^γ
```

Where:
- `Q` = order quantity
- `ADV` = average daily volume
- `k` = market impact coefficient (empirically estimated)
- `γ` = impact exponent (typically 0.5-1.0)

**Bid-Ask Spread:**

```
Spread Cost = (Pₐₛₖ - Pᵦᵢₐ) / 2
```

For a market order:

```
Cost = {
    Pₐₛₖ - Pₘᵢₐ,  for buy orders
    Pₘᵢₐ - Pᵦᵢₐ,  for sell orders
}
```

**Realized Slippage:**

```
Realized Slippage = |Pₑₓₑᶜᵤₜₑₐ - Pₛᵢ_ᵍₙₐₗ|
```

### Commission and Fees

**Fixed Commission:**

```
Commission = max(Cₘᵢₙ, Cᵦₐₛₑ + c·Q)
```

Where:
- `Cₘᵢₙ` = minimum commission per trade
- `Cᵦₐₛₑ` = base commission
- `c` = per-unit commission rate
- `Q` = quantity traded

**Percentage Commission:**

```
Commission = r·|Pₑₓₑᶜᵤₜₑₐ·Q|
```

**Exchange Fees:**

Clearing, regulatory, and exchange fees:

```
Fees = fₑₓᶜₕₐₙ_ᵍₑ + fₒₗₑₐᵣᵢₙ_ᵍ + fᵣₑ_ᵍᵤₗₐₜₒᵣᵧ
```

### Total Transaction Cost

```
Total Cost = |Slippage| + Commission + Fees
```

**As Percentage:**

```
Cost % = (Total Cost) / |Pₑₓₑᶜᵤₜₑₐ·Q| × 100%
```

### Opportunity Cost

**Delay Cost:**

If execution is delayed by Δt:

```
Opportunity Cost = Q·(Pₜ₊ₐₜ - Pₜ)
```

**Partial Fill Cost:**

If only fraction f of order fills:

```
Missed Opportunity = (1 - f)·Q·(Pᵧᵤₜᵤᵣₑ - Pₗᵢₘᵢₜ)
```

### Break-Even Analysis

**Minimum Price Move:**

Price must move by at least transaction costs to break even:

```
Pₘᵢₙ = (Total Cost) / Q
```

**Round-Trip Cost:**

Entering and exiting a position:

```
RT Cost = 2·(Spread Cost + Commission + Slippage)
```

### Implementation Models

**Fixed Percentage (Simple):**

```
Cost = bps·|Trade Value|
```

Common assumption: bps = 5-10 basis points (0.05%-0.10%)

**Tiered Model:**

```
Cost = {
    c₁·|Trade Value|,  if |Trade Value| < T₁
    c₂·|Trade Value|,  if T₁ ≤ |Trade Value| < T₂
    c₃·|Trade Value|,  if |Trade Value| ≥ T₂
}
```

**Volume-Based:**

```
Cost = c(Q) = {
    cₕᵢ_ᵍₕ,  if Q > threshold·ADV  (high impact)
    cₗₒᵥᵥ,   otherwise              (normal)
}
```

**Energy-Specific Considerations:**

- **Delivery Charges:** Physical delivery of electricity/gas
- **Balancing Costs:** Deviations from scheduled generation/consumption
- **Capacity Charges:** Transmission rights and reservations
- **Congestion Charges:** LMP differences across nodes

**Backtesting with Transaction Costs:**

```
PnL_net = PnL_gross - Σₜᵣₐₐₑₛ Transaction Cost(trade)
```

Adjust Sharpe ratio:

```
SR_net = (R̄_net - rᶠ) / σ_net
```

**Implementation:**
- Calibrate cost model from historical execution data
- Market microstructure analysis (order book depth)
- Optimize order execution (TWAP, VWAP, iceberg orders)
- Cost-aware portfolio optimization (turnover constraints)

---

## References and Further Reading

### Books

- **Time Series Analysis:** Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.
- **Machine Learning:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- **Portfolio Optimization:** Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*, 7(1), 77-91.
- **Risk Management:** Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*. McGraw-Hill.
- **Energy Trading:** Eydeland, A., & Wolyniec, K. (2003). *Energy and Power Risk Management*. Wiley.

### Research Papers

- **Black-Litterman:** Black, F., & Litterman, R. (1992). "Global Portfolio Optimization." *Financial Analysts Journal*, 48(5), 28-43.
- **CVaR:** Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." *Journal of Risk*, 2, 21-42.
- **Risk Parity:** Qian, E. (2005). "Risk Parity Portfolios." *PanAgora Asset Management*.
- **Wind Power:** Wan, Y. H. (2011). *Long-Term Wind Power Variability*. NREL Technical Report.
- **Solar Forecasting:** Inman, R. H., et al. (2013). "Solar forecasting methods for renewable energy integration." *Progress in Energy and Combustion Science*, 39(6), 535-576.

---

## Implementation Notes

All algorithms documented above are implemented in Python 3.11+ with the following key libraries:

- **Data Processing:** pandas, numpy
- **Time Series:** statsmodels, pmdarima
- **Machine Learning:** scikit-learn, xgboost, tensorflow/keras
- **Optimization:** scipy.optimize, cvxpy
- **Visualization:** plotly, matplotlib, seaborn
- **Backtesting:** vectorbt, custom BacktestEngine

**Code Organization:**

- `src/models/` - Forecasting models (ARIMA, XGBoost, LSTM)
- `src/strategies/` - Trading strategies (mean reversion, momentum, etc.)
- `src/optimization/` - Portfolio optimization algorithms
- `src/backtesting/` - Backtesting engine and performance metrics
- `src/data/` - Data fetching, processing, and synthetic generation

**For detailed API documentation, see [API.md](API.md)**

**For interactive examples, see [Jupyter Notebooks](../notebooks/)**

---

*Last Updated: 2025*

*This documentation is maintained as part of the Energy Trading & Portfolio Optimization System project.*
