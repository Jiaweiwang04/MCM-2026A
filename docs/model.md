# Task 1 Model (v1): SOC–Thermal–Network Coupled Dynamical Model (Implementable, Verifiable, Reproducible)

**v1.1 update (integrating 4 realism mechanisms):**
1. **SOH (State of Health) affects capacity:** aged batteries have reduced usable capacity  
2. **Temperature affects capacity:** usable capacity drops significantly at low temperature  
3. **Nonlinear discharge internal resistance:** internal resistance increases sharply at low SOC \(\big(R_{int}(T,s)\big)\)  
4. **Signal strength affects network power:** weaker signal increases power consumption  

This document refines the existing plan to be more “reviewer-friendly”. While keeping the **three-state structure** (SOC–temperature–network tail) and the **thermal throttling feedback loop**, it additionally ensures:

- **Closed units / dimensions** (avoid \(3600\times\) mistakes)
- **Log-to-model mapping & normalization** (reproducible)
- **Parameter identifiability & staged fitting** (defend against overfitting concerns)
- **Baselines / ablations & sensitivity plans** (demonstrate the value of innovations)

---

## 0. Global conventions (mandatory)

### 0.1 Unit system (recommended: internal SI units)

Internal computation uses SI:

- time: \(t\) in seconds (s)
- temperature: \(T\) in Kelvin (K)
- power: \(P\) in watts (W)
- energy: \(E\) in joules (J)

Conversion from battery capacity in Wh:

\[
E_{nom,J}=3600\,E_{nom,Wh}
\]

Therefore, the effective capacity in the SOC equation should be \(E_{cap,J}(T,H)\) (in J).

> If you prefer to compute in Wh and hours in code, that is also fine—but you must choose **one** system and explicitly state the conversion in both the paper and code. This v1 document uses SI as the default.

### 0.2 Normalization of log inputs (reproducible)

All external inputs come from observable logs \(u(t)\), with explicit normalization:

- \(u_{on}(t)\in\{0,1\}\): screen on/off (direct mapping)
- \(b(t)\in[0,1]\): brightness  
  - if 0–255: \(b=\mathrm{clip}(\mathrm{raw}/255,0,1)\)  
  - if 0–100: \(b=\mathrm{clip}(\mathrm{raw}/100,0,1)\)
- \(u_{cpu}(t)\in[0,1]\): CPU/GPU load  
  - if given as %: \(u_{cpu}=\mathrm{clip}(\mathrm{raw}/100,0,1)\)
- \(x_{net}(t)\ge 0\): network intensity (choose one definition from logs)
  - Option A (throughput): \(x_{net}=\) bytes/s (or KB/s)
  - Option B (packet rate): \(x_{net}=\) packets/s

> Key requirement: state clearly whether you use A or B, and stay consistent throughout. The unit of \(k_{net}\) depends on this choice.

### 0.3 Observables & evaluation metrics (must be explicit)

- Strongly observable (usually available): SOC (battery percentage) time series.
- If available (much stronger): core/battery temperature, voltage/current, power.

Suggested metrics:

- SOC prediction: MAE / RMSE (percentage points)
- If temperature is available: temperature MAE (K or °C)
- Explainability: show correspondence between \(\phi(T)\) activation windows and power changes

---

## 1. States and external inputs

### 1.1 State variables

- \(s(t)\in[0,100]\): SOC (%)
- \(T(t)\in[T_{env},T_{max}]\): core temperature (K)
- \(z(t)\ge 0\): network tail / hysteresis state (same unit as \(x_{net}\); interpretable as an exponential smoothing of \(x_{net}\))

### 1.2 External inputs (from logs only)

- \(u_{on}(t), b(t), u_{cpu}(t), x_{net}(t)\)
- \(type_{net}(t) \in \{\mathrm{wifi, cellular, none}\}\): network type (if missing, default to cellular)
- \(S_{signal}(t) \in (0,1]\): signal strength (1.0 is full; if missing, default to 1.0)
- \(T_{env}(t)\) (constant or time series; if unmeasured, treat as a constant parameter)

---

## 2. Core mechanism: Thermal Throttling Loop

Define the throttling factor:

\[
\phi(T)=\frac{1}{1+e^{\lambda\,(T-T_{thresh})}}
\]

Interpretation:

- if \(T<T_{thresh}\): \(\phi\approx 1\), almost no suppression
- if \(T\ge T_{thresh}\): \(\phi\to 0\), strong suppression of active power (frequency/brightness throttling)

**Optional enhancement (if reviewers question hysteresis):** use different on/off thresholds or add dependence on \(\dot T\). In v1 we keep logistic form to reduce parameters and improve identifiability.

---

## 3. Governing equations

### 3.1 Network tail-state equation (Tail Energy State)

**v1.2 extension: distinguish network types**

WiFi and cellular have significantly different tail behaviors: WiFi transitions faster (short tail), while cellular RRC state machines can produce longer tails.

\[
\frac{dz}{dt}=\frac{1}{\tau_{net}^{(type)}}\big(x_{net}(t)-z(t)\big)
\]

- \(\tau_{net}^{(type)}>0\): tail time constant (s), selected by \(type \in \{\mathrm{wifi, cellular, none}\}\)
  - \(\tau_{net}^{(wifi)} \approx 5\) s
  - \(\tau_{net}^{(cellular)} \approx 15\) s
  - \(\tau_{net}^{(none)} \approx 0.1\) s (fast decay)
- This is equivalent to exponential smoothing of \(x_{net}\) with time constant \(\tau_{net}^{(type)}\), so \(z\) and \(x_{net}\) share the same unit.

**Design principle:** keep only three states \((s,T,z)\); implement type differences via parameter switching (simpler, more identifiable).

### 3.2 Power decomposition

Total system power:

\[
P_{sys}(t)=P_{base}+\phi(T)\big(P_{scr}(u_{on},b)+P_{cpu}(u_{cpu})\big)+P_{net}(z, S_{signal})
\]

Components:

- Screen:
\[
P_{scr}=u_{on}(t)\,(\alpha_0+\alpha_1 b(t))
\]
- CPU/GPU:
\[
P_{cpu}=k_{cpu} \times (3\,u_{cpu}(t))^3
\]
- Network (linear in tail state + signal-strength correction + type split):
\[
P_{net}=\frac{1}{S_{signal}(t)}\Big(P_{idle}^{(type)}+k_{net}^{(type)} z(t)\Big)
\]

We model higher power under weak signal (\(S_{signal}<1\)) because the RF module must transmit harder to maintain connectivity. Different network types have different idle power and tail coefficients:
- \(P_{idle}^{(wifi)} < P_{idle}^{(cellular)}\)
- \(k_{net}^{(wifi)} \le k_{net}^{(cellular)}\)

**Parameter reduction tip:** if data is insufficient to fit all type-specific terms, fit WiFi-only and cellular-only segments separately, or keep only \(\tau_{net}^{(type)}\) type-splitting (tail duration is usually the most visible difference).

> Defensive statement (for reviewers): whether network power is throttled by \(\phi\) is uncertain. v1 assumes network power is **not** strongly suppressed by \(\phi\) (consistent with separate RF power management), and we test this via ablations in Section 7.

### 3.3 Internal resistance and loss

Use an Arrhenius relation for temperature dependence plus SOC nonlinearity:

\[
R_{int}(T,s)=R_{int}^{temp}(T)\times\Big[1+c\Big(\frac{s_{ref}}{s+\varepsilon}-1\Big)\Big]
\]

Temperature term:

\[
R_{int}^{temp}(T)=R_{ref}\exp\left[\frac{E_a}{k_B}\left(\frac{1}{T}-\frac{1}{T_{ref}}\right)\right]
\]

SOC nonlinearity interpretation:
- near \(s_{ref}\) (e.g., 50%), the factor \(\approx 1\)
- as \(s\to 0\), \(\frac{s_{ref}}{s+\varepsilon}\to\infty\) and resistance increases sharply  
Suggested parameters: \(c=0.3, s_{ref}=50, \varepsilon=1.0\).

Approximate current:

\[
I(t)\approx \frac{P_{sys}(t)}{V_{nom}}
\]

Internal resistance loss:

\[
P_{loss}(t)=I(t)^2 R_{int}(T,s)=\frac{P_{sys}(t)^2}{V_{nom}^2}R_{int}(T,s)
\]

Heat generation (approx.):

\[
P_{heat}(t)\approx P_{sys}(t)+P_{loss}(t)
\]

> Note: most electrical energy ends up as heat; a small fraction becomes light (screen) or RF radiation. We assess this approximation via sensitivity analysis and show it does not change conclusions.

### 3.4 Thermal dynamics

\[
\frac{dT}{dt}=\frac{1}{C_{eq}}\big(P_{heat}(t)-hA\,(T(t)-T_{env})\big)
\]

We use an equivalent thermal capacitance \(C_{eq}\) (J/K) to reduce parameter coupling.

### 3.5 SOC dynamics (SI form + realism mechanisms)

Effective capacity (J):

\[
E_{cap,J}(T,H)=3600\,E_{nom,Wh}\,\eta_{temp}(T)\,H
\]

**1) Temperature effect on capacity** \(\eta_{temp}(T)\)

\[
C(T) = 36.713\,(T-273.15)^3 - 826.14\,(T-273.15)^2 + 5988.6\,(T-273.15) -4204.4
\]
\[
C_{ref}= 36.713\,(T_{ref}-273.15)^3 - 826.14\,(T_{ref}-273.15)^2 + 5988.6\,(T_{ref}-273.15)-4204.4
\]
\[
\eta_{temp}(T) = \frac{C(T)}{C_{ref}}
\]

- \(T_{ref}=298.15\): reference temperature, capacity retention = 1.0

**2) SOH (health)** \(H\in(0,1]\)
- new battery: \(H=1.0\)
- aged battery: \(H<1\), capacity reduced proportionally
- if no log provides \(H\): treat as a parameter or fix by literature (e.g., \(H=0.85\) means 15% aging)

SOC evolution:

\[
\frac{ds}{dt}=-\frac{100}{E_{cap,J}(T,H)}\big(P_{sys}(t)+P_{loss}(t)\big)
\]

---

## 4. Parameter list (minimal + identifiability-first)

### 4.1 Thermal

- \(C_{eq}\) (J/K): equivalent thermal capacitance  
- \(hA\) (W/K): heat transfer coefficient  
- \(T_{thresh}\) (K): throttling threshold  
- \(\lambda\) (1/K): throttling steepness  
- \(T_{env}\) (K): ambient temperature (constant or time series)

### 4.2 Power

- \(P_{base}\) (W)
- \(\alpha_0,\alpha_1\) (W)
- \(k_{cpu}\)
- \(P_{idle}^{(type)}\) (W), \(k_{net}^{(type)}\) (W per unit of \(z\))

### 4.3 Battery

- \(E_{nom,Wh}\) (Wh), \(V_{nom}\) (V)
- \(R_{ref}\) (Ω), \(T_{ref}\) (K), \(E_a\) (J)
- \(c, s_{ref}, \varepsilon\): SOC nonlinearity parameters
- \(H\): SOH (0–1], constant or from logs
- Temperature–capacity relation parameters (if alternative forms are used)

**Strong v1 dimension-reduction advice (useful for defense):**
- If temperature is unobserved: fix the shape of \(R_{int}(T)\) or \(\eta_{temp}(T)\) and fit only a scaling factor (avoid non-identifiability between “capacity change” vs “resistance change”).
- Use \(C_{eq}\) instead of \(C_{th}\) and \(m\) separately to avoid product non-identifiability.
- Fix \((c, s_{ref}, \varepsilon)\) from literature to reduce degrees of freedom.

---

## 5. Numerical implementation (discrete time)

Let the sampling interval be \(\Delta t\) (s). Initial values:

\[
s_0=100,\quad T_0=T_{env},\quad z_0=0
\]

Per-step update (explicit Euler baseline):

1. Read inputs \(u_{on}, b, u_{cpu}, x_{net}, type_{net}, S_{signal}, T_{env}\)
2. Pick \(\tau_{net}^{(type)}\); update  
   \(z_{k+1}=z_k+\Delta t\,(x_{net,k}-z_k)/\tau_{net}^{(type)}\), then clamp \(z_{k+1}=\max(0,z_{k+1})\)
3. \(\phi_k=\phi(T_k)\)
4. Pick \((P_{idle}^{(type)}, k_{net}^{(type)})\); compute \(P_{scr,k}, P_{cpu,k}, P_{net,k}(z_k,S_{signal,k}), P_{sys,k}\)
5. Compute \(R_{int}(T_k, s_k), P_{loss,k}, P_{heat,k}\)
6. Compute \(\eta_{temp}(T_k)\) and \(E_{cap,J}(T_k,H)\)
7. Update temperature:  
   \(T_{k+1}=T_k+\Delta t\,(P_{heat,k}-hA(T_k-T_{env,k}))/C_{eq}\)
8. Update SOC:  
   \(s_{k+1}=s_k-\Delta t\,100\,(P_{sys,k}+P_{loss,k})/E_{cap,J}(T_k,H)\)
9. Clamp \(s\leftarrow\mathrm{clip}(s,0,100)\)

> If stiffness or oscillations appear (due to throttling feedback), switch to RK4 or lightly smooth \(\phi\) (without adding new parameters).

---

## 6. Fitting & identifiability (what reviewers care about)

### 6.1 Staged fitting (recommended workflow)

**Case A: temperature observations \(T_{obs}(t)\) available (best)**  
1. Fit thermal parameters \(C_{eq}, hA, T_{thresh}, \lambda\) using \(T_{obs}\)
2. Fix thermal parameters and fit power parameters \(P_{base}, \alpha\)'s, \(k_{cpu}\), \(P_{idle}\), \(k_{net}\) using SOC
3. Finally fine-tune resistance-related terms (or keep within literature ranges) to avoid overfitting

**Case B: no temperature observations (weak)**  
To avoid non-identifiability when \(T\) is hidden:
- Fix the shape of \(R_{int}(T)\) (or \(\eta_{temp}(T)\)); fit only a global multiplier
- Fix \(T_{thresh}\) to a typical smartphone range (e.g., 315–323K) and fit only \(\lambda\) or only the threshold
- Use physically plausible priors/ranges and report uncertainty

### 6.2 Objective function (examples)

SOC-only fitting:

\[
\min_\theta \sum_k\big(s_{model}(t_k;\theta)-s_{obs}(t_k)\big)^2
\]

With temperature (weighted multi-objective):

\[
\min_\theta \sum_k\Big(w_s\,(\Delta s_k)^2+w_T\,(\Delta T_k)^2\Big)
\]

---

## 7. Baselines & ablations (prove innovations help)

At minimum, run:

1. **No thermal feedback baseline:** set \(\phi(T)\equiv 1\); compare SOC/temperature errors and explainability.
2. **No tail-state baseline:** set \(z(t)=x_{net}(t)\) or \(\tau_{net}\to 0\); compare network-related energy fit.

Optional third (to address “does network get throttled?”):

3. **Alternative where network is throttled too:**  
   \(P_{sys}=P_{base}+\phi(T)\big(P_{scr}+P_{cpu}+P_{net}\big)\), compare fit quality.

---

## 8. Sensitivity analysis (robustness & credibility)

Perturb key parameters (e.g., ±10% or sample plausible ranges) and report output sensitivity:

- thermal: \(C_{eq}, hA\)
- throttling: \(T_{thresh}, \lambda\)
- network: \(\tau_{net}, k_{net}\)
- battery: \(E_{nom,Wh}, R_{ref}\) (or resistance scaling)

Suggested outputs:

- SOC error boxplots / uncertainty bands
- the effect of \(\phi(t)\) activation windows on power curves

---

## 9. Result presentation tips (easy “writing points”)

- Fig. 1: \(T(t)\), \(\phi(t)\), and \(P_{sys}(t)\) together (explain the “kink” when throttling triggers)
- Fig. 2: SOC error comparison with/without thermal feedback
- Fig. 3: SOC error comparison with/without network tail-state (or show network power contribution)
- Table 1: fitted parameters + plausible ranges + physical interpretation

---

## 10. Initialization and bounds (implementation details)

- Initial values: \(s_0\) from log start SOC; \(T_0=T_{env}\) if no temperature observation; \(z_0=0\) or \(x_{net}(0)\).
- Constraints:
  - clamp \(s\in[0,100]\)
  - clamp \(z\ge 0\)
  - keep \(T\) in K; convert for reporting via \(T_{\circ C}=T_K-273.15\)

---

## 11. One-sentence claim (for reviewers)

We map log-observable workloads to power and heat using **network tail-state + internal resistance loss + thermal throttling feedback**, and close the loop with thermodynamics and energy conservation to mechanistically explain SOC nonlinearity and reduced predictability under high-temperature scenarios.
