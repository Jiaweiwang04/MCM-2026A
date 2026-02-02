# Dataset Requirements (for the SOC–Thermal–Network Model)

This document defines the log-data fields, units, and normalization rules required **before coding**, and provides **fallback (degraded) options** when some fields are missing. The goal is to ensure the model is:

- **Implementable** (can simulate \(s(t), T(t), z(t)\))
- **Reproducible** (column names, units, and sampling interval are explicit)
- **Verifiable** (supports baselines/ablations and error evaluation)

---

## 1. Basic Dataset Form

### 1.1 Data granularity

- One **continuous discharge segment / single usage session** is treated as one sample (a *session*).
- Each session is a time-ordered sequence of logs.

### 1.2 Sampling requirement

- Recommended sampling interval: \(\Delta t = 1 \sim 5\) seconds.
- If sampling is irregular: you **must** provide a timestamp column `timestamp`, and compute \(\Delta t\) from consecutive differences in code.

### 1.3 Recommended file organization

- One CSV (or Parquet) per session.
- File names should include device / scenario / date (for easy grouping and comparison).

---

## 2. Minimum Runnable Data (Required — SOC simulation)

If you meet the fields in this section, you can run the **v1 model** and fit/predict SOC (but validating thermal mechanisms will be weaker).

### 2.1 Required columns

| Column (suggested) | Meaning | Unit / Range | Used for |
|---|---|---|---|
| `timestamp` or `t` | timestamp or time | seconds (or parseable time) | compute \(\Delta t\) |
| `soc` | battery SOC | % (0–100) | fitting/validation target \(s(t)\) |
| `screen_on` | screen on/off | 0/1 | build \(u_{on}(t)\) |
| `brightness_raw` | raw screen brightness | 0–255 or 0–100 | normalize to \(b(t)\) |
| `cpu_load_raw` | CPU/GPU load (raw) | 0–100% or 0–1 | normalize to \(u_{cpu}(t)\) |
| `x_net` | network intensity | see 2.2 | construct \(x_{net}(t)\) and \(z(t)\) |
| `t_env` (can be constant) | ambient temperature | K or °C | \(T_{env}\) (if missing, can be set constant and estimated) |

### 2.2 Definition of `x_net` (choose ONE and use consistently)

Pick a definition that can be directly extracted from logs:

- **Option A (throughput):** `x_net = bytes_per_second` (or `KB/s`, `MB/s`, but be consistent everywhere)
- **Option B (packet rate):** `x_net = packets_per_second`

> Note: \(z(t)\) has the **same unit** as \(x_{net}(t)\). Therefore, the dimension of \(k_{net}\) depends on whether you choose A or B.

---

## 3. Strongly Recommended Data (for validating the thermal throttling loop)

These fields substantially improve model credibility and parameter identifiability—especially thermal parameters and the plausibility of \(\phi(T)\).

### 3.1 Temperature observation (highly recommended to have at least one)

| Column (suggested) | Meaning | Unit | Used for |
|---|---|---|---|
| `t_core` or `t_batt` | core / battery temperature | °C or K | fit thermal params \(C_{eq}, hA\); validate \(\phi(T)\) |

If the temperature is in °C, convert to K in code:

\[
T_K = T_{\circ C} + 273.15
\]

### 3.2 Thermal-limiting observables (bonus if available)

| Column (suggested) | Meaning | Unit / Values | Used for |
|---|---|---|---|
| `cpu_freq` | CPU frequency | Hz/GHz | validate frequency throttling (linked to \(\phi(T)\)) |
| `throttle_flag` | throttling indicator | 0/1 | validate periods when \(\phi(T)\) should trigger |
| `brightness_effective` | system-applied brightness | 0–1 | validate brightness limiting |

---

## 4. Advanced Optional Data (better battery physics / internal resistance calibration)

These fields upgrade internal resistance loss from “approximate” to “calibratable”, and reduce identifiability risks.

| Column (suggested) | Meaning | Unit | Used for |
|---|---|---|---|
| `voltage` | battery voltage | V | compute power/current directly |
| `current` | battery current | A | fit \(R_{int}(T)\) or validate \(P_{loss}\) |
| `battery_power` | battery power (if provided) | W | replace approximation \(I \approx P/V_{nom}\) |
| `soh` / `health` | battery state of health | 0–1 or % | construct/calibrate \(\eta_{aging}(H)\) |
| `cycle_count` | cycle count | cycles | prior information for health |

---

## 5. Normalization & Preprocessing Rules (directly usable in code)

It is recommended to generate model inputs uniformly during data loading.

### 5.1 Brightness normalization

- If `brightness_raw` is 0–255:
\[
b=\mathrm{clip}(\mathrm{brightness\_raw}/255,0,1)
\]
- If `brightness_raw` is 0–100:
\[
b=\mathrm{clip}(\mathrm{brightness\_raw}/100,0,1)
\]

### 5.2 CPU/GPU load normalization

- If `cpu_load_raw` is 0–100:
\[
u_{cpu}=\mathrm{clip}(\mathrm{cpu\_load\_raw}/100,0,1)
\]
- If already 0–1:
\[
u_{cpu}=\mathrm{clip}(\mathrm{cpu\_load\_raw},0,1)
\]

### 5.3 Time handling

- If `timestamp` exists: compute \(\Delta t_k=t_{k+1}-t_k\) (seconds) from consecutive differences.
- If no `timestamp`: you must provide a constant `dt` (seconds).

---

## 6. Degraded Options When Fields Are Missing (must be documented in paper/code)

### 6.1 Missing temperature `t_core/t_batt`

- Run \(T(t)\) as a hidden state, but you must:
  - Fix or strongly regularize \(T_{thresh}\) (typical 315–323K) to avoid non-identifiability.
  - Fix the shape of \(R_{int}(T)\) or \(\eta_{temp}(T)\), and fit only one scaling parameter.
- Output \(\phi(t)\) and \(T(t)\) as “mechanistic explanations”, and explicitly note increased uncertainty.

### 6.2 Missing ambient temperature `t_env`

- Approximate with a constant: treat \(T_{env}\) as a per-session parameter to estimate.

### 6.3 Missing network intensity (no `x_net`)

- You cannot use the network tail-state module; degrade to a baseline using only screen + CPU (or a coarse “network on/off” proxy).
- In the paper: state that tail energy is not observable from logs, so it is excluded.

### 6.4 Missing screen fields

- Absorb screen power into \(P_{base}\), and state that screen behavior cannot be modeled.

---

## 7. Minimum Data Needed for Model Validation (control experiments)

Prepare at least enough data to support the following two baselines:

1. **No thermal feedback:** set \(\phi(T)\equiv 1\) and compare SOC errors.
2. **No network tail-state:** set \(z(t)=x_{net}(t)\) or \(\tau_{net}\to 0\), and compare SOC errors.

Therefore, the minimum validation set should include:
`soc + cpu_load_raw + screen_on + brightness_raw + x_net (+ timestamp)`.

---

## 8. Suggested CSV Column Templates

**Minimal (Required):**
- `timestamp, soc, screen_on, brightness_raw, cpu_load_raw, x_net`

**Enhanced (Recommended):**
- `timestamp, soc, screen_on, brightness_raw, cpu_load_raw, x_net, t_core, t_env`

**Advanced (Optional):**
- `timestamp, soc, screen_on, brightness_raw, cpu_load_raw, x_net, t_core, t_env, voltage, current, soh`
