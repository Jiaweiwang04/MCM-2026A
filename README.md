# SOC–Thermal–Network Coupled Dynamics Model

A three-state coupled model for **MCM/ICM 2026A Task 1**: SOC – temperature – network tail state.

## Key Innovations

- **Thermal Throttling Loop:** temperature triggers system throttling (CPU frequency / screen brightness), explaining nonlinear discharge under high temperature
- **Network Tail Energy Modeling:** a state variable \(z(t)\) captures the network module’s persistent “tail” power after traffic spikes
- **Internal Resistance Loss Coupled with Temperature:** an Arrhenius relationship models how battery internal resistance changes with temperature

**v1.1 realism enhancements (4 mechanisms):**
1. **SOH affects capacity:** aged batteries have reduced usable capacity (\(E_{cap}\) scaled by \(H\))
2. **Temperature affects capacity:** usable capacity decreases significantly at low temperature (via \(\eta_{temp}(T)\))
3. **Nonlinear discharge internal resistance:** resistance rises at low SOC (\(R_{int}(T,s)\))
4. **Signal strength affects network power:** weak signal increases network power (\(P_{net} \propto 1/S_{signal}\))

## Project Structure

```
2026A/
├── config/
│   └── default_params.yaml    # Model parameter configuration
├── src/
│   ├── model.py               # Core model (SOCThermalNetworkModel)
│   ├── data_loader.py         # Data loading & preprocessing
│   ├── visualizer.py          # Visualization tools (decoupled from computation)
│   └── utils.py               # Utility functions
├── generate_data.py           # Scenario data generator
├── main.py                    # Main runner script + examples
├── requirements.txt           # Python dependencies
├── model.md                   # Mathematical model document
└── data_requirement.md        # Dataset requirements
```

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Generate scenario data (optional)

If you do not have real logs, you can generate synthetic data for five typical scenarios:

```bash
python generate_data.py
```

This will create the following scenario files under `data/generated/`:
- `daily_use.csv`: daily mixed usage (light activities such as browsing, social, photo)
- `commute.csv`: commuting (music + navigation + intermittent browsing)
- `work.csv`: working (video meetings, document handling)
- `gaming.csv`: high-intensity gaming (sustained heavy load)
- `sleep.csv`: sleep/idle (mostly standby)

### 3) Run the demo

```bash
python main.py
```

Outputs and figures will be generated under `results/`.

## Usage

### Basic simulation (given an input sequence)

```python
from src.model import SOCThermalNetworkModel

# Initialize
model = SOCThermalNetworkModel(config_path='config/default_params.yaml')

# Prepare inputs
input_sequence = {
    't': [...],        # time (s)
    'u_on': [...],     # screen on/off 0/1
    'b': [...],        # brightness 0-1
    'u_cpu': [...],    # CPU load 0-1
    'x_net': [...],    # network intensity
    'S_signal': [...], # signal strength (0-1], optional; default = 1.0
    'T_env': [...]     # ambient temperature (K)
}

# Initial state
initial_state = {'s': 100.0, 'T': 298.15, 'z': 0.0}

# Simulate
trajectory = model.simulate(initial_state, input_sequence)

# Visualize
from src.visualizer import Visualizer
vis = Visualizer()
vis.plot_full_dashboard(trajectory)
```

### Time-to-Empty prediction (scenario application)

```python
from src.data_loader import generate_scenario_function

# Generate a scenario function
scenario_func = generate_scenario_function('gaming')  # or 'video', 'idle', 'browsing'

# Predict time to empty
tte, trajectory = model.predict_time_to_empty(
    initial_state={'s': 100.0, 'T': 298.15, 'z': 0.0},
    scenario_inputs=scenario_func,
    dt=5.0,
    soc_threshold=5.0
)

print(f"Time to Empty: {tte/60:.1f} minutes")
```

### Using real log data

```python
from src.data_loader import DataLoader

# Load data
loader = DataLoader(brightness_max=255, cpu_load_max=100)
df = loader.load_csv('your_log.csv')

# Prepare model inputs
inputs = loader.prepare_model_inputs(df, has_temperature=False)
initial_state = loader.get_initial_state(inputs)

# Simulate
trajectory = model.simulate(initial_state, inputs)
```

### Using generated scenario data

```python
from src.data_loader import DataLoader

loader = DataLoader(brightness_max=255, cpu_load_max=100)
df = loader.load_csv('data/generated/gaming.csv')

# Note: generated SOC is a placeholder and will be recomputed by the model
inputs = loader.prepare_model_inputs(df, has_temperature=False)
initial_state = {'s': 100.0, 'T': 298.15, 'z': 0.0}  # start from full charge

trajectory = model.simulate(initial_state, inputs)

from src.visualizer import Visualizer
vis = Visualizer()
vis.plot_full_dashboard(trajectory, save_path='results/gaming_scenario.png')
```

## Parameter Tuning

Edit `config/default_params.yaml` to adjust model parameters, including:

- **Thermal model:** `C_eq` (thermal capacity), `hA` (heat transfer), `T_thresh` (throttling threshold)
- **Power model:** `P_base`, screen/CPU/network coefficients
- **Battery model:** `E_nom_Wh` (capacity), `R_ref` (resistance), `H` (SOH)
- **Realism mechanisms:**
  - `c_soc, s_ref, epsilon`: SOC-nonlinear resistance
  - `capacity_temperature`: temperature–capacity curve parameters
  - `S_signal_default`: default signal strength
- **Baseline switches:** `no_thermal_throttling`, `no_network_tail` (for ablations)

## Baseline / Ablation Experiments

In the config file:

```yaml
baseline:
  no_thermal_throttling: true   # disable thermal throttling feedback
  no_network_tail: true         # disable network tail-state
```

You can also modify parameters dynamically in code.

## Output Fields

The model output `trajectory` dictionary includes:

- `t`: time array (s)
- `s`: SOC (%)
- `T`: temperature (K)
- `z`: network tail state
- `phi`: throttling factor \(\phi(T)\)
- `P_sys`: system power (W)
- `P_loss`: internal resistance loss (W)
- `P_heat`: total heat generation (W)

All arrays have equal length and can be used directly for downstream analysis and visualization.

## Visualization API

The `Visualizer` class provides:

- `plot_soc_comparison()`: predicted vs observed SOC
- `plot_thermal_mechanism()`: temperature–\(\phi\)–power triple plot
- `plot_baseline_comparison()`: compare multiple model variants
- `plot_time_to_empty_scenarios()`: compare TTE across scenarios
- `plot_full_dashboard()`: full dashboard of key variables

## Dataset Requirements

See `data_requirement.md`. Minimum required columns:

- `timestamp`, `soc`, `screen_on`, `brightness_raw`, `cpu_load_raw`, `x_net`

Recommended additions:

- `t_core` or `t_batt` (temperature observation; greatly improves credibility)
- `voltage`, `current` (advanced: calibrate resistance more accurately)

## Unit System (Important)

The model uses **SI units internally**:

- time: seconds (s)
- temperature: Kelvin (K)
- power: watts (W)
- energy: joules (J)

Log inputs are normalized/converted automatically, but please ensure parameter units in the config file are consistent.


**Contact:** for questions or improvements, please contact the teammate(s) or read the code comments.

### Thanks for my teammates https://github.com/Patrick1159/MCM2026_A