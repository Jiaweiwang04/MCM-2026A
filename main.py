"""
主运行脚本：模型训练/验证/应用示例
"""

import sys
import os

# 添加 src 到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import SOCThermalNetworkModel
from data_loader import DataLoader
from visualizer import Visualizer
import numpy as np
from argparse import ArgumentParser
from typing import Optional, Tuple, Dict, List
import pandas as pd


def example_simulate_with_synthetic_data():
    """
    示例1: 用合成数据验证模型可运行性
    """
    print("=" * 60)
    print("Example 1: Simulate with Synthetic Data")
    print("=" * 60)
    
    # 初始化模型
    config_path = 'config/default_params.yaml'
    model = SOCThermalNetworkModel(config_path=config_path)
    
    # 生成合成输入（模拟一段视频播放场景）
    t_max = 3600.0  # 1 小时
    dt = 1.0
    t = np.arange(0, t_max, dt)
    N = len(t)
    
    # 合成输入：前半段视频，后半段游戏
    input_sequence = {
        't': t,
        'u_on': np.ones(N),
        'b': np.where(t < t_max/2, 0.5, 0.8),  # 亮度切换
        'u_cpu': np.where(t < t_max/2, 0.3, 0.7),  # 负载切换
        'x_net': 3000.0 + 1000.0 * np.sin(2 * np.pi * t / 300.0),  # 周期网络
        'T_env': np.full(N, 298.15)
    }
    
    # 初始状态
    initial_state = {
        's': 100.0,
        'T': 298.15,
        'z': 0.0
    }
    
    # 仿真
    print("Running simulation...")
    trajectory = model.simulate(initial_state, input_sequence, dt=dt)
    
    # 可视化
    print("Generating visualizations...")
    vis = Visualizer()
    vis.plot_full_dashboard(trajectory, save_path='results/example1_dashboard.png')
    vis.plot_thermal_mechanism(trajectory, save_path='results/example1_thermal.png')
    
    print(f"Final SOC: {trajectory['s'][-1]:.2f}%")
    print(f"Max Temperature: {trajectory['T'].max() - 273.15:.2f}°C")
    print("✓ Example 1 completed.\n")


def example_baseline_comparison():
    """
    示例2: Baseline 对照实验（消融研究）
    """
    print("=" * 60)
    print("Example 2: Baseline Comparison (Ablation Study)")
    print("=" * 60)
    
    config_path = 'config/default_params.yaml'
    
    # 合成输入（模拟高负载场景，容易触发温控）
    t_max = 3600.0
    dt = 1.0
    t = np.arange(0, t_max, dt)
    N = len(t)
    
    input_sequence = {
        't': t,
        'u_on': np.ones(N),
        'b': np.full(N, 0.8),
        'u_cpu': np.full(N, 0.7),
        'x_net': np.full(N, 5000.0),
        'T_env': np.full(N, 298.15)
    }
    
    initial_state = {'s': 100.0, 'T': 298.15, 'z': 0.0}
    
    trajectories = {}
    
    # 完整模型
    print("1/3: Running full model...")
    model_full = SOCThermalNetworkModel(config_path=config_path)
    trajectories['Full Model'] = model_full.simulate(initial_state, input_sequence, dt=dt)
    
    # Baseline 1: 无温控反馈
    print("2/3: Running baseline (no thermal throttling)...")
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        params_no_throttle = yaml.safe_load(f)
    params_no_throttle['baseline']['no_thermal_throttling'] = True
    model_no_throttle = SOCThermalNetworkModel(params=params_no_throttle)
    trajectories['No Throttling'] = model_no_throttle.simulate(initial_state, input_sequence, dt=dt)
    
    # Baseline 2: 无网络尾态
    print("3/3: Running baseline (no network tail)...")
    with open(config_path, 'r', encoding='utf-8') as f:
        params_no_tail = yaml.safe_load(f)
    params_no_tail['baseline']['no_network_tail'] = True
    model_no_tail = SOCThermalNetworkModel(params=params_no_tail)
    trajectories['No Network Tail'] = model_no_tail.simulate(initial_state, input_sequence, dt=dt)
    
    # 可视化对比
    print("Generating comparison plots...")
    vis = Visualizer()
    vis.plot_baseline_comparison(trajectories, 
                                 title='Ablation Study: Baseline Comparison',
                                 save_path='results/example2_ablation.png')
    
    # 打印差异
    print("\nSOC at end (%):")
    for name, traj in trajectories.items():
        print(f"  {name:25s}: {traj['s'][-1]:.2f}%")
    
    print("✓ Example 2 completed.\n")


def example_time_to_empty_scenarios(scenario_paths: dict):
    """
    示例3: 多场景 Time-to-Empty 预测
    """
    print("=" * 60)
    print("Example 3: Time-to-Empty Prediction (Multiple Scenarios)")
    print("=" * 60)
    
    config_path = 'config/default_params.yaml'
    model = SOCThermalNetworkModel(config_path=config_path)
    
    # 测试多个场景
    scenario_names = ['music', 'video', 'localvideo', 'gaming', 'facebook']
    results = {}
    loader = DataLoader(brightness_max=255.0, cpu_load_max=100.0)
    
    for scenario_name in scenario_names:
        print(f"Predicting TTE for scenario: {scenario_name}...")
        csv_path = scenario_paths.get(scenario_name)
        if not csv_path:
            raise ValueError(f"Missing CSV path for scenario: {scenario_name}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = loader.load_csv(csv_path)
        inputs = loader.prepare_model_inputs(df, has_temperature=False, has_t_env=True)
        initial_state = loader.get_initial_state(inputs, use_observed_T=False)
        initial_state['s'] = 100.0
        scenario_func = loader.make_scenario_function(inputs)
        
        if len(inputs['t']) > 1:
            dt = float(np.median(np.diff(inputs['t'])))
            if dt <= 0:
                dt = 1.0
        else:
            dt = 1.0
        
        tte, trajectory = model.predict_time_to_empty(
            initial_state=initial_state,
            scenario_inputs=scenario_func,
            dt=dt,
            max_time=72000.0,  # 最大 20 小时
            soc_threshold=5.0  # 5% 视为空电
        )
        
        results[scenario_name] = (tte, trajectory)
        print(f"  → TTE: {tte/60:.1f} minutes ({tte/3600:.2f} hours)")
    
    # 可视化
    print("Generating scenario comparison plot...")
    vis = Visualizer()
    vis.plot_time_to_empty_scenarios(results, 
                                     save_path='results/example3_tte_scenarios.png')
    vis.plot_time_to_empty_bar(results,
                               save_path='results/example3_tte_bar.png')
    
    print("✓ Example 3 completed.\n")


def example_with_real_data(csv_path: str):
    """
    示例4: 使用真实日志数据（需提供 CSV 文件）
    
    Args:
        csv_path: CSV 文件路径
    """
    print("=" * 60)
    print("Example 4: Simulate with Real Log Data")
    print("=" * 60)
    
    if not os.path.exists(csv_path):
        print(f"⚠ CSV file not found: {csv_path}")
        print("  Please provide a CSV file with required columns.")
        print("  See data_requirement.md for details.\n")
        return
    
    # 数据加载
    print(f"Loading data from {csv_path}...")
    loader = DataLoader(brightness_max=255.0, cpu_load_max=100.0)
    
    # 列名映射（根据实际 CSV 调整）
    column_mapping = {
        # 示例映射，需根据实际日志调整
        # '原始列名': '标准列名'
        # 'time': 'timestamp',
        # 'battery_level': 'soc',
        # 'screen_state': 'screen_on',
        # ...
    }
    
    df = loader.load_csv(csv_path, column_mapping=column_mapping)
    
    # 兼容旧版合成输出（t/u_on/b/u_cpu/x_net/T_env）
    required = {'timestamp', 'soc', 'screen_on', 'brightness_raw', 'cpu_load_raw', 'x_net'}
    alt_cols = {'t', 'u_on', 'b', 'u_cpu', 'x_net', 'T_env'}
    if not required.issubset(df.columns) and alt_cols.issubset(df.columns):
        synth_inputs = {
            't': df['t'].values,
            'u_on': df['u_on'].values,
            'b': df['b'].values,
            'u_cpu': df['u_cpu'].values,
            'x_net': df['x_net'].values,
            'T_env': df['T_env'].values
        }
        init_state = {'s': 100.0, 'T': float(synth_inputs['T_env'][0]), 'z': 0.0}
        model_tmp = SOCThermalNetworkModel(config_path='config/default_params.yaml')
        traj_tmp = model_tmp.simulate(initial_state=init_state, input_sequence=synth_inputs)
        
        df = pd.DataFrame({
            'timestamp': synth_inputs['t'],
            'soc': traj_tmp['s'],
            'screen_on': synth_inputs['u_on'],
            'brightness_raw': np.clip(synth_inputs['b'] * 255.0, 0.0, 255.0),
            'cpu_load_raw': np.clip(synth_inputs['u_cpu'] * 100.0, 0.0, 100.0),
            'x_net': synth_inputs['x_net'],
            't_env': synth_inputs['T_env']
        })
    
    # 准备模型输入
    inputs = loader.prepare_model_inputs(df, has_temperature=False)
    initial_state = loader.get_initial_state(inputs)
    
    # 初始化模型并仿真
    print("Running simulation...")
    config_path = 'config/default_params.yaml'
    model = SOCThermalNetworkModel(config_path=config_path)
    trajectory = model.simulate(initial_state, inputs)
    
    # 可视化
    print("Generating visualizations...")
    vis = Visualizer()
    vis.plot_full_dashboard(trajectory, soc_obs=inputs['soc_obs'],
                           save_path=f'results/{os.path.basename(csv_path).replace(".csv", "")}_real_data.png')
    
    # 评估
    soc_pred = trajectory['s']
    soc_obs = inputs['soc_obs']
    mae = np.mean(np.abs(soc_pred - soc_obs))
    rmse = np.sqrt(np.mean((soc_pred - soc_obs)**2))
    
    print(f"MAE:  {mae:.2f}%")
    print(f"RMSE: {rmse:.2f}%")
    print("✓ Example 4 completed.\n")


def example_soc_comparison_music_video(music_csv: str, video_csv: str):
    """
    示例5: music/video 场景真值 SOC vs 预测 SOC 对比
    """
    print("=" * 60)
    print("Example 5: SOC Comparison (Music & Video)")
    print("=" * 60)
    
    for path in [music_csv, video_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
    
    config_path = 'config/default_params.yaml'
    model = SOCThermalNetworkModel(config_path=config_path)
    loader = DataLoader(brightness_max=255.0, cpu_load_max=100.0)
    
    def _run(csv_path: str):
        df = loader.load_csv(csv_path)
        inputs = loader.prepare_model_inputs(df, has_temperature=False, has_t_env=True)
        initial_state = loader.get_initial_state(inputs, use_observed_T=False)
        trajectory = model.simulate(initial_state=initial_state, input_sequence=inputs)
        return trajectory, inputs['soc_obs']
    
    music_traj, music_obs = _run(music_csv)
    video_traj, video_obs = _run(video_csv)
    
    vis = Visualizer()
    vis.plot_soc_comparison_two(music_traj, music_obs, video_traj, video_obs,
                                save_path='results/example5_soc_music_video.png')
    
    print("✓ Example 5 completed.\n")


def example_tte_bars_music_gaming(music_csv: str, gaming_csv: str,
                                  soc_high: float = 100.0, soc_low: float = 20.0):
    """
    示例6: 游戏/音乐 + 初始SOC高/低 四场景 TTE 柱状图
    """
    print("=" * 60)
    print("Example 6: TTE Bars (Music/Gaming, High/Low SOC)")
    print("=" * 60)
    
    for path in [music_csv, gaming_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
    
    config_path = 'config/default_params.yaml'
    model = SOCThermalNetworkModel(config_path=config_path)
    loader = DataLoader(brightness_max=255.0, cpu_load_max=100.0)
    
    def _prepare(csv_path: str):
        df = loader.load_csv(csv_path)
        inputs = loader.prepare_model_inputs(df, has_temperature=False, has_t_env=True)
        scenario_func = loader.make_scenario_function(inputs)
        if len(inputs['t']) > 1:
            dt = float(np.median(np.diff(inputs['t'])))
            if dt <= 0:
                dt = 1.0
        else:
            dt = 1.0
        return inputs, scenario_func, dt
    
    music_inputs, music_func, music_dt = _prepare(music_csv)
    gaming_inputs, gaming_func, gaming_dt = _prepare(gaming_csv)
    
    def _run(label: str, inputs, scenario_func, dt, soc_init: float):
        initial_state = loader.get_initial_state(inputs, use_observed_T=False)
        initial_state['s'] = soc_init
        tte, trajectory = model.predict_time_to_empty(
            initial_state=initial_state,
            scenario_inputs=scenario_func,
            dt=dt,
            max_time=72000.0,
            soc_threshold=5.0
        )
        return (tte, trajectory)
    
    results = {
        f"music_high({soc_high:g}%)": _run("music_high", music_inputs, music_func, music_dt, soc_high),
        f"music_low({soc_low:g}%)": _run("music_low", music_inputs, music_func, music_dt, soc_low),
        f"gaming_high({soc_high:g}%)": _run("gaming_high", gaming_inputs, gaming_func, gaming_dt, soc_high),
        f"gaming_low({soc_low:g}%)": _run("gaming_low", gaming_inputs, gaming_func, gaming_dt, soc_low)
    }
    
    vis = Visualizer()
    vis.plot_time_to_empty_bar(results,
                               title="TTE: Music/Gaming with High/Low Initial SOC",
                               save_path='results/example6_tte_bars_music_gaming.png')
    
    print("✓ Example 6 completed.\n")


def example_soc_soh_facebook(facebook_csv: str):
    """
    示例7: 不同 SOH 下 facebook 场景 SOC(t) 对比
    """
    print("=" * 60)
    print("Example 7: SOC vs Time under Different SOH (Facebook)")
    print("=" * 60)
    
    if not os.path.exists(facebook_csv):
        raise FileNotFoundError(f"CSV file not found: {facebook_csv}")
    
    config_path = 'config/default_params.yaml'
    base_model = SOCThermalNetworkModel(config_path=config_path)
    loader = DataLoader(brightness_max=255.0, cpu_load_max=100.0)
    
    df = loader.load_csv(facebook_csv)
    inputs = loader.prepare_model_inputs(df, has_temperature=False, has_t_env=True)
    initial_state = loader.get_initial_state(inputs, use_observed_T=False)
    
    sohs = [1.0, 0.8, 0.6, 0.4]
    trajectories = {}
    for soh in sohs:
        model = SOCThermalNetworkModel(config_path=config_path)
        model.params['battery']['H'] = float(soh)
        traj = model.simulate(initial_state=initial_state, input_sequence=inputs)
        trajectories[f"SOH={soh:g}"] = traj
    
    vis = Visualizer()
    vis.plot_soc_soh_comparison(trajectories,
                                title="Facebook Scenario: SOC vs Time for Different SOH",
                                save_path='results/example7_soc_soh_facebook.png')
    
    print("✓ Example 7 completed.\n")


def example_sensitivity_soc_rate(csv_path: str,
                                 t_thresh_vals: np.ndarray,
                                 lambda_vals: np.ndarray):
    """
    示例8: SOC 下降速率敏感性分析（T_thresh x lambda）
    """
    print("=" * 60)
    print("Example 8: Sensitivity (SOC decline rate)")
    print("=" * 60)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    loader = DataLoader(brightness_max=255.0, cpu_load_max=100.0)
    df = loader.load_csv(csv_path)
    inputs = loader.prepare_model_inputs(df, has_temperature=False, has_t_env=True)
    initial_state = loader.get_initial_state(inputs, use_observed_T=False)
    
    duration = float(inputs['t'][-1] - inputs['t'][0]) if len(inputs['t']) > 1 else 1.0
    if duration <= 0:
        duration = 1.0
    
    z = np.zeros((len(lambda_vals), len(t_thresh_vals)))
    
    for i, lam in enumerate(lambda_vals):
        for j, t_th in enumerate(t_thresh_vals):
            model = SOCThermalNetworkModel(config_path='config/default_params.yaml')
            model.params['thermal']['T_thresh'] = float(t_th)
            model.params['thermal']['lambda'] = float(lam)
            traj = model.simulate(initial_state=initial_state, input_sequence=inputs)
            s0 = float(traj['s'][0])
            s_end = float(traj['s'][-1])
            rate_per_hour = (s0 - s_end) / duration * 3600.0
            z[i, j] = rate_per_hour
    
    vis = Visualizer()
    vis.plot_sensitivity_heatmap(
        x_vals=t_thresh_vals,
        y_vals=lambda_vals,
        z_vals=z,
        xlabel="T_thresh (K)",
        ylabel="lambda (1/K)",
        title="Sensitivity: SOC Decline Rate",
        cbar_label="SOC decline rate (%/hour)",
        save_path='results/example8_sensitivity_soc_rate.png'
    )
    
    print("✓ Example 8 completed.\n")


def example_noise_sensitivity_sherlock(csv_path: str,
                                       strengths: Optional[np.ndarray] = None,
                                       n_runs: int = 8,
                                       seed: int = 42):
    """
    示例9: Sherlock 数据集噪声敏感性分析（SOC MSE）
    """
    print("=" * 60)
    print("Example 9: Noise Sensitivity (SOC MSE) - Sherlock")
    print("=" * 60)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if strengths is None:
        strengths = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    loader = DataLoader(brightness_max=255.0, cpu_load_max=100.0)
    df = loader.load_csv(csv_path)
    inputs = loader.prepare_model_inputs(df, has_temperature=False, has_t_env=True)
    initial_state = loader.get_initial_state(inputs, use_observed_T=False)
    soc_obs = inputs['soc_obs']
    
    base_model = SOCThermalNetworkModel(config_path='config/default_params.yaml')
    base_traj = base_model.simulate(initial_state=initial_state, input_sequence=inputs)
    base_mse = float(np.mean((base_traj['s'] - soc_obs) ** 2))
    
    # 可加噪特征（排除 u_on, T_env）
    candidate_features = ['b', 'u_cpu', 'x_net']
    features = [f for f in candidate_features if f in inputs]
    
    def _apply_noise(x: np.ndarray, strength: float, rng: np.random.Generator, clip: Optional[Tuple[Optional[float], Optional[float]]] = None):
        std = float(np.std(x))
        if std == 0.0:
            return x.copy()
        noisy = x + rng.normal(0.0, strength * std, size=x.shape)
        if clip is not None:
            low, high = clip
            if low is not None:
                noisy = np.maximum(noisy, low)
            if high is not None:
                noisy = np.minimum(noisy, high)
        # 防止数值异常传播
        noisy = np.where(np.isfinite(noisy), noisy, x)
        return noisy
    
    def _simulate_with_inputs(mod_inputs: Dict[str, np.ndarray]) -> float:
        model = SOCThermalNetworkModel(config_path='config/default_params.yaml')
        traj = model.simulate(initial_state=initial_state, input_sequence=mod_inputs)
        mask = np.isfinite(traj['s']) & np.isfinite(soc_obs)
        if not np.any(mask):
            return base_mse
        mse_val = float(np.mean((traj['s'][mask] - soc_obs[mask]) ** 2))
        if not np.isfinite(mse_val):
            return base_mse
        return mse_val
    
    mse_by_feature = {}
    delta_mse = {}
    rng = np.random.default_rng(seed)
    
    for feat in features:
        mse_vals = []
        for strength in strengths:
            runs = []
            for _ in range(n_runs):
                mod_inputs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in inputs.items()}
                if feat in ['b', 'u_cpu']:
                    mod_inputs[feat] = _apply_noise(mod_inputs[feat], strength, rng, clip=(0.0, 1.0))
                elif feat == 'x_net':
                    mod_inputs[feat] = _apply_noise(mod_inputs[feat], strength, rng, clip=(0.0, None))
                else:
                    mod_inputs[feat] = _apply_noise(mod_inputs[feat], strength, rng, clip=None)
                runs.append(_simulate_with_inputs(mod_inputs))
            m = float(np.mean(runs))
            if not np.isfinite(m):
                m = base_mse
            mse_vals.append(m)
        mse_vals = np.array(mse_vals)
        mse_by_feature[feat] = mse_vals - base_mse
        delta_mse[feat] = float(mse_vals[-1] - base_mse)
    
    # 全特征同时加噪（排除 u_on）
    mse_mean = []
    mse_std = []
    for strength in strengths:
        runs = []
        for _ in range(n_runs):
            mod_inputs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in inputs.items()}
            if 'b' in mod_inputs:
                mod_inputs['b'] = _apply_noise(mod_inputs['b'], strength, rng, clip=(0.0, 1.0))
            if 'u_cpu' in mod_inputs:
                mod_inputs['u_cpu'] = _apply_noise(mod_inputs['u_cpu'], strength, rng, clip=(0.0, 1.0))
            if 'x_net' in mod_inputs:
                mod_inputs['x_net'] = _apply_noise(mod_inputs['x_net'], strength, rng, clip=(0.0, None))
            runs.append(_simulate_with_inputs(mod_inputs))
        mse_mean.append(float(np.mean(runs)))
        mse_std.append(float(np.std(runs)))
    
    vis = Visualizer()
    vis.plot_noise_sensitivity_curves(
        strengths, mse_by_feature,
        title="Single-Feature Noise Sensitivity (SOC ΔMSE)",
        save_path='results/example9_noise_single_feature.png'
    )
    vis.plot_noise_sensitivity_bar(
        delta_mse,
        title="MSE Increase under Strong Noise (Single Feature)",
        save_path='results/example9_noise_feature_rank.png'
    )
    vis.plot_noise_stress_curve(
        strengths, np.array(mse_mean), np.array(mse_std),
        title="All-Feature Noise Stress Test (SOC MSE)",
        save_path='results/example9_noise_stress.png'
    )
    
    print("✓ Example 9 completed.\n")


def example_markov_synthetic_sequence(seed: int = 42,
                                      min_minutes: int = 10,
                                      max_minutes: int = 40):
    """
    示例10: 马尔可夫场景切换 + 合成序列 + 元数据输出
    """
    print("=" * 60)
    print("Example 10: Markov Synthetic Sequence")
    print("=" * 60)
    
    rng = np.random.default_rng(seed)
    
    scenario_paths = {
        'music': 'data/real/music.csv',
        'video': 'data/real/video.csv',
        'gaming': 'data/real/gaming.csv'
    }
    
    loader = DataLoader(brightness_max=255.0, cpu_load_max=100.0)
    
    # 读取并准备各场景输入
    scenario_inputs = {}
    dts = []
    for name, path in scenario_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        df = loader.load_csv(path)
        inputs = loader.prepare_model_inputs(df, has_temperature=False, has_t_env=True)
        scenario_inputs[name] = inputs
        if len(inputs['t']) > 1:
            dts.append(float(np.median(np.diff(inputs['t']))))
    
    dt = float(np.median(dts)) if dts else 1.0
    if dt <= 0:
        dt = 1.0
    
    def _resample(inputs: Dict[str, np.ndarray], dt: float) -> Dict[str, np.ndarray]:
        t = inputs['t']
        t_new = np.arange(0.0, t[-1] + 1e-9, dt)
        out = {'t': t_new}
        for k in ['u_on', 'b', 'u_cpu', 'x_net', 'T_env']:
            if k in inputs:
                out[k] = np.interp(t_new, t, inputs[k])
        return out
    
    scenario_resampled = {k: _resample(v, dt) for k, v in scenario_inputs.items()}
    
    # 马尔可夫转移矩阵（基于示例倾向）
    states = ['music', 'video', 'gaming']
    trans = {
        'music': {'music': 0.2, 'video': 0.6, 'gaming': 0.2},
        'video': {'music': 0.4, 'video': 0.2, 'gaming': 0.4},
        'gaming': {'music': 0.5, 'video': 0.3, 'gaming': 0.2}
    }
    
    def _next_state(curr: str) -> str:
        probs = [trans[curr][s] for s in states]
        return rng.choice(states, p=probs)
    
    # 对数正态持续时间（分钟）
    def _sample_duration_minutes() -> float:
        mean = (min_minutes + max_minutes) / 2.0
        sigma = 0.5
        mu = np.log(mean) - 0.5 * sigma * sigma
        dur = rng.lognormal(mean=mu, sigma=sigma)
        return float(np.clip(dur, min_minutes, max_minutes))
    
    def _crossfade(a: np.ndarray, b: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if n <= 0:
            return a, b
        n = min(n, len(a), len(b))
        w = np.linspace(0.0, 1.0, n, endpoint=False)
        a_tail = a[-n:] * (1.0 - w) + b[:n] * w
        b_head = b[:n] * 0.0  # 将被移除
        a_new = np.concatenate([a[:-n], a_tail])
        b_new = b[n:]  # 去掉头部
        return a_new, b_new
    
    # 生成合成序列（直到 SOC<=0 后截断）
    max_total_hours = 48
    total_time_target = 6 * 3600.0
    
    out = {'t': [], 'u_on': [], 'b': [], 'u_cpu': [], 'x_net': [], 'T_env': []}
    metadata: List[Dict] = []
    
    current_state = rng.choice(states)
    total_time = 0.0
    seg_idx = 0
    
    while total_time < max_total_hours * 3600.0:
        inputs = scenario_resampled[current_state]
        n_total = len(inputs['t'])
        dur_min = _sample_duration_minutes()
        dur_sec = dur_min * 60.0
        n_seg = max(1, int(dur_sec / dt))
        if n_seg >= n_total:
            n_seg = n_total
        start_idx = int(rng.integers(0, max(1, n_total - n_seg)))
        end_idx = start_idx + n_seg
        
        seg = {k: v[start_idx:end_idx].copy() for k, v in inputs.items() if k != 't'}
        
        # 过渡处理
        fade_sec = float(rng.uniform(1.0, 3.0))
        n_fade = int(fade_sec / dt)
        if len(out['t']) > 0 and n_fade > 0:
            for k in ['u_on', 'b', 'u_cpu', 'x_net', 'T_env']:
                a = np.array(out[k])
                b = seg[k]
                a_new, b_new = _crossfade(a, b, n_fade)
                out[k] = list(a_new)
                seg[k] = b_new
            # 对应时间也去掉 n_fade
            out['t'] = out['t'][:-n_fade]
        
        seg_len = len(seg['u_on'])
        t_start = total_time
        t_seg = (np.arange(seg_len) * dt + total_time).tolist()
        out['t'].extend(t_seg)
        for k in ['u_on', 'b', 'u_cpu', 'x_net', 'T_env']:
            out[k].extend(seg[k].tolist())
        
        total_time += seg_len * dt
        
        metadata.append({
            'segment_id': seg_idx,
            'scenario': current_state,
            'source_csv': scenario_paths[current_state],
            'start_time_s': t_start,
            'end_time_s': total_time,
            'duration_s': seg_len * dt,
            'source_start_idx': start_idx,
            'source_end_idx': end_idx,
            'fade_sec': fade_sec,
            'seed': seed
        })
        seg_idx += 1
        
        if total_time >= total_time_target:
            # 先模拟一次，看是否已耗尽 SOC
            synth_inputs = {k: np.array(v) for k, v in out.items()}
            initial_state = {'s': 100.0, 'T': synth_inputs['T_env'][0], 'z': 0.0}
            model = SOCThermalNetworkModel(config_path='config/default_params.yaml')
            traj = model.simulate(initial_state=initial_state, input_sequence=synth_inputs)
            s = traj['s']
            idx_zero = np.where(s <= 0.0)[0]
            if idx_zero.size > 0:
                cut_idx = int(idx_zero[0])
                for k in out:
                    out[k] = out[k][:cut_idx + 1]
                total_time = out['t'][-1] if out['t'] else 0.0
                # 修正最后一个 segment 的 end_time
                if metadata:
                    metadata[-1]['end_time_s'] = total_time
                    metadata[-1]['duration_s'] = total_time - metadata[-1]['start_time_s']
                break
            total_time_target += 3 * 3600.0
        
        current_state = _next_state(current_state)
    
    # 生成 SOC 并导出为 example_with_real_data 兼容列
    synth_inputs = {k: np.array(v) for k, v in out.items()}
    initial_state = {'s': 100.0, 'T': synth_inputs['T_env'][0], 'z': 0.0}
    model = SOCThermalNetworkModel(config_path='config/default_params.yaml')
    traj = model.simulate(initial_state=initial_state, input_sequence=synth_inputs)
    
    export_df = pd.DataFrame({
        'timestamp': synth_inputs['t'],
        'soc': traj['s'],
        'screen_on': synth_inputs['u_on'],
        'brightness_raw': np.clip(synth_inputs['b'] * 255.0, 0.0, 255.0),
        'cpu_load_raw': np.clip(synth_inputs['u_cpu'] * 100.0, 0.0, 100.0),
        'x_net': synth_inputs['x_net'],
        't_env': synth_inputs['T_env']
    })
    
    # 输出 CSV 与元数据
    os.makedirs('results', exist_ok=True)
    export_df.to_csv('results/synthetic_sequence.csv', index=False)
    pd.DataFrame(metadata).to_csv('results/synthetic_metadata.csv', index=False)
    
    print("✓ Example 10 completed.")
    print("  - results/synthetic_sequence.csv")
    print("  - results/synthetic_metadata.csv\n")


def example_power_contrib_radar_sherlock(csv_path: str = 'data/real/video.csv'):
    """
    示例11: P_scr/P_net/P_loss/P_cpu 雷达图
    """
    print("=" * 60)
    print("Example 11: Power Contribution Radar (Power Components)")
    print("=" * 60)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    loader = DataLoader(brightness_max=255.0, cpu_load_max=100.0)
    df = loader.load_csv(csv_path)
    inputs = loader.prepare_model_inputs(df, has_temperature=False, has_t_env=True)
    initial_state = loader.get_initial_state(inputs, use_observed_T=False)
    
    model = SOCThermalNetworkModel(config_path='config/default_params.yaml')
    traj = model.simulate(initial_state=initial_state, input_sequence=inputs)
    
    # 计算功耗分量（按时间平均）
    u_on = inputs['u_on']
    b = inputs['b']
    u_cpu = inputs['u_cpu']
    z = traj['z']
    T = traj['T']
    s = traj['s']
    
    P_scr = np.array([model.power_screen(u_on[i], b[i]) for i in range(len(u_on))])
    P_cpu = np.array([model.power_cpu(u_cpu[i]) for i in range(len(u_cpu))])
    P_net = np.array([model.power_network(z[i], 1.0, model.params['power']['network']['default_type'])
                      for i in range(len(z))])
    P_loss = np.array([model.power_loss(P_scr[i] + P_cpu[i] + P_net[i], T[i], s[i])
                       for i in range(len(s))])
    
    labels = ['P_scr', 'P_net', 'P_loss', 'P_cpu']
    values = [float(np.mean(P_scr)), float(np.mean(P_net)), float(np.mean(P_loss)), float(np.mean(P_cpu))]
    print(f"Using CSV: {csv_path}")
    print(f"Mean powers (W): {dict(zip(labels, values))}")
    
    vis = Visualizer()
    base = os.path.splitext(os.path.basename(csv_path))[0]
    vis.plot_power_contrib_pie(labels, values,
                               title=f"Power Contribution Pie ({base})",
                               save_path=f'results/example11_power_pie_{base}.png')
    
    print("✓ Example 11 completed.\n")


if __name__ == '__main__':
    parser = ArgumentParser(description="SOC-Thermal-Network Model Demo")
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to real log CSV file for Example 4')
    parser.add_argument('--music', type=str, default=None,
                        help='Path to music scenario CSV for Example 3')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video scenario CSV for Example 3')
    parser.add_argument('--localvideo', type=str, default=None,
                        help='Path to localvideo scenario CSV for Example 3')
    parser.add_argument('--gaming', type=str, default=None,
                        help='Path to gaming scenario CSV for Example 3')
    parser.add_argument('--facebook', type=str, default=None,
                        help='Path to facebook scenario CSV for Example 3')
    parser.add_argument('--soc-high', type=float, default=100.0,
                        help='High initial SOC for Example 6')
    parser.add_argument('--soc-low', type=float, default=20.0,
                        help='Low initial SOC for Example 6')
    parser.add_argument('--sens-csv', type=str, default=None,
                        help='CSV path for sensitivity analysis (Example 8)')
    parser.add_argument('--tthresh-min', type=float, default=295.0,
                        help='Min T_thresh (K) for sensitivity grid')
    parser.add_argument('--tthresh-max', type=float, default=310.0,
                        help='Max T_thresh (K) for sensitivity grid')
    parser.add_argument('--tthresh-steps', type=int, default=9,
                        help='Steps for T_thresh grid')
    parser.add_argument('--lambda-min', type=float, default=0.1,
                        help='Min lambda (1/K) for sensitivity grid')
    parser.add_argument('--lambda-max', type=float, default=1.0,
                        help='Max lambda (1/K) for sensitivity grid')
    parser.add_argument('--lambda-steps', type=int, default=9,
                        help='Steps for lambda grid')
    parser.add_argument('--noise-csv', type=str, default='data/real/facebook.csv',
                        help='CSV path for noise sensitivity')
    parser.add_argument('--markov-synth', action='store_true',
                        help='Run Markov synthetic sequence generation (Example 10)')
    parser.add_argument('--markov-seed', type=int, default=42,
                        help='Random seed for Markov synthesis')
    parser.add_argument('--power-radar', action='store_true',
                        help='Run power contribution radar (Example 11)')
    parser.add_argument('--power-radar-csv', type=str, default='data/real/video.csv',
                        help='CSV path for power contribution radar')
    args = parser.parse_args()

    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "SOC-Thermal-Network Model Demo" + " " * 17 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")
    
    # 运行示例
    try:
        # example_simulate_with_synthetic_data()
        # example_baseline_comparison()
        scenario_paths = {
            'music': args.music,
            'video': args.video,
            'localvideo': args.localvideo,
            'gaming': args.gaming,
            'facebook': args.facebook
        }
        example_time_to_empty_scenarios(scenario_paths)
        '''
        if args.music and args.video:
            example_soc_comparison_music_video(args.music, args.video)
        else:
            raise ValueError("Please provide both --music and --video for Example 5.")
        
        if args.music and args.gaming:
            example_tte_bars_music_gaming(args.music, args.gaming,
                                          soc_high=args.soc_high,
                                          soc_low=args.soc_low)
        
        if args.facebook:
            example_soc_soh_facebook(args.facebook)
        
        if args.sens_csv:
            t_thresh_vals = np.linspace(args.tthresh_min, args.tthresh_max, args.tthresh_steps)
            lambda_vals = np.linspace(args.lambda_min, args.lambda_max, args.lambda_steps)
            example_sensitivity_soc_rate(args.sens_csv, t_thresh_vals, lambda_vals)
        
        if args.noise_csv:
            example_noise_sensitivity_sherlock(args.noise_csv)
        
        if args.markov_synth:
            example_markov_synthetic_sequence(seed=args.markov_seed)
        
        if args.power_radar:
            example_power_contrib_radar_sherlock(args.power_radar_csv)
        
        if args.csv:
            example_with_real_data(args.csv)
        '''
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("All examples completed. Check 'results/' folder for outputs.")
    print("=" * 60 + "\n")
