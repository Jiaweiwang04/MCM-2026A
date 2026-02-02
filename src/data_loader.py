"""
数据加载与预处理模块
负责读取日志、归一化输入、生成模型可用的序列
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class DataLoader:
    """
    日志数据加载器
    
    功能：
    1. 读取 CSV 日志
    2. 归一化输入（亮度、负载等）
    3. 生成模型输入序列
    4. 计算时间步长
    """
    
    def __init__(self, 
                 brightness_max: float = 255.0,
                 cpu_load_max: float = 100.0):
        """
        Args:
            brightness_max: 亮度原始值最大值（255 或 100）
            cpu_load_max: CPU 负载最大值（通常 100）
        """
        self.brightness_max = brightness_max
        self.cpu_load_max = cpu_load_max
    
    def load_csv(self, filepath: str, 
                 column_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        加载 CSV 日志文件
        
        Args:
            filepath: CSV 文件路径
            column_mapping: 列名映射，例如 {'原始列名': '标准列名'}
                          标准列名：timestamp, soc, screen_on, brightness_raw, 
                                  cpu_load_raw, x_net, t_core, t_env
        
        Returns:
            DataFrame
        """
        df = pd.read_csv(filepath)
        
        if column_mapping is not None:
            df = df.rename(columns=column_mapping)
        
        return df


    def load_batteryage_csv(self, filepath: str) -> pd.DataFrame:
        """
        加载 BatteryAge / Zopper BatteryAge 这类“无表头”日志，并转成标准列。

        你当前的 daily_used.csv 就属于这种格式（首行是数据，不是 header）。
        转换后会生成：
            timestamp (秒, 相对起点),
            soc (%),
            screen_on (0/1),
            charging (0/1),
            t_batt (°C),
            voltage_mv,
            current_ma,
            network_type,
            x_net (proxy),
            brightness_raw (proxy),
            cpu_load_raw (proxy)

        ⚠ 重要：brightness_raw / cpu_load_raw / x_net 在该数据源中没有直接观测，
        这里给的是 *可复现的代理构造*，用于：
            - 无监督聚类提取“场景”
            - 给模型提供一个可跑通的 real-data 演示输入
        若你后面拿到更完整日志（亮度、CPU load、网络流量等），应替换这些 proxy。

        Returns:
            标准化 DataFrame
        """
        # 读取无表头 CSV
        raw = pd.read_csv(filepath, header=None)

        # 兼容列数不足/多余
        if raw.shape[1] < 15:
            raise ValueError(f"BatteryAge CSV columns unexpected: {raw.shape[1]} (<15).")
        if raw.shape[1] < 18:
            for j in range(raw.shape[1], 18):
                raw[j] = np.nan
        raw = raw.iloc[:, :18].copy()

        raw.columns = [
            "device_id", "device_model", "android_version", "battery_type",
            "battery_capacity_mAh", "timestamp_ms", "screen_on", "soc",
            "logger_pkg", "unknown0", "t_batt", "voltage_mv", "current_ma",
            "network_type", "charging", "unused15", "unused16", "unused17"
        ]

        # 类型转换
        raw["timestamp_ms"] = pd.to_numeric(raw["timestamp_ms"], errors="coerce")
        raw = raw.dropna(subset=["timestamp_ms"]).copy()

        for c in ["soc", "t_batt", "voltage_mv", "current_ma"]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

        # bool -> 0/1
        for c in ["screen_on", "charging"]:
            if raw[c].dtype != bool:
                raw[c] = raw[c].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
                raw[c] = raw[c].fillna(False)
            raw[c] = raw[c].astype(int)

        # timestamp: ms -> relative seconds
        t0 = raw["timestamp_ms"].iloc[0]
        raw["timestamp"] = (raw["timestamp_ms"] - t0) / 1000.0

        # --------- proxy 构造（可替换） ---------
        # 代理亮度：屏幕亮时取 128（中亮度），屏幕灭时 0
        raw["brightness_raw"] = raw["screen_on"] * 128.0

        # 网络代理：Wi-Fi 给高网络强度，否则低
        net_lower = raw["network_type"].astype(str).str.lower()
        raw["x_net"] = np.where(net_lower == "wi-fi", 5000.0, 100.0)

        # CPU 代理：用功率 proxy 粗略映射到 0-100 的“负载”
        # power_proxy(W) ≈ V(mV)/1000 * I(mA)/1000
        power_w = (raw["voltage_mv"] / 1000.0) * (raw["current_ma"] / 1000.0)
        # 以 0~5W 映射到 0~100（可调）
        raw["cpu_load_raw"] = np.clip((power_w / 5.0) * 100.0, 0.0, 100.0)

        # 最终输出列（保持与 prepare_model_inputs 的 required 对齐）
        df = pd.DataFrame({
            "timestamp": raw["timestamp"].values,
            "soc": raw["soc"].values,
            "screen_on": raw["screen_on"].values,
            "brightness_raw": raw["brightness_raw"].values,
            "cpu_load_raw": raw["cpu_load_raw"].values,
            "x_net": raw["x_net"].values,
            # 可选列
            "t_batt": raw["t_batt"].values,
            "t_env": np.nan,
            "charging": raw["charging"].values,
            "network_type": raw["network_type"].values,
            "voltage_mv": raw["voltage_mv"].values,
            "current_ma": raw["current_ma"].values,
        })

        return df
    
    def normalize_brightness(self, brightness_raw: np.ndarray) -> np.ndarray:
        """
        归一化亮度 b ∈ [0, 1]
        
        Args:
            brightness_raw: 原始亮度值
        
        Returns:
            归一化亮度
        """
        b = brightness_raw / self.brightness_max
        return np.clip(b, 0.0, 1.0)
    
    def normalize_cpu_load(self, cpu_load_raw: np.ndarray) -> np.ndarray:
        """
        归一化 CPU 负载 u_cpu ∈ [0, 1]
        
        Args:
            cpu_load_raw: 原始 CPU 负载值（通常 0-100）
        
        Returns:
            归一化负载
        """
        u_cpu = cpu_load_raw / self.cpu_load_max
        return np.clip(u_cpu, 0.0, 1.0)
    
    def compute_time_steps(self, timestamp: np.ndarray) -> np.ndarray:
        """
        从时间戳计算时间步长序列 Δt
        
        Args:
            timestamp: 时间戳数组（秒）
        
        Returns:
            时间步长数组（第一个元素为 NaN）
        """
        dt = np.diff(timestamp)
        dt = np.concatenate([[np.nan], dt])  # 第一个时刻无 Δt
        return dt
    
    def celsius_to_kelvin(self, temp_celsius: np.ndarray) -> np.ndarray:
        """
        温度转换：°C -> K
        
        Args:
            temp_celsius: 摄氏温度
        
        Returns:
            开尔文温度
        """
        return temp_celsius + 273.15
    
    def prepare_model_inputs(self, df: pd.DataFrame,
                            has_temperature: bool = False,
                            has_t_env: bool = False,
                            T_env_default: float = 298.15) -> Dict[str, np.ndarray]:
        """
        准备模型输入序列
        
        Args:
            df: 包含必需列的 DataFrame
                必需：timestamp, soc, screen_on, brightness_raw, cpu_load_raw, x_net
                可选：t_core (或 t_batt), t_env
            has_temperature: 是否有温度观测
            has_t_env: 是否有环境温度序列
            T_env_default: 默认环境温度 (K)
        
        Returns:
            模型输入字典：
            {
                't': 时间数组 (s),
                'soc_obs': 观测 SOC (%),
                'u_on': 屏幕开关,
                'b': 归一化亮度,
                'u_cpu': 归一化负载,
                'x_net': 网络强度,
                'T_env': 环境温度 (K),
                'T_obs': 观测温度 (K) [若有]
            }
        """
        # 必需字段检查
        required = ['timestamp', 'soc', 'screen_on', 'brightness_raw', 
                   'cpu_load_raw', 'x_net']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # 时间处理（转为相对时间，单位秒）
        if df['timestamp'].dtype == 'object':
            # 若为时间字符串，转换
            t_abs = pd.to_datetime(df['timestamp'])
            t = (t_abs - t_abs.iloc[0]).dt.total_seconds().values
        else:
            # 若已为数值（秒），转为相对
            t = df['timestamp'].values
            t = t - t[0]
        
        # 归一化输入
        u_on = df['screen_on'].values.astype(float)
        b = self.normalize_brightness(df['brightness_raw'].values)
        u_cpu = self.normalize_cpu_load(df['cpu_load_raw'].values)
        x_net = df['x_net'].values
        
        # 构造输出字典
        inputs = {
            't': t,
            'soc_obs': df['soc'].values,
            'u_on': u_on,
            'b': b,
            'u_cpu': u_cpu,
            'x_net': x_net
        }
        
        # 信号强度（可选列，默认1.0满信号）
        if 'S_signal' in df.columns:
            inputs['S_signal'] = df['S_signal'].values
        else:
            inputs['S_signal'] = np.ones_like(t)
        
        # 网络类型（可选列，默认'cellular'）
        if 'net_type' in df.columns:
            inputs['net_type'] = df['net_type'].values
        else:
            inputs['net_type'] = np.full(len(t), 'cellular', dtype=object)
        
        # 环境温度
        if has_t_env and 't_env' in df.columns:
            T_env_raw = df['t_env'].values
            # 判断是否需要转换 °C -> K（若值域小于 200 则视为 °C）
            if T_env_raw.mean() < 200:
                inputs['T_env'] = self.celsius_to_kelvin(T_env_raw)
            else:
                inputs['T_env'] = T_env_raw
        else:
            inputs['T_env'] = np.full_like(t, T_env_default)
        
        # 观测温度（若有）
        if has_temperature:
            temp_col = 't_core' if 't_core' in df.columns else 't_batt'
            if temp_col in df.columns:
                T_obs_raw = df[temp_col].values
                if T_obs_raw.mean() < 200:
                    inputs['T_obs'] = self.celsius_to_kelvin(T_obs_raw)
                else:
                    inputs['T_obs'] = T_obs_raw
        
        return inputs

    def make_scenario_function(self, inputs: Dict[str, np.ndarray]) -> callable:
        """
        将时间序列输入转换为 scenario_inputs(t) 函数（分段常数取值）
        
        Args:
            inputs: prepare_model_inputs 的输出
        
        Returns:
            scenario_inputs(t) -> Dict
        """
        t_series = inputs['t']
        n = len(t_series)
        if n == 0:
            raise ValueError("inputs['t'] is empty.")
        
        duration = t_series[-1] - t_series[0]
        
        def scenario_inputs(t: float) -> Dict:
            if duration > 0:
                t = ((t - t_series[0]) % duration) + t_series[0]
            idx = int(np.searchsorted(t_series, t, side='right') - 1)
            if idx < 0:
                idx = 0
            elif idx >= n:
                idx = n - 1
            
            return {
                'u_on': float(inputs['u_on'][idx]),
                'b': float(inputs['b'][idx]),
                'u_cpu': float(inputs['u_cpu'][idx]),
                'u_gpu': float(inputs.get('u_gpu', np.zeros(n))[idx]),
                'x_net': float(inputs['x_net'][idx]),
                'T_env': float(inputs.get('T_env', np.full(n, 298.15))[idx]),
                'S_signal': float(inputs.get('S_signal', np.ones(n))[idx]),
                'net_type': inputs.get('net_type', np.full(n, 'cellular', dtype=object))[idx]
            }
        
        return scenario_inputs
    
    def split_train_test(self, inputs: Dict[str, np.ndarray],
                        train_ratio: float = 0.7) -> Tuple[Dict, Dict]:
        """
        按时间顺序划分训练/测试集
        
        Args:
            inputs: 完整输入字典
            train_ratio: 训练集比例
        
        Returns:
            (train_inputs, test_inputs)
        """
        N = len(inputs['t'])
        split_idx = int(N * train_ratio)
        
        train_inputs = {k: v[:split_idx] for k, v in inputs.items()}
        test_inputs = {k: v[split_idx:] for k, v in inputs.items()}
        
        # 测试集时间重置为从 0 开始
        test_inputs['t'] = test_inputs['t'] - test_inputs['t'][0]
        
        return train_inputs, test_inputs
    
    def get_initial_state(self, inputs: Dict[str, np.ndarray],
                         use_observed_T: bool = False) -> Dict:
        """
        从输入序列提取初始状态
        
        Args:
            inputs: 输入字典
            use_observed_T: 是否使用观测温度作为初值（若有）
        
        Returns:
            初始状态 {'s': SOC, 'T': 温度, 'z': 网络尾态}
        """
        state = {
            's': inputs['soc_obs'][0],
            'z': 0.0  # 网络尾态初值默认为 0
        }
        
        # 温度初值
        if use_observed_T and 'T_obs' in inputs:
            state['T'] = inputs['T_obs'][0]
        else:
            state['T'] = inputs['T_env'][0]
        
        return state


def generate_scenario_function(scenario_type: str) -> callable:
    """
    生成场景输入函数（用于 time-to-empty 预测）
    
    Args:
        scenario_type: 场景类型
            - 'music': 待机
            - 'video': 视频播放
            - 'localvideo': 本地视频
            - 'gaming': 游戏
            - 'facebook': facebook
    
    Returns:
        场景函数 f(t) -> inputs_dict
    """
    
    def music_scenario(t: float) -> Dict:
        """音乐场景：屏幕关闭，中等 CPU，低网络"""
        return {
            'u_on': 0.0,
            'b': 0.0,
            'u_cpu': 0.4613,
            'x_net': 1000.0,  # 少量后台同步
            'T_env': 298.15
        }
    
    def video_scenario(t: float) -> Dict:
        """视频播放：屏幕开启中亮度，低 CPU，中网络"""
        return {
            'u_on': 1.0,
            'b': 0.9585,
            'u_cpu': 0.1116,
            'x_net': 4000.0,  # 视频流量
            'T_env': 298.15
        }
    
    def localvideo_scenario(t: float) -> Dict:
        """视频播放：屏幕开启中亮度，高 CPU，无网络"""
        return {
            'u_on': 1.0,
            'b': 0.9288,
            'u_cpu': 0.5236,
            'x_net': 0,  # 视频流量
            'T_env': 298.15
        }
    
    def gaming_scenario(t: float) -> Dict:
        """游戏场景：屏幕开启高亮度，高 CPU，高网络"""
        return {
            'u_on': 1.0,
            'b': 1.0,
            'u_cpu': 1.0,
            'x_net': 5000.0,
            'T_env': 298.15
        }
    
    def facebook_scenario(t: float) -> Dict:
        """浏览facebook：屏幕开启中亮度，高 CPU，中等网络"""
        return {
            'u_on': 1.0,
            'b': 0.5577,
            'u_cpu': 0.5477,
            'x_net': 3000,
            'T_env': 298.15
        }
    
    scenarios = {
        'music': music_scenario,
        'video': video_scenario,
        'localvideo': localvideo_scenario,
        'gaming': gaming_scenario,
        'facebook': facebook_scenario
    }
    
    if scenario_type not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_type}. "
                        f"Available: {list(scenarios.keys())}")
    
    return scenarios[scenario_type]
