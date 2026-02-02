"""
场景数据生成器
生成5种典型使用场景的时序日志数据，可直接用于模型仿真
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import os


class ScenarioDataGenerator:
    """
    场景数据生成器
    
    场景：
    1. 日常使用 (daily_use): 混合轻度使用（浏览、社交、拍照）
    2. 通勤 (commute): 音乐+导航+间歇浏览
    3. 工作 (work): 长时间屏幕开启、视频会议、文档处理
    4. 高强度游戏 (gaming): 持续高负载、高亮度、高网络
    5. 夜间睡眠 (sleep): 待机为主、偶尔消息推送
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: 随机种子（保证可复现）
        """
        np.random.seed(seed)
        self.T_env = 298.15  # 环境温度 25°C
    
    def _add_noise(self, base: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """
        添加随机噪声（模拟真实波动）
        
        Args:
            base: 基础信号
            noise_level: 噪声水平（相对于均值）
        
        Returns:
            带噪声的信号
        """
        noise = np.random.randn(len(base)) * noise_level * np.mean(np.abs(base))
        return base + noise
    
    def _generate_burst_pattern(self, N: int, burst_prob: float = 0.1, 
                                burst_duration: int = 60) -> np.ndarray:
        """
        生成突发模式（用于网络流量）
        
        Args:
            N: 序列长度
            burst_prob: 突发概率
            burst_duration: 突发持续时间（秒）
        
        Returns:
            突发模式掩码
        """
        pattern = np.zeros(N)
        i = 0
        while i < N:
            if np.random.rand() < burst_prob:
                duration = min(burst_duration, N - i)
                pattern[i:i+duration] = 1.0
                i += duration
            else:
                i += 1
        return pattern
    
    def generate_daily_use(self, duration_hours: float = 3.0, 
                          dt: float = 1.0) -> pd.DataFrame:
        """
        日常使用场景：混合轻度活动
        
        特点：
        - 屏幕间歇开关（刷社交、拍照、浏览）
        - 中低负载
        - 中等亮度
        - 网络流量有突发
        
        Args:
            duration_hours: 持续时间（小时）
            dt: 采样间隔（秒）
        
        Returns:
            DataFrame with columns: timestamp, soc, screen_on, brightness_raw, 
                                   cpu_load_raw, x_net, t_env
        """
        N = int(duration_hours * 3600 / dt)
        t = np.arange(0, N * dt, dt)
        
        # 屏幕开关：60% 时间开启，周期约 5 分钟
        screen_on = (np.sin(2 * np.pi * t / 300 + np.random.rand(N) * 3) > -0.2).astype(float)
        
        # 亮度：屏幕开时 40-70%
        brightness_raw = screen_on * (100 + 80 * (0.5 + 0.3 * np.sin(2 * np.pi * t / 600)))
        brightness_raw = self._add_noise(brightness_raw, 0.03)
        brightness_raw = np.clip(brightness_raw, 0, 255)
        
        # CPU 负载：低-中（10-40%）
        cpu_load_raw = 15 + 15 * np.sin(2 * np.pi * t / 400) + 10 * screen_on
        cpu_load_raw = self._add_noise(cpu_load_raw, 0.1)
        cpu_load_raw = np.clip(cpu_load_raw, 0, 100)
        
        # 网络流量：突发模式（刷新、加载）
        burst = self._generate_burst_pattern(N, burst_prob=0.08, burst_duration=30)
        x_net = 500 + burst * (2000 + 1000 * np.random.rand(N))
        x_net = self._add_noise(x_net, 0.15)
        x_net = np.clip(x_net, 0, None)
        
        # 初始 SOC 80%，根据功耗粗略递减（实际由模型计算）
        soc_init = 80.0
        # 这里给一个占位值，实际 SOC 应由模型仿真得到
        soc = np.linspace(soc_init, soc_init - 15, N)  # 占位：约消耗 15%
        
        df = pd.DataFrame({
            'timestamp': t,
            'soc': soc,
            'screen_on': screen_on,
            'brightness_raw': brightness_raw,
            'cpu_load_raw': cpu_load_raw,
            'x_net': x_net,
            't_env': np.full(N, self.T_env)
        })
        
        return df
    
    def generate_commute(self, duration_hours: float = 1.5, 
                        dt: float = 1.0) -> pd.DataFrame:
        """
        通勤场景：音乐+导航+间歇浏览
        
        特点：
        - 屏幕多数时间关闭（音乐后台）或低亮度（导航）
        - 持续中低 CPU（音频解码）
        - 持续网络流量（音乐流媒体）
        - 偶尔屏幕亮起（看消息）
        """
        N = int(duration_hours * 3600 / dt)
        t = np.arange(0, N * dt, dt)
        
        # 屏幕开关：30% 时间开启（偶尔看导航/消息）
        screen_on = (np.random.rand(N) < 0.3).astype(float)
        
        # 亮度：开启时低亮度（30-50%）
        brightness_raw = screen_on * (80 + 50 * np.random.rand(N))
        brightness_raw = np.clip(brightness_raw, 0, 255)
        
        # CPU 负载：持续低负载（15-25%，音频解码）
        cpu_load_raw = 20 + 5 * np.sin(2 * np.pi * t / 300)
        cpu_load_raw = self._add_noise(cpu_load_raw, 0.08)
        cpu_load_raw = np.clip(cpu_load_raw, 5, 100)
        
        # 网络流量：持续中等（音乐流）
        x_net = 1500 + 500 * np.sin(2 * np.pi * t / 200)
        x_net = self._add_noise(x_net, 0.1)
        x_net = np.clip(x_net, 0, None)
        
        soc_init = 65.0
        soc = np.linspace(soc_init, soc_init - 8, N)  # 占位
        
        df = pd.DataFrame({
            'timestamp': t,
            'soc': soc,
            'screen_on': screen_on,
            'brightness_raw': brightness_raw,
            'cpu_load_raw': cpu_load_raw,
            'x_net': x_net,
            't_env': np.full(N, self.T_env)
        })
        
        return df
    
    def generate_work(self, duration_hours: float = 4.0, 
                     dt: float = 1.0) -> pd.DataFrame:
        """
        工作场景：视频会议、文档处理
        
        特点：
        - 屏幕长时间开启
        - 中等亮度
        - CPU 负载有波动（视频会议时高）
        - 网络流量周期性（会议时高）
        """
        N = int(duration_hours * 3600 / dt)
        t = np.arange(0, N * dt, dt)
        
        # 屏幕开关：90% 时间开启
        screen_on = np.ones(N)
        screen_on[np.random.rand(N) < 0.1] = 0  # 偶尔关闭
        
        # 亮度：中等（50-70%）
        brightness_raw = 130 + 50 * np.sin(2 * np.pi * t / 1800)
        brightness_raw = self._add_noise(brightness_raw, 0.05)
        brightness_raw = np.clip(brightness_raw, 0, 255)
        
        # CPU 负载：有会议高峰（40-60%）
        # 模拟每小时一次视频会议
        meeting_pattern = np.zeros(N)
        for meeting_start in range(0, N, 3600):
            meeting_end = min(meeting_start + 1800, N)  # 30分钟会议
            meeting_pattern[meeting_start:meeting_end] = 1.0
        
        cpu_load_raw = 25 + meeting_pattern * 30 + 10 * np.sin(2 * np.pi * t / 600)
        cpu_load_raw = self._add_noise(cpu_load_raw, 0.12)
        cpu_load_raw = np.clip(cpu_load_raw, 10, 100)
        
        # 网络流量：会议时高（视频流）
        x_net = 1000 + meeting_pattern * 4000 + 500 * np.random.rand(N)
        x_net = self._add_noise(x_net, 0.1)
        x_net = np.clip(x_net, 0, None)
        
        soc_init = 90.0
        soc = np.linspace(soc_init, soc_init - 25, N)  # 占位
        
        df = pd.DataFrame({
            'timestamp': t,
            'soc': soc,
            'screen_on': screen_on,
            'brightness_raw': brightness_raw,
            'cpu_load_raw': cpu_load_raw,
            'x_net': x_net,
            't_env': np.full(N, self.T_env)
        })
        
        return df
    
    def generate_gaming(self, duration_hours: float = 2.0, 
                       dt: float = 1.0) -> pd.DataFrame:
        """
        高强度游戏场景
        
        特点：
        - 屏幕持续开启、高亮度
        - CPU/GPU 高负载（60-80%）
        - 网络流量中等（多人游戏）
        - 易触发温控（高产热）
        """
        N = int(duration_hours * 3600 / dt)
        t = np.arange(0, N * dt, dt)
        
        # 屏幕持续开启
        screen_on = np.ones(N)
        
        # 亮度：高（80-100%）
        brightness_raw = 200 + 40 * np.sin(2 * np.pi * t / 300)
        brightness_raw = self._add_noise(brightness_raw, 0.03)
        brightness_raw = np.clip(brightness_raw, 180, 255)
        
        # CPU 负载：持续高（60-80%）
        cpu_load_raw = 70 + 10 * np.sin(2 * np.pi * t / 180)
        cpu_load_raw = self._add_noise(cpu_load_raw, 0.08)
        cpu_load_raw = np.clip(cpu_load_raw, 50, 100)
        
        # 网络流量：中等但稳定（游戏数据包）
        x_net = 2500 + 800 * np.sin(2 * np.pi * t / 120)
        x_net = self._add_noise(x_net, 0.15)
        x_net = np.clip(x_net, 1000, None)
        
        soc_init = 100.0
        soc = np.linspace(soc_init, soc_init - 45, N)  # 占位：高耗电
        
        df = pd.DataFrame({
            'timestamp': t,
            'soc': soc,
            'screen_on': screen_on,
            'brightness_raw': brightness_raw,
            'cpu_load_raw': cpu_load_raw,
            'x_net': x_net,
            't_env': np.full(N, self.T_env)
        })
        
        return df
    
    def generate_sleep(self, duration_hours: float = 8.0, 
                      dt: float = 1.0) -> pd.DataFrame:
        """
        夜间睡眠场景
        
        特点：
        - 屏幕几乎一直关闭
        - CPU 极低负载（后台维持）
        - 网络流量极低（偶尔推送）
        - 温度接近环境温度
        """
        N = int(duration_hours * 3600 / dt)
        t = np.arange(0, N * dt, dt)
        
        # 屏幕几乎不开（<1% 时间，偶尔看闹钟）
        screen_on = np.zeros(N)
        screen_on[np.random.rand(N) < 0.005] = 1
        
        # 亮度：偶尔开启时低亮度
        brightness_raw = screen_on * (30 + 20 * np.random.rand(N))
        brightness_raw = np.clip(brightness_raw, 0, 255)
        
        # CPU 负载：极低（2-5%）
        cpu_load_raw = 3 + 2 * np.sin(2 * np.pi * t / 1800)
        cpu_load_raw = self._add_noise(cpu_load_raw, 0.2)
        cpu_load_raw = np.clip(cpu_load_raw, 1, 100)
        
        # 网络流量：极低（偶尔推送）
        push_events = self._generate_burst_pattern(N, burst_prob=0.01, burst_duration=5)
        x_net = 50 + push_events * 200
        x_net = self._add_noise(x_net, 0.3)
        x_net = np.clip(x_net, 10, None)
        
        soc_init = 85.0
        soc = np.linspace(soc_init, soc_init - 3, N)  # 占位：极低耗电
        
        df = pd.DataFrame({
            'timestamp': t,
            'soc': soc,
            'screen_on': screen_on,
            'brightness_raw': brightness_raw,
            'cpu_load_raw': cpu_load_raw,
            'x_net': x_net,
            't_env': np.full(N, self.T_env)
        })
        
        return df
    
    def generate_all_scenarios(self, output_dir: str = 'data/generated') -> Dict[str, str]:
        """
        生成所有场景的数据集并保存为 CSV
        
        Args:
            output_dir: 输出目录
        
        Returns:
            场景名 -> 文件路径的字典
        """
        os.makedirs(output_dir, exist_ok=True)
        
        scenarios = {
            'daily_use': (self.generate_daily_use, {}),
            'commute': (self.generate_commute, {}),
            'work': (self.generate_work, {}),
            'gaming': (self.generate_gaming, {}),
            'sleep': (self.generate_sleep, {})
        }
        
        file_paths = {}
        
        for scenario_name, (gen_func, kwargs) in scenarios.items():
            print(f"Generating {scenario_name}...")
            df = gen_func(**kwargs)
            
            filepath = os.path.join(output_dir, f'{scenario_name}.csv')
            df.to_csv(filepath, index=False)
            file_paths[scenario_name] = filepath
            
            print(f"  → Saved to {filepath} ({len(df)} samples)")
        
        print(f"\n✓ All scenarios generated in '{output_dir}/'")
        return file_paths


if __name__ == '__main__':
    """直接运行此脚本生成所有场景数据"""
    
    print("=" * 60)
    print("Scenario Data Generator")
    print("=" * 60 + "\n")
    
    generator = ScenarioDataGenerator(seed=42)
    file_paths = generator.generate_all_scenarios()
    
    print("\nGenerated scenarios:")
    for name, path in file_paths.items():
        print(f"  {name:15s}: {path}")
    
    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("Use these CSV files with DataLoader for model simulation.")
    print("=" * 60)
