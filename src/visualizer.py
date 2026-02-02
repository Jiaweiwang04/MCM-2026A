"""
可视化模块（与模型计算解耦）
提供标准化的绘图函数，用于模型结果分析与论文展示
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List, Tuple
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


class Visualizer:
    """
    模型可视化工具
    
    功能：
    1. SOC 预测 vs 观测对比
    2. 温控机制解释（T, φ, P_sys 三联图）
    3. Baseline 对照（消融实验）
    4. Time-to-empty 场景对比
    """
    
    def __init__(self, style: Optional[str] = None):
        """
        Args:
            style: matplotlib 样式
        """
        # 颜色体系（白底、低饱和蓝/绿/橙，深灰文字/线条，浅灰网格）
        self.colors = {
            'C_BG': '#FFFFFF',
            'C_TEXT': '#2F2F2F',
            'C_AXES': '#5A5A5A',
            'C_GRID': '#E6E6E6',
            'C_BLUE_MAIN': '#AEC6CF',
            'C_BLUE_DARK': '#34495E',
            'C_GREEN_MAIN': '#A0BC92',
            'C_GREEN_LIGHT': '#DDE9D7',
            'C_ORANGE_MAIN': '#D6BA9C',
            'C_ORANGE_DARK': '#B07A3F'
        }
        self.palette = [
            self.colors['C_BLUE_MAIN'],
            self.colors['C_GREEN_MAIN'],
            self.colors['C_ORANGE_MAIN'],
            self.colors['C_BLUE_DARK'],
            self.colors['C_GREEN_LIGHT']
        ]
        
        if style:
            try:
                plt.style.use(style)
            except:
                pass  # 若样式不存在则使用默认
        
        # 全局 rc 参数
        plt.rcParams.update({
            'figure.facecolor': self.colors['C_BG'],
            'axes.facecolor': self.colors['C_BG'],
            'axes.edgecolor': self.colors['C_AXES'],
            'axes.labelcolor': self.colors['C_TEXT'],
            'xtick.color': self.colors['C_AXES'],
            'ytick.color': self.colors['C_AXES'],
            'text.color': self.colors['C_TEXT'],
            'grid.color': self.colors['C_GRID'],
            'grid.alpha': 0.8,
            'axes.prop_cycle': plt.cycler(color=self.palette)
        })
        
        # 设置中文字体（避免乱码）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_soc_comparison(self, 
                           trajectory: Dict,
                           soc_obs: Optional[np.ndarray] = None,
                           title: str = "SOC Prediction vs Observation",
                           save_path: Optional[str] = None):
        """
        绘制 SOC 预测 vs 观测对比
        
        Args:
            trajectory: 模型输出轨迹（必须包含 't', 's'）
            soc_obs: 观测 SOC（若提供则绘制对比）
            title: 图标题
            save_path: 保存路径（可选）
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        t = trajectory['t']
        s_pred = trajectory['s']
        
        ax.plot(t, s_pred, label='Predicted SOC', linewidth=2)
        
        if soc_obs is not None:
            ax.plot(t, soc_obs, label='Observed SOC', 
                   linestyle='--', linewidth=1.5, alpha=0.7)
            
            # 计算误差
            mae = np.mean(np.abs(s_pred - soc_obs))
            rmse = np.sqrt(np.mean((s_pred - soc_obs)**2))
            ax.text(0.02, 0.98, f'MAE: {mae:.2f}%\nRMSE: {rmse:.2f}%',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('SOC (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_soc_comparison_two(self,
                                music_traj: Dict,
                                music_obs: np.ndarray,
                                video_traj: Dict,
                                video_obs: np.ndarray,
                                title: str = "SOC Prediction vs Observation (Music & Video)",
                                save_path: Optional[str] = None):
        """
        音乐/视频场景 SOC 真值 vs 预测对比（双子图）
        
        Args:
            music_traj: 音乐场景轨迹（含 't','s'）
            music_obs: 音乐场景观测 SOC
            video_traj: 视频场景轨迹（含 't','s'）
            video_obs: 视频场景观测 SOC
            title: 图标题
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        fig.suptitle(title, fontsize=12, fontweight='bold')
        
        def _plot_one(ax, traj, obs, label):
            t = traj['t']
            s_pred = traj['s']
            ax.plot(t, s_pred, label='Predicted SOC', linewidth=2, color=self.colors['C_BLUE_DARK'])
            ax.plot(t, obs, label='Observed SOC', linestyle='--', linewidth=1.6,
                    alpha=0.9, color=self.colors['C_ORANGE_MAIN'])
            
            mse = float(np.mean((s_pred - obs) ** 2))
            ax.text(0.98, 0.98, f"MSE: {mse:.2f}",
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor=self.colors['C_BG'],
                              edgecolor=self.colors['C_AXES'], alpha=0.9))
            
            ax.set_title(label, fontsize=11)
            ax.set_xlabel('Time (s)')
            ax.grid(True, alpha=0.3)
        
        _plot_one(axes[0], music_traj, music_obs, 'Music')
        _plot_one(axes[1], video_traj, video_obs, 'Video')
        
        axes[0].set_ylabel('SOC (%)')
        axes[1].legend(loc='best')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_soc_soh_comparison(self,
                                trajectories: Dict[str, Dict],
                                title: str = "SOC vs Time under Different SOH",
                                save_path: Optional[str] = None):
        """
        不同 SOH 下的 SOC 曲线对比
        
        Args:
            trajectories: {'soh_label': trajectory_dict}
            title: 图标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        for i, (label, traj) in enumerate(trajectories.items()):
            t = traj['t']
            s = traj['s']
            color = self.palette[i % len(self.palette)]
            ax.plot(t, s, label=label, linewidth=2, color=color)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('SOC (%)')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_sensitivity_heatmap(self,
                                 x_vals: np.ndarray,
                                 y_vals: np.ndarray,
                                 z_vals: np.ndarray,
                                 xlabel: str = "T_thresh (K)",
                                 ylabel: str = "lambda (1/K)",
                                 title: str = "Sensitivity Heatmap",
                                 cbar_label: str = "SOC decline rate (%/hour)",
                                 save_path: Optional[str] = None):
        """
        二维敏感性热力图/等高线图
        
        Args:
            x_vals: 横轴取值
            y_vals: 纵轴取值
            z_vals: 指标矩阵 (len(y_vals) x len(x_vals))
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cmap = LinearSegmentedColormap.from_list(
            "custom_sens",
            [self.colors['C_GREEN_LIGHT'], self.colors['C_GREEN_MAIN'],
             self.colors['C_ORANGE_MAIN'], self.colors['C_ORANGE_DARK']]
        )
        
        im = ax.imshow(z_vals, origin='lower', aspect='auto',
                       extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
                       cmap=cmap)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(False)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_noise_sensitivity_curves(self,
                                      strengths: np.ndarray,
                                      mse_by_feature: Dict[str, np.ndarray],
                                      title: str = "Single-Feature Noise Sensitivity (ΔMSE)",
                                      save_path: Optional[str] = None):
        """
        单特征加噪 MSE 曲线
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
        
        for i, (feat, mse_vals) in enumerate(mse_by_feature.items()):
            color = self.palette[i % len(self.palette)]
            marker = markers[i % len(markers)]
            ax.plot(strengths, mse_vals, label=feat, linewidth=2, color=color,
                    marker=marker, markersize=4)
        
        ax.set_xlabel('Noise strength (normalized by feature std)')
        ax.set_ylabel('SOC ΔMSE')
        ax.set_title(title)
        ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_noise_sensitivity_bar(self,
                                   delta_mse: Dict[str, float],
                                   title: str = "MSE Increase under Strong Noise",
                                   save_path: Optional[str] = None):
        """
        强噪声下 MSE 增量条形图
        """
        items = sorted(delta_mse.items(), key=lambda x: x[1], reverse=True)
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [self.colors['C_ORANGE_MAIN']] * len(labels)
        ax.barh(labels, values, color=colors, edgecolor=self.colors['C_AXES'], linewidth=0.6)
        ax.invert_yaxis()
        ax.set_xlabel('MSE increase')
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_noise_stress_curve(self,
                                strengths: np.ndarray,
                                mse_mean: np.ndarray,
                                mse_std: np.ndarray,
                                title: str = "All-Feature Noise Stress Test",
                                save_path: Optional[str] = None):
        """
        全特征加噪 MSE 曲线（均值 + 方差带）
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(strengths, mse_mean, color=self.colors['C_BLUE_DARK'],
                linewidth=2, label='Mean MSE')
        ax.fill_between(strengths, mse_mean - mse_std, mse_mean + mse_std,
                        color=self.colors['C_BLUE_MAIN'], alpha=0.3, label='±1 std')
        
        ax.set_xlabel('Noise strength (normalized by feature std)')
        ax.set_ylabel('SOC MSE')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_power_contrib_radar(self,
                                 labels: List[str],
                                 values: List[float],
                                 title: str = "Power Contribution Radar",
                                 save_path: Optional[str] = None,
                                 log_scale: bool = False):
        """
        功耗贡献雷达图
        """
        if len(labels) != len(values):
            raise ValueError("labels and values must have the same length.")
        
        vals = np.array(values, dtype=float)
        if log_scale:
            vals_for_plot = np.log10(np.maximum(vals, 1e-6))
            vmin = float(np.min(vals_for_plot))
            vmax = float(np.max(vals_for_plot))
            if vmax > vmin:
                vals_norm = (vals_for_plot - vmin) / (vmax - vmin)
            else:
                vals_norm = np.zeros_like(vals_for_plot)
        else:
            total = float(np.sum(vals)) if np.sum(vals) > 0 else 1.0
            vals_norm = vals / total
        
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        vals_plot = vals_norm.tolist() + vals_norm[:1].tolist()
        
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        
        ax.plot(angles, vals_plot, color=self.colors['C_ORANGE_DARK'], linewidth=2)
        ax.fill(angles, vals_plot, color=self.colors['C_ORANGE_MAIN'], alpha=0.35)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels([])
        ax.set_ylim(0, max(0.2, float(np.max(vals_norm)) * 1.15))
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # 标注实际数值与占比
        for i, (ang, v, vn) in enumerate(zip(angles[:-1], vals, vals_norm)):
            txt = f"{v:.2f} W\n{vn*100:.1f}%"
            ax.text(ang, vn + 0.03, txt, ha='center', va='center',
                    fontsize=8, color=self.colors['C_TEXT'])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_power_contrib_pie(self,
                               labels: List[str],
                               values: List[float],
                               title: str = "Power Contribution Pie",
                               save_path: Optional[str] = None):
        """
        功耗贡献饼图
        """
        if len(labels) != len(values):
            raise ValueError("labels and values must have the same length.")
        
        vals = np.array(values, dtype=float)
        total = float(np.sum(vals)) if np.sum(vals) > 0 else 1.0
        perc = vals / total * 100.0
        
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = [self.colors['C_BLUE_MAIN'],
                  self.colors['C_ORANGE_MAIN'],
                  self.colors['C_GREEN_MAIN'],
                  self.colors['C_BLUE_DARK']]
        colors = [colors[i % len(colors)] for i in range(len(labels))]
        
        ax.pie(perc, labels=None, colors=colors, startangle=90,
               wedgeprops=dict(edgecolor=self.colors['C_AXES'], linewidth=0.8))
        ax.axis('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # 图例带数值
        legend_labels = [f"{lab}: {val:.2f} W ({p:.1f}%)"
                         for lab, val, p in zip(labels, vals, perc)]
        ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_thermal_mechanism(self,
                              trajectory: Dict,
                              title: str = "Thermal Throttling Mechanism",
                              save_path: Optional[str] = None):
        """
        三联图：温度、温控系数、功耗（解释热限制回路）
        
        Args:
            trajectory: 必须包含 't', 'T', 'phi', 'P_sys'
            title: 图标题
            save_path: 保存路径
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        t = trajectory['t']
        T_celsius = trajectory['T'] - 273.15  # 转换为 °C 便于阅读
        phi = trajectory['phi']
        P_sys = trajectory['P_sys']
        
        # 子图1: 温度
        axes[0].plot(t, T_celsius, color='red', linewidth=2)
        axes[0].set_ylabel('Temperature (°C)', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(title, fontsize=13, fontweight='bold')
        
        # 子图2: 温控系数 φ(T)
        axes[1].plot(t, phi, color='orange', linewidth=2)
        axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No throttling')
        axes[1].set_ylabel('Throttling Factor φ(T)', fontsize=11)
        axes[1].set_ylim([0, 1.1])
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # 子图3: 系统功耗
        axes[2].plot(t, P_sys, color='blue', linewidth=2)
        axes[2].set_xlabel('Time (s)', fontsize=11)
        axes[2].set_ylabel('System Power (W)', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_power_decomposition(self,
                                trajectory: Dict,
                                title: str = "Power Decomposition",
                                save_path: Optional[str] = None):
        """
        功耗分解：P_sys, P_loss, P_heat
        
        Args:
            trajectory: 必须包含 't', 'P_sys', 'P_loss', 'P_heat'
            title: 图标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        t = trajectory['t']
        P_sys = trajectory['P_sys']
        P_loss = trajectory['P_loss']
        P_heat = trajectory['P_heat']
        
        ax.plot(t, P_sys, label='System Power ($P_{sys}$)', linewidth=2)
        ax.plot(t, P_loss, label='Resistive Loss ($P_{loss}$)', 
               linewidth=2, linestyle='--')
        ax.plot(t, P_heat, label='Total Heat ($P_{heat}$)', 
               linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (W)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_baseline_comparison(self,
                                trajectories: Dict[str, Dict],
                                soc_obs: Optional[np.ndarray] = None,
                                title: str = "Baseline Comparison (Ablation Study)",
                                save_path: Optional[str] = None):
        """
        Baseline 对照图（消融实验）
        
        Args:
            trajectories: 多条轨迹字典 {'label': trajectory_dict, ...}
            soc_obs: 观测 SOC
            title: 图标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        linestyles = ['-', '--', '-.', ':']
        
        for i, (label, traj) in enumerate(trajectories.items()):
            t = traj['t']
            s = traj['s']
            ax.plot(t, s, label=label, 
                   color=colors[i % len(colors)],
                   linestyle=linestyles[i % len(linestyles)],
                   linewidth=2)
        
        if soc_obs is not None:
            # 假设所有轨迹时间对齐
            t_obs = trajectories[list(trajectories.keys())[0]]['t']
            ax.plot(t_obs, soc_obs, label='Observed', 
                   color='black', linewidth=2, alpha=0.7, linestyle=':')
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('SOC (%)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_time_to_empty_scenarios(self,
                                    scenarios: Dict[str, Tuple[float, Dict]],
                                    title: str = "Time to Empty: Scenario Comparison",
                                    save_path: Optional[str] = None):
        """
        多场景 TTE 对比
        
        Args:
            scenarios: {'scenario_name': (tte, trajectory), ...}
            title: 图标题
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # 字体与边框加粗（仅本图）
        title_fs = 14
        label_fs = 12
        tick_fs = 11
        spine_lw = 1.2
        
        colors = [self.palette[i % len(self.palette)] for i in range(len(scenarios))]
        scenario_color_override = {
            'facebook': self.colors['C_ORANGE_DARK'],
            'localvideo': '#C05A5A'
        }
        
        # 子图1: SOC 演化
        for i, (name, (tte, traj)) in enumerate(scenarios.items()):
            t = traj['t']
            s = traj['s']
            color = scenario_color_override.get(name, colors[i])
            axes[0].plot(t, s, label=f'{name} (TTE={tte/60:.1f} min)',
                        color=color, linewidth=2)
        
        axes[0].set_ylabel('SOC (%)', fontsize=label_fs)
        axes[0].legend(loc='best', fontsize=11, frameon=True, framealpha=0.9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(title, fontsize=title_fs, fontweight='bold')
        
        # 子图2: 温度演化
        for i, (name, (tte, traj)) in enumerate(scenarios.items()):
            t = traj['t']
            T_celsius = traj['T'] - 273.15
            color = scenario_color_override.get(name, colors[i])
            axes[1].plot(t, T_celsius, label=name, 
                        color=color, linewidth=2)
        
        axes[1].set_xlabel('Time (s)', fontsize=label_fs)
        axes[1].set_ylabel('Temperature (°C)', fontsize=label_fs)
        axes[1].legend(loc='best', fontsize=11, frameon=True, framealpha=0.9)
        axes[1].grid(True, alpha=0.3)
        
        for ax in axes:
            ax.tick_params(axis='both', labelsize=tick_fs, width=spine_lw)
            for spine in ax.spines.values():
                spine.set_linewidth(spine_lw)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_time_to_empty_bar(self,
                               scenarios: Dict[str, Tuple[float, Dict]],
                               title: str = "Time to Empty (TTE) by Scenario",
                               save_path: Optional[str] = None):
        """
        多场景 TTE 柱状图
        
        Args:
            scenarios: {'scenario_name': (tte, trajectory), ...}
            title: 图标题
            save_path: 保存路径
        """
        names = list(scenarios.keys())
        tte_minutes = [scenarios[name][0] / 60.0 for name in names]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [self.palette[i % len(self.palette)] for i in range(len(names))]
        bars = ax.bar(names, tte_minutes, color=colors, edgecolor='black', linewidth=0.6)
        
        ax.set_ylabel('TTE (minutes)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, tte_minutes):
            ax.text(bar.get_x() + bar.get_width() / 2, val,
                    f"{val:.1f}", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_full_dashboard(self,
                           trajectory: Dict,
                           soc_obs: Optional[np.ndarray] = None,
                           save_path: Optional[str] = None):
        """
        综合仪表板（所有关键信息）
        
        Args:
            trajectory: 完整轨迹
            soc_obs: 观测 SOC
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        t = trajectory['t']
        
        # (0,0): SOC
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t, trajectory['s'], label='Predicted', linewidth=1.5)
        # 真值对比
        '''if soc_obs is not None:
            ax1.plot(t, soc_obs, label='Observed', linestyle='--', linewidth=1.5)
            mae = np.mean(np.abs(trajectory['s'] - soc_obs))
            ax1.text(0.02, 0.98, f'MAE: {mae:.2f}%',
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))'''
        
        ax1.set_ylabel('SOC (%)')
        ax1.set_title('Battery State of Charge', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # (0,1): 温度
        ax2 = fig.add_subplot(gs[0, 1])
        T_celsius = trajectory['T'] - 273.15
        ax2.plot(t, T_celsius, color='red', linewidth=1.5)
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Core Temperature', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # (1,0): 温控系数
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(t, trajectory['phi'], color='orange', linewidth=1.5)
        ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_ylabel('Throttling Factor φ(T)')
        ax3.set_title('Thermal Throttling Mechanism', fontweight='bold')
        ax3.set_ylim([0, 1.1])
        ax3.grid(True, alpha=0.3)
        
        # (1,1): 功耗
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(t, trajectory['P_sys'], label='$P_{sys}$', linewidth=1.5)
        ax4.plot(t, trajectory['P_loss'], label='$P_{loss}$', 
                linestyle='--', linewidth=1)
        #ax4.plot(t, trajectory['P_heat'], label='$P_{heat}$', 
                #alpha=0.7, linewidth=2)
        ax4.set_ylabel('Power (W)')
        ax4.set_title('Power Decomposition', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # (2,0): 网络尾态
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(t, trajectory['z'], color='green', linewidth=1.5)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Network Tail State z(t)')
        ax5.set_title('Network Tail Energy State', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # (2,1): SOC 误差（若有观测）
        ax6 = fig.add_subplot(gs[2, 1])
        if soc_obs is not None:
            error = trajectory['s'] - soc_obs
            ax6.plot(t, error, color='purple', linewidth=1.5)
            ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax6.fill_between(t, 0, error, alpha=0.3, color='purple')
            ax6.set_ylabel('SOC Error (%)')
            ax6.set_title('Prediction Error', fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No Observation Available',
                    ha='center', va='center', transform=ax6.transAxes,
                    fontsize=12, color='gray')
            ax6.set_title('Prediction Error', fontweight='bold')
        ax6.set_xlabel('Time (s)')
        ax6.grid(True, alpha=0.3)
        
        fig.suptitle('SOC-Thermal-Network Model Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # ==================== Clustering Visualizations ====================

    def plot_cluster_timeline(self,
                              clustered_windows,
                              time_col: str = "window_start_dt",
                              cluster_col: str = "cluster",
                              title: str = "Unsupervised Scenario Clusters Over Time",
                              save_path: Optional[str] = None):
        """
        Plot cluster id over time (each point is one window).

        Args:
            clustered_windows: DataFrame with at least [time_col, cluster_col]
            time_col: datetime column (preferred) or seconds
            cluster_col: cluster label column
            title: plot title
            save_path: path to save png
        """
        import pandas as pd
        df = clustered_windows.copy()

        # Choose x axis
        if time_col not in df.columns:
            time_col = "window_start_s"
        x = df[time_col]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.scatter(x, df[cluster_col], s=18, alpha=0.85)

        ax.set_xlabel("Time" if "dt" in time_col else "Time (s)")
        ax.set_ylabel("Cluster ID")
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # pretty xticks if datetime
        if "dt" in time_col:
            fig.autofmt_xdate()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_cluster_pca(self,
                         clustered_windows,
                         x_col: str = "pca1",
                         y_col: str = "pca2",
                         cluster_col: str = "cluster",
                         title: str = "Cluster Visualization (PCA 2D)",
                         save_path: Optional[str] = None):
        """
        Plot PCA scatter colored by cluster.
        (Colors handled by matplotlib default colormap to avoid hardcoding.)

        Args:
            clustered_windows: DataFrame containing pca1/pca2 and cluster labels
        """
        df = clustered_windows.copy()

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(df[x_col], df[y_col], c=df[cluster_col], s=20, alpha=0.85)

        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Cluster ID")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
