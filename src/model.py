"""
SOC-Thermal-Network 耦合动力学模型（核心实现）
基于 model_v1.md 的完整方程组

单位体系（SI）：
- 时间 t: 秒 (s)
- 温度 T: 开尔文 (K)
- 功率 P: 瓦 (W)
- 能量 E: 焦耳 (J)
"""

import numpy as np
from typing import Dict, Tuple, Optional
import yaml


class SOCThermalNetworkModel:
    """
    三状态耦合模型：SOC-温度-网络尾态
    
    状态变量：
    - s(t): SOC (%)
    - T(t): 核心温度 (K)
    - z(t): 网络尾态/滞后状态
    """
    
    def __init__(self, config_path: str = None, params: Dict = None):
        """
        初始化模型参数
        
        Args:
            config_path: YAML 配置文件路径
            params: 参数字典（若提供，优先于 config_path）
        """
        if params is not None:
            self.params = params
        elif config_path is not None:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.params = yaml.safe_load(f)
        else:
            raise ValueError("Must provide either config_path or params")
        
        # 预处理：将 eV 转换为 J
        E_a_eV = self.params['battery']['internal_resistance']['E_a']
        k_B_eV = self.params['battery']['internal_resistance']['k_B']
        self.E_a_J = E_a_eV * 1.602e-19  # J
        self.k_B_J = k_B_eV * 1.602e-19  # J/K
        
        # 电池容量换算 Wh -> J
        self.E_cap_J = 3600.0 * self.params['battery']['E_nom_Wh']
    
    # ==================== 核心机制函数 ====================
    
    def thermal_throttling_factor(self, T: float) -> float:
        """
        温控系数 φ(T)（Thermal Throttling Factor）
        
        φ(T) = 1 / (1 + exp[λ(T - T_thresh)])
        
        Args:
            T: 当前温度 (K)
        
        Returns:
            φ ∈ (0,1]，接近 0 表示强抑制
        """
        if self.params['baseline']['no_thermal_throttling']:
            return 1.0  # Baseline: 无温控反馈
        
        T_thresh = self.params['thermal']['T_thresh']
        lam = self.params['thermal']['lambda']
        
        exponent = lam * (T - T_thresh)
        # 防止数值溢出
        exponent = np.clip(exponent, -50, 50)
        
        return 1.0 / (1.0 + np.exp(exponent))
    
    def internal_resistance(self, T: float, s: float) -> float:
        """
        电池内阻（Arrhenius 关系 + SOC 非线性）
        
        R_int(T,s) = R_int(T) * [1 + c*(s_ref/(s+ε) - 1)]
        
        Args:
            T: 当前温度 (K)
            s: 当前 SOC (%)
        
        Returns:
            内阻 (Ω)
        """
        ir_params = self.params['battery']['internal_resistance']
        R_ref = ir_params['R_ref']
        T_ref = ir_params['T_ref']
        
        # Arrhenius 温度依赖
        exponent = (self.E_a_J / self.k_B_J) * (1.0 / T - 1.0 / T_ref)
        exponent = np.clip(exponent, -50, 50)
        R_temp = R_ref * np.exp(exponent)
        
        # SOC 非线性因子（低电量内阻增大）
        c_soc = ir_params['c_soc']
        s_ref = ir_params['s_ref']
        epsilon = ir_params['epsilon']
        
        soc_factor = 1.0 + c_soc * (s_ref / (s + epsilon) - 1.0)
        
        return R_temp * soc_factor
    
    # ==================== 功率分解函数 ====================
    
    def power_screen(self, u_on: float, b: float) -> float:
        """
        屏幕功耗 P_scr = u_on * (α_0 + α_1 * b)
        
        Args:
            u_on: 屏幕开关 (0/1)
            b: 亮度 (0-1)
        
        Returns:
            屏幕功耗 (W)
        """
        scr_params = self.params['power']['screen']
        return u_on * (scr_params['alpha_0'] + scr_params['alpha_1'] * b)
    
    def power_cpu(self, u_cpu: float) -> float:
        """
        CPU/GPU 功耗 P_cpu = β_1 * u_cpu + β_2 * u_cpu^2
        
        Args:
            u_cpu: CPU 负载 (0-1)
            3*u_cpu: 频率
        
        Returns:
            CPU 功耗 (W)
        """
        cpu_params = self.params['power']['cpu']
        '''k_cpu = cpu_params['k_cpu']
        k_cpu = float(np.asarray(k_cpu).reshape(-1)[0])'''
        
        return cpu_params['k_cpu'] * (3*u_cpu)**3
    
    
    def power_network(self, z: float, S_signal: float = 1.0, net_type: str = 'cellular') -> float:
        """
        网络功耗 P_net = (1/S_signal) * (P_idle + k_net * z)
        
        Args:
            z: 网络尾态
            S_signal: 信号强度 (0-1]，1.0为满信号，小于1时耗电增加
            net_type: 网络类型 ('wifi', 'cellular', 'none')
        
        Returns:
            网络功耗 (W)
        """
        # 根据网络类型选取对应的参数
        net_params = self.params['power']['network'].get(net_type, self.params['power']['network']['cellular'])
        P_base_net = net_params['P_idle'] + net_params['k_net'] * z
        
        # 信号强度修正：信号差时功耗增大
        return P_base_net / max(S_signal, 0.1)  # 防止除零
    
    def power_system(self, u_on: float, b: float, u_cpu: float, 
                    z: float, T: float, S_signal: float = 1.0, net_type: str = 'cellular') -> float:
        """
        系统总功耗 P_sys(t)
        
        P_sys = P_base + φ(T)*(P_scr + P_cpu) + P_net(z, S_signal, net_type)
        
        Args:
            u_on, b, u_cpu: 输入负载
            z: 网络尾态
            T: 当前温度 (K)
            S_signal: 信号强度
            net_type: 网络类型
        
        Returns:
            系统功耗 (W)
        """
        P_base = self.params['power']['P_base']
        P_scr = self.power_screen(u_on, b)
        P_cpu = self.power_cpu(u_cpu)
        P_net = self.power_network(z, S_signal, net_type)
        
        phi = self.thermal_throttling_factor(T)
        
        # 判断网络是否受温控抑制（备择假设）
        if self.params['baseline']['network_in_throttle']:
            return P_base + phi * (P_scr + P_cpu + P_net)
        else:
            return P_base + phi * (P_scr + P_cpu) + P_net
    
    def power_loss(self, P_sys: float, T: float, s: float) -> float:
        """
        电池内阻损耗 P_loss = I^2 * R_int(T, s)
        
        其中 I ≈ P_sys / V_nom
        
        Args:
            P_sys: 系统功耗 (W)
            T: 当前温度 (K)
            s: 当前 SOC (%)
        
        Returns:
            内阻损耗 (W)
        """
        V_nom = self.params['battery']['V_nom']
        R_int = self.internal_resistance(T, s)
        
        I = P_sys / V_nom
        return I**2 * R_int
    
    def power_heat(self, P_sys: float, P_loss: float) -> float:
        """
        总产热功率 P_heat ≈ P_sys + P_loss
        
        Args:
            P_sys: 系统功耗 (W)
            P_loss: 内阻损耗 (W)
        
        Returns:
            产热功率 (W)
        """
        return P_sys + P_loss
    
    def capacity_factor(self, T: float) -> float:
        """
        温度对电池容量的影响因子 η_temp(T)
        
        使用倒钟形曲线：
        - T_optimal 时容量最大（因子=1.0）
        - T < T_low 时容量显著下降
        - T > T_high 时容量略有下降
        
        Args:
            T: 当前温度 (K)
        
        Returns:
            容量保留率 (0-1]
        """
        cap_params = self.params['battery']['capacity_temperature']
        T_opt = cap_params['T_optimal']
        T_low = cap_params['T_low']
        T_high = cap_params['T_high']
        f_low = cap_params['capacity_factor_low']
        f_high = cap_params['capacity_factor_high']

        t = T - 273.15

        t_clamped = max(T_low-273.15, min(t, T_high-273.15))

        # 3. 计算绝对容量C_t
        C_t = (36.716 * (t_clamped**3) - 
               826.14 * (t_clamped**2) +
               5988.6 * t_clamped -
               4204.4)
        # 4 寻找参考值（比如25摄氏度为标准100%容量）
        t_ref = 25.0
        C_ref = (36.716 * (t_ref**3) - 
               826.14 * (t_ref**2) +
               5988.6 * t_ref -
               4204.4)
        
        # 5 返回容量保留率
        factor = C_t / C_ref

        # 6 最终安全门限：即使公式失败，保留率也不应低于10%
        return max(0.1, min(factor, 1.05))
        ''' if T < T_opt:
            # 低温段：线性插值
            if T <= T_low:
                return f_low
            else:
                return f_low + (1.0 - f_low) * (T - T_low) / (T_opt - T_low)
        else:
            # 高温段：线性插值
            if T >= T_high:
                return f_high
            else:
                return 1.0 - (1.0 - f_high) * (T - T_opt) / (T_high - T_opt)'''
       
    
    # ==================== 状态演化方程（微分方程右端）====================
    
    def dz_dt(self, z: float, x_net: float, net_type: str = 'cellular') -> float:
        """
        网络尾态方程 dz/dt = (x_net - z) / τ_net
        
        Args:
            z: 当前网络尾态
            x_net: 当前网络强度（吞吐/包率）
            net_type: 网络类型 ('wifi', 'cellular', 'none')
        
        Returns:
            dz/dt
        """
        if self.params['baseline']['no_network_tail']:
            # Baseline: 无尾态，直接跟随
            return 0.0  # z 将直接设为 x_net
        
        # 根据网络类型选取对应的tau_net
        tau_net = self.params['network_tail'].get(net_type, self.params['network_tail']['cellular'])['tau_net']
        return (x_net - z) / tau_net
    
    def dT_dt(self, T: float, P_heat: float, T_env: float) -> float:
        """
        热力学方程 dT/dt = (P_heat - hA*(T - T_env)) / C_eq
        
        Args:
            T: 当前温度 (K)
            P_heat: 产热功率 (W)
            T_env: 环境温度 (K)
        
        Returns:
            dT/dt (K/s)
        """
        C_eq = self.params['thermal']['C_eq']
        hA = self.params['thermal']['hA']
        
        return (P_heat - hA * (T - T_env)) / C_eq
    
    def ds_dt(self, P_sys: float, P_loss: float, T: float, s: float) -> float:
        """
        SOC 方程 ds/dt = -100 * (P_sys + P_loss) / E_cap(T,H)
        
        包含：
        1. SOH (H) 影响：电池老化后有效容量变小
        2. 温度影响：低温下容量下降
        
        Args:
            P_sys: 系统功耗 (W)
            P_loss: 内阻损耗 (W)
            T: 当前温度 (K)
            s: 当前 SOC (%)
        
        Returns:
            ds/dt (%/s)
        """
        # 温度对容量的影响
        eta_temp = self.capacity_factor(T)
        
        # SOH 对容量的影响
        H = self.params['battery']['H']
        
        # 有效容量 = 名义容量 * 温度因子 * 健康度
        E_cap_effective = self.E_cap_J * eta_temp * H
        
        return -100.0 * (P_sys + P_loss) / E_cap_effective
    
    # ==================== 单步更新（显式 Euler）====================
    
    def step(self, state: Dict, inputs: Dict, dt: float) -> Dict:
        """
        单步时间更新（显式 Euler 方法）
        
        Args:
            state: 当前状态 {'s': SOC (%), 'T': 温度 (K), 'z': 网络尾态}
            inputs: 外部输入 {'u_on', 'b', 'u_cpu', 'x_net', 'T_env', 'S_signal', 'net_type'}
            dt: 时间步长 (s)
        
        Returns:
            新状态字典 + 中间诊断量（φ, P_sys, P_loss, P_heat）
        """
        s, T, z = state['s'], state['T'], state['z']
        u_on = inputs['u_on']
        b = inputs['b']
        u_cpu = inputs['u_cpu']
        x_net = inputs['x_net']
        T_env = inputs.get('T_env', self.params['thermal']['T_env'])
        S_signal = inputs.get('S_signal', self.params['power']['network']['S_signal_default'])
        net_type = inputs.get('net_type', self.params['power']['network']['default_type'])
        
        # 1. 更新网络尾态
        if self.params['baseline']['no_network_tail']:
            z_new = x_net  # Baseline: 直接跟随
        else:
            dz = self.dz_dt(z, x_net, net_type)
            z_new = z + dt * dz
            z_new = max(z_new, self.params['constraints']['z_min'])
        
        # 2. 计算温控系数
        phi = self.thermal_throttling_factor(T)
        
        # 3. 计算功耗
        P_sys = self.power_system(u_on, b, u_cpu, z, T, S_signal, net_type)
        P_loss = self.power_loss(P_sys, T, s)
        P_heat = self.power_heat(P_sys, P_loss)
        
        # 4. 更新温度
        dT = self.dT_dt(T, P_heat, T_env)
        T_new = T + dt * dT
        
        # 5. 更新 SOC
        ds = self.ds_dt(P_sys, P_loss, T, s)
        s_new = s + dt * ds
        s_new = np.clip(s_new, 
                       self.params['constraints']['s_min'],
                       self.params['constraints']['s_max'])
        
        # 返回新状态 + 诊断量
        return {
            's': s_new,
            'T': T_new,
            'z': z_new,
            # 诊断量（用于可视化与分析）
            'phi': phi,
            'P_sys': P_sys,
            'P_loss': P_loss,
            'P_heat': P_heat
        }
    
    # ==================== 应用接口：预测 Time to Empty ====================
    
    def predict_time_to_empty(self, 
                              initial_state: Dict,
                              scenario_inputs: callable,
                              dt: float = 1.0,
                              max_time: float = 36000.0,
                              soc_threshold: float = 0.0) -> Tuple[float, Dict]:
        """
        预测"到空"时间（Time to Empty, TTE）
        
        Args:
            initial_state: 初始状态 {'s', 'T', 'z'}
            scenario_inputs: 场景输入函数 f(t) -> {'u_on', 'b', 'u_cpu', 'x_net', 'T_env'}
            dt: 时间步长 (s)
            max_time: 最大仿真时间 (s)
            soc_threshold: SOC 阈值 (%)，低于此视为空电
        
        Returns:
            (time_to_empty, trajectory) 
            - time_to_empty: 到空时间 (s)，若未到空则返回 max_time
            - trajectory: 完整轨迹字典（用于分析）
        """
        state = initial_state.copy()
        trajectory = {
            't': [0.0],
            's': [state['s']],
            'T': [state['T']],
            'z': [state['z']],
            'phi': [],
            'P_sys': [],
            'P_loss': [],
            'P_heat': []
        }
        
        t = 0.0
        while t < max_time and state['s'] > soc_threshold:
            inputs = scenario_inputs(t)
            result = self.step(state, inputs, dt)
            
            # 更新状态
            state['s'] = result['s']
            state['T'] = result['T']
            state['z'] = result['z']
            
            # 记录轨迹
            t += dt
            trajectory['t'].append(t)
            trajectory['s'].append(result['s'])
            trajectory['T'].append(result['T'])
            trajectory['z'].append(result['z'])
            trajectory['phi'].append(result['phi'])
            trajectory['P_sys'].append(result['P_sys'])
            trajectory['P_loss'].append(result['P_loss'])
            trajectory['P_heat'].append(result['P_heat'])
        
        # 转换为 numpy 数组（便于后续处理）
        for key in trajectory:
            trajectory[key] = np.array(trajectory[key])
        
        return t, trajectory
    
    # ==================== 批量仿真接口 ====================
    
    def simulate(self, 
                initial_state: Dict,
                input_sequence: Dict,
                dt: Optional[float] = None) -> Dict:
        """
        批量仿真（给定完整输入时间序列）
        
        Args:
            initial_state: 初始状态 {'s', 'T', 'z'}
            input_sequence: 输入序列字典，每个键对应数组
                           {'t', 'u_on', 'b', 'u_cpu', 'x_net', 'T_env'}
            dt: 时间步长 (s)，若为 None 则从 input_sequence['t'] 推断
        
        Returns:
            完整轨迹字典
        """
        t_array = input_sequence['t']
        N = len(t_array)
        
        # 初始化输出
        trajectory = {
            't': t_array.copy(),
            's': np.zeros(N),
            'T': np.zeros(N),
            'z': np.zeros(N),
            'phi': np.zeros(N),
            'P_sys': np.zeros(N),
            'P_loss': np.zeros(N),
            'P_heat': np.zeros(N)
        }
        
        # 设置初值
        state = initial_state.copy()
        trajectory['s'][0] = state['s']
        trajectory['T'][0] = state['T']
        trajectory['z'][0] = state['z']
        
        # 时间步迭代
        for i in range(1, N):
            if dt is None:
                dt_i = t_array[i] - t_array[i-1]
            else:
                dt_i = dt
            
            inputs = {
                'u_on': input_sequence['u_on'][i-1],
                'b': input_sequence['b'][i-1],
                'u_cpu': input_sequence['u_cpu'][i-1],
                'x_net': input_sequence['x_net'][i-1],
                'T_env': input_sequence.get('T_env', [self.params['thermal']['T_env']])[i-1]
            }
            
            result = self.step(state, inputs, dt_i)
            
            # 更新状态
            state['s'] = result['s']
            state['T'] = result['T']
            state['z'] = result['z']
            
            # 记录
            trajectory['s'][i] = result['s']
            trajectory['T'][i] = result['T']
            trajectory['z'][i] = result['z']
            trajectory['phi'][i] = result['phi']
            trajectory['P_sys'][i] = result['P_sys']
            trajectory['P_loss'][i] = result['P_loss']
            trajectory['P_heat'][i] = result['P_heat']
        
        return trajectory
