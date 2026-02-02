"""
工具函数模块
"""

import numpy as np
from typing import Dict


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """
    计算预测误差指标
    
    Args:
        y_pred: 预测值
        y_true: 真实值
    
    Returns:
        {'mae': ..., 'rmse': ..., 'mape': ...}
    """
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    
    # MAPE（避免除零）
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }


def exponential_moving_average(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    指数移动平均（EMA）
    
    Args:
        x: 输入序列
        alpha: 平滑系数 (0, 1)
    
    Returns:
        平滑后序列
    """
    ema = np.zeros_like(x)
    ema[0] = x[0]
    
    for i in range(1, len(x)):
        ema[i] = alpha * x[i] + (1 - alpha) * ema[i-1]
    
    return ema


def resample_timeseries(t: np.ndarray, y: np.ndarray, 
                        t_new: np.ndarray) -> np.ndarray:
    """
    时间序列重采样（线性插值）
    
    Args:
        t: 原始时间
        y: 原始值
        t_new: 新时间网格
    
    Returns:
        插值后的值
    """
    return np.interp(t_new, t, y)
