# OptionPreprocessor.py
import os
import glob
import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm, skew
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
import math
from scipy.optimize import minimize_scalar

class OptionPreprocessor:
    """独立预处理模块"""
    def __init__(self, exchange: str, column_mapping: Dict[str, str]):
        self.exchange = exchange.lower()
        self.col = column_mapping
        self.logger = logging.getLogger('Preprocessor')
        self._init_patterns()
        self._init_parsers()

    def _init_patterns(self):
        """初始化各交易所的正则模式"""
        self.patterns = {
            'sse': {
                'strike': r'A0(\d{5})$',  # 匹配510050C1803A02750中的02750
                'maturity': r'[A-Z](\d{2})(\d{2})[A-Z]',  # 匹配1803中的月份
                'type': r'([CP])\d'  # 匹配C或P
            },
            'dce': {
                'strike': r'-(\d+)$',  # 匹配M2405-C-2750中的2750
                'maturity': r'(\d{4})-[CP]-',  # 匹配2405
                'type': r'-([CP])-'  # 匹配C或P
            }
        }

    def _init_parsers(self):
        """初始化解析方法"""
        self.parsers = {
            'strike': self._parse_strike,
            'maturity': self._parse_maturity,
            'type': self._parse_type
        }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行完整预处理流程"""
        # 字段存在性检查与解析
        for field in ['type','strike', 'maturity']:
            if self.col[field] not in df.columns:
                self.logger.info(f"解析缺失字段: {field}")
                df = self.parsers[field](df)
        return self._post_process(df)

    def _parse_strike(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析执行价格"""
        pattern = self.patterns[self.exchange]['strike']
        extract = df[self.col['option_code']].str.extract(pattern)
        
        # 执行价格转换逻辑
        if self.exchange == 'sse':
            df[self.col['strike']] = extract[0].astype(float) / 1000  # 假设格式为02750→27.50
        elif self.exchange == 'dce':
            df[self.col['strike']] = extract[0].astype(float)
        return df

    def _parse_maturity(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析到期日"""
        pattern = self.patterns[self.exchange]['maturity']
        extract = df[self.col['option_code']].str.extract(pattern)
        
        if self.exchange == 'sse':
            # 上交所格式示例：1803 → 2018年3月
            df['contract_month'] = pd.to_datetime(
                '20' + extract[0] + extract[1], 
                format='%Y%m'
            )
            df[self.col['maturity']] = df['contract_month'].apply(
                lambda x: self._sse_maturity(x)
            )
        elif self.exchange == 'dce':
            # 大商所格式示例：2405 → 2024年5月
            df['contract_month'] = pd.to_datetime(
                extract[0], 
                format='%y%m'
            )
            df[self.col['maturity']] = df['contract_month'].apply(
                lambda x: self._dce_maturity(x)
            )
        return df.drop(columns=['contract_month'])

    def _parse_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析期权类型"""
        pattern = self.patterns[self.exchange]['type']
        df['type'] = df[self.col['option_code']].str.extract(pattern)
        return df

    @staticmethod
    def _sse_maturity(contract_month: pd.Timestamp) -> pd.Timestamp:
        """上交所实际到期日计算（第四个周三）"""
        return nth_weekday(contract_month, 4, 2)

    @staticmethod
    def _dce_maturity(contract_month: pd.Timestamp) -> pd.Timestamp:
        """商品期权实际到期日计算（前月第五个交易日）"""
        prev_month = contract_month - pd.DateOffset(months=1)
        return get_nth_trading_day(prev_month, 5)

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """后处理：计算剩余期限"""

        df[self.col['maturity']] = pd.to_datetime(df[self.col['maturity']], errors='coerce')
        df[self.col['date']] = pd.to_datetime(df[self.col['date']], errors='coerce')
        
        df['T'] = (df[self.col['maturity']] - 
                  df[self.col['date']]).dt.days / 365.25
        return df

# 工具函数 --------------------------------------------------
def nth_weekday(year_month: pd.Timestamp, n: int, weekday: int) -> pd.Timestamp:
    """计算某月第n个星期几"""
    first_day = year_month.replace(day=1)
    adjust = (weekday - first_day.weekday()) % 7
    return first_day + pd.Timedelta(days=adjust + 7*(n-1))

def get_nth_trading_day(month: pd.Timestamp, n: int = 5) -> pd.Timestamp:
    """获取某月第n个交易日（简化版）"""
    return pd.date_range(
        start=month.replace(day=1),
        end=month + pd.offsets.MonthEnd(1),
        freq='B'
    )[n-1]
