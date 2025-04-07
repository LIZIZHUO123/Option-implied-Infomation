# strategy_backtest.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
from typing import List, Dict
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import math

class OptionBacktester:
    def __init__(
        self,
        data_path: str,
        date_col: str = '日期',
        price_col: str = '标的收盘价',
        features: List[str] = None,
        benchmark_col: str = None,
        fee_rate: float = 0.0003,
        slippage: float = 0.0001,
        initial_capital: float = 1e6,
        allow_short: bool = True
    ):
        """
        初始化回测器
        Parameters:
            data_path: 数据文件路径
            date_col: 日期列名
            price_col: 标的资产价格列名
            features: 需要使用的特征列名列表
            benchmark_col: 基准价格列名，默认使用标的资产价格
            fee_rate: 单边交易手续费率
            slippage: 滑点比例
            initial_capital: 初始本金
        """
        # 数据加载
        self.data = pd.read_excel(data_path, parse_dates=[date_col]).set_index(date_col)
        self.price_col = price_col
        self.benchmark_col = benchmark_col if benchmark_col else price_col
        self.features = features if features else [c for c in self.data.columns if c not in [price_col]]
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.allow_short = allow_short
        
        # 预处理
        self._prepare_data()
        
    def _prepare_data(self):
        """数据预处理"""
        # 计算标的收益率
        self.data['benchmark_ret'] = self.data[self.benchmark_col].pct_change()
        # 前向填充特征数据
        self.data[self.features] = self.data[self.features].ffill()
        # 删除无效数据
        self.data = self.data.dropna(subset=['benchmark_ret']+self.features)
        
    def generate_signal(
        self,
        factor_rules: Dict[str, Dict],
        combine_method: str = 'all'
    ) -> pd.DataFrame:
        """
        生成交易信号
        Parameters:
            factor_rules: 因子信号规则字典，格式为：
                {
                    '因子名': {
                        'type': 'quantile'/'std',  # 分位数类型或标准差类型
                        'window': 滚动窗口长度，None表示全历史,
                        'threshold': (lower, upper) 阈值,
                        'direction': 1/-1  # 因子方向，1表示因子越大越看多
                    },
                    ...
                }
            combine_method: 多因子信号组合方式，'all'需要所有信号一致，'any'任一信号触发
        """
        signals = pd.DataFrame(index=self.data.index)
        
        # 为每个因子生成信号
        for factor, config in factor_rules.items():
            if factor not in self.features:
                raise ValueError(f"数据中不存在因子: {factor}")
                
            # 计算阈值
            if config['type'] == 'quantile':
                lower_q, upper_q = config['threshold']
                if config['window']:
                    # 滚动分位数
                    signals[f'{factor}_lower'] = self.data[factor].shift(1).rolling(config['window']).quantile(lower_q)
                    signals[f'{factor}_upper'] = self.data[factor].shift(1).rolling(config['window']).quantile(upper_q)
                else:
                    # 全历史分位数
                    signals[f'{factor}_lower'] = self.data[factor].shift(1).expanding().quantile(lower_q)
                    signals[f'{factor}_upper'] = self.data[factor].shift(1).expanding().quantile(upper_q)
                
                # 生成原始信号
                raw_signal = np.zeros(len(self.data))
                raw_signal[self.data[factor] > signals[f'{factor}_upper']] = 1
                raw_signal[self.data[factor] < signals[f'{factor}_lower']] = -1
                
            elif config['type'] == 'std':
                n_std_lower, n_std_upper = config['threshold']
                if config['window']:
                    # 滚动标准差
                    mean = self.data[factor].shift(1).rolling(config['window']).mean()
                    std = self.data[factor].shift(1).rolling(config['window']).std()
                else:
                    # 全历史统计
                    mean = self.data[factor].shift(1).expanding().mean()
                    std = self.data[factor].shift(1).expanding().std()
                
                signals[f'{factor}_lower'] = mean + n_std_lower * std
                signals[f'{factor}_upper'] = mean + n_std_upper * std
                
                # 生成原始信号
                raw_signal = np.zeros(len(self.data))
                raw_signal[self.data[factor] > signals[f'{factor}_upper']] = 1
                raw_signal[self.data[factor] < signals[f'{factor}_lower']] = -1
                
            else:
                raise ValueError("不支持的信号类型，可选 'quantile' 或 'std'")
            
            # 处理因子方向
            raw_signal *= config.get('direction', 1)
            signals[factor] = raw_signal
        
        # 组合信号
        if len(factor_rules) > 1:
            if combine_method == 'all':
                # 所有信号同方向时触发
                combined = (signals[self.features].sum(axis=1) / len(factor_rules)).map(
                    lambda x: 1 if x >= 1 else (-1 if x <= -1 else 0)
                )
            elif combine_method == 'any':
                # 任一信号触发
                combined = signals[self.features].sum(axis=1).map(
                    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
                )
            else:
                raise ValueError("combine_method 应为 'all' 或 'any'")
            signals['combined'] = combined
        else:
            signals['combined'] = signals[self.features[0]]
            
        return signals

    def backtest(
        self,
        signals: pd.Series,
        stop_loss: float = None,
        take_profit: float = None
    ) -> pd.DataFrame:
        """
        执行回测
        Parameters:
            signals: 交易信号序列，取值为1(做多), -1(做空), 0(空仓)
            stop_loss: 止损比例，如0.05表示5%止损
            take_profit: 止盈比例
        Returns:
            回测结果DataFrame
        """
        # 初始化持仓数据
        positions = pd.DataFrame(index=signals.index, columns=[
            'position', 'entry_price', 'stop_price', 'take_price'
        ])
        positions['position'] = 0
        positions['capital'] = self.initial_capital

        # 添加基准收益率列
        positions['benchmark_ret'] = self.data['benchmark_ret']
        
        # 在回测前处理信号
        if not self.allow_short:
            signals = signals.replace(-1, 0)  # 将做空信号转为空仓

        # 交易循环
        for i in range(1, len(positions)):
            current_date = positions.index[i]
            prev_date = positions.index[i-1]
            
            # 当前信号
            signal = signals.loc[current_date]
            price = self.data[self.price_col].loc[current_date]
            
            # 处理止损止盈
            if positions.loc[prev_date, 'position'] != 0:
                if stop_loss and price <= positions.loc[prev_date, 'stop_price']:
                    signal = 0
                elif take_profit and price >= positions.loc[prev_date, 'take_price']:
                    signal = 0
            
            # 计算仓位变化
            if signal != positions.loc[prev_date, 'position']:
                # 计算交易价格（考虑滑点）
                trade_price = price * (1 + self.slippage) if signal == 1 else price * (1 - self.slippage)
                # 计算手续费
                fee = positions.loc[prev_date, 'capital'] * self.fee_rate
                positions.loc[current_date, 'capital'] = positions.loc[prev_date, 'capital'] - fee
                # 更新持仓
                positions.loc[current_date, 'position'] = signal
                positions.loc[current_date, 'entry_price'] = trade_price
                positions.loc[current_date, 'stop_price'] = trade_price * (1 - stop_loss) if stop_loss else None
                positions.loc[current_date, 'take_price'] = trade_price * (1 + take_profit) if take_profit else None
            else:
                # 持仓不变，计算收益
                if positions.loc[prev_date, 'position'] != 0:
                    ret_pct = (price / self.data[self.price_col].loc[prev_date] - 1)
                    positions.loc[current_date, 'capital'] = positions.loc[prev_date, 'capital'] * (1 + ret_pct * positions.loc[prev_date, 'position'])
                else:
                    positions.loc[current_date, 'capital'] = positions.loc[prev_date, 'capital']
                positions.loc[current_date, 'position'] = positions.loc[prev_date, 'position']
        
        # 计算净值
        positions['strategy_ret'] = positions['capital'].pct_change()
        positions['strategy_ret'].iloc[0] = 0
        return positions

    def analyze_performance(self, result_df: pd.DataFrame) -> Dict:
        """绩效分析"""
        # 策略累计收益
        strategy_cum = (1 + result_df['strategy_ret']).cumprod()
        # 基准累计收益
        benchmark_cum = (1 + result_df['benchmark_ret']).cumprod()
        
        # 计算最大回撤
        max_drawdown = (strategy_cum / strategy_cum.cummax() - 1).min()
        
        # 年化收益
        annual_ret = strategy_cum.iloc[-1] ** (252/len(result_df)) - 1
        
        # 波动率
        annual_vol = result_df['strategy_ret'].std() * np.sqrt(252)
        
        # 夏普比率
        sharpe = annual_ret / annual_vol if annual_vol !=0 else np.nan
        
        # 创建一个 DataFrame 包含策略净值和基准净值
        performance_df = pd.DataFrame({
            '策略净值': strategy_cum,
            '基准净值': benchmark_cum
        })
        
        # 创建一个字典包含其他绩效指标
        performance_metrics = {
            '累计收益': strategy_cum.iloc[-1] - 1,
            '年化收益': annual_ret,
            '最大回撤': max_drawdown,
            '夏普比率': sharpe
        }
        
        return performance_metrics, performance_df
        
    def plot_distribution(self, features: List[str] = None, figsize=(12, 8)):
        """绘制因子分布图"""
        features = features if features else self.features
        n_cols = 2
        n_rows = math.ceil(len(features)/n_cols)
        
        plt.figure(figsize=figsize)
        for i, col in enumerate(features, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(self.data[col], kde=True)
            plt.title(f'{col}分布')
        plt.tight_layout()
        plt.show()
    
    def plot_results(self, result_df: pd.DataFrame, figsize=(14, 8)):
        """绘制回测结果"""
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # 净值曲线
        ax1.plot(result_df['策略净值'], label='策略净值', color='tab:red')
        ax1.plot(result_df['基准净值'], label='基准净值', color='tab:blue')
        ax1.set_ylabel('净值')
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}倍'.format(y)))
        ax1.legend(loc='upper left')
        
        # 回撤曲线
        ax2 = ax1.twinx()
        drawdown = (result_df['策略净值'] / result_df['策略净值'].cummax() - 1)
        ax2.fill_between(result_df.index, drawdown, 0, alpha=0.3, color='tab:grey')
        ax2.set_ylabel('回撤')
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.title('策略表现')
        plt.show()
