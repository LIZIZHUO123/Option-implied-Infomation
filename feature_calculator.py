# feature_calculator.py
import math
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer

class FeatureCalculator:
    def __init__(
        self, 
        column_mapping: Dict[str, str],
        r: float,
        logger
    ):
        self.col = column_mapping
        self.r = r
        self.logger = logger

    def compute_features(
        self,
        feature_list: List[str],
        merged_data: pd.DataFrame
    ) -> pd.DataFrame:
        """主特征计算入口"""
        feature_map = {
            'VIX': self._calc_vix,
            'Skew': self._calc_skew,
            'Delta': self._calc_delta,
            'Gamma': self._calc_gamma,
            'PCRatio': self._calc_pcr,
            'TermSlope': self._calc_term_structure,
            'Vega': self._calc_vega,
            'Theta': self._calc_theta,
            'ITG': self._calc_itg,
            'ITL': self._calc_itl
        }
        
        results = []
        for feature in feature_list:
            if feature not in feature_map:
                raise ValueError(f"不支持的特征: {feature}")
            results.append(feature_map[feature](merged_data))
        
        return pd.concat(results, axis=1)
    
    
    
    def find_K0_CBOE(self, calls, puts, common_strikes, underlying_price):
        candidates = []
        
        for K in sorted(common_strikes):
            try:
                call_price = calls.loc[calls[self.col['strike']] == K, self.col['settle']].mean()
                put_price = puts.loc[puts[self.col['strike']] == K, self.col['settle']].mean()
                
                if pd.isna(call_price) or pd.isna(put_price):
                    continue
                    
                price_diff = abs(call_price - put_price)
                moneyness = abs(K - underlying_price)
                
                candidates.append({
                    'strike': K,
                    'price_diff': price_diff,
                    'moneyness': moneyness
                })
                
            except Exception as e:
                continue
        
        if not candidates:
            raise ValueError("无有效候选行权价")
        
        # 先按价差排序，再按接近平值程度排序
        candidates.sort(key=lambda x: (x['price_diff'], x['moneyness']))
        return candidates[0]['strike']

    def _calc_vix(self, merged_data: pd.DataFrame) -> pd.Series:
        """严格遵循CBOE标准的VIX计算（期权链方法）"""
        # 辅助函数：计算单个到期日的方差贡献
        def calculate_sigma_sq(maturity_group):
            try:
                # Step 1: 确定远期价格F (使用看跌-看涨平价)
                calls = maturity_group[maturity_group['type'] == 'C']
                puts = maturity_group[maturity_group['type'] == 'P']
                
                # 获取共同行权价
                common_strikes = set(calls[self.col['strike']]).intersection(puts[self.col['strike']])
                underlying_price = maturity_group[self.col['underlying_price']].iloc[0]
                if not common_strikes:
                    return np.nan

                K0 = self.find_K0_CBOE(calls, puts, common_strikes, underlying_price)
                
                # 计算远期价格
                T = maturity_group['T'].iloc[0]
                call_price = calls.loc[calls[self.col['strike']] == K0, self.col['settle']].mean()
                put_price = puts.loc[puts[self.col['strike']] == K0, self.col['settle']].mean()
                F = K0 + math.exp(self.r * T) * (call_price - put_price)

                
                '''
                # Step 2: 筛选行权价范围 (80%-120% of F)
                valid_strikes = maturity_group[
                    (maturity_group[self.col['strike']] >= 0.8 * F) & 
                    (maturity_group[self.col['strike']] <= 1.2 * F)
                ]
                '''
                valid_strikes = maturity_group
                # Step 3: 计算行权价贡献
                strikes = sorted(valid_strikes[self.col['strike']].unique())

                if len(strikes) < 2:
                    self.logger.error("len(strikes) < 2")
                    return np.nan
                    
                contributions = []
                for i, K in enumerate(strikes):
                    # 优先选择 OTM 期权
                    if K >= F:
                        option = valid_strikes[(valid_strikes[self.col['strike']] == K) & 
                                            (valid_strikes['type'] == 'C')]  # 虚值 Call
                        if option.empty:
                            # 如果 OTM Call 缺失，使用 ITM Put（同一行权价）
                            option = valid_strikes[(valid_strikes[self.col['strike']] == K) & 
                                                (valid_strikes['type'] == 'P')]
                            self.logger.warning(f"行权价 {K} 使用 ITM Put 代替 OTM Call")
                    else:
                        option = valid_strikes[(valid_strikes[self.col['strike']] == K) & 
                                            (valid_strikes['type'] == 'P')]  # 虚值 Put
                        if option.empty:
                            # 如果 OTM Put 缺失，使用 ITM Call（同一行权价）
                            option = valid_strikes[(valid_strikes[self.col['strike']] == K) & 
                                                (valid_strikes['type'] == 'C')]
                            self.logger.warning(f"行权价 {K} 使用 ITM Call 代替 OTM Put")
                    
                    if option.empty:
                        self.logger.error(f"行权价 {K} 无任何有效期权")
                        continue
                        
                    # 计算delta_K
                    if i == 0:
                        delta_K = strikes[1] - strikes[0]
                    elif i == len(strikes)-1:
                        delta_K = strikes[-1] - strikes[-2]
                    else:
                        delta_K = (strikes[i+1] - strikes[i-1]) / 2
                    
                    # 计算贡献值
                    Q = option[self.col['settle']].iloc[0]
                    contributions.append( (delta_K / K**2) * math.exp(self.r * T) * Q )
                
                # Step 4: 计算方差项
                if not contributions:
                    self.logger.error("not contributions")
                    return np.nan
                    
                sigma_sq = (2 / T) * sum(contributions) - (1 / T) * (F / K0 - 1)**2
                return sigma_sq
                
            except Exception as e:
                self.logger.error(f"计算方差时出错: {str(e)}")
                return np.nan

        # 主计算流程
        try:
            # 计算隐含波动率
            merged_data['iv'] = merged_data.apply(
                lambda x: self._implied_vol(x), 
                axis=1
            )
            '''
            # 筛选有效到期日（23-37天）
            valid = merged_data[
                merged_data['iv'].notna()) &
                merged_data['T'].between(23/365, 37/365, inclusive='both'))
            ]
            '''

            valid = merged_data
            if valid.empty:
                return pd.Series(name='VIX', dtype=float)
            
            # 按日期和到期日双层分组
            daily_results = []
            for date, date_group in valid.groupby(self.col['date']):
                try:
                    # 按到期日分组计算
                    maturity_groups = date_group.groupby(self.col['maturity'])
                    sigma_sqs = maturity_groups.apply(calculate_sigma_sq).dropna()
                    
                    if len(sigma_sqs) < 2:
                        continue
                        
                    # 获取每个到期日对应的T值并按T排序
                    Ts = date_group.groupby(self.col['maturity'])['T'].first().loc[sigma_sqs.index]
                    combined = pd.DataFrame({'sigma_sq': sigma_sqs, 'T': Ts}).sort_values('T')
                    
                    # 提取前两个最近的到期日数据
                    T1 = combined['T'].iloc[0]
                    T2 = combined['T'].iloc[1]
                    sigma_sq1 = combined['sigma_sq'].iloc[0]
                    sigma_sq2 = combined['sigma_sq'].iloc[1]
                    
                    # 计算插值权重（确保转换为年单位）
                    distance1 = abs(T1 - 30/365)
                    distance2 = abs(T2 - 30/365)
                    w = distance2 / (distance1 + distance2)
                    
                    # 插值计算30天方差
                    final_variance = w * sigma_sq1 + (1 - w) * sigma_sq2
                    
                    # 转换为VIX指数
                    vix = 100 * math.sqrt(final_variance * (365 / 30))
                    daily_results.append((date, vix))
                    
                except Exception as e:
                    self.logger.error(f"日期{date}计算失败: {str(e)}")
                    continue
            
            if not daily_results:
                return pd.Series(name='VIX', dtype=float)
                
            # 构建时间序列
            dates, vix_values = zip(*daily_results)
            return pd.Series(vix_values, index=pd.to_datetime(dates), name='VIX').sort_index()
            
        except Exception as e:
            self.logger.error(f"VIX计算失败: {str(e)}")
            return pd.Series(name='VIX', dtype=float)

    def _calc_skew(self, merged_data: pd.DataFrame) -> pd.Series:
        """CBOE式隐含偏度计算（风险中性偏度）"""
        def calculate_daily_skew(group):
            try:
                # Step 1: 确定远期价格F和K0
                calls = group[group['type'] == 'C']
                puts = group[group['type'] == 'P']
                common_strikes = set(calls[self.col['strike']]).intersection(puts[self.col['strike']])
                if not common_strikes:
                    self.logger.error(f"common_strikes do not exist")
                    return np.nan
                
                underlying_price = group[self.col['underlying_price']].iloc[0]
                K0 = self.find_K0_CBOE(calls, puts, common_strikes, underlying_price)
                
                T = group['T'].iloc[0]
                call_price = calls.loc[calls[self.col['strike']] == K0, self.col['settle']].mean()
                put_price = puts.loc[puts[self.col['strike']] == K0, self.col['settle']].mean()
                F = K0 + math.exp(self.r * T) * (call_price - put_price)

                # Step 2: 筛选OTM期权并计算贡献
                '''
                valid_strikes = group[
                    (group[self.col['strike']] >= 0.8 * F) & 
                    (group[self.col['strike']] <= 1.2 * F)
                ]
                '''

                valid_strikes = group

                strikes = sorted(valid_strikes[self.col['strike']].unique())
                
                if len(strikes) < 3:
                    return np.nan

                # 计算三阶矩（偏度）和二阶矩（方差）
                contributions_W = []  # 三阶矩贡献
                contributions_V = []  # 二阶矩贡献
                
                for i, K in enumerate(strikes):
                    # 选择OTM期权（允许ITM作为备用）
                    if K > F:
                        option = valid_strikes[(valid_strikes[self.col['strike']] == K) & 
                                            (valid_strikes['type'] == 'C')]
                        if option.empty:
                            option = valid_strikes[(valid_strikes[self.col['strike']] == K) & 
                                                (valid_strikes['type'] == 'P')]
                    elif K < F:
                        option = valid_strikes[(valid_strikes[self.col['strike']] == K) & 
                                            (valid_strikes['type'] == 'P')]
                        if option.empty:
                            option = valid_strikes[(valid_strikes[self.col['strike']] == K) & 
                                                (valid_strikes['type'] == 'C')]
                    else:
                        continue  # 跳过平值期权
                    
                    if option.empty:
                        continue
                    
                    # 计算delta_K
                    if i == 0:
                        delta_K = strikes[1] - strikes[0]
                    elif i == len(strikes)-1:
                        delta_K = strikes[-1] - strikes[-2]
                    else:
                        delta_K = (strikes[i+1] - strikes[i-1]) / 2
                    
                    Q = option[self.col['settle']].iloc[0]
                    term = delta_K / K**2 * math.exp(self.r * T)
                    
                    # 三阶矩贡献
                    ln_ratio = math.log(K/F)
                    contributions_W.append(term * Q * ln_ratio**3)
                    
                    # 二阶矩贡献
                    contributions_V.append(term * Q * ln_ratio**2)
                
                if not contributions_W or not contributions_V:
                    self.logger.error(f"contributions_W 或 contributions_V为空")
                    return np.nan
                
                W = sum(contributions_W)
                V = sum(contributions_V)
                
                # 计算偏度
                mu = math.exp(self.r * T) - 1 - 0.5 * math.exp(self.r * T) * V
                skewness = (math.exp(self.r * T) * W - 3 * mu * math.exp(self.r * T) * V + 2 * mu**3) / (V**1.5)
                
                # 转换为CBOE SKEW格式
                return 100 - 10 * skewness
            except Exception as e:
                self.logger.error(f"偏度计算失败: {str(e)}")
                return np.nan

        # 按日期分组计算
        return merged_data.groupby(self.col['date']).apply(calculate_daily_skew).rename('Skew')

    def _calc_delta(self, merged_data: pd.DataFrame) -> pd.Series:
        """Delta均值"""
        deltas = merged_data.groupby(self.col['date']).apply(
            lambda g: g.apply(self._compute_delta, axis=1).mean()
        )
        return deltas.rename('Delta')

    def _calc_gamma(self, merged_data: pd.DataFrame) -> pd.Series:
        """Gamma时间序列"""
        def compute_gamma(row):
            try:
                S = row[self.col['underlying_price']]
                K = row[self.col['strike']]
                T = row['T']
                sigma = row['iv']
                
                d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                return norm.pdf(d1) / (S * sigma * np.sqrt(T))
            except:
                return np.nan
        
        gamma = merged_data.groupby(self.col['date']).apply(
            lambda g: g.apply(compute_gamma, axis=1).mean()
        )
        return gamma.rename('Gamma')

    def _calc_pcr(self, merged_data: pd.DataFrame) -> pd.Series:
        """成交量Put-Call比率"""
        def pcr(group):
            puts = group[group['type'] == 'P'][self.col['volume']].sum()
            calls = group[group['type'] == 'C'][self.col['volume']].sum()
            return puts / (calls + 1e-6)  # 防止除零
        
        return merged_data.groupby(self.col['date']).apply(pcr).rename('PCRatio')

    def _calc_term_structure(self, merged_data: pd.DataFrame) -> pd.Series:
        """波动率期限结构斜率"""
        def term_slope(group):
            if len(group) < 2:
                return np.nan
            X = group['T'].values.reshape(-1,1)
            y = group['iv'].values
            # 检查 y 中是否存在空值
            if np.isnan(y).any():
                # 使用 KNNImputer 对 y 中的空值进行填充
                imputer = KNNImputer(n_neighbors=3)  # 选择合适的邻居数
                y = imputer.fit_transform(y.reshape(-1, 1)).flatten()

            model = LinearRegression().fit(X, y)
            return model.coef_[0]
        
        return merged_data.groupby(self.col['date']).apply(term_slope).rename('TermSlope')

    def _compute_delta(self, row) -> float:
        """修正的Delta计算"""
        try:
            S = row[self.col['underlying_price']]
            K = row[self.col['strike']]
            T = max(row['T'], 1e-4)  # 防止零除错
            sigma = row.get('iv', 0.2)  # 默认波动率
            
            d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            if row['type'] == 'C':
                return norm.cdf(d1)
            else:
                return norm.cdf(d1) - 1  # 修正看跌期权Delta公式
        except:
            return np.nan

    def _implied_vol(self, row) -> float:
        """隐含波动率计算"""
        S = row[self.col['underlying_price']]
        K = row[self.col['strike']]
        T = max(row['T'], 1/365)  # 防止零日到期
        price = row[self.col['settle']]
        option_type = row['type']
        
        def bsm(sigma):
            d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            if option_type == 'C':
                return S * norm.cdf(d1) - K * np.exp(-self.r*T) * norm.cdf(d2)
            else:
                return K * np.exp(-self.r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        try:
            return brentq(lambda s: bsm(s) - price, 0.001, 5.0, xtol=1e-4)
        except:
            # 使用市场价与理论价最小化的数值方法
            sigmas = np.linspace(0.001, 5.0, 50)
            errors = [abs(bsm(s) - price) for s in sigmas]
            return sigmas[np.argmin(errors)]
        
    def _interpolate_vix(self, group: pd.DataFrame) -> float:
        """波动率插值计算"""
        if len(group) == 0:
            return np.nan
            
        # 提取时间与波动率数据
        times = group['minutes_to_expiry'].values / (1440 * 365.25)  # 转换为年
        ivs = group['iv'].values
        
        # 构建插值函数
        from scipy.interpolate import interp1d
        try:
            f = interp1d(times, ivs, kind='linear', fill_value='extrapolate')
            
            # 计算30天（约0.082年）的波动率
            iv_30 = f(30/365)
            
            # 返回年化波动率
            return 100 * iv_30 * np.sqrt(365.25/30)
        except:
            # 插值失败时使用最近合约
            closest = group.iloc[np.argmin(np.abs(times - 30/365))]
            return 100 * closest['iv'] * np.sqrt(closest['T'] / (30/365))

    def _calc_vega(self, merged_data: pd.DataFrame) -> pd.Series:
        """Vega风险暴露"""
        def compute_vega(row):
            try:
                S = row[self.col['underlying_price']]
                K = row[self.col['strike']]
                T = row['T']
                sigma = row['iv']
                
                d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                return S * np.sqrt(T) * norm.pdf(d1)  # 标准Vega公式（每波动率百分点）
            except:
                return np.nan
        
        vega = merged_data.groupby(self.col['date']).apply(
            lambda g: g.apply(compute_vega, axis=1).mean()
        )
        return vega.rename('Vega')

    def _calc_theta(self, merged_data: pd.DataFrame) -> pd.Series:
        """时间衰减"""
        def compute_theta(row):
            try:
                S = row[self.col['underlying_price']]
                K = row[self.col['strike']]
                T = row['T']
                sigma = row['iv']
                r = self.r
                
                d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                
                term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                if row['type'] == 'C':
                    term2 = - r * K * np.exp(-r*T) * norm.cdf(d2)
                else:
                    term2 = r * K * np.exp(-r*T) * norm.cdf(-d2)
                
                return (term1 + term2) / 365  # 转换为每日theta
            except:
                return np.nan
        
        theta = merged_data.groupby(self.col['date']).apply(
            lambda g: g.apply(compute_theta, axis=1).mean()
        )
        return theta.rename('Theta')
    
    def compute_theta(self, row) -> float:
        """计算单个期权的Theta值"""
        try:
            S = row[self.col['underlying_price']]
            K = row[self.col['strike']]
            T = max(row['T'], 1e-4)
            sigma = row.get('iv', 0.2)
            r = self.r
            
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            if row['type'] == 'C':
                term2 = - r * K * np.exp(-r*T) * norm.cdf(d2)
            else:
                term2 = r * K * np.exp(-r*T) * norm.cdf(-d2)
            
            return (term1 + term2) / 365  # 转换为每日theta
        except:
            return np.nan

    def _calc_itg(self, merged_data: pd.DataFrame) -> pd.Series:
        """隐含尾部收益因子"""
        # 获取列名映射
        date_col = self.col['date']
        strike_col = self.col['strike']
        underlying_price_col = self.col['underlying_price']
        settle_col = self.col['settle']
        volume_col = self.col['volume']
        type_col = 'type'  # 假设type列名正确
        
        # 计算每个期权的Theta
        merged_data['Theta'] = merged_data.apply(self.compute_theta, axis=1)
        
        # 筛选虚值看涨期权（行权价>标的价格，且Theta < 0.8）
        '''
        call_condition = (
            (merged_data[type_col] == 'C') & 
            (merged_data[strike_col] > merged_data[underlying_price_col]) & 
            (merged_data['Theta'] < 0.8)
        )
        '''
        call_condition = (
            (merged_data[type_col] == 'C') & 
            (merged_data[strike_col] > merged_data[underlying_price_col])
        )

        call_options = merged_data[call_condition].copy()
        
        if call_options.empty:
            self.logger.warning("无有效虚值看涨期权数据")
            return pd.Series(name='ITG', dtype=float)
        
        results = []
        for date, group in call_options.groupby(date_col):
            try:
                # 提取必要数据
                strikes = group[strike_col].values
                market_prices = group[settle_col].values
                volumes = group[volume_col].values
                S = group[underlying_price_col].iloc[0]
                
                # 确定参考行权价K（中位数）
                median_idx = np.argsort(strikes)[len(strikes) // 2]
                K = strikes[median_idx]
                C_K = market_prices[median_idx]
                
                # 筛选K_i > K的期权
                valid = strikes > K
                strikes = strikes[valid]
                market_prices = market_prices[valid]
                volumes = volumes[valid]
                
                if len(strikes) == 0:
                    continue
                
                # 计算权重（成交量占比）
                total_volume = volumes.sum()
                if total_volume <= 0:
                    continue
                weights = volumes / total_volume
                
                # 定义优化目标函数
                def objective(params):
                    xi, beta = params
                    theoretical = C_K * (xi / beta * (strikes - K) + 1) ** (1 - 1/xi)
                    errors = market_prices - theoretical
                    return np.sum(weights * (errors ** 2))
                
                # 参数优化
                initial_guess = [0.01, 1.0]
                bounds = [(1e-10, None), (1e-10, None)]
                result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
                
                if not result.success:
                    self.logger.warning(f"日期 {date} 参数优化失败: {result.message}")
                    continue
                
                xi_hat, beta_hat = result.x
                
                # 计算ITG因子
                itg = beta_hat / ((1 - xi_hat) * S)
                results.append((date, itg))
                
            except Exception as e:
                self.logger.error(f"日期 {date} 计算异常: {str(e)}")
                continue
        
        if not results:
            return pd.Series(name='ITG', dtype=float)
        
        # 转换为时间序列
        dates, itg_values = zip(*results)
        return pd.Series(itg_values, index=pd.to_datetime(dates), name='ITG').sort_index()
    
    def _calc_itl(self, merged_data: pd.DataFrame) -> pd.Series:
        """隐含尾部损失因子（针对虚值看跌期权）"""
        # 获取列名映射
        date_col = self.col['date']
        strike_col = self.col['strike']
        underlying_price_col = self.col['underlying_price']
        settle_col = self.col['settle']
        volume_col = self.col['volume']
        type_col = 'type'

        '''
        # 筛选虚值看跌期权（行权价<标的价格，且Theta > -0.8）
        put_condition = (
            (merged_data[type_col] == 'P') & 
            (merged_data[strike_col] < merged_data[underlying_price_col]) & 
            (merged_data['Theta'] > -0.8)  # 使用计算好的Theta列
        )

        '''
        put_condition = (
            (merged_data[type_col] == 'P') & 
            (merged_data[strike_col] < merged_data[underlying_price_col])
        )
        

        put_options = merged_data[put_condition].copy()

        if put_options.empty:
            self.logger.warning("无有效虚值看跌期权数据")
            return pd.Series(name='ITL', dtype=float)

        results = []
        for date, group in put_options.groupby(date_col):
            try:
                # 提取必要数据
                strikes = group[strike_col].values
                market_prices = group[settle_col].values
                volumes = group[volume_col].values
                S = group[underlying_price_col].iloc[0]

                # 确定参考行权价K（中位数）
                median_idx = np.argsort(strikes)[len(strikes) // 2]
                K = strikes[median_idx]
                P_K = market_prices[median_idx]

                # 筛选K_i < K的期权
                valid = strikes < K
                strikes = strikes[valid]
                market_prices = market_prices[valid]
                volumes = volumes[valid]

                if len(strikes) == 0:
                    continue

                # 计算权重（成交量占比）
                total_volume = volumes.sum()
                if total_volume <= 0:
                    continue
                weights = volumes / total_volume

                # 定义优化目标函数（调整理论价格公式）
                def objective(params):
                    xi, beta = params
                    theoretical = P_K * (xi / beta * (K - strikes) + 1) ** (1 - 1/xi)
                    errors = market_prices - theoretical
                    return np.sum(weights * (errors ** 2))

                # 参数优化
                initial_guess = [0.01, 1.0]
                bounds = [(1e-10, None), (1e-10, None)]
                result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

                if not result.success:
                    self.logger.warning(f"日期 {date} 参数优化失败: {result.message}")
                    continue

                xi_hat, beta_hat = result.x

                # 计算ITL因子（公式形式与ITG相同）
                itl = beta_hat / ((1 - xi_hat) * S)
                results.append((date, itl))

            except Exception as e:
                self.logger.error(f"日期 {date} 计算异常: {str(e)}")
                continue

        if not results:
            return pd.Series(name='ITL', dtype=float)

        # 转换为时间序列
        dates, itl_values = zip(*results)
        return pd.Series(itl_values, index=pd.to_datetime(dates), name='ITL').sort_index()