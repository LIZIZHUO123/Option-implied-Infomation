## option-implied-information
本项目意在探究期权隐含信息与其标的资产价格的关联，包含数据清洗、特征计算、可视化、策略回测四大功能。

## 示例数据来源：Wind
示例数据包括来自上海证券交易所的股指ETF合约、大连商品交易所的商品期货期权

## 文件说明
### main.py 主程序
### OptionPreprocessor.py 数据清洗
### feature_calculator.py 特征计算
实现以下期权隐含特征计算，核心逻辑如下：

#### VIX (波动率指数)
- 借鉴CBOE计算方法，采用近月与次近月合约插值
- 关键步骤：
  1. 通过看跌-看涨平价确定远期价格F
  2. 筛选行权价范围（理论为F的80%-120%）
  3. 计算各行权价的方差贡献量
  4. 加权合成30日波动率预期
  5. 最终公式：VIX = 100 × √(σ² × 365/30)

#### Skew (风险中性偏度)
- 基于CBOE SKEW指数计算方法
- 通过OTM期权价格计算三阶矩
- 转换公式：SKEW = 100 - 10 × 偏度
- 偏度计算：
  - W = Σ[Q(K) × (ln(K/F))³ × ΔK/K²]
  - V = Σ[Q(K) × (ln(K/F))² × ΔK/K²]
  - 偏度 = (e^{rT}W - 3μV + 2μ³)/V^{1.5}

#### Delta (方向暴露)
- 采用BSM模型Delta公式
- 看涨期权：Δ = N(d1)
- 看跌期权：Δ = N(d1) - 1
- 全市场合约Delta均值

#### Gamma (凸性暴露)
- BSM模型Gamma公式：
  Γ = N'(d1)/(Sσ√T)
- 反映Delta对标的资产价格的敏感度

#### PCRatio (量能情绪)
- Put-Call成交量比率
- 计算公式：PCR = Put成交量 / Call成交量

#### TermSlope (期限结构)
- 隐含波动率期限结构斜率
- 通过线性回归计算：iv ~ β×T
- 使用KNN填充缺失值后拟合

#### Vega (波动率暴露)
- BSM模型Vega公式：
  ν = S√T × N'(d1)
- 衡量波动率变动1%对期权价格的影响

#### Theta (时间衰减)
- BSM模型Theta公式：
  - 看涨：Θ = [-SσN'(d1)/(2√T)] - rKe^{-rT}N(d2)
  - 看跌：Θ = [-SσN'(d1)/(2√T)] + rKe^{-rT}N(-d2)
- 转换为每日时间衰减值

#### ITG (隐含尾部收益因子)
- 针对虚值看涨期权计算，定价公式见参考文献
- 通过优化拟合期权价格与行权价之间的关系，得到参数估计xi_hat和beta_hat
- 最终计算公式：ITG(t) = beta_hat(t) / ((1 - xi_hat(t)) × S(t))
- 该因子反映了市场对标的资产极端上涨收益的预期
- 参考文献：The information content of option-implied tail risk on the future
returns of the underlying asset

#### ITL (隐含尾部损失因子)
- 针对虚值看跌期权计算，定价公式见参考文献
- 通过优化拟合期权价格与行权价之间的关系，得到参数估计xi_hat和beta_hat
- 最终计算公式：ITL = beta_hat(t) / ((1 - xi_hat(t)) × S(t))
- 该因子反映了市场对标的资产极端下跌损失的预期
- 参考文献：The information content of option-implied tail risk on the future
returns of the underlying asset

### FeaturePlotter.py 可视化
可视化各特征与标的资产价格走势

### strategy_backtest.py 策略回测
择时策略框架：
1. 信号生成：基于特征阈值生成交易信号
2. 仓位管理：
   - signal=1: 全仓做多
   - signal=-1: 全仓做空 
   - signal=0: 平仓至现金
3. 扩展支持：
   - 金字塔加仓机制
   - 动态头寸调整
   - 风险管理模块

### Enhanced_Backtest.py 增强回测
- 加入二次特征衍生
- 加入机器学习模型生成信号
