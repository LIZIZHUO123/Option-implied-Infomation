# FeaturePlotter.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


class FeaturePlotter:
    """
    用于绘制特征与标的资产价格双轴图的类。
    """
    
    def __init__(self, results_file: str = "results.xlsx", 
                 underlying_price_col: str = "标的收盘价", 
                 date_col: str = "日期",  # 新增日期列参数
                 date_format: str = "%Y-%m-%d",  # 新增日期格式参数
                 save_dir: str = "plots"):
        """
        初始化类实例。
        
        参数:
            results_file (str): 包含特征和标的资产价格的Excel文件路径，默认为 "results.xlsx"。
            underlying_price_col (str): 标的资产价格的列名，默认为 "标的收盘价"。
            date_col (str): 日期列的列名，默认为 "日期"。
            date_format (str): 日期列的格式，默认为 "%Y-%m-%d"。
            save_dir (str): 保存图像的目录，默认为 "plots"。
        """
        self.results_file = results_file
        self.underlying_price_col = underlying_price_col
        self.date_col = date_col
        self.date_format = date_format
        self.save_dir = save_dir
        self.data = None  # 初始化为空，后续加载数据

    def load_data(self):
        """
        加载数据。
        """
        try:
            # 读取 Excel 文件，跳过第一列（索引列），并指定日期列为日期格式
            self.data = pd.read_excel(self.results_file, sheet_name="特征指标", parse_dates=[self.date_col])
            # 将日期列格式化为指定的字符串格式
            self.data[self.date_col] = self.data[self.date_col].dt.strftime(self.date_format)
            # 将日期列重新转换为日期类型（为了绘图）
            self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])

        except Exception as e:
            print(f"读取文件时出错: {e}")
            self.data = None

    def validate_columns(self):
        """
        检查必要的列是否存在。
        """
        if self.data is None:
            print("数据未成功加载，无法进行列检查。")
            return False

        if self.underlying_price_col not in self.data.columns:
            print(f"指定的标的资产价格列名 '{self.underlying_price_col}' 不存在于数据中。")
            return False
        return True

    def plot_features(self):
        """
        绘制特征与标的资产价格的双轴图。
        """
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 获取所有特征列（除了日期列和标的资产价格列）
        feature_cols = [col for col in self.data.columns if col not in [self.date_col, self.underlying_price_col]]
        
        # 绘图
        for feature in feature_cols:
            plt.figure(figsize=(12, 6))
            
            # 左轴：特征
            ax1 = plt.gca()  # 获取当前轴
            ax1.plot(self.data[self.date_col], self.data[feature], color="blue", label=feature)
            ax1.set_xlabel("日期")
            ax1.set_ylabel(feature, color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")
            
            # 右轴：标的资产价格
            ax2 = ax1.twinx()  # 创建一个共享X轴但独立Y轴的轴
            ax2.plot(self.data[self.date_col], self.data[self.underlying_price_col], color="red", label=self.underlying_price_col, linestyle="--")
            ax2.set_ylabel(self.underlying_price_col, color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            
            # 自动调整图例
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            
            # 自动调整布局
            plt.title(f"{feature} 与 {self.underlying_price_col}")
            plt.tight_layout()
            
            # 保存图像
            plt.savefig(os.path.join(self.save_dir, f"{feature}_with_{self.underlying_price_col}.png"))
            plt.close()
        
        print(f"所有图像已保存到目录 '{self.save_dir}' 中。")

    def run(self):
        """
        运行绘图流程。
        """
        self.load_data()
        if self.data is not None and self.validate_columns():
            self.plot_features()
        else:
            print("绘图流程终止。")