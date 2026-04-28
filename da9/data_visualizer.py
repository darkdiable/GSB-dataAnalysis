import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties, findfont, FontManager
import seaborn as sns
import os

# 全局字体属性
_chinese_font = None
_selected_font_name = None

def setup_chinese_font():
    """
    设置matplotlib中文显示 - 最终修复版
    """
    global _chinese_font, _selected_font_name
    
    # ===== 重要：首先设置风格，这必须在字体设置之前 =====
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # macOS常见中文字体列表（按优先级排序）
    font_candidates = [
        'PingFang SC',           # macOS系统默认中文字体
        'STHeiti',                # 华文黑体
        'Heiti TC',               # 黑体-繁
        'Heiti SC',               # 黑体-简
        'Hiragino Sans GB',       # 冬青黑体
        'Arial Unicode MS',       # Arial Unicode
        'Microsoft YaHei',        # 微软雅黑（如果安装了Office）
        'SimHei',                 # 黑体
        'Songti SC',              # 宋体
        'KaiTi',                  # 楷体
    ]
    
    # 查找可用的中文字体
    fm = FontManager()
    available_fonts = [f.name for f in fm.ttflist]
    
    selected_font = None
    for font_name in font_candidates:
        if font_name in available_fonts:
            selected_font = font_name
            print(f"找到中文字体: {font_name}")
            break
    
    # 如果没有找到，尝试直接查找字体文件
    if selected_font is None:
        # macOS常见的中文字体文件路径
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    _chinese_font = FontProperties(fname=font_path)
                    print(f"使用字体文件: {font_path}")
                    _selected_font_name = os.path.basename(font_path)
                    break
                except Exception as e:
                    print(f"加载字体文件失败: {font_path}")
                    continue
    
    # ===== 设置matplotlib参数（在style之后） =====
    if selected_font:
        # 方法1: 使用rcParams设置全局字体
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = [selected_font] + rcParams['font.sans-serif']
        rcParams['axes.unicode_minus'] = False
        rcParams['figure.figsize'] = (12, 8)
        rcParams['figure.dpi'] = 100
        rcParams['font.size'] = 12
        _selected_font_name = selected_font
        print(f"已设置字体: {selected_font}")
    elif _chinese_font is not None:
        # 有字体文件，通过FontProperties使用
        rcParams['axes.unicode_minus'] = False
        rcParams['figure.figsize'] = (12, 8)
        rcParams['figure.dpi'] = 100
        rcParams['font.size'] = 12
        print(f"已设置字体文件: {_selected_font_name}")
    else:
        # 最后尝试：使用DejaVu Sans（可能不支持中文，但至少不会报错）
        print("警告: 未找到中文字体，图表中的中文可能显示为方框")
        rcParams['font.family'] = 'DejaVu Sans'
        rcParams['axes.unicode_minus'] = False
        rcParams['figure.figsize'] = (12, 8)
        rcParams['figure.dpi'] = 100
        rcParams['font.size'] = 12


def get_chinese_font(size=12):
    """
    获取中文字体属性对象
    """
    global _chinese_font, _selected_font_name
    if _chinese_font:
        return FontProperties(fname=_chinese_font.get_file(), size=size)
    elif _selected_font_name:
        return FontProperties(family=[_selected_font_name], size=size)
    return FontProperties(size=size)


def set_axes_text_font(ax, fontsize=12):
    """
    为坐标轴的所有文本元素设置中文字体
    """
    font = get_chinese_font(fontsize)
    title_font = get_chinese_font(fontsize + 2)
    
    # 设置标题
    title = ax.get_title()
    if title:
        ax.set_title(title, fontproperties=title_font, fontweight='bold')
    
    # 设置x轴标签
    xlabel = ax.get_xlabel()
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=font)
    
    # 设置y轴标签
    ylabel = ax.get_ylabel()
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=font)
    
    # 设置刻度标签
    for label in ax.get_xticklabels():
        label.set_fontproperties(font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font)
    
    return ax


class DataVisualizer:
    """
    数据可视化类，生成折线图、柱状图、饼图、热力图
    """
    
    def __init__(self, analysis_results, output_dir='da9/output'):
        self.analysis_results = analysis_results
        self.output_dir = output_dir
        self.generated_charts = []
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        setup_chinese_font()
    
    def plot_monthly_sales_trend(self):
        """
        绘制月度销售额趋势图（折线图）
        """
        print("\n绘制月度销售额趋势图...")
        
        if 'monthly_sales' not in self.analysis_results:
            print("错误：未找到月度销售数据")
            return None
        
        monthly_data = self.analysis_results['monthly_sales']['monthly_data'].copy()
        
        # 获取中文字体
        font_title = get_chinese_font(16)
        font_label = get_chinese_font(12)
        font_legend = get_chinese_font(10)
        font_annot = get_chinese_font(10)
        
        # 创建图表
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # 绘制销售额折线
        color1 = '#1f77b4'
        ax1.set_xlabel('月份', fontproperties=font_label)
        ax1.set_ylabel('销售额 (元)', color=color1, fontproperties=font_label)
        line1 = ax1.plot(monthly_data['order_month'], monthly_data['销售额'], 
                         color=color1, marker='o', linewidth=2, markersize=8)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # 设置y轴刻度格式
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{x:,.0f}'))
        
        # 创建第二个y轴显示订单数
        ax2 = ax1.twinx()
        color2 = '#ff7f0e'
        ax2.set_ylabel('订单数', color=color2, fontproperties=font_label)
        line2 = ax2.bar(monthly_data['order_month'], monthly_data['订单数'], 
                        color=color2, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 设置标题（显式使用中文字体）
        ax1.set_title('月度销售额与订单数趋势', fontproperties=font_title, fontweight='bold', pad=20)
        
        # 添加数据标签（销售额）
        for i, val in enumerate(monthly_data['销售额']):
            ax1.annotate(f'{val:,.0f}', 
                        (i, val), 
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center', 
                        fontproperties=font_annot)
        
        # 旋转x轴标签
        plt.xticks(rotation=45, ha='right')
        
        # 添加图例
        ax1.legend([line1[0], line2], ['销售额', '订单数'], loc='upper left', prop=font_legend)
        
        # 为所有刻度标签设置字体
        for label in ax1.get_xticklabels():
            label.set_fontproperties(font_annot)
        for label in ax1.get_yticklabels():
            label.set_fontproperties(font_annot)
        for label in ax2.get_yticklabels():
            label.set_fontproperties(font_annot)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, '1_月度销售额趋势.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.generated_charts.append(output_path)
        print(f"图表已保存: {output_path}")
        
        return output_path
    
    def plot_rfm_segmentation(self):
        """
        绘制RFM用户分层图（柱状图）
        """
        print("\n绘制RFM用户分层图...")
        
        if 'rfm' not in self.analysis_results:
            print("错误：未找到RFM分析数据")
            return None
        
        segment_stats = self.analysis_results['rfm']['segment_stats'].copy()
        
        # 获取中文字体
        font_suptitle = get_chinese_font(16)
        font_title = get_chinese_font(14)
        font_label = get_chinese_font(12)
        font_legend = get_chinese_font(10)
        font_annot = get_chinese_font(10)
        font_tick = get_chinese_font(10)
        
        # 按用户数排序
        segment_stats = segment_stats.sort_values('用户数', ascending=True)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 子图1：用户数分布（横向柱状图）
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(segment_stats)))
        bars = ax1.barh(segment_stats['Segment'], segment_stats['用户数'], color=colors, alpha=0.8)
        
        ax1.set_xlabel('用户数', fontproperties=font_label)
        ax1.set_title('RFM用户分层 - 用户数分布', fontproperties=font_title, fontweight='bold', pad=15)
        
        # 添加数据标签
        for bar in bars:
            width = bar.get_width()
            ax1.annotate(f'{int(width)}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0),
                        textcoords='offset points',
                        va='center',
                        fontproperties=font_annot)
        
        # 子图2：平均消费金额和消费次数
        x = np.arange(len(segment_stats))
        width = 0.35
        
        bars2 = ax2.bar(x - width/2, segment_stats['平均消费金额'], width, 
                        label='平均消费金额', color='#1f77b4', alpha=0.8)
        bars3 = ax2.bar(x + width/2, segment_stats['平均消费次数'], width, 
                        label='平均消费次数', color='#ff7f0e', alpha=0.8)
        
        ax2.set_xlabel('用户分层', fontproperties=font_label)
        ax2.set_ylabel('数值', fontproperties=font_label)
        ax2.set_title('RFM用户分层 - 消费特征', fontproperties=font_title, fontweight='bold', pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(segment_stats['Segment'], fontproperties=font_tick, rotation=45, ha='right')
        ax2.legend(prop=font_legend)
        
        # 为所有刻度标签设置字体
        for label in ax1.get_xticklabels():
            label.set_fontproperties(font_tick)
        for label in ax1.get_yticklabels():
            label.set_fontproperties(font_tick)
        for label in ax2.get_yticklabels():
            label.set_fontproperties(font_tick)
        
        # 总标题
        plt.suptitle('RFM用户分层分析', fontproperties=font_suptitle, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, '2_RFM用户分层.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.generated_charts.append(output_path)
        print(f"图表已保存: {output_path}")
        
        return output_path
    
    def plot_category_contribution(self):
        """
        绘制产品类别贡献度图（饼图）
        """
        print("\n绘制产品类别贡献度图...")
        
        if 'category_contribution' not in self.analysis_results:
            print("错误：未找到类别贡献度数据")
            return None
        
        category_stats = self.analysis_results['category_contribution']['category_stats'].copy()
        
        # 获取中文字体
        font_suptitle = get_chinese_font(16)
        font_title = get_chinese_font(14)
        font_label = get_chinese_font(12)
        font_pie = get_chinese_font(11)
        font_annot = get_chinese_font(10)
        font_tick = get_chinese_font(10)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 子图1：销售额占比饼图
        # 合并小类别为"其他"
        threshold = 5  # 小于5%的合并为其他
        mask = category_stats['销售额占比'] >= threshold
        main_categories = category_stats[mask].copy()
        other_sales = category_stats[~mask]['销售额'].sum()
        other_share = category_stats[~mask]['销售额占比'].sum()
        
        if other_sales > 0:
            other_row = pd.DataFrame({
                'category': ['其他'],
                '销售额': [other_sales],
                '销售额占比': [other_share]
            })
            main_categories = pd.concat([main_categories, other_row], ignore_index=True)
        
        # 绘制饼图
        colors = plt.cm.Set3(np.linspace(0, 1, len(main_categories)))
        explode = [0.05] + [0] * (len(main_categories) - 1)  # 突出显示第一个
        
        # 关键：为pie图显式设置字体属性
        wedges, texts, autotexts = ax1.pie(
            main_categories['销售额'],
            labels=main_categories['category'],
            colors=colors,
            autopct='%1.1f%%',
            explode=explode,
            startangle=90,
            textprops={'fontproperties': font_pie},
            pctdistance=0.85
        )
        
        # 为百分比文本也设置字体
        for autotext in autotexts:
            autotext.set_fontproperties(font_pie)
        
        # 添加中心圆（环形图效果）
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax1.add_artist(centre_circle)
        
        ax1.set_title('产品类别销售额占比', fontproperties=font_title, fontweight='bold', pad=15)
        ax1.axis('equal')  # 确保饼图是圆形
        
        # 子图2：类别销售额柱状图
        bars = ax2.bar(category_stats['category'], category_stats['销售额'], 
                       color=plt.cm.viridis(np.linspace(0.2, 0.8, len(category_stats))),
                       alpha=0.8)
        
        ax2.set_xlabel('产品类别', fontproperties=font_label)
        ax2.set_ylabel('销售额 (元)', fontproperties=font_label)
        ax2.set_title('各产品类别销售额', fontproperties=font_title, fontweight='bold', pad=15)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{x:,.0f}'))
        
        # 设置x轴刻度标签字体
        ax2.set_xticklabels(category_stats['category'], fontproperties=font_tick, rotation=45, ha='right')
        
        # 为所有刻度标签设置字体
        for label in ax2.get_yticklabels():
            label.set_fontproperties(font_tick)
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center',
                        fontproperties=font_annot)
        
        # 总标题
        plt.suptitle('产品类别贡献度分析', fontproperties=font_suptitle, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, '3_产品类别贡献度.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.generated_charts.append(output_path)
        print(f"图表已保存: {output_path}")
        
        return output_path
    
    def plot_price_quantity_heatmap(self):
        """
        绘制价格-销量相关性热力图
        """
        print("\n绘制价格-销量相关性热力图...")
        
        if 'price_quantity_correlation' not in self.analysis_results:
            print("错误：未找到价格-销量相关性数据")
            return None
        
        product_stats = self.analysis_results['price_quantity_correlation']['product_stats'].copy()
        price_range_stats = self.analysis_results['price_quantity_correlation']['price_range_stats'].copy()
        
        # 获取中文字体
        font_suptitle = get_chinese_font(16)
        font_title = get_chinese_font(14)
        font_label = get_chinese_font(12)
        font_legend = get_chinese_font(10)
        font_annot = get_chinese_font(9)
        font_tick = get_chinese_font(10)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 子图1：价格区间统计（堆叠柱状图）
        price_order = ['低价 (<50)', '中低价 (50-200)', '中高价 (200-500)', '高价 (500-2000)', '超高价 (≥2000)']
        price_range_stats = price_range_stats.set_index('价格区间').reindex(price_order).reset_index()
        
        x = np.arange(len(price_range_stats))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, price_range_stats['产品数'], width,
                        label='产品数', color='#1f77b4', alpha=0.8)
        bars2 = ax1.bar(x + width/2, price_range_stats['平均销量'], width,
                        label='平均销量', color='#ff7f0e', alpha=0.8)
        
        ax1.set_xlabel('价格区间', fontproperties=font_label)
        ax1.set_ylabel('数量', fontproperties=font_label)
        ax1.set_title('不同价格区间的产品数与平均销量', fontproperties=font_title, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(price_range_stats['价格区间'], fontproperties=font_tick, rotation=45, ha='right')
        ax1.legend(prop=font_legend)
        
        # 为y轴刻度设置字体
        for label in ax1.get_yticklabels():
            label.set_fontproperties(font_tick)
        
        # 添加数据标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.0f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords='offset points',
                            ha='center',
                            fontproperties=font_annot)
        
        # 子图2：价格-销量散点图与热力图
        # 准备数据用于热力图
        # 创建价格和销量的分箱
        price_bins = pd.qcut(product_stats['unit_price'], q=5, duplicates='drop')
        quantity_bins = pd.qcut(product_stats['总销量'], q=5, duplicates='drop')
        
        # 创建交叉表
        cross_tab = pd.crosstab(price_bins.astype(str), quantity_bins.astype(str))
        
        # 绘制热力图 - 使用fontproperties设置annot的字体
        heatmap = sns.heatmap(cross_tab, 
                             annot=True, 
                             fmt='d', 
                             cmap='YlGnBu',
                             ax=ax2,
                             cbar_kws={'label': '产品数量'},
                             annot_kws={'fontproperties': font_annot})
        
        # 设置标题和标签
        ax2.set_xlabel('销量区间', fontproperties=font_label)
        ax2.set_ylabel('价格区间', fontproperties=font_label)
        ax2.set_title('价格-销量分布热力图', fontproperties=font_title, fontweight='bold', pad=15)
        
        # 为x轴和y轴刻度标签设置字体
        for label in ax2.get_xticklabels():
            label.set_fontproperties(font_tick)
        for label in ax2.get_yticklabels():
            label.set_fontproperties(font_tick)
        
        # 设置colorbar的标签字体
        cbar = ax2.collections[0].colorbar
        cbar.ax.yaxis.label.set_fontproperties(font_label)
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(font_tick)
        
        # 旋转x轴标签
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 添加相关系数文本
        corr_data = self.analysis_results['price_quantity_correlation']['overall_correlation']
        corr_text = f"价格-销量相关系数: {corr_data['价格-销量相关系数']:.4f}\n"
        corr_text += f"价格-订单数相关系数: {corr_data['价格-订单数相关系数']:.4f}"
        
        ax1.text(0.02, 0.98, corr_text, transform=ax1.transAxes, 
                fontproperties=font_annot, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 总标题
        plt.suptitle('价格与销量相关性分析', fontproperties=font_suptitle, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, '4_价格与销量相关性.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.generated_charts.append(output_path)
        print(f"图表已保存: {output_path}")
        
        return output_path
    
    def plot_all(self):
        """
        绘制所有图表
        """
        print("\n" + "="*50)
        print("开始数据可视化")
        print("="*50)
        
        self.plot_monthly_sales_trend()
        self.plot_rfm_segmentation()
        self.plot_category_contribution()
        self.plot_price_quantity_heatmap()
        
        print("\n" + "="*50)
        print("数据可视化完成")
        print("="*50)
        print(f"共生成 {len(self.generated_charts)} 个图表:")
        for chart in self.generated_charts:
            print(f"  - {chart}")
        
        return self.generated_charts
    
    def get_generated_charts(self):
        """
        获取生成的图表列表
        """
        return self.generated_charts


if __name__ == '__main__':
    from data_generator import load_datasets
    from data_cleaner import DataCleaner
    from data_analyzer import DataAnalyzer
    
    # 完整流程测试
    print("="*60)
    print("电商销售分析系统 - 可视化测试")
    print("="*60)
    
    # 加载并清洗数据
    users, products, orders, order_details = load_datasets()
    cleaner = DataCleaner(users, products, orders, order_details)
    cleaned_users, cleaned_products, cleaned_orders, cleaned_order_details = cleaner.clean_all()
    
    # 执行分析
    analyzer = DataAnalyzer(cleaned_users, cleaned_products, cleaned_orders, cleaned_order_details)
    results = analyzer.run_all_analyses()
    
    # 生成可视化
    visualizer = DataVisualizer(results)
    charts = visualizer.plot_all()
