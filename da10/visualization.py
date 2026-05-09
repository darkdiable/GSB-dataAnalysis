import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import seaborn as sns
import os

sns.set_style("whitegrid")
sns.set_palette("husl")

def get_chinese_font():
    font_paths = [
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/Songti.ttc',
        '/System/Library/Fonts/Kai.ttc',
        '/Library/Fonts/SimHei.ttf',
        '/Library/Fonts/Arial Unicode.ttf',
    ]
    
    selected_font_path = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            selected_font_path = font_path
            break
    
    font_names = ['STHeiti', 'Heiti TC', 'Songti SC', 'PingFang HK', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
    
    plt.rcParams['font.sans-serif'] = font_names + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    if selected_font_path:
        try:
            font_prop = fm.FontProperties(fname=selected_font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            return font_prop
        except:
            pass
    
    for font_name in font_names:
        try:
            font_prop = fm.FontProperties(family=font_name)
            plt.rcParams['font.family'] = font_name
            return font_prop
        except:
            continue
    
    return None

chinese_font = get_chinese_font()

def set_chinese_labels(ax, title=None, xlabel=None, ylabel=None):
    if chinese_font:
        if title:
            ax.set_title(title, fontproperties=chinese_font, fontsize=14, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel, fontproperties=chinese_font, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontproperties=chinese_font, fontsize=12)
    else:
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)

def set_chinese_suptitle(fig, title):
    if chinese_font:
        fig.suptitle(title, fontproperties=chinese_font, fontsize=16, fontweight='bold')
    else:
        fig.suptitle(title, fontsize=16, fontweight='bold')


def ensure_output_dir(output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_time_series(df, output_dir='output', save_png=True, save_pdf=False):
    output_dir = ensure_output_dir(output_dir)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    set_chinese_suptitle(fig, '近3年逐日气象数据时间序列')
    fig.subplots_adjust(top=0.96)
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    labels = ['温度 (°C)', '湿度 (%)', '降水量 (mm)', '风速 (m/s)']
    columns = ['temperature', 'humidity', 'precipitation', 'wind_speed']
    
    for i, (ax, col, color, label) in enumerate(zip(axes, columns, colors, labels)):
        ax.plot(df['date'], df[col], color=color, alpha=0.7, linewidth=0.8)
        
        rolling_window = 30
        rolling_mean = df[col].rolling(window=rolling_window, center=True).mean()
        ax.plot(df['date'], rolling_mean, color=color, linewidth=2, label=f'{rolling_window}天移动平均')
        
        set_chinese_labels(ax, ylabel=label)
        
        if chinese_font:
            ax.legend(loc='upper right', prop=chinese_font)
        else:
            ax.legend(loc='upper right')
        
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    saved_files = []
    if save_png:
        png_path = os.path.join(output_dir, 'time_series.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        saved_files.append(png_path)
    if save_pdf:
        pdf_path = os.path.join(output_dir, 'time_series.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        saved_files.append(pdf_path)
    
    plt.close()
    print(f"时间序列图已保存: {saved_files}")
    return saved_files


def plot_monthly_boxplots(df, output_dir='output', save_png=True, save_pdf=False):
    output_dir = ensure_output_dir(output_dir)
    
    month_names = ['1月', '2月', '3月', '4月', '5月', '6月', 
                   '7月', '8月', '9月', '10月', '11月', '12月']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    set_chinese_suptitle(fig, '各月气象指标分布箱线图')
    fig.subplots_adjust(top=0.95)
    
    plot_configs = [
        ('temperature', '温度 (°C)', 'Reds', axes[0, 0]),
        ('humidity', '湿度 (%)', 'Blues', axes[0, 1]),
        ('precipitation', '降水量 (mm)', 'Greens', axes[1, 0]),
        ('wind_speed', '风速 (m/s)', 'Oranges', axes[1, 1])
    ]
    
    for col, ylabel, palette, ax in plot_configs:
        sns.boxplot(x='month', y=col, data=df, ax=ax, palette=palette, showfliers=False)
        
        set_chinese_labels(ax, xlabel='月份', ylabel=ylabel)
        
        if chinese_font:
            ax.set_xticklabels(month_names, fontproperties=chinese_font)
        else:
            ax.set_xticklabels(month_names)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    saved_files = []
    if save_png:
        png_path = os.path.join(output_dir, 'monthly_boxplots.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        saved_files.append(png_path)
    if save_pdf:
        pdf_path = os.path.join(output_dir, 'monthly_boxplots.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        saved_files.append(pdf_path)
    
    plt.close()
    print(f"月度箱线图已保存: {saved_files}")
    return saved_files


def plot_correlation_heatmap(correlation_matrix, output_dir='output', save_png=True, save_pdf=False):
    output_dir = ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = ['温度', '湿度', '降水量', '风速']
    corr_matrix = correlation_matrix.copy()
    corr_matrix.columns = labels
    corr_matrix.index = labels
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    cbar_kws = {'shrink': 0.8}
    
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True, 
                linewidths=0.5, 
                cbar_kws=cbar_kws,
                ax=ax,
                vmin=-1,
                vmax=1,
                mask=mask,
                fmt='.3f',
                annot_kws={'fontproperties': chinese_font} if chinese_font else {})
    
    if chinese_font:
        ax.set_title('气象变量相关系数热力图', fontproperties=chinese_font, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticklabels(labels, fontproperties=chinese_font, fontsize=11)
        ax.set_yticklabels(labels, fontproperties=chinese_font, fontsize=11)
        
        cbar = ax.collections[0].colorbar
        cbar.set_label('相关系数', fontproperties=chinese_font, fontsize=12)
        for t in cbar.ax.get_yticklabels():
            t.set_fontproperties(chinese_font)
    else:
        ax.set_title('气象变量相关系数热力图', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    saved_files = []
    if save_png:
        png_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        saved_files.append(png_path)
    if save_pdf:
        pdf_path = os.path.join(output_dir, 'correlation_heatmap.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        saved_files.append(pdf_path)
    
    plt.close()
    print(f"相关系数热力图已保存: {saved_files}")
    return saved_files


def plot_temperature_humidity_dual_axis(df, output_dir='output', save_png=True, save_pdf=False):
    output_dir = ensure_output_dir(output_dir)
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    monthly_avg = df.groupby(['year', 'month']).agg({
        'temperature': 'mean',
        'humidity': 'mean'
    }).reset_index()
    
    monthly_avg['date'] = pd.to_datetime(
        monthly_avg['year'].astype(str) + '-' + monthly_avg['month'].astype(str) + '-01'
    )
    monthly_avg = monthly_avg.sort_values('date')
    
    color_temp = '#e74c3c'
    set_chinese_labels(ax1, xlabel='日期', ylabel='月平均温度 (°C)')
    ax1.tick_params(axis='y', labelcolor=color_temp)
    
    line1 = ax1.plot(monthly_avg['date'], monthly_avg['temperature'], 
                     color=color_temp, linewidth=2, marker='o', markersize=6, label='温度')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color_humid = '#3498db'
    
    if chinese_font:
        ax2.set_ylabel('月平均湿度 (%)', fontproperties=chinese_font, fontsize=12, color=color_humid)
    else:
        ax2.set_ylabel('月平均湿度 (%)', fontsize=12, color=color_humid)
    ax2.tick_params(axis='y', labelcolor=color_humid)
    
    line2 = ax2.plot(monthly_avg['date'], monthly_avg['humidity'], 
                     color=color_humid, linewidth=2, marker='s', markersize=6, label='湿度')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    if chinese_font:
        ax1.legend(lines, labels, loc='upper left', prop=chinese_font)
    else:
        ax1.legend(lines, labels, loc='upper left')
    
    set_chinese_labels(ax1, title='月平均温度与湿度双轴对比图')
    
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    saved_files = []
    if save_png:
        png_path = os.path.join(output_dir, 'temp_humid_dual_axis.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        saved_files.append(png_path)
    if save_pdf:
        pdf_path = os.path.join(output_dir, 'temp_humid_dual_axis.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        saved_files.append(pdf_path)
    
    plt.close()
    print(f"温湿双轴图已保存: {saved_files}")
    return saved_files


def plot_extreme_weather_summary(extreme_summary, output_dir='output', save_png=True, save_pdf=False):
    output_dir = ensure_output_dir(output_dir)
    
    event_names_cn = {
        'high_temp': '高温(≥35°C)',
        'low_temp': '低温(≤0°C)',
        'heavy_wind': '大风(≥15m/s)',
        'heavy_rain': '强降水(≥50mm)'
    }
    
    events = list(extreme_summary.keys())
    counts = [extreme_summary[e]['total_count'] for e in events]
    event_labels = [event_names_cn[e] for e in events]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
    x_pos = np.arange(len(event_labels))
    bars = axes[0].bar(x_pos, counts, color=colors, alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{int(height)}',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    set_chinese_labels(axes[0], title='近3年极端天气事件统计', ylabel='事件次数')
    
    axes[0].set_xticks(x_pos)
    if chinese_font:
        axes[0].set_xticklabels(event_labels, fontproperties=chinese_font, rotation=15, ha='right')
    else:
        axes[0].set_xticklabels(event_labels, rotation=15, ha='right')
    
    axes[0].grid(True, alpha=0.3, axis='y')
    
    yearly_data = {}
    years = set()
    for event in events:
        yearly_data[event] = extreme_summary[event]['yearly_count']
        years.update(yearly_data[event].keys())
    
    years = sorted(years)
    x = np.arange(len(years))
    width = 0.2
    
    for i, (event, color) in enumerate(zip(events, colors)):
        counts_year = [yearly_data[event].get(year, 0) for year in years]
        axes[1].bar(x + i * width, counts_year, width, 
                    label=event_names_cn[event], color=color, alpha=0.7)
    
    set_chinese_labels(axes[1], title='各年极端天气事件分布', xlabel='年份', ylabel='事件次数')
    
    axes[1].set_xticks(x + width * 1.5)
    axes[1].set_xticklabels([str(y) for y in years])
    
    if chinese_font:
        axes[1].legend(prop=chinese_font)
    else:
        axes[1].legend()
    
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    saved_files = []
    if save_png:
        png_path = os.path.join(output_dir, 'extreme_weather_summary.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        saved_files.append(png_path)
    if save_pdf:
        pdf_path = os.path.join(output_dir, 'extreme_weather_summary.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        saved_files.append(pdf_path)
    
    plt.close()
    print(f"极端天气统计图已保存: {saved_files}")
    return saved_files


def run_visualization(df, analysis_results, output_dir='output', save_png=True, save_pdf=False):
    print("\n开始生成可视化图表...")
    
    if chinese_font:
        print(f"已找到中文字体: {chinese_font.get_name()}")
    else:
        print("警告: 未找到中文字体，图表可能无法正确显示中文")
    
    all_saved_files = []
    
    print("\n1. 生成时间序列图...")
    files = plot_time_series(df, output_dir, save_png, save_pdf)
    all_saved_files.extend(files)
    
    print("\n2. 生成月度箱线图...")
    files = plot_monthly_boxplots(df, output_dir, save_png, save_pdf)
    all_saved_files.extend(files)
    
    print("\n3. 生成相关系数热力图...")
    files = plot_correlation_heatmap(analysis_results['correlation_matrix'], output_dir, save_png, save_pdf)
    all_saved_files.extend(files)
    
    print("\n4. 生成温湿双轴对比图...")
    files = plot_temperature_humidity_dual_axis(df, output_dir, save_png, save_pdf)
    all_saved_files.extend(files)
    
    print("\n5. 生成极端天气统计图...")
    files = plot_extreme_weather_summary(analysis_results['extreme_summary'], output_dir, save_png, save_pdf)
    all_saved_files.extend(files)
    
    print(f"\n可视化完成！共生成 {len(all_saved_files)} 个文件")
    
    return all_saved_files


if __name__ == '__main__':
    from data_loader import generate_sample_data
    from preprocess import preprocess_data
    from analysis import run_analysis
    
    df = generate_sample_data()
    df_processed = preprocess_data(df)
    results = run_analysis(df_processed)
    run_visualization(df_processed, results, save_png=True, save_pdf=False)
