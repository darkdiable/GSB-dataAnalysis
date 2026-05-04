# 气象数据分析与可视化系统

## 项目简介

本项目是一个完整的Python数据分析与可视化项目，用于分析某城市近3年逐日气象数据（温度、湿度、降水量、风速），揭示季节性规律及极端天气趋势。

## 技术栈

- **Python 3.x**
- **pandas** (>=1.3.0) - 数据处理与分析
- **matplotlib** (>=3.4.0) - 数据可视化
- **numpy** (>=1.20.0) - 数值计算
- **seaborn** (>=0.11.0) - 统计可视化
- **scipy** (>=1.7.0) - 科学计算

## 项目结构

```
da10/
├── main.py              # 主程序入口
├── data_loader.py       # 数据加载模块
├── preprocess.py        # 数据预处理模块
├── analysis.py          # 数据分析模块
├── visualization.py     # 可视化模块
├── requirements.txt     # 依赖包列表
├── README.md            # 项目说明文档
└── output/              # 输出目录（运行后自动生成）
    ├── raw_data.csv              # 原始数据
    ├── time_series.png           # 时间序列图
    ├── monthly_boxplots.png      # 月度箱线图
    ├── correlation_heatmap.png   # 相关系数热力图
    ├── temp_humid_dual_axis.png  # 温湿双轴图
    ├── extreme_weather_summary.png # 极端天气统计图
    ├── summary_report_overview.csv   # 汇总概览
    ├── summary_report_monthly.csv    # 月度统计
    ├── summary_report_quarterly.csv  # 季度统计
    └── summary_report_correlation.csv # 相关系数
```

## 功能模块说明

### 1. data_loader.py - 数据加载模块
- `generate_sample_data()`: 生成近3年（2023-2025）的模拟气象数据，包含季节性规律
- `load_data()`: 尝试加载外部CSV文件，如果不存在则生成示例数据
- `save_raw_data()`: 保存原始数据到output目录

### 2. preprocess.py - 数据预处理模块
- `handle_missing_values()`: 处理缺失值（插值、前向填充、后向填充、均值填充）
- `detect_outliers_zscore()`/`detect_outliers_iqr()`: 异常值检测
- `handle_outliers()`: 处理异常值（截断或移除）
- `add_time_features()`: 添加时间特征（年、月、季度、年内天数、星期几）
- `validate_data_ranges()`: 验证数据范围有效性
- `preprocess_data()`: 主预处理函数

### 3. analysis.py - 数据分析模块
- `calculate_monthly_stats()`: 计算月度统计指标
- `calculate_quarterly_stats()`: 计算季度统计指标
- `calculate_correlation_matrix()`: 计算变量相关系数矩阵
- `detect_extreme_weather()`: 检测极端天气事件
- `analyze_trends()`: 分析年度趋势
- `save_summary_report()`: 保存汇总报表
- `run_analysis()`: 主分析函数

### 4. visualization.py - 可视化模块
- `plot_time_series()`: 绘制时间序列图（含30天移动平均）
- `plot_monthly_boxplots()`: 绘制各月分布箱线图
- `plot_correlation_heatmap()`: 绘制相关系数热力图
- `plot_temperature_humidity_dual_axis()`: 绘制温湿双轴对比图
- `plot_extreme_weather_summary()`: 绘制极端天气统计图
- `run_visualization()`: 主可视化函数

### 5. main.py - 主程序
串联整个数据分析流程，包括：
1. 加载数据
2. 数据预处理
3. 数据分析
4. 生成可视化图表
5. 保存汇总报表

## 安装与运行

### 步骤1: 安装依赖包

```bash
cd da10
pip install -r requirements.txt
```

### 步骤2: 运行程序

```bash
python main.py
```

程序将自动：
1. 生成示例气象数据（或加载外部数据）
2. 执行数据清洗和预处理
3. 执行统计分析
4. 生成可视化图表
5. 保存所有输出到`output/`目录

## 输出文件说明

### 图表文件（PNG格式）

1. **time_series.png** - 时间序列图
   - 展示近3年温度、湿度、降水量、风速的逐日变化
   - 包含30天移动平均线

2. **monthly_boxplots.png** - 月度箱线图
   - 展示各月气象指标的分布情况
   - 便于观察季节性规律

3. **correlation_heatmap.png** - 相关系数热力图
   - 展示温度、湿度、降水量、风速之间的相关性

4. **temp_humid_dual_axis.png** - 温湿双轴图
   - 双轴对比展示月平均温度和湿度的变化趋势

5. **extreme_weather_summary.png** - 极端天气统计图
   - 统计高温、低温、大风、强降水等极端天气事件

### 数据报表（CSV格式）

1. **summary_report_overview.csv** - 数据概览
   - 数据范围、总体统计、极端天气统计

2. **summary_report_monthly.csv** - 月度统计
   - 各月温度、湿度、降水量、风速的统计指标

3. **summary_report_quarterly.csv** - 季度统计
   - 各季度统计指标

4. **summary_report_correlation.csv** - 相关系数矩阵

## 使用外部数据

如果要使用自己的气象数据，请准备一个CSV文件，包含以下列：
- `date`: 日期（格式：YYYY-MM-DD）
- `temperature`: 温度（°C）
- `humidity`: 湿度（%）
- `precipitation`: 降水量（mm）
- `wind_speed`: 风速（m/s）

然后修改`main.py`中的`load_data()`调用，传入文件路径：

```python
df = load_data(file_path='your_data.csv')
```

## 极端天气阈值

默认的极端天气检测阈值：
- 高温：≥ 35°C
- 低温：≤ 0°C
- 大风：≥ 15 m/s
- 强降水：≥ 50 mm

如需修改阈值，请编辑`analysis.py`中的`detect_extreme_weather()`函数参数。

## 注意事项

1. 程序支持中文显示，已配置matplotlib中文字体
2. 生成的模拟数据包含真实的季节性规律（夏季高温多雨，冬季低温干燥）
3. 所有输出文件默认保存在`output/`目录下
4. 如需PDF格式的图表，请修改`visualization.py`中的`save_pdf=True`

## 许可证

MIT License
