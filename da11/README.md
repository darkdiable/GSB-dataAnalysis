# 防火墙策略配置模拟器

一个功能完整的防火墙策略配置模拟器，用于测试和分析防火墙规则的匹配性能和效果。

## 功能特性

1. **规则管理**
   - 支持CSV和JSON格式的规则文件
   - 规则字段：id, priority, src_ip, dst_ip, protocol, port, action
   - 支持CIDR格式IP（如192.168.1.0/24）和单个IP
   - 支持端口范围（如22-25）和"any"
   - 按priority升序匹配

2. **默认隐式拒绝**
   - 未匹配到任何规则的数据包默认被拒绝
   - 在统计中单独计数

3. **流量生成器**
   - 可生成指定数量的随机数据包（默认500个）
   - 支持协议、端口范围、IP子网内随机生成

4. **匹配引擎**
   - 基于ipaddress库实现精确的子网匹配
   - 记录每条规则的匹配耗时（微秒级）
   - 输出规则平均匹配时间

5. **可视化图表**
   - 饼图：规则命中分布
   - 柱状图：Top 10命中规则
   - 散点图：规则效率（横轴优先级，纵轴耗时，气泡大小=命中次数）
   - 热力图：协议命中分布（tcp/udp/icmp vs allow/deny）

6. **HTML报告**
   - 展示所有统计图表
   - 规则详情表格（含命中数、平均耗时）
   - 未命中规则折叠列表

7. **性能测试**
   - 支持--benchmark模式
   - 生成20000条随机规则 + 10000个包
   - 打印总匹配时间和每秒处理包数

## 安装

```bash
cd da11
pip install -r requirements.txt
```

## 使用方法

### 1. 创建示例规则文件

```bash
python main.py --create-sample
```

这会在当前目录生成 `sample_rules.csv` 文件。

### 2. 基本运行

```bash
python main.py --rules sample_rules.csv --output output
```

### 3. 指定数据包数量

```bash
python main.py --rules sample_rules.csv --packets 1000 --output output
```

### 4. 导出规则为JSON

```bash
python main.py --rules sample_rules.csv --export-json rules.json
```

### 5. 从JSON导入规则

```bash
python main.py --rules rules.json --output output
```

### 6. 运行性能测试

```bash
python main.py --benchmark
```

## 规则文件格式

### CSV格式示例 (sample_rules.csv)

```csv
id,priority,src_ip,dst_ip,protocol,port,action
1,10,192.168.1.0/24,10.0.0.0/8,tcp,80,allow
2,20,192.168.2.0/24,10.0.0.0/8,udp,53,allow
3,30,0.0.0.0/0,10.0.0.0/8,icmp,any,deny
4,40,192.168.3.100,10.0.0.1,tcp,22-23,allow
5,50,192.168.4.0/24,10.0.0.0/8,tcp,443,deny
```

### JSON格式示例

```json
[
  {
    "id": 1,
    "priority": 10,
    "src_ip": "192.168.1.0/24",
    "dst_ip": "10.0.0.0/8",
    "protocol": "tcp",
    "port": "80",
    "action": "allow"
  }
]
```

## 规则字段说明

| 字段 | 说明 | 示例 |
|------|------|------|
| id | 规则ID | 1, 2, 3... |
| priority | 优先级（数值越小优先级越高） | 10, 20, 100 |
| src_ip | 源IP（支持CIDR） | 192.168.1.0/24 或 192.168.1.100 |
| dst_ip | 目的IP（支持CIDR） | 10.0.0.0/8 或 10.0.0.1 |
| protocol | 协议 | tcp, udp, icmp |
| port | 端口 | 80, 22-25, any |
| action | 动作 | allow, deny |

## 命令行参数

```
--rules PATH          规则文件路径（CSV或JSON）
--packets N           生成数据包数量（默认500）
--output DIR          输出目录
--export-json PATH    导出当前规则为JSON文件
--benchmark           运行性能测试模式
--create-sample       创建示例规则文件
```

## 运行单元测试

```bash
cd da11
python -m pytest tests/ -v
```

## 输出说明

运行后，指定的输出目录将包含：

- `report.html` - 完整的HTML分析报告
- `pie_chart.png` - 规则命中分布饼图
- `top10_bar_chart.png` - Top 10命中规则柱状图
- `efficiency_scatter.png` - 规则效率散点图
- `protocol_heatmap.png` - 协议命中热力图
- `rules.json` - 导出的规则JSON文件

## 性能测试预期结果

性能测试模式（--benchmark）会生成：
- 20000条随机规则
- 10000个随机数据包

预期输出包含：
- 总匹配时间（毫秒）
- 每秒处理包数
- 匹配成功/默认拒绝的统计

## 项目结构

```
da11/
├── main.py                      # 主程序入口
├── requirements.txt             # 依赖包
├── README.md                    # 本文件
├── src/
│   └── firewall_simulator/
│       ├── __init__.py
│       ├── rules.py             # 规则解析模块
│       ├── traffic_generator.py # 流量生成器
│       ├── matcher.py           # 匹配引擎
│       ├── visualization.py     # 可视化模块
│       └── report.py            # HTML报告生成
└── tests/
    ├── __init__.py
    └── test_rules.py            # 单元测试
```

## 许可证

MIT License
