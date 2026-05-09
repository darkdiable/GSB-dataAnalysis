import os
from typing import List, Dict
from datetime import datetime
from .rules import Rule
from .matcher import MatchResult

class ReportGenerator:
    @staticmethod
    def generate_html_report(
        rules: List[Rule],
        match_results: List[MatchResult],
        stats: Dict,
        output_dir: str
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        
        default_deny_hits = sum(1 for r in match_results if r.is_default_deny)
        hit_rules = [r for r in rules if r.hit_count > 0]
        missed_rules = [r for r in rules if r.hit_count == 0]
        
        # 按命中数排序规则
        sorted_rules = sorted(rules, key=lambda r: r.hit_count, reverse=True)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>防火墙策略分析报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card:nth-child(2) {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .stat-card:nth-child(3) {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
        .stat-card:nth-child(4) {{ background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }}
        .stat-card:nth-child(5) {{ background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }}
        .stat-value {{ font-size: 28px; font-weight: bold; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; }}
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }}
        .chart-container img {{ width: 100%; height: auto; border-radius: 5px; }}
        .full-width {{ grid-column: 1 / -1; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            margin-bottom: 30px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{ background-color: #f5f5f5; }}
        .action-allow {{ color: #27ae60; font-weight: bold; }}
        .action-deny {{ color: #e74c3c; font-weight: bold; }}
        .collapsible {{
            background-color: #95a5a6;
            color: white;
            cursor: pointer;
            padding: 18px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 16px;
            font-weight: 600;
            border-radius: 5px;
            transition: 0.3s;
        }}
        .collapsible:hover, .collapsible.active {{ background-color: #7f8c8d; }}
        .collapsible-content {{
            padding: 0 18px;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.2s ease-out;
            background-color: #f1f1f1;
            border-radius: 0 0 5px 5px;
            margin-bottom: 10px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }}
        .badge-hit {{ background-color: #d4edda; color: #155724; }}
        .badge-miss {{ background-color: #f8d7da; color: #721c24; }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 防火墙策略分析报告</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(rules)}</div>
                <div class="stat-label">总规则数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(hit_rules)}</div>
                <div class="stat-label">命中规则数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(missed_rules)}</div>
                <div class="stat-label">未命中规则数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['total_hits'] + default_deny_hits}</div>
                <div class="stat-label">总数据包数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{default_deny_hits}</div>
                <div class="stat-label">默认拒绝数</div>
            </div>
        </div>
        
        <h2>📊 可视化图表</h2>
        <div class="charts-grid">
            <div class="chart-container">
                <h3>规则命中分布</h3>
                <img src="pie_chart.png" alt="Pie Chart">
            </div>
            <div class="chart-container">
                <h3>Top 10 命中规则</h3>
                <img src="top10_bar_chart.png" alt="Bar Chart">
            </div>
            <div class="chart-container">
                <h3>规则效率散点图</h3>
                <img src="efficiency_scatter.png" alt="Scatter Chart">
            </div>
            <div class="chart-container">
                <h3>协议命中热力图</h3>
                <img src="protocol_heatmap.png" alt="Heatmap">
            </div>
        </div>
        
        <h2>📋 规则详情</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>优先级</th>
                    <th>源IP</th>
                    <th>目的IP</th>
                    <th>协议</th>
                    <th>端口</th>
                    <th>动作</th>
                    <th>命中次数</th>
                    <th>平均耗时(μs)</th>
                    <th>状态</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for rule in sorted_rules:
            status_class = "badge-hit" if rule.hit_count > 0 else "badge-miss"
            status_text = "命中" if rule.hit_count > 0 else "未命中"
            action_class = "action-allow" if rule.action == "allow" else "action-deny"
            avg_time = f"{rule.avg_match_time:.4f}" if rule.avg_match_time > 0 else "-"
            
            html_content += f"""
                <tr>
                    <td>{rule.id}</td>
                    <td>{rule.priority}</td>
                    <td>{rule.src_ip}</td>
                    <td>{rule.dst_ip}</td>
                    <td>{rule.protocol}</td>
                    <td>{rule.port}</td>
                    <td class="{action_class}">{rule.action.upper()}</td>
                    <td>{rule.hit_count}</td>
                    <td>{avg_time}</td>
                    <td><span class="badge {status_class}">{status_text}</span></td>
                </tr>
            """
        
        html_content += """
            </tbody>
        </table>
        """
        
        # 添加未命中规则折叠列表
        if missed_rules:
            html_content += f"""
        <h2>⚠️ 未命中规则 ({len(missed_rules)} 条)</h2>
        <button class="collapsible">点击展开查看 {len(missed_rules)} 条未命中规则</button>
        <div class="collapsible-content">
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>优先级</th>
                        <th>源IP</th>
                        <th>目的IP</th>
                        <th>协议</th>
                        <th>端口</th>
                        <th>动作</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for rule in missed_rules:
                action_class = "action-allow" if rule.action == "allow" else "action-deny"
                html_content += f"""
                    <tr>
                        <td>{rule.id}</td>
                        <td>{rule.priority}</td>
                        <td>{rule.src_ip}</td>
                        <td>{rule.dst_ip}</td>
                        <td>{rule.protocol}</td>
                        <td>{rule.port}</td>
                        <td class="{action_class}">{rule.action.upper()}</td>
                    </tr>
                """
            
            html_content += """
                </tbody>
            </table>
        </div>
        <script>
            var coll = document.getElementsByClassName("collapsible");
            for (var i = 0; i < coll.length; i++) {
                coll[i].addEventListener("click", function() {
                    this.classList.toggle("active");
                    var content = this.nextElementSibling;
                    if (content.style.maxHeight) {
                        content.style.maxHeight = null;
                    } else {
                        content.style.maxHeight = content.scrollHeight + "px";
                    }
                });
            }
        </script>
            """
        
        html_content += f"""
        <div class="footer">
            <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>平均规则匹配时间: {stats['avg_match_time_per_rule_us']:.4f} 微秒</p>
        </div>
    </div>
</body>
</html>
        """
        
        report_path = os.path.join(output_dir, 'report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
