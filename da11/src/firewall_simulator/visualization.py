import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple
from .rules import Rule
from .matcher import MatchResult

plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS',
    'PingFang SC',
    'Hiragino Sans GB',
    'Microsoft YaHei',
    'SimHei',
    'STHeiti',
    'Heiti TC'
]
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    @staticmethod
    def ensure_output_dir(output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
    
    @staticmethod
    def create_pie_chart(
        rules: List[Rule],
        default_deny_hits: int,
        output_path: str
    ):
        hit_data = []
        labels = []
        colors = []
        
        for rule in rules:
            if rule.hit_count > 0:
                hit_data.append(rule.hit_count)
                labels.append(f"规则 {rule.id}")
                colors.append(plt.cm.tab20(rule.id % 20))
        
        if default_deny_hits > 0:
            hit_data.append(default_deny_hits)
            labels.append("默认拒绝")
            colors.append('#FF6B6B')
        
        plt.figure(figsize=(10, 10))
        plt.pie(
            hit_data,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90
        )
        plt.title('规则命中分布')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def create_top10_bar_chart(
        rules: List[Rule],
        default_deny_hits: int,
        output_path: str
    ):
        rule_hits = [(rule.id, rule.hit_count, False) for rule in rules]
        if default_deny_hits > 0:
            rule_hits.append((-1, default_deny_hits, True))
        
        rule_hits.sort(key=lambda x: x[1], reverse=True)
        top10 = rule_hits[:10]
        
        if not top10:
            return
        
        ids = []
        hits = []
        colors = []
        
        for item in top10:
            if item[2]:
                ids.append("默认拒绝")
                colors.append('#FF6B6B')
            else:
                ids.append(f"规则 {item[0]}")
                colors.append('#4ECDC4')
            hits.append(item[1])
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(ids, hits, color=colors)
        plt.xlabel('规则ID')
        plt.ylabel('命中次数')
        plt.title('Top 10 命中规则')
        plt.xticks(rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom'
            )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def create_efficiency_scatter_plot(
        rules: List[Rule],
        output_path: str
    ):
        priorities = []
        avg_times = []
        hit_counts = []
        rule_ids = []
        
        for rule in rules:
            if rule.hit_count > 0:
                priorities.append(rule.priority)
                avg_times.append(rule.avg_match_time)
                hit_counts.append(rule.hit_count)
                rule_ids.append(rule.id)
        
        if not priorities:
            return
        
        max_hits = max(hit_counts) if hit_counts else 1
        bubble_sizes = [count / max_hits * 1000 for count in hit_counts]
        
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(
            priorities,
            avg_times,
            s=bubble_sizes,
            c=hit_counts,
            cmap='viridis',
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        
        plt.colorbar(scatter, label='命中次数')
        plt.xlabel('优先级')
        plt.ylabel('平均匹配时间 (微秒)')
        plt.title('规则效率散点图')
        plt.grid(True, alpha=0.3)
        
        for i, rule_id in enumerate(rule_ids):
            plt.annotate(
                f'R{rule_id}',
                (priorities[i], avg_times[i]),
                fontsize=8,
                alpha=0.7
            )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def create_protocol_heatmap(
        rules: List[Rule],
        match_results: List[MatchResult],
        packets: List,
        output_path: str
    ):
        protocols = ['tcp', 'udp', 'icmp']
        actions = ['allow', 'deny']
        
        heatmap_data = np.zeros((len(protocols), len(actions)))
        
        for i, result in enumerate(match_results):
            if result.is_default_deny:
                if i < len(packets):
                    packet = packets[i]
                    if packet.protocol in protocols:
                        row = protocols.index(packet.protocol)
                        col = actions.index('deny')
                        heatmap_data[row, col] += 1
            elif result.matched_rule:
                protocol = result.matched_rule.protocol
                action = result.action
                if protocol in protocols and action in actions:
                    row = protocols.index(protocol)
                    col = actions.index(action)
                    heatmap_data[row, col] += 1
        
        plt.figure(figsize=(10, 6))
        heatmap = plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        
        plt.colorbar(heatmap, label='命中次数')
        plt.xticks(range(len(actions)), actions)
        plt.yticks(range(len(protocols)), protocols)
        plt.xlabel('动作')
        plt.ylabel('协议')
        plt.title('协议命中热力图')
        
        for i in range(len(protocols)):
            for j in range(len(actions)):
                plt.text(
                    j, i,
                    f'{int(heatmap_data[i, j])}',
                    ha='center',
                    va='center',
                    fontsize=12,
                    color='black'
                )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def create_all_visualizations(
        rules: List[Rule],
        match_results: List[MatchResult],
        packets: List,
        output_dir: str
    ):
        Visualizer.ensure_output_dir(output_dir)
        
        default_deny_hits = sum(1 for r in match_results if r.is_default_deny)
        
        Visualizer.create_pie_chart(rules, default_deny_hits, os.path.join(output_dir, 'pie_chart.png'))
        Visualizer.create_top10_bar_chart(rules, default_deny_hits, os.path.join(output_dir, 'top10_bar_chart.png'))
        Visualizer.create_efficiency_scatter_plot(rules, os.path.join(output_dir, 'efficiency_scatter.png'))
        Visualizer.create_protocol_heatmap(rules, match_results, packets, os.path.join(output_dir, 'protocol_heatmap.png'))
