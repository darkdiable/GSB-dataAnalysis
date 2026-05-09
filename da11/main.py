#!/usr/bin/env python3
"""
防火墙策略配置模拟器
"""

import argparse
import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from firewall_simulator import Rule, RuleParser, TrafficGenerator, MatchEngine, Visualizer, ReportGenerator

def generate_random_rules(count: int) -> list:
    rules = []
    protocols = ['tcp', 'udp', 'icmp']
    actions = ['allow', 'deny']
    
    for i in range(count):
        src_ip = f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.0/{random.choice([24, 16, 8])}"
        dst_ip = f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.0/{random.choice([24, 16, 8])}"
        protocol = random.choice(protocols)
        
        if protocol == 'icmp':
            port = 'any'
        else:
            port_type = random.choice(['single', 'range', 'any'])
            if port_type == 'single':
                port = str(random.randint(1, 65535))
            elif port_type == 'range':
                start = random.randint(1, 60000)
                end = random.randint(start, 65535)
                port = f"{start}-{end}"
            else:
                port = 'any'
        
        rules.append(Rule(
            id=i + 1,
            priority=random.randint(1, 100),
            src_ip=src_ip,
            dst_ip=dst_ip,
            protocol=protocol,
            port=port,
            action=random.choice(actions)
        ))
    
    return rules

def create_sample_rules():
    sample_csv_content = """id,priority,src_ip,dst_ip,protocol,port,action
1,10,192.168.1.0/24,10.0.0.0/8,tcp,80,allow
2,20,192.168.2.0/24,10.0.0.0/8,udp,53,allow
3,30,0.0.0.0/0,10.0.0.0/8,icmp,any,deny
4,40,192.168.3.100,10.0.0.1,tcp,22-23,allow
5,50,192.168.4.0/24,10.0.0.0/8,tcp,443,deny
6,60,172.16.0.0/12,10.0.0.0/8,udp,any,allow
7,70,192.168.5.50,10.0.0.5,tcp,3306,allow
8,80,10.0.0.0/8,192.168.0.0/16,tcp,any,deny
9,90,192.168.6.0/24,192.168.7.0/24,icmp,any,allow
10,100,0.0.0.0/0,0.0.0.0/0,tcp,8080,allow
"""
    with open('sample_rules.csv', 'w', encoding='utf-8') as f:
        f.write(sample_csv_content)
    print("已创建示例规则文件: sample_rules.csv")

def run_normal_mode(args):
    print("=== 防火墙策略配置模拟器 ===")
    
    if not os.path.exists(args.rules):
        print(f"错误: 规则文件不存在: {args.rules}")
        return
    
    print(f"1. 加载规则文件: {args.rules}")
    rules = RuleParser.load_rules(args.rules)
    print(f"   成功加载 {len(rules)} 条规则")
    
    if args.export_json:
        json_path = args.export_json
        RuleParser.save_json(rules, json_path)
        print(f"   已导出规则到: {json_path}")
        return
    
    print(f"2. 生成 {args.packets} 个随机数据包")
    packets = TrafficGenerator.generate_packets(count=args.packets)
    
    print("3. 初始化匹配引擎")
    engine = MatchEngine(rules)
    
    print("4. 开始匹配数据包...")
    start_time = time.perf_counter()
    results = engine.match_packets(packets)
    end_time = time.perf_counter()
    
    total_time = (end_time - start_time) * 1000
    stats = engine.get_statistics()
    default_deny_hits = sum(1 for r in results if r.is_default_deny)
    
    print("5. 匹配完成！统计信息：")
    print(f"   - 总规则数: {stats['total_rules']}")
    print(f"   - 命中规则数: {stats['rules_with_hits']}")
    print(f"   - 总匹配次数: {stats['total_hits']}")
    print(f"   - 默认拒绝数: {default_deny_hits}")
    print(f"   - 总匹配时间: {total_time:.2f} ms")
    print(f"   - 平均规则匹配时间: {stats['avg_match_time_per_rule_us']:.4f} μs")
    print(f"   - 处理速度: {args.packets / (end_time - start_time):.2f} 包/秒")
    
    if args.output:
        print(f"6. 生成输出到目录: {args.output}")
        os.makedirs(args.output, exist_ok=True)
        
        print("   - 生成可视化图表...")
        Visualizer.create_all_visualizations(rules, results, packets, args.output)
        
        print("   - 生成HTML报告...")
        report_path = ReportGenerator.generate_html_report(rules, results, stats, args.output)
        
        print(f"   - 导出JSON规则...")
        json_path = os.path.join(args.output, 'rules.json')
        RuleParser.save_json(rules, json_path)
        
        print(f"7. 完成！报告已生成: {report_path}")

def run_benchmark_mode():
    print("=== 性能测试模式 ===")
    
    num_rules = 20000
    num_packets = 10000
    
    print(f"1. 生成 {num_rules} 条随机规则...")
    rules = generate_random_rules(num_rules)
    print(f"   完成")
    
    print(f"2. 生成 {num_packets} 个随机数据包...")
    packets = TrafficGenerator.generate_packets(count=num_packets)
    print(f"   完成")
    
    print("3. 初始化匹配引擎...")
    engine = MatchEngine(rules)
    print(f"   完成")
    
    print("4. 开始性能测试...")
    start_time = time.perf_counter()
    results = engine.match_packets(packets)
    end_time = time.perf_counter()
    
    total_time_seconds = end_time - start_time
    total_time_ms = total_time_seconds * 1000
    packets_per_second = num_packets / total_time_seconds if total_time_seconds > 0 else 0
    
    default_deny = sum(1 for r in results if r.is_default_deny)
    matched = len(results) - default_deny
    
    print("5. 性能测试结果：")
    print(f"   - 测试规则数: {num_rules}")
    print(f"   - 测试数据包数: {num_packets}")
    print(f"   - 总匹配时间: {total_time_ms:.2f} ms ({total_time_seconds:.4f} 秒)")
    print(f"   - 每秒处理包数: {packets_per_second:.2f} 包/秒")
    print(f"   - 匹配成功: {matched} ({matched/num_packets*100:.2f}%)")
    print(f"   - 默认拒绝: {default_deny} ({default_deny/num_packets*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='防火墙策略配置模拟器')
    
    parser.add_argument('--rules', type=str, help='规则文件路径（CSV或JSON）')
    parser.add_argument('--packets', type=int, default=500, help='生成数据包数量（默认500）')
    parser.add_argument('--output', type=str, help='输出目录')
    parser.add_argument('--export-json', type=str, help='导出当前规则为JSON文件')
    parser.add_argument('--benchmark', action='store_true', help='运行性能测试模式')
    parser.add_argument('--create-sample', action='store_true', help='创建示例规则文件')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_rules()
        return
    
    if args.benchmark:
        run_benchmark_mode()
        return
    
    if not args.rules:
        print("错误: 必须指定 --rules 参数或使用 --benchmark 模式")
        parser.print_help()
        return
    
    run_normal_mode(args)

if __name__ == '__main__':
    main()
