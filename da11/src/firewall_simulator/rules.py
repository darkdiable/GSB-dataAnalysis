import csv
import json
from dataclasses import dataclass, asdict
from typing import List, Optional
from ipaddress import ip_network, ip_address
import re

@dataclass
class Rule:
    id: int
    priority: int
    src_ip: str
    dst_ip: str
    protocol: str
    port: str
    action: str
    hit_count: int = 0
    total_match_time: float = 0.0
    avg_match_time: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Rule':
        return cls(
            id=int(data['id']),
            priority=int(data['priority']),
            src_ip=data['src_ip'],
            dst_ip=data['dst_ip'],
            protocol=data['protocol'],
            port=data['port'],
            action=data['action'],
            hit_count=int(data.get('hit_count', 0)),
            total_match_time=float(data.get('total_match_time', 0.0)),
            avg_match_time=float(data.get('avg_match_time', 0.0))
        )

class RuleParser:
    PROTOCOLS = ['tcp', 'udp', 'icmp']
    ACTIONS = ['allow', 'deny']
    
    @staticmethod
    def parse_port(port_str: str) -> List[int]:
        if port_str.lower() == 'any':
            return list(range(1, 65536))
        if '-' in port_str:
            start, end = map(int, port_str.split('-'))
            return list(range(start, end + 1))
        return [int(port_str)]
    
    @staticmethod
    def validate_rule(rule: Rule) -> bool:
        try:
            if rule.protocol not in RuleParser.PROTOCOLS:
                return False
            if rule.action not in RuleParser.ACTIONS:
                return False
            
            # 验证源IP
            try:
                if '/' in rule.src_ip:
                    ip_network(rule.src_ip, strict=False)
                else:
                    ip_address(rule.src_ip)
            except ValueError:
                return False
            
            # 验证目的IP
            try:
                if '/' in rule.dst_ip:
                    ip_network(rule.dst_ip, strict=False)
                else:
                    ip_address(rule.dst_ip)
            except ValueError:
                return False
            
            # 验证端口
            try:
                RuleParser.parse_port(rule.port)
            except ValueError:
                return False
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def load_csv(file_path: str) -> List[Rule]:
        rules = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rule = Rule(
                    id=int(row['id']),
                    priority=int(row['priority']),
                    src_ip=row['src_ip'],
                    dst_ip=row['dst_ip'],
                    protocol=row['protocol'],
                    port=row['port'],
                    action=row['action'])
                
                if not RuleParser.validate_rule(rule):
                    raise ValueError(f"无效规则: {rule}")
                rules.append(rule)
        return rules
    
    @staticmethod
    def load_json(file_path: str) -> List[Rule]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rules = []
        for item in data:
            rule = Rule.from_dict(item)
            if not RuleParser.validate_rule(rule):
                raise ValueError(f"无效规则: {rule}")
            rules.append(rule)
        return rules
    
    @staticmethod
    def load_rules(file_path: str) -> List[Rule]:
        if file_path.endswith('.csv'):
            return RuleParser.load_csv(file_path)
        elif file_path.endswith('.json'):
            return RuleParser.load_json(file_path)
        else:
            raise ValueError("只支持CSV和JSON格式")
    
    @staticmethod
    def save_json(rules: List[Rule], file_path: str) -> None:
        data = [rule.to_dict() for rule in rules]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def sort_rules(rules: List[Rule]) -> List[Rule]:
        return sorted(rules, key=lambda r: r.priority)
