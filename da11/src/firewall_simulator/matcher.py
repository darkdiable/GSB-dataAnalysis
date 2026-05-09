import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
from ipaddress import ip_network, ip_address
from .rules import Rule, RuleParser
from .traffic_generator import Packet

@dataclass
class MatchResult:
    rule_id: Optional[int]
    rule_priority: Optional[int]
    action: str
    matched_rule: Optional[Rule]
    match_time_us: float
    is_default_deny: bool = False

class MatchEngine:
    def __init__(self, rules: List[Rule]):
        self.rules = RuleParser.sort_rules(rules)
        self._parse_rules_cache()
    
    def _parse_rules_cache(self):
        self._rule_caches = []
        for rule in self.rules:
            # 解析源IP
            if '/' in rule.src_ip:
                src_net = ip_network(rule.src_ip, strict=False)
            else:
                src_net = ip_network(rule.src_ip + '/32')
            
            # 解析目的IP
            if '/' in rule.dst_ip:
                dst_net = ip_network(rule.dst_ip, strict=False)
            else:
                dst_net = ip_network(rule.dst_ip + '/32')
            
            # 解析端口
            ports = RuleParser.parse_port(rule.port)
            ports_set = set(ports)
            
            self._rule_caches.append({
                'rule': rule,
                'src_net': src_net,
                'dst_net': dst_net,
                'ports_set': ports_set
            })
    
    def _match_rule(self, packet: Packet, cache: dict) -> bool:
        rule = cache['rule']
        
        # 匹配协议
        if rule.protocol != packet.protocol:
            return False
        
        # 匹配源IP
        try:
            src_ip = ip_address(packet.src_ip)
            if src_ip not in cache['src_net']:
                return False
        except ValueError:
            return False
        
        # 匹配目的IP
        try:
            dst_ip = ip_address(packet.dst_ip)
            if dst_ip not in cache['dst_net']:
                return False
        except ValueError:
            return False
        
        # 匹配端口（ICMP协议不检查端口）
        if packet.protocol != 'icmp' and packet.port not in cache['ports_set']:
            return False
        
        return True
    
    def match_packet(self, packet: Packet) -> MatchResult:
        start_time = time.perf_counter()
        total_match_time = 0.0
        
        for cache in self._rule_caches:
            rule_start = time.perf_counter()
            if self._match_rule(packet, cache):
                rule_end = time.perf_counter()
                rule_match_time = (rule_end - rule_start) * 1_000_000
                
                rule = cache['rule']
                rule.hit_count += 1
                rule.total_match_time += rule_match_time
                rule.avg_match_time = rule.total_match_time / rule.hit_count
                
                total_end = time.perf_counter()
                total_time = (total_end - start_time) * 1_000_000
                
                return MatchResult(
                    rule_id=rule.id,
                    rule_priority=rule.priority,
                    action=rule.action,
                    matched_rule=rule,
                    match_time_us=total_time,
                    is_default_deny=False
                )
        
        # 没有匹配到规则，应用默认拒绝
        total_end = time.perf_counter()
        total_time = (total_end - start_time) * 1_000_000
        
        return MatchResult(
            rule_id=None,
            rule_priority=None,
            action='deny',
            matched_rule=None,
            match_time_us=total_time,
            is_default_deny=True
        )
    
    def match_packets(self, packets: List[Packet]) -> List[MatchResult]:
        return [self.match_packet(packet) for packet in packets]
    
    def get_statistics(self) -> dict:
        total_hits = sum(rule.hit_count for rule in self.rules)
        default_deny_hits = 0
        total_match_time = 0.0
        rules_with_hits = 0
        
        for rule in self.rules:
            total_match_time += rule.total_match_time
            if rule.hit_count > 0:
                rules_with_hits += 1
        
        avg_match_time_per_rule = total_match_time / total_hits if total_hits > 0 else 0.0
        
        return {
            'total_rules': len(self.rules),
            'total_hits': total_hits,
            'rules_with_hits': rules_with_hits,
            'total_match_time_us': total_match_time,
            'avg_match_time_per_rule_us': avg_match_time_per_rule
        }
    
    def reset_statistics(self):
        for rule in self.rules:
            rule.hit_count = 0
            rule.total_match_time = 0.0
            rule.avg_match_time = 0.0
