import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from firewall_simulator import Rule, RuleParser, TrafficGenerator, MatchEngine
from firewall_simulator.traffic_generator import Packet

class TestPortParsing:
    def test_single_port(self):
        ports = RuleParser.parse_port("80")
        assert ports == [80]
    
    def test_port_range(self):
        ports = RuleParser.parse_port("22-25")
        assert ports == [22, 23, 24, 25]
    
    def test_port_any(self):
        ports = RuleParser.parse_port("any")
        assert len(ports) == 65535
        assert 1 in ports
        assert 65535 in ports
    
    def test_port_any_case_insensitive(self):
        ports = RuleParser.parse_port("Any")
        assert len(ports) == 65535

class TestRuleValidation:
    def test_valid_rule(self):
        rule = Rule(
            id=1,
            priority=10,
            src_ip="192.168.1.0/24",
            dst_ip="10.0.0.0/8",
            protocol="tcp",
            port="80",
            action="allow"
        )
        assert RuleParser.validate_rule(rule) == True
    
    def test_invalid_protocol(self):
        rule = Rule(
            id=1,
            priority=10,
            src_ip="192.168.1.0/24",
            dst_ip="10.0.0.0/8",
            protocol="http",
            port="80",
            action="allow"
        )
        assert RuleParser.validate_rule(rule) == False
    
    def test_invalid_action(self):
        rule = Rule(
            id=1,
            priority=10,
            src_ip="192.168.1.0/24",
            dst_ip="10.0.0.0/8",
            protocol="tcp",
            port="80",
            action="drop"
        )
        assert RuleParser.validate_rule(rule) == False
    
    def test_invalid_ip(self):
        rule = Rule(
            id=1,
            priority=10,
            src_ip="256.1.1.1",
            dst_ip="10.0.0.0/8",
            protocol="tcp",
            port="80",
            action="allow"
        )
        assert RuleParser.validate_rule(rule) == False

class TestPrioritySorting:
    def test_sort_rules_by_priority(self):
        rules = [
            Rule(id=1, priority=50, src_ip="192.168.1.1", dst_ip="10.0.0.1", protocol="tcp", port="80", action="allow"),
            Rule(id=2, priority=10, src_ip="192.168.1.2", dst_ip="10.0.0.2", protocol="tcp", port="80", action="allow"),
            Rule(id=3, priority=30, src_ip="192.168.1.3", dst_ip="10.0.0.3", protocol="tcp", port="80", action="allow"),
        ]
        sorted_rules = RuleParser.sort_rules(rules)
        assert sorted_rules[0].priority == 10
        assert sorted_rules[1].priority == 30
        assert sorted_rules[2].priority == 50
        assert sorted_rules[0].id == 2

class TestCIDRMatch:
    def test_cidr_match_inside_subnet(self):
        rule = Rule(
            id=1,
            priority=10,
            src_ip="192.168.1.0/24",
            dst_ip="10.0.0.0/8",
            protocol="tcp",
            port="80",
            action="allow"
        )
        engine = MatchEngine([rule])
        
        packet = Packet(
            src_ip="192.168.1.100",
            dst_ip="10.0.0.5",
            protocol="tcp",
            port=80
        )
        
        result = engine.match_packet(packet)
        assert result.rule_id == 1
        assert result.action == "allow"
        assert result.is_default_deny == False
    
    def test_cidr_no_match_outside_subnet(self):
        rule = Rule(
            id=1,
            priority=10,
            src_ip="192.168.1.0/24",
            dst_ip="10.0.0.0/8",
            protocol="tcp",
            port="80",
            action="allow"
        )
        engine = MatchEngine([rule])
        
        packet = Packet(
            src_ip="192.168.2.100",
            dst_ip="10.0.0.5",
            protocol="tcp",
            port=80
        )
        
        result = engine.match_packet(packet)
        assert result.rule_id is None
        assert result.action == "deny"
        assert result.is_default_deny == True
    
    def test_single_ip_match(self):
        rule = Rule(
            id=1,
            priority=10,
            src_ip="192.168.1.100",
            dst_ip="10.0.0.5",
            protocol="tcp",
            port="80",
            action="allow"
        )
        engine = MatchEngine([rule])
        
        packet = Packet(
            src_ip="192.168.1.100",
            dst_ip="10.0.0.5",
            protocol="tcp",
            port=80
        )
        
        result = engine.match_packet(packet)
        assert result.rule_id == 1
    
    def test_protocol_mismatch(self):
        rule = Rule(
            id=1,
            priority=10,
            src_ip="192.168.1.0/24",
            dst_ip="10.0.0.0/8",
            protocol="udp",
            port="80",
            action="allow"
        )
        engine = MatchEngine([rule])
        
        packet = Packet(
            src_ip="192.168.1.100",
            dst_ip="10.0.0.5",
            protocol="tcp",
            port=80
        )
        
        result = engine.match_packet(packet)
        assert result.is_default_deny == True

class TestPortMatch:
    def test_single_port_match(self):
        rule = Rule(
            id=1,
            priority=10,
            src_ip="192.168.1.0/24",
            dst_ip="10.0.0.0/8",
            protocol="tcp",
            port="80",
            action="allow"
        )
        engine = MatchEngine([rule])
        
        packet = Packet(
            src_ip="192.168.1.100",
            dst_ip="10.0.0.5",
            protocol="tcp",
            port=80
        )
        
        result = engine.match_packet(packet)
        assert result.rule_id == 1
    
    def test_single_port_no_match(self):
        rule = Rule(
            id=1,
            priority=10,
            src_ip="192.168.1.0/24",
            dst_ip="10.0.0.0/8",
            protocol="tcp",
            port="80",
            action="allow"
        )
        engine = MatchEngine([rule])
        
        packet = Packet(
            src_ip="192.168.1.100",
            dst_ip="10.0.0.5",
            protocol="tcp",
            port=443
        )
        
        result = engine.match_packet(packet)
        assert result.is_default_deny == True
    
    def test_port_range_match(self):
        rule = Rule(
            id=1,
            priority=10,
            src_ip="192.168.1.0/24",
            dst_ip="10.0.0.0/8",
            protocol="tcp",
            port="22-25",
            action="allow"
        )
        engine = MatchEngine([rule])
        
        packet = Packet(
            src_ip="192.168.1.100",
            dst_ip="10.0.0.5",
            protocol="tcp",
            port=23
        )
        
        result = engine.match_packet(packet)
        assert result.rule_id == 1
    
    def test_port_any_match(self):
        rule = Rule(
            id=1,
            priority=10,
            src_ip="192.168.1.0/24",
            dst_ip="10.0.0.0/8",
            protocol="tcp",
            port="any",
            action="allow"
        )
        engine = MatchEngine([rule])
        
        for port in [1, 80, 443, 65535]:
            packet = Packet(
                src_ip="192.168.1.100",
                dst_ip="10.0.0.5",
                protocol="tcp",
                port=port
            )
            result = engine.match_packet(packet)
            assert result.rule_id == 1
    
    def test_icmp_no_port_check(self):
        rule = Rule(
            id=1,
            priority=10,
            src_ip="192.168.1.0/24",
            dst_ip="10.0.0.0/8",
            protocol="icmp",
            port="any",
            action="allow"
        )
        engine = MatchEngine([rule])
        
        packet = Packet(
            src_ip="192.168.1.100",
            dst_ip="10.0.0.5",
            protocol="icmp",
            port=0
        )
        
        result = engine.match_packet(packet)
        assert result.rule_id == 1

class TestJSONImportExport:
    def test_json_export_import(self, tmp_path):
        rules = [
            Rule(id=1, priority=10, src_ip="192.168.1.0/24", dst_ip="10.0.0.0/8", protocol="tcp", port="80", action="allow"),
            Rule(id=2, priority=20, src_ip="192.168.2.0/24", dst_ip="10.0.0.0/8", protocol="udp", port="53", action="deny"),
        ]
        
        json_file = tmp_path / "rules.json"
        RuleParser.save_json(rules, str(json_file))
        
        assert json_file.exists()
        
        loaded_rules = RuleParser.load_json(str(json_file))
        
        assert len(loaded_rules) == 2
        assert loaded_rules[0].id == 1
        assert loaded_rules[0].priority == 10
        assert loaded_rules[0].src_ip == "192.168.1.0/24"
        assert loaded_rules[0].protocol == "tcp"
        assert loaded_rules[0].action == "allow"
        
        assert loaded_rules[1].id == 2
        assert loaded_rules[1].action == "deny"

class TestPriorityOrderMatch:
    def test_higher_priority_matches_first(self):
        rules = [
            Rule(id=2, priority=20, src_ip="192.168.1.0/24", dst_ip="10.0.0.0/8", protocol="tcp", port="80", action="deny"),
            Rule(id=1, priority=10, src_ip="192.168.1.0/24", dst_ip="10.0.0.0/8", protocol="tcp", port="80", action="allow"),
        ]
        
        engine = MatchEngine(rules)
        
        packet = Packet(
            src_ip="192.168.1.100",
            dst_ip="10.0.0.5",
            protocol="tcp",
            port=80
        )
        
        result = engine.match_packet(packet)
        assert result.rule_id == 1
        assert result.action == "allow"
        assert result.rule_priority == 10

class TestDefaultDeny:
    def test_no_match_default_deny(self):
        rule = Rule(
            id=1,
            priority=10,
            src_ip="10.0.0.0/8",
            dst_ip="192.168.1.0/24",
            protocol="tcp",
            port="80",
            action="allow"
        )
        engine = MatchEngine([rule])
        
        packet = Packet(
            src_ip="172.16.0.1",
            dst_ip="192.168.2.1",
            protocol="tcp",
            port=80
        )
        
        result = engine.match_packet(packet)
        assert result.is_default_deny == True
        assert result.action == "deny"
        assert result.rule_id is None
