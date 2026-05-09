import random
from dataclasses import dataclass
from typing import List, Optional
from ipaddress import ip_network, ip_address
import struct

@dataclass
class Packet:
    src_ip: str
    dst_ip: str
    protocol: str
    port: int

class TrafficGenerator:
    PROTOCOLS = ['tcp', 'udp', 'icmp']
    
    @staticmethod
    def random_ip() -> str:
        return ".".join(str(random.randint(0, 255)) for _ in range(4))
    
    @staticmethod
    def random_ip_in_cidr(cidr: str) -> str:
        network = ip_network(cidr, strict=False)
        if network.num_addresses == 1:
            return str(network.network_address)
        # 生成网络内的随机IP（排除网络地址和广播地址）
        start = int(network.network_address) + 1
        end = int(network.broadcast_address) - 1
        if start > end:
            return str(network.network_address)
        random_addr = random.randint(start, end)
        return str(ip_address(random_addr))
    
    @staticmethod
    def random_port(min_port: int = 1, max_port: int = 65535) -> int:
        return random.randint(min_port, max_port)
    
    @staticmethod
    def random_protocol() -> str:
        return random.choice(TrafficGenerator.PROTOCOLS)
    
    @staticmethod
    def generate_packet(
        src_cidr: Optional[str] = None,
        dst_cidr: Optional[str] = None,
        protocols: Optional[List[str]] = None,
        port_range: Optional[List[int]] = None
    ) -> Packet:
        # 生成源IP
        if src_cidr:
            src_ip = TrafficGenerator.random_ip_in_cidr(src_cidr)
        else:
            src_ip = TrafficGenerator.random_ip()
        
        # 生成目的IP
        if dst_cidr:
            dst_ip = TrafficGenerator.random_ip_in_cidr(dst_cidr)
        else:
            dst_ip = TrafficGenerator.random_ip()
        
        # 生成协议
        if protocols:
            protocol = random.choice(protocols)
        else:
            protocol = TrafficGenerator.random_protocol()
        
        # 生成端口（ICMP协议端口固定为0）
        if protocol == 'icmp':
            port = 0
        elif port_range:
            port = random.randint(port_range[0], port_range[1])
        else:
            port = TrafficGenerator.random_port()
        
        return Packet(src_ip=src_ip, dst_ip=dst_ip, protocol=protocol, port=port)
    
    @staticmethod
    def generate_packets(
        count: int = 500,
        src_cidr: Optional[str] = None,
        dst_cidr: Optional[str] = None,
        protocols: Optional[List[str]] = None,
        port_range: Optional[List[int]] = None
    ) -> List[Packet]:
        return [
            TrafficGenerator.generate_packet(src_cidr, dst_cidr, protocols, port_range)
            for _ in range(count)
        ]
