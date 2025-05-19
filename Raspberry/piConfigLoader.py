import yaml
import os
from dacite import from_dict
from dataclasses import dataclass


@dataclass
class PiNetwork:
    listening_ip: str
    server_ip: str
    server_port: int
    receive_port: int
    sample_rate: int  # should not be touched
    channels: int  # should not be touched
    chunk_size: int
    output_volume: float
    buffer_ms: int
    max_udp_packet: int


@dataclass
class PiConfig:
    network: PiNetwork


class Loader:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(current_dir, "piConfig.yaml")
        with open(file, "r") as config:
            self.config = yaml.safe_load(config)

    def load_pi_config(self):
        return from_dict(data_class=PiConfig, data=self.config["Pi"])
