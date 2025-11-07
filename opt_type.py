from dataclasses import dataclass
from typing import Any, List


@dataclass
class SSHopt:
    dir_name: str
    task_name: str
    
    @dataclass
    class SSHargs:
        n_nodes: int
        group_nodes:int
        seq_length: int
        batch_size: int
        nhid: int
        in_dim:int
        total_epoch: int
        update_every: int
        show_graph_every: int
        
        @dataclass
        class data_pred:
            model: str
            pred_step: int
            mlp_hid: int
            mlp_layers: int
            lr_data_start: float
            lr_data_end: float
            weight_decay: int

        @dataclass
        class graph_discov:
            lambda_s_end=0.1
            lambda_s_start=0.1
            lambda_s: 0.1
            lr_graph_start: float
            lr_graph_end: float
            start_tau: 0.3
            end_tau: 0.01
            dynamic_sampling_milestones: list
            dynamic_sampling_periods: list

    causal_thres: str
    