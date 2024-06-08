import logging
from typing import Dict, List

import torch

"""
simple demo
not thread safely
"""
class LocalPsServer:
    def __init__(self):
        logging.info("init ps server")
        self.GLOBAL_HASH_TENSOR:Dict[str,Dict[int,torch.Tensor]] = {}
        self.GLOBAL_EXP_AVG: Dict[str, Dict[int, torch.Tensor]] = {}
        self.GLOBAL_EXP_SP_AVG: Dict[str, Dict[int, torch.Tensor]] = {}
        self.GLOBAL_STEP:Dict[str,int] = {}

    def init_hash_tensor(self,emb_name:str):
        self.GLOBAL_HASH_TENSOR[emb_name]={}

    def get_weight(self,emb_name:str,emb_size:int,dim:int) -> torch.Tensor:
        data:List[torch.Tensor] = []
        p_dic:Dict[int,torch.Tensor] = self.GLOBAL_HASH_TENSOR[emb_name]
        for i in range(emb_size):
            if i not in p_dic:
                data.append(torch.zeros(dim,dtype=torch.float32))
            else:
                data.append(p_dic[i])

        return torch.stack(data)

    def get_tensor_from_dic(self,emb_name:str,id:int):
        if id not in self.GLOBAL_HASH_TENSOR[emb_name]:
            return None
        else:
            return self.GLOBAL_HASH_TENSOR[emb_name][id]

    def set_tensor_to_dic(self,emb_name: str, id:int, value: torch.Tensor):
        self.GLOBAL_HASH_TENSOR[emb_name][id] = value

    def update_tensor_to_dic(self,emb_name: str, indices: List[int], values: torch.Tensor):
        if emb_name not in self.GLOBAL_HASH_TENSOR:
            raise RuntimeError(emb_name + " not in GLOBAL_HASH_TENSOR")

        p_dic = self.GLOBAL_HASH_TENSOR[emb_name]
        for ind, id in enumerate(indices):
            p_dic[id] = torch.index_select(values, 0,torch.tensor([ind], dtype=torch.long)).squeeze()

    def get_exp_avg(self,param_name: str, indices: List[int], dim: int) -> torch.Tensor:
        if param_name not in self.GLOBAL_EXP_AVG:
            self.GLOBAL_EXP_AVG[param_name] = {}

        data = []
        p_dic = self.GLOBAL_EXP_AVG[param_name]
        for i, v in enumerate(indices):
            if v not in p_dic:
                p_dic[v] = torch.zeros(dim, dtype=torch.float32)

            data.append(p_dic[v])
        return torch.stack(data)

    def set_exp_avg(self,param_name: str, indices: List[int], values: torch.Tensor):
        if param_name not in self.GLOBAL_EXP_AVG:
            raise RuntimeError(param_name + " not in GLOBAL_EXP_AVG")

        p_dic = self.GLOBAL_EXP_AVG[param_name]
        for ind, id in enumerate(indices):
            p_dic[id] = torch.index_select(values, 0, torch.tensor([ind], dtype=torch.long)).squeeze()

    def get_exp_sp_avg(self,param_name: str, indices: List[int], dim: int) -> torch.Tensor:
        if param_name not in self.GLOBAL_EXP_SP_AVG:
            self.GLOBAL_EXP_SP_AVG[param_name] = {}

        data = []
        p_dic = self.GLOBAL_EXP_SP_AVG[param_name]
        for i, v in enumerate(indices):
            if v not in p_dic:
                p_dic[v] = torch.zeros(dim, dtype=torch.float32)

            data.append(p_dic[v])

        return torch.stack(data)

    def set_exp_sp_avg(self,param_name: str, indices: List[int], values: torch.Tensor):
        if param_name not in self.GLOBAL_EXP_SP_AVG:
            raise RuntimeError(param_name + " not in GLOBAL_EXP_SP_AVG")

        p_dic = self.GLOBAL_EXP_SP_AVG[param_name]
        for ind, id in enumerate(indices):
            p_dic[id] = torch.index_select(values, 0, torch.tensor([ind], dtype=torch.long)).squeeze()

    def set_step(self,param_name:str,step:int):
        if param_name not in self.GLOBAL_STEP:
            self.GLOBAL_STEP[param_name] = step
            return

        self.GLOBAL_STEP[param_name] +=1

    def get_step(self,param_name:str):
        if param_name not in self.GLOBAL_STEP:
            self.GLOBAL_STEP[param_name] = 0
        return self.GLOBAL_STEP[param_name]

