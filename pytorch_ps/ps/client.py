from typing import Dict, List

import torch
import logging
from pytorch_ps.ps.server import LocalPsServer


class PSClient:
    def __init__(self, addr: str=None):
        self.addr = addr
        self.EMB_CONTEXT_OUTPUT: Dict[str, torch.Tensor] = {}
        self.remote = LocalPsServer()

    def _add_emb_context(self, name: str, output: torch.Tensor):
        if name in self.EMB_CONTEXT_OUTPUT:
            logging.debug("duplicate embedding name:", name)
            raise RuntimeError("duplicate embedding name:" + name)

        self.EMB_CONTEXT_OUTPUT[name] = output

    def _get_emb_context(self, name: str) -> torch.Tensor:
        return self.EMB_CONTEXT_OUTPUT[name]

    def _clear_emb_context(self):
        logging.debug("clear EMB_CONTEXT_OUTPUT")
        self.EMB_CONTEXT_OUTPUT.clear()

    def register(self, emb_name: str):
        self.remote.init_hash_tensor(emb_name)

    def get_tensor_from_dic(self, emb_name: str, indices: List[int], dim:int,init_value:List[float]=None)->torch.Tensor:
        tensors: List[torch.Tensor] = []

        for idx in indices:
            emb1 = self._get_tensor_from_dic(emb_name, int(idx))
            if emb1 is None:
                if init_value is None:
                    emb1 = torch.randn(dim, dtype=torch.float32, requires_grad=False)
                else:
                    emb1 = torch.tensor(init_value, dtype=torch.float32, requires_grad=False)
                self._set_tensor_to_dic(emb_name, int(idx), emb1)

            tensors.append(emb1)

        return torch.stack(tensors)

    def _get_tensor_from_dic(self, emb_name: str, idx: int):
        return self.remote.get_tensor_from_dic(emb_name, idx)

    def _set_tensor_to_dic(self, emb_name: str, idx: int, value: torch.Tensor):
        self.remote.set_tensor_to_dic(emb_name, idx, value)

    def update_tensor_to_dic(self, emb_name: str, indices: List[int], values: torch.Tensor):
        self.remote.update_tensor_to_dic(emb_name, indices, values)

    def get_weight(self, emb_name: str, emb_size: int,dim:int) -> torch.Tensor:
        return self.remote.get_weight(emb_name,emb_size,dim)

    def get_training_state(self, param_name: str, indices: List[int], dim: int) -> Dict[str, torch.Tensor]:
        stat = {}
        stat["exp_avg"] = self._get_exp_avg(param_name, indices, dim)
        stat["exp_sp_avg"] = self._get_exp_sp_avg(param_name, indices, dim)
        stat["step"] = torch.tensor(self._get_step(param_name), dtype=torch.long)
        return stat

    def set_training_state(self, param_name: str, indices: List[int], exp_avg: torch.Tensor, exp_sq_avg: torch.Tensor,
                           step: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._set_exp_avg(param_name, indices, exp_avg)
        self._set_exp_sp_avg(param_name, indices, exp_sq_avg)
        self._set_step(param_name, int(step.item()))

    def _get_exp_avg(self, param_name: str, indices: List[int], dim: int) -> torch.Tensor:
        return self.remote.get_exp_avg(param_name, indices, dim)

    def _set_exp_avg(self, param_name: str, indices: List[int], values: torch.Tensor):
        self.remote.set_exp_avg(param_name, indices, values)

    def _get_exp_sp_avg(self, param_name: str, indices: List[int], dim: int) -> torch.Tensor:
        return self.remote.get_exp_sp_avg(param_name, indices, dim)

    def _set_exp_sp_avg(self, param_name: str, indices: List[int], values: torch.Tensor):
        self.remote.set_exp_sp_avg(param_name, indices, values)

    def _set_step(self, param_name: str, step: int):
        self.remote.set_step(param_name, step)

    def _get_step(self, param_name: str) -> int:
        return self.remote.get_step(param_name)
