
import logging

from typing import List
import torch

from pytorch_ps.ps.client import PSClient

""" store embedding"""

emb_logger = logging.getLogger("emb")

class HashEmbeddingFunc1(torch.autograd.Function):

    @staticmethod
    def forward(selected_dense: torch.Tensor,input:torch.Tensor,weight:torch.Tensor) -> torch.Tensor:
        """
        RuntimeError: A input that has been returned as-is as output is being saved for backward. This is not supported if you override setup_context. You should return and save a view of the input instead, e.g. with x.view_as(x) or setup ctx inside the forward function itself

        """
        return selected_dense.clone()


    def setup_context(ctx, inputs, output):
        _selected_dense,input, weight = inputs
        ctx.save_for_backward(input, weight,output)
        logging.info("HashEmbeddingFunc1 setup_context")

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor) -> torch.Tensor:
        indices,weight,output = ctx.saved_tensors
        if emb_logger.isEnabledFor(logging.DEBUG):
            emb_logger.debug(f"weight size:{weight.size()} grad.is_sparse:{grad_output.is_sparse} grad_output:{grad_output}")

        """todo: use torch operator """
        sparse_indices = [[], []]
        values = []
        rows = grad_output.tolist()
        for ind1, idx in enumerate(indices.tolist()):
            for index, v in enumerate(rows[ind1]):
                sparse_indices[0].append(int(idx))
                sparse_indices[1].append(index)
                values.append(v)

        if emb_logger.isEnabledFor(logging.DEBUG):
            emb_logger.debug(f"weight size:{weight.size()} sparse_indices:{sparse_indices},values:{values}")

        grad1 = torch.sparse_coo_tensor(indices=sparse_indices, values=values, size=weight.size())
        return None,None,grad1


class HashEmbeddingLayer(torch.nn.Module):
    def __init__(self,ps_client:PSClient, name:str,emb_size:int,dim:int,init_value:List[float]=None):
        super().__init__()
        self.emb_size = emb_size
        self.dim = dim
        self.init_value = init_value
        self._param_name = "hash_emb_weight_"+name
        self.ps_client = ps_client
        param = torch.sparse_coo_tensor(indices=[[],[]], values=[], size=[self.emb_size, self.dim])
        """set parameter"""
        self.__setattr__(self._param_name,torch.nn.Parameter(data=param,requires_grad=True))
        self.ps_client.register(self._param_name)

    def get_hash_parameter(self) -> torch.Tensor:
        """
        just for test
        :return: real weight
        """
        return self.ps_client.get_weight(self._param_name,self.emb_size,self.dim)

    def forward(self,input):
        idx = input.tolist()
        selected_dense = self.ps_client.get_tensor_from_dic(self._param_name,idx,self.dim,self.init_value)

        output = HashEmbeddingFunc1.apply(selected_dense,input,self.__getattr__(self._param_name))
        # cann't duplicate or reuse embedding instance
        self.ps_client._add_emb_context(self._param_name, output)
        return output




