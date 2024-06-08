import logging
import math
import time

import torch

from pytorch_ps.nn.hash_embedding import HashEmbeddingLayer
from pytorch_ps.nn.sparse_adam import HashSparseAdam
from pytorch_ps.ps.client import PSClient

def test_one():
    inputs1 = torch.tensor([2, 4, 5], dtype=torch.long, requires_grad=False)
    inputs2 = torch.tensor([10, 11, 12], dtype=torch.long, requires_grad=False)

    labels = torch.tensor([0.20, 0.44, 0.60], dtype=torch.float32, requires_grad=False)
    ps_client = PSClient()
    emb_layer1 = HashEmbeddingLayer(ps_client,"sparse_test1", 10_000_000_000, 4,[0.1,0.2,0.3,0.4])
    emb_layer2 = HashEmbeddingLayer(ps_client,"sparse_test2", 10_000_000_000, 8,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])

    params_name = []
    params_v = []


    for n, v in emb_layer1.named_parameters():
        params_name.append(n)
        params_v.append(v)

    for n, v in emb_layer2.named_parameters():
        params_name.append(n)
        params_v.append(v)

    output1: torch.Tensor = emb_layer1(inputs1)
    output2: torch.Tensor = emb_layer2(inputs2)

    hidden_inputs = torch.concat([output1, output2], dim=1)
    print(f"hidden_inputs:%s" % (hidden_inputs))

    loss = torch.mean(hidden_inputs)
    loss.backward()

def test_lr():
    inputs1 = torch.tensor([2, 4, 5], dtype=torch.long, requires_grad=False)
    inputs2 = torch.tensor([10, 11, 12], dtype=torch.long, requires_grad=False)

    labels = torch.tensor([0.20, 0.44, 0.60], dtype=torch.float32, requires_grad=False)
    ps_client = PSClient()

    emb_layer1 = HashEmbeddingLayer(ps_client, "sparse_test1", 10_000_000_000, 4, [0.1, 0.2, 0.3, 0.4])
    emb_layer2 = HashEmbeddingLayer(ps_client, "sparse_test2", 10_000_000_000, 8,
                                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    params_name = []
    params_v = []

    mlp = torch.nn.Sequential(
        torch.nn.Linear(12, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 1)
    )

    for n, v in emb_layer1.named_parameters():
        params_name.append(n)
        params_v.append(v)

    for n, v in emb_layer2.named_parameters():
        params_name.append(n)
        params_v.append(v)

    sparse_optim = HashSparseAdam(ps_client,params_name, params_v, lr=0.002)
    dense_optim = torch.optim.Adam(mlp.parameters(), lr=0.002)
    lossfn = torch.nn.L1Loss()

    for epoch in range(1000):
        output1: torch.Tensor = emb_layer1(inputs1)
        output2: torch.Tensor = emb_layer2(inputs2)

        hidden_inputs = torch.concat([output1, output2], dim=1)
        #print(f"hidden_inputs:%s" % (hidden_inputs))

        pred_y = mlp(hidden_inputs)
        print("pred_y", pred_y)

        loss = lossfn(pred_y.squeeze(), labels)

        loss.backward()
        sparse_optim.step()
        dense_optim.zero_grad()

        sparse_optim.zero_grad()
        dense_optim.zero_grad()

        print(f"************%dloss:%s" % (epoch, loss))
        if loss < 0.05:
            break

    time.sleep(1)

def _compare_grad1(init_data:list,init_data2:list,init_data3:list):

    ps_client = PSClient()
    hash_emb_layer = HashEmbeddingLayer(ps_client,"test1",10,4,[0.0,0.0,0.0,0.0])

    params_name = []
    params_v = []

    for n, v in hash_emb_layer.named_parameters():
        params_name.append(n)
        params_v.append(v)

    sparse_optim = HashSparseAdam(ps_client, params_name, params_v, lr=0.002)

    mlp1 = torch.nn.Sequential(
        TestLinear(len(init_data[0]), len(init_data), _weight=init_data),
        torch.nn.ReLU(),
        TestLinear(len(init_data2[0]), len(init_data2), _weight=init_data2),
        torch.nn.Sigmoid()
    )
    dense_optim1 = torch.optim.Adam(mlp1.parameters(), lr=0.002)
    lossfn1 = torch.nn.BCEWithLogitsLoss()

    inputs1 = torch.tensor([2, 4, 5], dtype=torch.long, requires_grad=False)
    labels1 = torch.tensor([0.5, 1, 0.9], dtype=torch.float32, requires_grad=False)

    inputs1_other = torch.tensor(init_data3,dtype=torch.float32)

    for epoch in range(10):
        hidden_output1 = hash_emb_layer(inputs1)
        pred_y1 = mlp1(torch.concat([hidden_output1,inputs1_other],dim=-1))
        loss1 = lossfn1(pred_y1.squeeze(), labels1)

        loss1.backward()
        sparse_optim.step()
        dense_optim1.step()

        sparse_optim.zero_grad()
        dense_optim1.zero_grad()

        yield hash_emb_layer.get_hash_parameter()

def _compare_grad2(init_data:list,init_data2:list,init_data3:list):
    torch_emb_layer = torch.nn.Embedding(10, 4, _weight=torch.zeros((10, 4), dtype=torch.float32))
    emb_optim2 = torch.optim.Adam(torch_emb_layer.parameters(), lr=0.002)

    mlp2 = torch.nn.Sequential(

        TestLinear(len(init_data[0]), len(init_data), _weight=init_data),
        torch.nn.ReLU(),
        TestLinear(len(init_data2[0]), len(init_data2), _weight=init_data2),
        torch.nn.Sigmoid()
    )
    dense_optim2 = torch.optim.Adam(mlp2.parameters(), lr=0.002)
    lossfn2 = torch.nn.BCEWithLogitsLoss()

    inputs2 = torch.tensor([2, 4, 5], dtype=torch.long, requires_grad=False)

    labels2 = torch.tensor([0.5, 1, 0.9], dtype=torch.float32, requires_grad=False)

    inputs2_other = torch.tensor(init_data3, dtype=torch.float32)
    for epoch in range(10):
        hidden_output2 = torch_emb_layer(inputs2)
        pred_y2 = mlp2(torch.concat([hidden_output2,inputs2_other],dim=-1))
        loss2 = lossfn2(pred_y2.squeeze(), labels2)

        loss2.backward()
        emb_optim2.step()
        dense_optim2.step()

        emb_optim2.zero_grad()
        dense_optim2.zero_grad()

        yield torch_emb_layer.parameters().__next__()

def run_compare_grad():
    input_dim = 16
    d1 = torch.randn((input_dim,32),dtype=torch.float32).t()
    d2 = torch.randn((32, 1), dtype=torch.float32).t()
    d3 = torch.randn((3, input_dim-4), dtype=torch.float32)

    for v1,v2 in zip(_compare_grad1(d1.tolist(),d2.tolist(),d3.tolist()),_compare_grad2(d1.tolist(),d2.tolist(),d3.tolist())):
        print("hash embedding:", torch.index_select(input=v1,dim=0,index=torch.tensor([2,4,5])))
        print("torch embedding:", torch.index_select(input=v1,dim=0,index=torch.tensor([2,4,5])))

        print(f"===================equal items:{torch.sum((v1.sub(v2)<0.00001).to(torch.int)).item()}====================")





class TestLinear(torch.nn.Module):
    """
    copy from torch.nn.Linear
    fix parameter for testing
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int,_weight:list = None, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if _weight is not None:
            self.weight = torch.nn.Parameter(torch.tensor(_weight))
            self.bias = torch.nn.Parameter(torch.zeros(out_features)+0.1)
        else:
            self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

if __name__=="__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.ERROR)
    test_one()
    test_lr()
    run_compare_grad()