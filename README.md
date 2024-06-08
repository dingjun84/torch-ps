# torch-ps
In the recommendation scenario, the dimension of users or items is very large, and it may be impossible for the single machine memory to store it. Although there is the torchrec project under the pytorch framework to solve the problem of model training in the recommendation scenario, sometimes it may be necessary to align the training scheme of data parallelism + model parameter sharing of tensorflow in the CPU scenario.
The main purpose of this project is to verify the remote parameter server scheme under the pytorch framework.
## Implementation details
- Customized embedding and optimizer
- The weight of the embedding is a sparse tensor that does not store the actual data, only as the carrier of the gradient
- In the optimizer, according to the name bound by the weight and the row number of the first dimension of the embedding that needs to update the gradient, the weight on the remote parameter server is updated
- The weight in the embedding can be distributed across multiple parameter server nodes according to the first dimension
## Test demo
Python >= 3.9  
Pytorch >= 2.2  
`
python3 test_hash_embedding.py
`  
[ Translated by https://doubao.com ]


# torch-ps
在推荐场景，用户或者物品维度非常大，可能单机内存无法存放，在pytorch框架下虽然有torchrec项目来解决推荐场景模型训练的问题，但是有时可能需要在CPU场景下对齐tensorflow的数据并行+模型参数共享的训练方案。
本项目主要目的为验证pytorch框架下的remote parameter server方案。

## 实现细节
- 定制embedding和optimizer
- embedding的weight为sparse tensor不存放实际数据，只作为梯度的载体
- 在optimizer中，根据weight绑定的名称和需要更新梯度的embedding第一维的行号，更新远程parameter server上的weight
- embedding中的weight可以按第一维分布在多个parameter server 节点上

## 测试demo
python >= 3.9  
pytorch >= 2.2  
`
python3 test_hash_embedding.py
`