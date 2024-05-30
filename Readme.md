## 任务描述

写一个脚本测试单机多卡(2，4，8卡)以及多机多卡（16卡）的GPU带宽，能够测量all-reduce, all-gather，broadcast和all-to-all的不同通信原语下的通信带宽，只能使用PyTorch以及python内置的包。

## 任务执行
多机器：

torchrun --nnode 2  --node_rank=1  --master-addr 172.27.183.198  --master-port 55555 --nproc_per_node 8 dist_bandwidth.py

单机器：

torchrun  --nproc_per_node 8 dist_bandwidth.py
## 任务结果


| 指标      |    2卡 | 4卡  |8卡  | 2机器*4卡 |2机器*8卡|
| :--- | ---:| :--: |:--: |:--: |:--: |
|all-to-all   | 617 |436|369|11.2
| all-reduce  | 319| 224|253|14.0
all-gather     |369| 313|290|7.9
broadcast     | 344| 338|338|6.0

