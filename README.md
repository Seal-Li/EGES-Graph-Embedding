## 算法效果
在公开Amazon数据集上，仅使用1/20的特征，链接预测任务AUC可达96.6%；

原EGES算法，使用全量特征时，链接预测任务AUC可达97.0%。

## 算法能力
①完整复现了工业界常用的EGES图算法；

② 通过对其进行改进，目前已经支持对不同类型的节点进行metapath随机游走采样；

③ 充分利用了节点的特征信息，对节点的每个特征都会生成一个embedding，然后通过对节点的base embedding和特征的embedding进行聚合，得到节点的最终embedding；

④ 算法可以应对冷启动问题，对新节点的特征生成embedding，然后对使用特征的embedding进行聚合，得到新节点的初始化embedding；


## 算法参数
### 路径参数
root_path:存放数据的文件夹  

save_path:保存embedding的文件夹

### 随机游走参数
num_walks:随机游走的轮次  

walk_length:每次随机游走的序列长度  

schema:带节点类型的随机游走路径格式


### 模型参数
dim:节点base embedding和特征embedding的维度

batch_size:每个batch的节点数量

window_size:进行skip gram生成pair时选择的窗口宽度

num_negative:负采样比率

lr:初始学习率

shrinkage:学习率衰减因子

shrink_step:学习率衰减步长，即每隔多少个epoch衰减一次学习率

num_features:节点特征的维度

epoch:模型训练的轮次

### 其他参数
log_every:打印日志的step间隔


## 数据格式
### 1 edge.txt
共两列，第一列为源节点，第二列为目标节点，中间以空格分隔，格式示例：src dst

### 2 node_type.txt
共两列，第一列为节点id，第二列为节点类型，中间以空格分隔，格式示例：node_id node_type

### 3 side_info.txt
第一列为节点id，其余列为节点特征，节点与特征、特征与特征之间以空格分隔，示例：node_id feature_1 feature_2 ... feature_n