# Ultralytics 剪枝项目说明

本项目基于 Ultralytics 框架，支持模型训练与剪枝。通过灵活配置剪枝参数，可实现模型压缩与加速。

## 剪枝相关参数说明

在 `train.py` 中，模型训练与剪枝的主要参数如下：

- **prune_method**  
  剪枝方法。  
  例如：`'abs'` 表示按权重绝对值大小进行剪枝。
    `'wanda'` 表示按wanda++的方法进行剪枝。
    `'None'` 表示进行不剪枝训练。

- **prune_fre**  
  剪枝频率。  
  表示每隔多少个epoch进行一次剪枝。  
  例如：`prune_fre=100` 表示每100个epoch剪枝一次。

- **prune_final_sparsity**  
  剪枝稀疏率。  
  表示最终希望达到的稀疏率（被剪掉的比例）。  
  例如：`prune_final_sparsity=0.875` 表示最终保留12.5%的参数，其余剪除。

- **patience**  
  训练早停参数。  
  当验证集指标在指定epoch内无提升时，提前终止训练。

## 训练与剪枝示例

```python
model.train(
    data='coco.yaml',
    imgsz=640,
    device='0,1,2,3',
    epochs=150,
    batch=96,
    prune_method='wanda',
    patience=150,
    prune_fre=100,
    prune_final_sparsity=0.875
)
```

## 剪枝流程简介

1. **选择剪枝方法**：通过 `prune_method` 参数指定。
2. **设置剪枝频率**：通过 `prune_fre` 控制剪枝间隔。
3. **设定目标稀疏率**：通过 `prune_final_sparsity` 设定最终稀疏目标。
4. **训练与剪枝交替进行**：训练过程中按设定频率自动剪枝，逐步逼近目标稀疏率。


