### 第一次模型训练结果
```shell
200 epochs completed in 0.231 hours.
Optimizer stripped from runs\train\exp8\weights\last.pt, 173.1MB
Optimizer stripped from runs\train\exp8\weights\best.pt, 173.1MB

Validating runs\train\exp8\weights\best.pt...
Fusing layers... 
Model summary: 322 layers, 86180143 parameters, 0 gradients, 203.8 GFLOPs
Class Images Instances P R mAP50 mAP50-95: 100%|██████████| 1/1 [00:00<00:00, 3.25it/s]
all 7 95 0.84 0.884 0.877 0.657
luggage 7 55 0.823 0.818 0.832 0.621
person 7 40 0.857 0.95 0.921 0.692
Results saved to runs\train\exp8
```
#### 测试结果评价 
人物检测精度足够，行李箱精度足够，背包，挎包等出现混乱和识别不出的情况
#### 改进措施
1.增加背包挎包手提包等多样性行李的数据集进行训练，2.提高训练轮数。

### 第二次训练结果
```shell
Stopping training early as no improvement observed in last 100 epochs. Best results observed at epoch 185, best model saved as best.pt.
To update EarlyStopping(patience=100) pass a new patience value, i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.

286 epochs completed in 0.402 hours.
Optimizer stripped from runs\train\exp\weights\last.pt, 173.1MB
Optimizer stripped from runs\train\exp\weights\best.pt, 173.1MB

Validating runs\train\exp\weights\best.pt...
Fusing layers...
Model summary: 322 layers, 86180143 parameters, 0 gradients, 203.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 1/1 [00:00<00:00,  2.70it/s]
                   all          9        129      0.899      0.804      0.864      0.623
               luggage          9         77      0.861      0.766      0.832      0.606
                person          9         52      0.936      0.843      0.896      0.641
Results saved to runs\train\exp
```

> 本次训练中，yolo自动终止了训练，因为在过去的100轮中，并没有任何改善，故保存了最佳的第185次