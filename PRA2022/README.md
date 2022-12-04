# 数据集处理以及预训练模型

需要将初赛、初赛补充以及复赛的所有数据集汇总在一起， 这步我们已经完成，汇总的数据集分别放在 `datasets/det_all` 和 `datasets/seg_all` 文件夹中。

训练用到了Paddle官方提供的预训练模型yolov3_r34_270e_coco.pdparams 和 hardnet.pdparams。这两个文件已经分别放在 `pretrained_weights/det/` 和 `pretrained_weights/seg/` 文件夹下。

# 检测模型训练

进入目标检测文件目录：

```
cd PaddleDet
```

运行训练脚本：

```
bash scripts/train_yolov3_res34.sh
```

训练结束后，将 `output/yolov3_res34/best_model.pdparams` 移动到 `../model/det/` 目录下， 并将其重命名为 `yolov3_res34.pdparams`。

```
mv output/yolov3_res34/best_model.pdparams ../model/det/yolov3_res34.pdparams
```

# 分割模型训练

进入语义分割文件目录：

```
cd ppseg
```

运行训练脚本：

```
bash scripts/train_hardnet.sh
```

训练结束后，将 `output/best_model/model.pdparams` 移动到 `../model/seg/` 目录下， 并将其重命名为 `hardnet.pdparams`。

```
mv output/best_model/model.pdparams ../model/seg/hardnet.pdparams
```


# 推理

将训练好的检测模型和分割模型分别置于 `model/det` 和 `model/seg` 文件夹后，在 PaddleDet 和 ppseg 的上层目录下， 运行如下脚本：

```
python3 predict.py data_val.txt result.json
```

将 `data_val.txt` 替换为测试数据。 