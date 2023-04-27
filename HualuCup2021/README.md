

# Docker镜像

使用以下命令拉取Docker镜像：

```bash
sudo docker pull registry.cn-shanghai.aliyuncs.com/shawnd/hlb:v0929
```

使用以下命令进入容器：

```bash
sudo docker run -it --shm-size 8G --gpus '"device=0,1,2,3"' registry.cn-shanghai.aliyuncs.com/shawnd/hlb:v0929 /bin/bash
```

# CLIPCap

​		CLIP(Contrastive Language–Image Pre-training)  模型通过文本编码器（本项目使用经过预训练的 GPT 模型）将文本编码为特征向量， 通过图像编码器（本项目使用在 ImageNet 上预训练过的 Res2Next50 模型）将图像编码为特征向量， 通过计算文本特征向量与图像特征向量的内积， 得到文本与图像的相似度。CLIP 模型的任务就是使图像和文本编码器编码出的向量尽可能地相似。

​		受到 CLIP 模型的启发， 我们认为 CLIP 模型中的图像编码器学习到了文本编码器中的知识， 同样地， 文本编码器也应该学习到了图像编码器中的知识。 因此，CLIP 模型中的图像编码器编码了更复杂的语义信息， 我们选择 CLIP 中的图像编码器作为 Image Caption任务中的图像编码器。

​		我们希望构造一个生成器将图像编码器中编码的复杂的语义信息提取出来， 用以完成 Image Caption 任务。受到 VisualGPT 的启发， 他们认为经过预训练的语言模型中已经学习到了文本的知识，可以将此部分知识迁移，使用预训练过的 GPT 模型作为 Caption 的生成器，可以减少训练所需要的样本数。 因此， 我们选择使用经过预训练的 GPT 模型作为 Image Caption 任务中的 Caption 生成器。此外， 因为 Image Caption 中的文本生成器 与 CLIP 模型中文本编码器具有完全相同的结构， 为了进一步利用模型学习的知识， 我们将 Image Caption 中的文本生成器 加载 预训练 CLIP 模型文本编码器的权重。

# 数据集

​		**Flickr8k-CN & Flickr30k-CN**： 针对图像描述任务（Image Captioning)，将英文公开数据集Flickr8K（8 千图像、4 万英文描述）、Flickr30K（3 万图像、15 万英文描述）中的英文描述使用机器翻译将翻译为中文描述，其中，测试集通过人工翻译成中文描述，构建了Flickr8K-CN、和 Flickr30k-CN 图像描述数据集。

​		**AIC-IIC**： AIC-ICC(Image Chinese Captioning from AI Challenge)包含 21 万图像和 105万图像的中文描述，涵盖日常生活常见的 200 多个场景，如足球场、草地等场景，唱歌、跑步等动作，是目前图像描述领域最大的中文数据集。

​		以及**华录杯官方提供数据集**。

​		以上为打榜期间使用数据集， 因为复赛B榜与A榜分布差异较大，许多模型并未见过视觉概念重复出现， 因此在此次提交的数据集中额外增加了以下数据集：

​		**COCO-CN**：  COCO-CN数据集是一个添加了人工标注中文句子和标签的双语图像描述数据集。该数据集可用于跨语言的包括图像标注、图像描述和图像检索等多个任务。

​		COCO-CN数据集中出现许多与复赛B榜数据集中相似的视觉概念， 因此复现成绩应该比打榜成绩更好。

# 算法结构

## 网络结构

​		训练主要包括预训练和图像描述生成微调两个阶段。

​		**CLIP预训练阶段**： CLIP 模型的文本编码器选择使用 GPT 语言模型，而图像编码器选择使用经过预训练的 Res2Next50 模型。 将图像输入 CLIP 模型的图像编码器得到图像特征， 将文本输入 CLIP 模型的文本编码器得到文本特征， 将文本特征与图像特征的内积作为模型的输出， 理想情况下其输出应为对角线上值大，而其他位置输出值小，因此其监督信息为一个对角矩阵。通过大量的图像文本对进行无监督训练， CLIP 模型的图像编码器学习文本中的概念， 编码复杂的语义信息， 相应的， CLIP 模型的文本编码器学习图像中的视觉概念。



![图片1](./figure/图片1.png)

​		我们使用 GPT 模型的 表示句子结束的符号(EOS) 作为整个句子的表征， 使用Res2Next50 模型全连接层前的池化层输出作为图像的表征， 再分别使用两个多层感知机分别将句子的表征和图像的表征投影到一个 768维的跨模态联合表征空间 $\mathbb{R}^{768 \times 1}$。

​		**图像描述生成微调阶段**： 既然 CLIP 模型的图像编码器编码了复杂的语义信息， 因此我们想要通过一个强大的文本生成器提取 CLIP模型的图像编码器中编码的语义信息。受到 VisualGPT 的启发，经过预训练的语言模型中已经学习到了文本的知识，可以将此部分知识迁移，减少训练所需要的样本数。 因此，我们选择使用经过预训练的 GPT 模型作为 Image Caption 任务中的 Caption 生成器。

​		为了进一步利用 CLIP 模型中的文本编码器学习到的图像中的视觉概念， 我们将 Image Caption任务中的 GPT 模型 与 CLIP 模型的文本编码器(也使用 GPT 模型) 共享权重。

![图片2](./figure/图片2.png)



## 损失函数

​		**CLIP预训练阶段**： 预训练阶段对相似度分数计算对称交叉熵损失。给定一个批量大小为 $N$ 的 图像文本对， CLIP模型预测 $N \times N$ 个 图像文本对的余弦相似度 。 CLIP模型训练一个图像编码器和一个文本编码器， 图像编码器输出图像嵌入， 文本编码器输出文本嵌入， 最大化在一个批量中真实匹配的文本嵌入和图像嵌入的余弦相似度， 同时最小化 $N^2–N$ 个不正确匹配的文本嵌入和图像嵌入的余弦相似度， 以此学习一个多模态嵌入空间。使用对称交叉熵损失函数优化相似度分数。其伪代码如下：

```python
# image_encoder - ResNet 或者 Vision Transformer 
# text_encoder  - CBOW 或者 Text Transformer 
# I[n, h, w, c] - 批量图像文本对 中的图像 
# T[n, l]       - 批量图像文本对 中的文本  
# W_i[d_i, d_e] - 图像学习到的投影嵌入
# W_t[d_t, d_e] - 文本学习到的投影嵌入 
# t             - 学习的参数 

# 提取每个模态的特征表征
I_f = image_encoder(I) #[n, d_i] 
T_f = text_encoder(T)  #[n, d_t] 

# 多模态联合嵌入空间 [n, d_e] 
I_e = l2_normalize(np.dot(I_f, W_i), axis=1) 
T_e = l2_normalize(np.dot(T_f, W_t), axis=1) 

# 缩放成对的余弦相似度 [n, n] 
logits = np.dot(I_e, T_e.T) * np.exp(t) 

# 对称交叉熵损失函数 
labels = np.arange(n) 
loss_i = cross_entropy_loss(logits, labels, axis=0) 
loss_t = cross_entropy_loss(logits, labels, axis=1) 
loss   = (loss_i + loss_t)/2 
```

**图像描述生成微调阶段**： 

​		给定图像， 我们的目标是最大化 正确描述 的概率， 公式如下：

$$
\theta^* = \arg \max_{\theta} \sum_{(I, S)} \log p(S \mid I; \theta)
$$

​		其中 $\theta$ 是模型的参数， $I$ 是图像， $S$ 是图像的正确描述。 通常对模型中 $S_0, ..., S_N$ 的联合概率使用链式法则：


$$
\log p(S \mid I; \theta) = \sum_{t=0}^N \log p(S_t \mid I, S_0, ..., S_{t-1}; \theta)
$$

​		在训练时， $(S, I)$ 是一对训练样本， 我们在所有训练集上使用随机梯度梯度下降优化上式中的对数概率之和。 我们选择使用 GPT 模型建模自然语言模型 $p(S_t \mid I, S_0, ..., S_{t-1})$。



​		最终， 图像描述生成微调阶段的损失函数为对每个时间步生成的词计算负对数似然函数之和， 其公式如下：


$$
L(I, S) = - \sum_{t=1}^N \log p_t(S_t) 
$$


## 优化器

​        预训练阶段 和 图像描述生成阶段 都使用 AdamW 优化器， 学习率 $3 \times 10^{-4}$， 权重衰减
0.01。



# 算法运行说明

运行容器后进入文件夹 `HLB_Last`

```bash
cd /home/HLB_Last
```



## 文件结构

├── checkpoints										  # 模型保存的位置
├── config													# 保存 配置文件 以及 vocabulary 的位置
│   ├── cfg.py
│   └── vocab_cn.json
├── data 													# 读取数据集
│   ├── coco_dataset.py
│   ├── hlb_dataset.py
│   ├── icap_dataset.py
│   └── tokenizer.py
├── evaluation 										# 评价指标
│   ├── bleu
│   ├── cider
│   ├── meteor
│   ├── rouge
│   └── tokenizer.py
├── logs 													# 保存训练记录
├── models 											# 定义 模型 的位置
│   ├── clip.py
│   ├── gpt.py
│   └── vit.py
├── scripts 											# 脚本文件

└── utils 												# 工具类函数
├── cap_inference.py
├── clip_inference.py
├── predict.py
├── ensemble.py
├── train_cap.py
├── train_clip.py
├── train.py
├── README.md

# 检查代码能否顺利执行

```bash
python3 train.py --debug
```

建议等预训练的5个 epoch 结束后， 在微调阶段分别训练和测试一个epoch后停止测试程序。

运行正式训练代码。

## 训练

```bash
python3 train.py --seed 42 --clip_save_checkpoint_path ./checkpoints/AiO_clip_best_model_0923.pth.tar --cap_save_checkpoint_path ./checkpoints/AiO_cap_best_model_0923.pth.tar 
```

## 推理

```bash
python3 predict.py --cap_save_checkpoint_path ./checkpoints/AiO_cap_best_model_0923.pth.tar 
```

输出 `res.json` ， 即为推理的结果。

因为复赛B榜与A榜分布差异较大，许多模型并未见过视觉概念重复出现， 因此在此次提交的数据集中额外增加了  COCO-CN 数据集。  COCO-CN数据集中出现许多与复赛B榜数据集中相似的视觉概念， 因此复现成绩应该比打榜成绩更好。

## 模型融合

由于训练时间比较久， 模型融合使用提交的本地训练权重， 其性能在 **算法性能** 部分展示。

```bash
python3 ensemble.py --model1 ./checkpoints/AiO_cap_best_model_0923.pth.tar --model2 ./checkpoints/AiO_cap_best_model_0920.pth.tar --model3 ./checkpoints/AiO_cap_best_model_0913.pth.tar
```

输出 `res_ensemble.json` ， 即为模型融合推理的结果

**注：本地权重非最好结果， 如在训练脚本 `train.py` 中使用不同随机种子训练得到不同权重， 再进行融合可得到更好的表现**

# 算法性能

|模型|复赛A榜|复赛B榜|
|:-:|:-:|:-:|
|model1|30.02|-|
|model2|28.54|-|
|model3|28.52|-|
|model4|-|19.78|
|ensemble 1+2+3|30.72|20.19|
|ensemble 1+2+4|-|20.49|


训练及推理的线上环境为 双卡3090， 1卡占用显存 18559MiB， 2卡占用 17661 MiB

单模型平均每张图片推理时间为 187.4ms
多模型平均每张图片推理时间为 512.3ms



# 改进创新点

我们的创新点主要如下：

1. 首次基于 CLIP 模型实现中文图文多模态模型。
2. 提出 CLIP 模型的图像编码器编码了复杂的语义信息， 使用预训练的 GPT 模型作为文本生成器提取 CLIP 模型的图像编码器中编码的语义信息， 有效减少训练所需的样本数量。
3. 通过文本生成器与 CLIP 模型的文本编码器共享权重，利用了无监督对比学习过程中文本编码器所学习到的知识， 同时进一步减少训练所需的样本数量。

# Reference

1. Radford A, Kim J W, Hallacy C, et al. Learning transferable visual models from natural language supervision[J]. arXiv preprint arXiv:2103.00020, 2021.
2. Radford A, Narasimhan K, Salimans T, et al. Improving language understanding by generative pre-training[J]. 2018.
3. Radford A, Wu J, Child R, et al. Language models are unsupervised multitask learners[J]. OpenAI blog, 2019, 1(8): 9.
4. Chen J, Guo H, Yi K, et al. VisualGPT: Data-efficient Adaptation of Pretrained Language Models for Image Captioning[J]. arXiv preprint arXiv:2102.10407, 2021.
5. Vinyals O, Toshev A, Bengio S, et al. Show and tell: A neural image caption generator[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 3156-3164.