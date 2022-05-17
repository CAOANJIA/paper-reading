# Paper Reading

[TOC]

个人论文阅读笔记

## Image Caption

### Show and Tell(ensemble)

Vinyals, Oriol, et al. "Show and tell: A neural image caption generator." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2015. [[pdf](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)]

1. 用预训练CNN, Inceptionv3
2. LSTM输入只有第一个cell接受了图像编码；`word embedding`和`hiddensize`都是512d；SGD
3. BeamSearch
4. BLEU

```
总结：
机器翻译得到的启发encoder+decoder；
作者实验结论是只给第一个cell图像信息就够了
作者发现在语料库中建立的词表没啥作用，因此简单地随机初始化了word embedding weights
```



### Show, Attend and Tell [[PyTorch实现](https://github.com/CAOANJIA/show-attend-and-tell)]

Xu, Kelvin, et al. "Show, attend and tell: Neural image caption generation with visual attention." *International conference on machine learning*. PMLR, 2015. [[pdf](http://proceedings.mlr.press/v37/xuc15.pdf)]

1. VGG19
2. LSTM输入每一个cell都考虑了zt，即attend之后的图像编码，zt的计算也要用到ht-1
3. hard: REINFORCE; soft: SGD
4. soft是对于一个`14*14*512`的feature map，每个像素可以表示为一个512维的向量`aL`，考虑到感受野，其实每个像素都代表了一个区域，
   然后为每个像素(196个)一个512维的权值alphaL(用ht-1和aL计算)，相对于对每个像素都有权重，相乘就得到zt，这就是spatial attention
5. Flickr8k: official split; 	Flickr30k, MSCOCO: Karpathy splits; 
6. `batchsize 64`, 每次选相同长度的	`Adam`	避免过拟合：`Dropout`	*early stopping*
7. BLEU, METEOR


```
总结：
attention效果很好；
一般来说硬注意力比软注意力效果好；
双重正则化，鼓励模型看到每个像素；门控*注意力结果，鼓励模型关注物体
trick--每次选相同长度的作为一个batch，加快训练速度;
```



### SCA-CNN

Chen, Long, et al. "Sca-cnn: Spatial and channel-wise attention in convolutional networks for image captioning." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2017. [[pdf](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_SCA-CNN_Spatial_and_CVPR_2017_paper.pdf)]

1. CNN: `VGG19`/ `Res-152`	`LSTM`: word-embedding 100d, hidden-state 1000d	Attention Space: 512d both	
2. `spatial` + `channel-wise` + `multi-layer`，多层迭代，层内是先channel再spatial
3. channel-wise相当于选择语义的过程，因为可以说每个filter(channel)对应了一种模式
3. channel-wise：先看作[u1, ..., uC]共C个向量，每个向量都是`H*W`维的，然后可以*meanpool*，得到[v1, ..., vC]，共C个标量，因为只需要考虑平均值就可以衡量这个channel对应的模式存在的可能性
       spatial：[v1, ..., vm]共m个向量，m=H*W，每个向量都是C维的
5. attend之后继续接入后续的层，得到最终表示。结果表明：VGG只用spatial比较好（有FC，保留了空间信息），Res则是C-S好（meanpool，丢失了空间信息）C效果好的原因应该是channel多（2048），
6. Flickr8k: official split; 	Flickr30k, MSCOCO: Karpathy splits;
7. `batchsize 16，64，64`	`Adadelta`	BeamSearch-5结合长度归一化(testing时)	避免过拟合：`Dropout`	*early stopping*
8. BLEU, METEOR, CIDEr, ROUGE-L

```
总结：
需要通道多的时候可以发挥channelwise的作用，比如Res152有2048个channel，因此Res效果比VGG好;
C-S略比S-C好，相差不太大；
multi-layer有一定作用，但多了就可能过拟合；
trick--beamsearch结合lengthnorm效果更好，引用别人的结论；
```



### Meshed-Memory-Transformer

Cornia, Marcella, et al. "Meshed-memory transformer for image captioning." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cornia_Meshed-Memory_Transformer_for_Image_Captioning_CVPR_2020_paper.pdf)]

1. `multi-layer encoder`  +  `multi-layer decoder`, 每个layer内部是类transformer结构（multihead-self-attention + addnorm），但是加上了先验记忆向量
2. mesh结构，并且每条边都有权重
3. decoder是`masked self attention`
4. region-based-Encoder: `Faster-RCNN` + `ResNet-101(pretrained on Visual Genome)` 每个区域都对应一个2048维的向量 
5. Decoder: `one-hot vector linearly projection` + `sin positional embedding`
6. COCO上：`d=512`，`head=8`，`memory vector=40`，`dropout=0.1`，`Adam`（pretrain时用warmup，finetuneCIDEr时5*10-6），`batchsize=50`，beamsize=5
7. nocap上：`GloVe`
8. 去除低频5的词，区域最大数量50，nn.Module.register_buffer减少3倍训练时间，记忆向量初始化



### ClipCap

Mokady, Ron, Amir Hertz, and Amit H. Bermano. "Clipcap: Clip prefix for image captioning." *arXiv preprint arXiv:2111.09734* (2021). [[pdf](https://arxiv.org/pdf/2111.09734.pdf)]

1. `CLIP` + `Mapping Network` + `GPT2`（轻量级）

   以前两者的输出作为语言模型的`prefix`，然后自回归获得caption

2. 可以只训练Mapping Network：那么需要能力更强的`transformer`

   也可以训练MN+LM：只需要`MLP`，甚至一层都可以

3. 在*COCO*上~~不SOTA~~

   在更有挑战的*nocap*、*Conceptual Captions*数据集上<u>SOTA</u>

   *nocap*只有验证、测试集，训练集是直接用*COCO*，其有三大部分...

4. **KEY: Mapping Network**，其输入为`CLIP encoding`和一个`constant input`，constant input可以：

   - 获得多头注意力的信息

   - 调整模型，使其适应frozen的语言模型

5. baseline对比：`VLP`、`Oscar` （Oscar还用了额外的监督：对象标签）

   MLP+GPT2总体上都不如transformer，只有在Conceptual Captions上较好，可能是因为微调使其更具风格多样性，但微调容易过拟合

6. 训练时间短，说明lightweight，下面是COCO上的

   - [ ] VLP: V100----------------48h

   - [ ] Oscar: V100-------------74h


   - [ ] ClipCap: GTX 1080----7h(w/ fine-tuning)	GTX 1080----6H(w/o fine-tuning)

7. prefix的探究：

   - 根据余弦相似度，将其转换为接近的word embedding，看看是哪些单词

     在微调时prefix非常有用，单词和内容很相关

     不微调时可能是因为MN需要操纵固定的LM，且prefix在不同图像之间需要共享信息

   - 长度

     MLP架构受限，前缀较短，不能扩展到长前缀

     transformer可以增加前缀长度

   

### VSR(Verb-specific Semantic Roles)

Chen, Long, et al. "Human-like controllable image captioning with verb-specific semantic roles." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2021. [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Human-Like_Controllable_Image_Captioning_With_Verb-Specific_Semantic_Roles_CVPR_2021_paper.pdf)]

1. *"human-like"*的控制信号需要满足：

   - **Event-compatible**：人在潜意识下会保证所描述的事件是兼容的
   - **Sample-suitable**：需要预先知道这个控制信号对于此样本是合适的，比如简单的内容不能给很复杂的要求

2. 流行的内容控制和结构控制都有局限性，本文提出`VSR`，一个面向事件（活动）的客观控制信号，满足上述两个条件，其包含：

   - **一个动词**`verb`：捕获活动的范围

   - **一些用户感兴趣的语义角色**`semantic roles`：参与活动的的对象（参加此活动的对象都是兼容的，满足了第一个条件；只限制了所设计的语义角色，理论上适合所有具有活动的图像，满足第二个条件）

3. 三个主要模型：

   - `GSRL`（定位语义角色标记模型）：识别、定位每个语义角色（受NLP中的semantic roles labeling任务的启发）
   - `SSP`（语义结构规划器）：对给定的`verb`和`semantic roles`进行排序，输出一些“*human like*”描述性语义结构
   - `RNN-based role-shift captioning model`：依次关注不同的角色，给出caption

4. 利用语义角色解析工具包获得VSR注释

5. 数据集：*COCO、Flickr30k*

6. $$
   VSR = \{v, <s_1, n_1>, ..., <s_m, n_m>\}
   $$

   其中，*v*代表动词，*s*i代表语义角色，*n*i代表其实体的数量

7. 可以自动构建`VSR`

   - [ ] 对于动词，可以用一个`off-the-shelf action recognition network`

   - [ ] 对于语义角色，可以用*verb lexicon*（如`PropBank`或`FrameNet`）检索获得，然后自己选出得到的角色的子集，并随机分配实体号

8. 建模：
   $$
   p(y|I,VSR) &=& p(y|pattern)p(pattern|I,VSR)\\
              &=& p(y|S,R)p(S,R|\tilde{R},VSR)p(\tilde{R}|I,VSR)
   $$
   其中，*pattern*的作用是，先构建一个描述性的模式，然后完成细节的描述
   
   下面三个概率分别对应captioner、SSP、GSRL
   
   
   
   

## GAN

### GAN

Goodfellow, Ian, et al. "Generative adversarial nets." *Advances in neural information processing systems* 27 (2014). [[pdf](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)]





### DCGAN

Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." *arXiv preprint arXiv:1511.06434* (2015). [[pdf](https://arxiv.org/pdf/1511.06434.pdf%C3)]





### CycleGAN

Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." *Proceedings of the IEEE international conference on computer vision*. 2017. [[pdf](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)]





### CartoonGAN

Chen, Yang, Yu-Kun Lai, and Yong-Jin Liu. "Cartoongan: Generative adversarial networks for photo cartoonization." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018. [[pdf](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)]





### AnimeGAN

Chen, Jie, Gang Liu, and Xin Chen. "AnimeGAN: A novel lightweight gan for photo animation." *International Symposium on Intelligence Computation and Applications*. Springer, Singapore, 2019. [[pdf](https://link.springer.com/chapter/10.1007/978-981-15-5577-0_18)]





### StyleGAN

Karras, Tero, Samuli Laine, and Timo Aila. "A style-based generator architecture for generative adversarial networks." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2019. [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf)]





### StyleGANv2

Karras, Tero, et al. "Analyzing and improving the image quality of stylegan." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2020. [[pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.pdf)]





### StyleCLIP

Patashnik, Or, et al. "Styleclip: Text-driven manipulation of stylegan imagery." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2021. [[pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Patashnik_StyleCLIP_Text-Driven_Manipulation_of_StyleGAN_Imagery_ICCV_2021_paper.pdf)]

