# Paper Reading

* [Paper Reading](#paper-reading)
  * [Transformer](#transformer)
    * [Transformer](#transformer-1)
    * [BERT](#bert)
    * [ViT](#vit)
    * [Swin Transformer](#swin-transformer)
    * [ViLT](#vilt)
  * [Image Caption](#image-caption)
    * [Show and Tell(ensemble)](#show-and-tellensemble)
    * [Show, Attend and Tell](#show-attend-and-tell)
    * [SCA\-CNN](#sca-cnn)
    * [Meshed\-Memory\-Transformer](#meshed-memory-transformer)
    * [ClipCap](#clipcap)
    * [VSR(Verb\-specific Semantic Roles)](#vsrverb-specific-semantic-roles)
  * [Human Pose Estimation](#human-pose-estimation)
    * [Deep Learning\-Based Human Pose Estimation: A Survey](#deep-learning-based-human-pose-estimation-a-survey)
  * [GAN](#gan)
    * [GAN](#gan-1)
    * [DCGAN](#dcgan)
    * [CycleGAN](#cyclegan)
    * [CartoonGAN](#cartoongan)
    * [AnimeGAN](#animegan)
    * [StyleGAN](#stylegan)
    * [StyleGANv2](#styleganv2)
    * [StyleCLIP](#styleclip)



**个人论文阅读笔记**



## Transformer

### Transformer

Vaswani, Ashish, et al. "Attention is all you need." *Advances in neural information processing systems* 30 (2017). [[pdf](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)]

#### 论文解读

- 纯靠注意力机制实现，抛弃了 RNN 和 CNN ，并行化程度高

- 整个transformer架构是以 堆叠的自注意力层 和 point-wise的FC构成

   <img src="img/transformer.png" alt="transformer" style="zoom: 33%;" />

- **Encoder**：
   
   - 6层
   - 每层都有2个子层：**multi-head self-attention** + **position-wise FC**
   - 且子层之间：**residual** + **LayerNorm**
   - 输出512维
   
- **Decoder**：

   - 相对于Encoder，加了一层交互层，也是 multi-head attention
   - **mask操作**，只看到前面时间步的部分，实现自回归

- **Scaled Dot-Product Attention**：

   - 和additional attention比较，两者理论上**复杂度相似**，但是实际应用中sdpa更**快**，**节省空间**，因为是优化过的矩阵乘法
   - scaled原因是点积的**数量级（方差）太大**的时候，softmax**梯度太小**

- **multi-head**：

   - 多头分别得到结果，**有不同侧重点**，最后concat一起并过一个linear
   - 因为输出512维，用8个头的话 q 和 k 的维度都是512 / 8  = 64

- **交互层**：

   - Q来自decoder的上一个子层
   - K 和 V 来自encoder的输出
   - 因此，**Q查询了输入序列的所有位置**

- Encoder可以使**每个token都被每个token关注**，Decoder也是，不过是mask的基础上

- position-wise FC：

   - **两层FC**，可以表示为 512 -> 2048 + ReLU -> 512

- 两个embedding层和pre-softmax-linear层的权重共享

- positional encoding：

   - 因为没有递归和卷积，所以要加入序列信息
   - 直接相加 encoding 和 embedding ，可以解释为**不同频率的信息叠加**

- 自注意力的原因：

   - 减小每层的总计算**复杂度**

   - 增加可**并行化**计算量

   - 去除**长程依赖问题**带来的影响



#### **个人理解**

- token之间的**相似度**越高，那么关系越接近，更应该关注对方。

  

- 发挥attention作用的关键就是**加权**。

  

- 优点

  - 最关键的是：有**捕获长期依赖**的能力，即容易学到**全局的信息**，因为token两两之间都算了attention。

- 缺点

  - **计算量大**，因此如何**减少计算、加速推理**（比如不用全局信息）可以研究。

    

- 为何多头，多头后为何要降维？

  - 相当于CNN的核数量，有**不同的侧重点**，学到更丰富的信息。

  - 将高维空间转换到低维空间，使得**concat后仍然满足维度dk**，丰富特征信息。

    

- 为什么Q、K、V都要通过一个Linear？

  - 可以实现**在不同空间进行投影**，增强泛化能力。

  - 和加法相比，复杂度差不多。

    

- 为什么scaled，为什么维度开根号？

  - 如果softmax内计算的数数量级太大，会输出近似**one-hot**编码的形式，导致**梯度消失**的问题，所以需要scale。

  - 为什么需要用维度开根号，假设向量q，k满足各分量独立同分布，均值为0，方差为1，那么qk点积均值为0，方差为dk，从统计学计算，若果让qk点积的**方差控制在1**，需要将其除以dk的平方根，使得softmax更加平滑。

    

- 为什么LayNorm而不是BN？

  - **LayNorm**：对一个序列的不同特征维度进行Norm。

  - **BatchNorm**：对batch进行Norm。

  - CV使用BN是认为channel维度的信息对cv方面有重要意义，如果对channel维度也归一化会造成不同通道信息一定的损失。而同理nlp领域认为**句子长度不一致**，并且**各个batch的信息没什么关系**，因此只考虑句子内信息的归一化，也就是LN。

    

- decoder并行化训练如何实现？

  - **mask**。

  - 推理的时候不能并行。

    

- 非线性体现在哪？

  - self attention 用了 **softmax** ，非线性。
  - FFN 中用了 **ReLU** 。

- 1

- 1

- 



### BERT

Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." *arXiv preprint arXiv:1810.04805* (2018). [[pdf](https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ)]



### ViT

Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." *arXiv preprint arXiv:2010.11929* (2020). [[pdf](https://arxiv.org/pdf/2010.11929.pdf?ref=https://githubhelp.com)]

### Swin Transformer

Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2021. [[pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)]

### ViLT

Kim, Wonjae, Bokyung Son, and Ildoo Kim. "Vilt: Vision-and-language transformer without convolution or region supervision." *International Conference on Machine Learning*. PMLR, 2021. [[pdf](http://proceedings.mlr.press/v139/kim21k/kim21k.pdf)]





## Image Caption

### Show and Tell(ensemble)

Vinyals, Oriol, et al. "Show and tell: A neural image caption generator." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2015. [[pdf](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)]

- 用预训练CNN, Inceptionv3
- LSTM输入只有第一个cell接受了图像编码；`word embedding`和`hiddensize`都是512d；SGD
- BeamSearch
- BLEU

```
总结：
机器翻译得到的启发encoder+decoder；
作者实验结论是只给第一个cell图像信息就够了
作者发现在语料库中建立的词表没啥作用，因此简单地随机初始化了word embedding weights
```



### Show, Attend and Tell

Xu, Kelvin, et al. "Show, attend and tell: Neural image caption generation with visual attention." *International conference on machine learning*. PMLR, 2015. [[pdf](http://proceedings.mlr.press/v37/xuc15.pdf)]

[[PyTorch实现](https://github.com/CAOANJIA/image-caption)]

- VGG19
- LSTM输入每一个cell都考虑了zt，即attend之后的图像编码，zt的计算也要用到ht-1
- hard: REINFORCE; soft: SGD
- soft是对于一个`14*14*512`的feature map，每个像素可以表示为一个512维的向量`aL`，考虑到感受野，其实每个像素都代表了一个区域，
   然后为每个像素(196个)一个512维的权值alphaL(用ht-1和aL计算)，相对于对每个像素都有权重，相乘就得到zt，这就是spatial attention
- Flickr8k: official split; 	Flickr30k, MSCOCO: Karpathy splits; 
- `batchsize 64`, 每次选相同长度的	`Adam`	避免过拟合：`Dropout`	*early stopping*
- BLEU, METEOR


```
总结：
attention效果很好；
一般来说硬注意力比软注意力效果好；
双重正则化，鼓励模型看到每个像素；门控*注意力结果，鼓励模型关注物体
trick--每次选相同长度的作为一个batch，加快训练速度;
```



### SCA-CNN

Chen, Long, et al. "Sca-cnn: Spatial and channel-wise attention in convolutional networks for image captioning." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2017. [[pdf](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_SCA-CNN_Spatial_and_CVPR_2017_paper.pdf)]

- CNN: `VGG19`/ `Res-152`	`LSTM`: word-embedding 100d, hidden-state 1000d	Attention Space: 512d both	
- `spatial` + `channel-wise` + `multi-layer`，多层迭代，层内是先channel再spatial
- channel-wise相当于选择**语义**的过程，因为可以说**每个filter(channel)对应了一种模式**
- channel-wise：先看作[u1, ..., uC]共C个向量，每个向量都是`H*W`维的，然后可以*meanpool*，得到[v1, ..., vC]，共C个标量，因为只需要考虑平均值就可以衡量这个channel对应的模式存在的可能性
       spatial：[v1, ..., vm]共m个向量，m=H*W，每个向量都是C维的
- attend之后继续接入后续的层，得到最终表示。结果表明：VGG只用spatial比较好（有FC，保留了空间信息），Res则是C-S好（meanpool，丢失了空间信息）C效果好的原因应该是channel多（2048），
- Flickr8k: official split; 	Flickr30k, MSCOCO: Karpathy splits;
- `batchsize 16，64，64`	`Adadelta`	BeamSearch-5结合长度归一化(testing时)	避免过拟合：`Dropout`	*early stopping*
- BLEU, METEOR, CIDEr, ROUGE-L

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
3. decoder是 **masked self attention**
4. region-based-Encoder: `Faster-RCNN` + `ResNet-101(pretrained on Visual Genome)` 每个区域都对应一个2048维的向量 
5. Decoder: `one-hot vector linearly projection` + `sin positional embedding`
6. COCO上：`d=512`，`head=8`，`memory vector=40`，`dropout=0.1`，`Adam`（pretrain时用warmup，finetuneCIDEr时5*10-6），`batchsize=50`，beamsize=5
7. nocap上：`GloVe`
8. 去除低频5的词，区域最大数量50，nn.Module.register_buffer减少3倍训练时间，记忆向量初始化



### ClipCap

Mokady, Ron, Amir Hertz, and Amit H. Bermano. "Clipcap: Clip prefix for image captioning." *arXiv preprint arXiv:2111.09734* (2021). [[pdf](https://arxiv.org/pdf/2111.09734.pdf)]

- `CLIP` + `Mapping Network` + `GPT2`（轻量级）

   以前两者的输出作为语言模型的`prefix`，然后自回归获得caption

- 可以只训练Mapping Network：那么需要能力更强的`transformer`

   也可以训练MN+LM：只需要`MLP`，甚至一层都可以

- 在*COCO*上~~不SOTA~~

   在更有挑战的*nocap*、*Conceptual Captions*数据集上<u>SOTA</u>

   *nocap*只有验证、测试集，训练集是直接用*COCO*，其有三大部分...

- **KEY: Mapping Network**，其输入为`CLIP encoding`和一个`constant input`，constant input可以：

   - 获得多头注意力的信息

   - 调整模型，使其适应frozen的语言模型

- baseline对比：`VLP`、`Oscar` （Oscar还用了额外的监督：对象标签）

   MLP+GPT2总体上都不如transformer，只有在Conceptual Captions上较好，可能是因为微调使其更具风格多样性，但微调容易过拟合

- 训练时间短，说明lightweight，下面是COCO上的

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
   
   
   



## Human Pose Estimation

### Deep Learning-Based Human Pose Estimation: A Survey

Zheng, Ce, et al. "Deep learning-based human pose estimation: A survey." *arXiv preprint arXiv:2012.13392* (2020). [[pdf](https://arxiv.org/pdf/2012.13392.pdf)]

1. Abstract

   人体姿态估计的目的：定位人体关键部位 + 构建人体表征。

2. Intro

   2D单人HPE已经获得高性能，当前关注于复杂场景的多人高遮挡场景的2D HPE；而3D HPE则挑战较大，单目图像的挑战是 ''`depth ambiguities`'' 深度歧义，多视图设置下的关键问题是 "`viewpoints association`" 视点关联。

   ![](img/taxonomy-of-hpe.png)

3. 人体建模

   大多数使用 `N-joints rigid kinematic model` 。

   三种建模方法：

   - 运动学模型 kinematic model (2D/3D)，使用一组关节位置和肢体方向来表示人体结构，优点是灵活的图表示，缺点是表示纹理和新装信息的能力有限，PSM是被广泛使用的图模型。
   - 平面模型 planar model (2D)，用近似于人体轮廓的矩形，来表示身体部位，广泛用于PCA捕获整个人体图形和剪影变形。
   - 体积模型 volumetric model (3D)，SMPL 皮肤多人线性模型是 3D HPE 广泛采用的模型，可以用表现出软组织动力学的自然姿态依赖变形进行建模。

4. 2D HPE

   2D HPE方法估计人体关键点的二维位置或者空间位置。

5. 

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

