# Towards Large-Scale Small Object Detection:Survey and Benchmarks 的研读、对第V部分进行细致解读 11.08
## 名词
### Benchmarks 来自标题
Benchmark 在机器学习里的定义：

>Benchmarking measures performance using a specific indicator, resulting in a metric that is then compared to others.Key performance indicators typically measured here are **data capacity, training speed, inference speed, and model precision**.

"Benchmark" 允许我们以客观的方式测量不同算法、模型或方法在特定任务上的表现，就像比较两种产品的性能一样。在机器学习中，"Benchmark"（基准测试）通常指的是一种对算法、模型或方法性能的标准化评估和比较方法。这是一种重要的工具，用于衡量和比较不同机器学习算法或模型的表现，以确定哪个方法在特定任务或数据集上表现最佳。
### Small Object Detection (SOD)
来自Abstract
>Then,to catalyze the development of SOD, we construct two large-scale Small Object Detection dAtasets(SODA), SODA-D and SODA-A, which focus on the **Driving and Aerial scenarios** respectively

用两个数据集分别针对驾驶和空中场景
#### convolutional neural networks 卷积神经网络

### IoU（Intersection over Union）II.A

   >是一种用于衡量目标检测算法性能的常用指标。它通过计算两个边界框（通常是一个预测的边界框和一个真实的边界框）之间的重叠程度来评估检测结果的准确性。

IoU的计算方法是将两个边界框的交集面积除以它们的并集面积。公式如下：

IoU=预测框∩真实框预测框∪真实框IoU=预测框∪真实框预测框∩真实框?

这个值的范围在0到1之间，表示了两个框重叠的程度。IoU值越高，表示两个边界框之间的重叠越大，通常用来衡量预测框与真实框之间的匹配程度。在目标检测任务中，一般IoU达到一定的阈值（如0.5或0.75）会被认为是一个成功的检测结果。
 
## I. Introduction
### A.Problem Definition
>Object detection aims to classify and locate instances. the terms tiny and small are typically defined by an area threshold  or length threshold 

目标检测目的为分类与定位实例。
物体的“小”由面积阈值或长度阈值定义
### B. Comparisons With Previous Reviews
before:
>concentrate on either generic object detection or specific object detection task such as pedestrian detection
大多数先前的综述(如表I所示)集中在通用目标检测[13],[14],[15]或特定目标检测任务，如行人检测[16],[17],文本检测[18],遥感图像中的检测[19],[20]和交通场景下的检测[21],[22]等。
now:
>we provide a systematic survey of small object detection and an understandable and highly structured taxonomy(分类法), which organizes SOD approaches into six major categories based on the techniques involved and is radically different from previous ones.

### conclusion
the main contributions of this paper 
1. **Reviewing** the development of small object detection in the deep-learning era and providing a systematic survey of the recent progress in this field, which can be grouped into **six categories**:
   > sample-oriented methods, scale-aware methods,attention-based methods, feature-imitation methods, context modeling methods, and focus-and-detect approaches.
2. **Releasing two large-scale benchmarks** for small object detection, where the first one was dedicated to driving scenarios and the other was specialized for aerial scenes.
3. **Investigating the performance** of several representative object detection methods on our datasets
## II. REVIEW ON SMALL OBJECT DETECTION
### A. Main Challenges
#### 1. object information loss
   >Such information loss will scarcely impair the performance of large or medium-sized objects to a certain extent, considering that the final features still retain enough information of them. Unfortunately, this is fatal for small objects, because the detection head can **hardly give accurate predictions** on top of the highly structural representations, in which the weak signals of small objects were almost wiped out.

#### 2. noisy feature representation
   >To sum up, the feature representations of small objects are apt to suffer from the noise, hindering the subsequent detection.

#### 3. low tolerance for bounding box perturbation and inadequate samples.(对边界框扰动的低容忍度和样本不足)
   >Union(IoU) metric was adopted to evaluate the accuracy. 
   IoU表示真实框与相关预测框之间的交集比联合。

   - **对边界框扰动的低容忍度**：目标检测中的一个主要任务是定位，它在大多数检测范式中被构建为回归问题。
   >As shown inFig. 1, a slight deviation (6 pixels along the diagonal direction)of predicted box for a small object causes significant drop onIoU (from 100% to 32.5%) compared to medium and large objects (56.6% and 71.8%). Meanwhile, a greater variance (say, 12 pixels) further exacerbates the situation, and the IoU drops to poorly 8.7% for small objects
   ![Fig.1](D:\桌面\Fig.1.png)
   
   这表明，与大目标相比，小目标对边界框扰动的容忍度较低，加剧了回归分支的学习困难。

   - **训练样本不足**：选择正负样本是训练高性能检测器不可或缺的步骤。
   >positive and negative samples:在机器学习中，特别是在目标检测任务中，为了训练模型，需要从数据集中选择具有代表性的   **正样本**(*包含目标*)和负样本（*不包含目标*）。选择这些样本是为了让模型学习如何区分目标和非目标，从而实现准确的目标检测。选择不足的样本可能会导致模型训练不充分，影响最终检测器的性能和准确度。

   > Concretely, small instances occupy fairly small regions and have limited overlaps to priors (anchors or points). 
   
   这对传统的标签分配策略构成了巨大挑战，这些策略是基于边界框或中心区域的重叠来收集正负样本的，导致在训练过程中分配给小目标的正样本不足。

### B. Review of Small Object Detection Algorithms
   基于深度学习的通用目标检测方法可以分为两类：两阶段和单阶段检测，前者通过粗到细的步骤进行目标检测，而后者则一次性完成检测。
   -  detection head
   >在目标检测中，"detection heads"（检测头部）通常指的是神经网络模型的最后几层或特定部分，专门用于处理生成的特征，并执行目标检测的关键步骤，如分类和定位。这些头部层可以理解为网络结构的一部分，接受来自前面层（通常是提取的特征图）的信息，并进行最终的目标分类和定位。 
#### 1. 两阶段目标检测方法
   
   在两阶段目标检测方法中，首先通过一些机制（如Region Proposal Network）生成候选区域，然后这些区域的特征被传递到检测头部进行最终的目标分类和边界框回归。 
#### 2. 一阶段方法
   
   在一阶段方法中，网络直接在密集的锚点或网格上执行检测，并直接预测分类分数和坐标。**由于不需要生成候选区域，一阶段检测器在计算效率上表现出优势，但通常在准确性上稍显落后.**

   为了解决小目标检测中的挑战性问题，现有的方法通常在通用目标检测的强大范式中引入刻意设计。
#### 3.1 Sample-Oriented Methods 样本取样方法
   
   > 困境：一是当前数据集中尺寸较小的目标只占据了一小部分区域；二是基于重叠的匹配方法对于样本提取过于严格，无法采样到足够的正向锚点或点，因为先验框与小目标区域之间的重叠有限。主要分为以下两个方向:
   
##### *数据增强策略：*

   - 一些方法如Kisantal等人的工作通过将小目标进行复制，并在同一图像中的不同位置进行随机变换来增广小实例。
   - RRNet引入了AdaResampling，利用先验分割图来指导有效位置的抽样过程，进一步减少了粘贴对象的尺度差异。
   - Zhang等人和Wang等人使用了基于分割的操作来获得更多小目标的训练样本，例如分割、图像修复和图像融合等。

##### *优化的标签分配：*

   - 另一些方法通过优化标签分配策略来改善重叠或距离匹配策略带来的子优采样结果，并减少回归过程中的扰动。
   - 例如，S3FD通过设计的尺度补偿锚点匹配策略增加了微小人脸的匹配锚点，从而提高了召回率。
   - Zhu等人提出了Expected Max Overlapping（EMO）分数，考虑了锚点步长在计算重叠时的影响，为小人脸提供更好的锚点设置。
   - 还有其他方法，比如DotD、RFLA等，都采用了不同的方式来改进标签分配策略，提高主流检测器在小目标上的性能表现。

#### 3.2 Scale-Aware Methods 尺度感知方法
   尺度变化可能非常显著，导致同一个检测器面临着不同的检测难题
   > the following works mainly follow two paths. 
   - __multi-branch architecture or tailored training scheme__ 设计多支路架构或定制训练方案来构建特定尺度的检测器
   - __fuse the hierarchical features__ 将分层特征进行融合 for powerful representations of small objects.
   Both of these approaches actually minimize the information loss during feature extraction to a certain extent.

##### *Scale-Specific Detectors:尺度特定检测器*
   - The nature behind this line is simple: 不同深度或级别的特征负责检测相应尺度的物体。
   - 如利用尺度相关池化（SDP）选择适合小目标的特征层进行池化操作，或者在不同中间层生成物体提议，每个层专注于特定尺度范围内的物体。这种方法通过在不同尺度上产生多尺度预测，使高分辨率特征负责小目标，从而提高了小目标检测的效果。
  
   > **尺度相关池化(SDP)** 
   是一种针对不同尺度目标的特定池化操作。它的主要目的是为了在特征图的不同深度或层级上进行池化操作，以便更有效地处理不同尺度物体的特征信息。通过尺度相关池化，可以选择性地对不同尺度的特征图进行处理，使得针对不同尺度目标的特征能够更好地被模型所利用。这种池化方法有助于提高对小目标或大目标等不同尺度目标的检测精度。

   >**在不同中间层生成物体提议**
   指的是一种生成物体提议（通常是候选区域或感兴趣区域）的方法，该方法是基于深度神经网络中的不同中间层次来产生针对不同尺度的物体提议。通常，深度神经网络会在不同深度产生不同级别、不同分辨率的特征图，这些特征图对应着不同层次的语义信息和图像细节。在物体检测中，利用这些不同层次的特征图可以产生不同尺度的物体提议，使得模型能够更好地适应不同尺度目标的检测需求。这样的方法可以增加模型对不同尺度物体的感知能力，并提高检测的准确性。

##### *Hierarchical Feature Fusion：层级特征融合的方法*
   - 困境：在小目标检测任务中，深层特征可能难以捕捉到小物体的响应，而早期阶段的特征图则容易受到光照、变形和物体姿态等因素的影响，使得分类任务更加具有挑战性。
   - 采用特征融合的方法，即集成不同深度的特征，以获得更好的小目标特征表示。
   - 如PANet、BiFPN等，以增强深层特征并提高其对准确定位的信号的敏感性。其他方法如Zhang等人提出的多深度RoI特征汇聚、Woo等人的StairNet、M2Det中的并行分支、IPG-Net的IPG变换模块、Gong等人的基于统计的融合因子、SSPNet中突出特定尺度特征等方法.
   
   - **为什么特征融合后就可以更好的表示小目标特征了？**
   >特征融合有助于更好地表示小目标特征的原因主要有以下几点：
   **1.信息丰富性增强：** 融合不同深度或不同层级的特征能够将多个层次的信息结合起来，从而提供更丰富、更全面的视角。低层特征包含更多局部细节和定位信息，而高层特征包含更丰富的语义信息。将这些不同层次的信息融合在一起可以提供更全面的、更有利于小目标检测的特征表示。
   **2.增强鲁棒性：** 在面对光照变化、姿态变化或者小物体消失响应等问题时，单一层级的特征可能会受到较大影响。特征融合可以在一定程度上提高特征的鲁棒性，使得模型更能够适应不同场景下的小目标检测需求。
   **3.语义和定位的结合：** 通过将底层的定位信息与高层的语义信息相结合，特征融合能够实现更准确的定位和更高层次的语义理解。这对于小目标而言尤其重要，因为小目标往往具有较少的特征信息，融合不同层次的信息可以帮助更好地识别和定位这些目标。

总的来说，特征融合可以将不同层次、不同层级的信息结合起来，为小目标提供更全面、更准确的特征表示，从而改善小目标的检测效果。

##### conclusion
   特定尺度架构旨在以最合适的尺度处理小目标，而融合方法旨在弥合**金字塔层级**中不同级别特征之间的空间和语义差异。
   作者的目标是在特征的不同层次中提供更多语义信息，同时防止深层特征的影响过大而掩盖了小目标的原始响应。
   但实现这两者之间的平衡是个难题，因为要既强化低层特征的语义信息，又避免深层特征压制小目标的信息。这个问题需要仔细权衡和解决。
   - **金字塔层级**
   >在计算机视觉中，金字塔通常指的是由不同分辨率的图像组成的层级结构。金字塔级别则指的是这个金字塔结构中的不同层次或不同分辨率的层级。
   图像金字塔是一种常用的图像处理技术，它包含了同一图像的多个版本，这些版本具有不同的分辨率。金字塔结构通常由原始图像开始，然后通过降采样或上采样等方式生成一系列具有不同分辨率的图像。最常见的金字塔类型是高斯金字塔和拉普拉斯金字塔。
   在文中，提到了不同金字塔级别的特征。这指的是在一个图像金字塔结构中，不同层级或分辨率的特征图。在物体检测或图像处理中，这些不同级别的特征可以用于对不同尺度的目标进行检测或分析。
#### 3.3 Attention-Based Methods 基于注意力机制的方法
   为特征图的不同部分分配不同的权重，强调有价值的区域，抑制无关紧要的区域。这种方法可以用于突出小目标，因为小目标往往容易被背景和噪声模式所掩盖，从而在特征表示中部分减少了干扰。
#### 3.4 Feature-Imitation Methods 特征模拟方法
在小目标检测中，一个主要挑战是由于小目标的信息较少而导致的低质量特征表示。因此，**减轻这种低质量特征**表示问题的直接方式是通过**模拟**较大目标的区域特征来**丰富**小目标的特征表示。

方法分为两类：**相似性学习和基于超分辨率的特征模拟框架。**  

这些方法旨在通过模仿较大目标的特征来增强小目标的特征表示，从而提高小目标检测的准确性和鲁棒性。
##### *Similarity Learning-Based Methods 基于相似性学习的方法*
- 即在通用检测器上施加额外的相似性约束，以弥合小目标和大目标之间的特征表示差距。
- 受人类视觉理解机制的记忆过程启发，通过大尺度行人的记忆来优化整体架构，引导小尺度和大尺度行人特征的相似性。
##### *Super-Resolution-Based Frameworks 基于超分辨率的框架*
- 旨在恢复小目标的扭曲结构，而不仅仅是放大它们模糊的外观。
- 
## V. EXPERIMENTS


https://github.com/open-mmlab/mmdetection
https://github.com/open-mmlab/mmrotate
https://github.com/open-mmlab/OpenMMLabCourse