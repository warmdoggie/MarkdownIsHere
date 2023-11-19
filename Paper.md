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
#### A.Problem Definition
>Object detection aims to classify and locate instances. the terms tiny and small are typically defined by an area threshold  or length threshold 

目标检测目的为分类与定位实例。
物体的“小”由面积阈值或长度阈值定义
#### B. Comparisons With Previous Reviews
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
   
   >Two-stage detection methods [1], [46], [49] produce high-quality proposals with a well-designed architecture such as RegionProposal Network (RPN) [1] at first, then the detection head stake regional features as input and perform subsequent classification and localization respectively.
   
   在两阶段目标检测方法中，首先通过一些机制（如Region Proposal Network）生成候选区域，然后这些区域的特征被传递到检测头部进行最终的目标分类和边界框回归。
   
   >Compared with two-stage algorithms, one-stage approaches [3], [44], [50] tile dense anchors on feature maps and predict the classification scores and coordinates directly. Benefiting from proposal-free setting,one-stage detectors enjoy high computational efficiency but often lag behind in accuracy.
   
   而在一阶段方法中，网络直接在密集的锚点或网格上执行检测，并直接预测分类分数和坐标。**由于不需要生成候选区域，一阶段检测器在计算效率上表现出优势，但通常在准确性上稍显落后.**

   为了解决小目标检测中的挑战性问题，现有的方法通常在通用目标检测的强大范式中引入刻意设计。
   
   **1. Sample-Oriented Methods 样本取样方法**
   
      >Such predicament originates from two aspects: __the targets with limited sizes only occupy a small portion__ in current datasets [6], [30], [31]; current overlap-based matching schemes [1], [3], [4], [47], [48] are __too  rigorous to sample sufficient positive anchors or points__ owing to the limited overlaps between priors and the regions of small objects.
   
   *数据增强策略：*

   - 一些方法如Kisantal等人的工作通过将小目标进行复制，并在同一图像中的不同位置进行随机变换来增广小实例。
   - RRNet引入了AdaResampling，利用先验分割图来指导有效位置的抽样过程，进一步减少了粘贴对象的尺度差异。
   - Zhang等人和Wang等人使用了基于分割的操作来获得更多小目标的训练样本，例如分割、图像修复和图像融合等。

   *优化的标签分配：*

   - 另一些方法通过优化标签分配策略来改善重叠或距离匹配策略带来的子优采样结果，并减少回归过程中的扰动。
   - 例如，S3FD通过设计的尺度补偿锚点匹配策略增加了微小人脸的匹配锚点，从而提高了召回率。
   - Zhu等人提出了Expected Max Overlapping（EMO）分数，考虑了锚点步长在计算重叠时的影响，为小人脸提供更好的锚点设置。
   - 还有其他方法，比如DotD、RFLA等，都采用了不同的方式来改进标签分配策略，提高主流检测器在小目标上的性能表现。

   **2. Scale-Aware Methods 尺度感知方法**
   > the following works mainly follow two paths. 
   One refers to construct scale-specific detectors by devising __multi-branch architecture or tailored training scheme__, and the other line of efforts intends to __fuse the hierarchical features__ for powerful representations of small objects.
   Both of these approaches actually minimize the information loss during feature extraction to a certain extent.


   *Scale-Specific Detectors:尺度特定检测器*
   - The nature behind this line is simple: the features at different depths or levels were responsible for detecting the objects of corresponding scales only.
