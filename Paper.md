# Towards Large-Scale Small Object Detection:Survey and Benchmarks 的研读、对第V部分进行细致解读 11.08
## 名词
### Benchmarks 来自标题
Benchmark 在机器学习里的定义：

>Benchmarking measures performance using a specific indicator, resulting in a metric that is then compared to others.Key performance indicators typically measured here are **data capacity, training speed, inference speed, and model precision**.

"Benchmark" 允许我们以客观的方式测量不同算法、模型或方法在特定任务上的表现，就像比较两种产品的性能一样。在机器学习中，"Benchmark"（基准测试）通常指的是一种对算法、模型或方法性能的标准化评估和比较方法。这是一种重要的工具，用于衡量和比较不同机器学习算法或模型的表现，以确定哪个方法在特定任务或数据集上表现最佳。
### Small Object Detection (SOD)
来自Abstract
>Then,to catalyze the development of SOD, we construct two large-scale Small Object Detection dAtasets(SODA), SODA-D and SODA-A, which focus on the **Driving and Aerial scenarios** respectively

用两个数据集分别集中于驾驶和空中场景
#### convolutional neural networks 卷积神经网络
 
## I. Introduction
#### A.Problem Definition
>Object detection aims to classify and locate instances. the terms tiny and small are typically defined by an area threshold  or length threshold 

目标检测目的为分类与定位实例。
物体的“小”由面积阈值或长度阈值定义
#### B. Comparisons With Previous Reviews
before:
>concentrate on either generic object detection or specific object detection task such as pedestrian detection

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

#### 3. low tolerance for bounding box perturbation and inadequate samples.(对边界盒扰动的低公差和样本不足)
