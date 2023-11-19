# Towards Large-Scale Small Object Detection:Survey and Benchmarks ���ж����Ե�V���ֽ���ϸ�½�� 11.08
## ����
### Benchmarks ���Ա���
Benchmark �ڻ���ѧϰ��Ķ��壺

>Benchmarking measures performance using a specific indicator, resulting in a metric that is then compared to others.Key performance indicators typically measured here are **data capacity, training speed, inference speed, and model precision**.

"Benchmark" ���������Կ͹۵ķ�ʽ������ͬ�㷨��ģ�ͻ򷽷����ض������ϵı��֣�����Ƚ����ֲ�Ʒ������һ�����ڻ���ѧϰ�У�"Benchmark"����׼���ԣ�ͨ��ָ����һ�ֶ��㷨��ģ�ͻ򷽷����ܵı�׼�������ͱȽϷ���������һ����Ҫ�Ĺ��ߣ����ں����ͱȽϲ�ͬ����ѧϰ�㷨��ģ�͵ı��֣���ȷ���ĸ��������ض���������ݼ��ϱ�����ѡ�
### Small Object Detection (SOD)
����Abstract
>Then,to catalyze the development of SOD, we construct two large-scale Small Object Detection dAtasets(SODA), SODA-D and SODA-A, which focus on the **Driving and Aerial scenarios** respectively

���������ݼ��ֱ���Լ�ʻ�Ϳ��г���
#### convolutional neural networks ���������
 
## I. Introduction
#### A.Problem Definition
>Object detection aims to classify and locate instances. the terms tiny and small are typically defined by an area threshold  or length threshold 

Ŀ����Ŀ��Ϊ�����붨λʵ����
����ġ�С���������ֵ�򳤶���ֵ����
#### B. Comparisons With Previous Reviews
before:
>concentrate on either generic object detection or specific object detection task such as pedestrian detection
�������ǰ������(���I��ʾ)������ͨ��Ŀ����[13],[14],[15]���ض�Ŀ�������������˼��[16],[17],�ı����[18],ң��ͼ���еļ��[19],[20]�ͽ�ͨ�����µļ��[21],[22]�ȡ�
now:
>we provide a systematic survey of small object detection and an understandable and highly structured taxonomy(���෨), which organizes SOD approaches into six major categories based on the techniques involved and is radically different from previous ones.

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

#### 3. low tolerance for bounding box perturbation and inadequate samples.(�Ա߽���Ŷ��ĵ����̶Ⱥ���������)
   >Union(IoU) metric was adopted to evaluate the accuracy. 
   >As shown inFig. 1, a slight deviation (6 pixels along the diagonal direction)of predicted box for a small object causes significant drop onIoU (from 100% to 32.5%) compared to medium and large objects (56.6% and 71.8%). Meanwhile, a greater variance (say, 12 pixels) further exacerbates the situation, and the IoU drops to poorly 8.7% for small objects
   ![Fig.1](D:\����\Fig.1.png)
   
