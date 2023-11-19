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

### IoU��Intersection over Union��II.A

   >��һ�����ں���Ŀ�����㷨���ܵĳ���ָ�ꡣ��ͨ�����������߽��ͨ����һ��Ԥ��ı߽���һ����ʵ�ı߽��֮����ص��̶��������������׼ȷ�ԡ�

IoU�ļ��㷽���ǽ������߽��Ľ�������������ǵĲ����������ʽ���£�

IoU=Ԥ������ʵ��Ԥ������ʵ��IoU=Ԥ������ʵ��Ԥ������ʵ��?

���ֵ�ķ�Χ��0��1֮�䣬��ʾ���������ص��ĳ̶ȡ�IoUֵԽ�ߣ���ʾ�����߽��֮����ص�Խ��ͨ����������Ԥ�������ʵ��֮���ƥ��̶ȡ���Ŀ���������У�һ��IoU�ﵽһ������ֵ����0.5��0.75���ᱻ��Ϊ��һ���ɹ��ļ������
 
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
   IoU��ʾ��ʵ�������Ԥ���֮��Ľ��������ϡ�

   - **�Ա߽���Ŷ��ĵ����̶�**��Ŀ�����е�һ����Ҫ�����Ƕ�λ�����ڴ������ⷶʽ�б�����Ϊ�ع����⡣
   >As shown inFig. 1, a slight deviation (6 pixels along the diagonal direction)of predicted box for a small object causes significant drop onIoU (from 100% to 32.5%) compared to medium and large objects (56.6% and 71.8%). Meanwhile, a greater variance (say, 12 pixels) further exacerbates the situation, and the IoU drops to poorly 8.7% for small objects
   ![Fig.1](D:\����\Fig.1.png)
   
   ����������Ŀ����ȣ�СĿ��Ա߽���Ŷ������̶Ƚϵͣ��Ӿ��˻ع��֧��ѧϰ���ѡ�

   - **ѵ����������**��ѡ������������ѵ�������ܼ�������ɻ�ȱ�Ĳ��衣
   >positive and negative samples:�ڻ���ѧϰ�У��ر�����Ŀ���������У�Ϊ��ѵ��ģ�ͣ���Ҫ�����ݼ���ѡ����д����Ե�   **������**(*����Ŀ��*)�͸�������*������Ŀ��*����ѡ����Щ������Ϊ����ģ��ѧϰ�������Ŀ��ͷ�Ŀ�꣬�Ӷ�ʵ��׼ȷ��Ŀ���⡣ѡ������������ܻᵼ��ģ��ѵ������֣�Ӱ�����ռ���������ܺ�׼ȷ�ȡ�

   > Concretely, small instances occupy fairly small regions and have limited overlaps to priors (anchors or points). 
   
   ��Դ�ͳ�ı�ǩ������Թ����˾޴���ս����Щ�����ǻ��ڱ߽�������������ص����ռ����������ģ�������ѵ�������з����СĿ������������㡣
### B. Review of Small Object Detection Algorithms
�������ѧϰ��ͨ��Ŀ���ⷽ�����Է�Ϊ���ࣺ���׶κ͵��׶μ�⣬ǰ��ͨ���ֵ�ϸ�Ĳ������Ŀ���⣬��������һ������ɼ�⡣
   -  detection head
   >��Ŀ�����У�"detection heads"�����ͷ����ͨ��ָ����������ģ�͵���󼸲���ض����֣�ר�����ڴ������ɵ���������ִ��Ŀ����Ĺؼ����裬�����Ͷ�λ����Щͷ����������Ϊ����ṹ��һ���֣���������ǰ��㣨ͨ������ȡ������ͼ������Ϣ�����������յ�Ŀ�����Ͷ�λ��
   
   >Two-stage detection methods [1], [46], [49] produce high-quality proposals with a well-designed architecture such as RegionProposal Network (RPN) [1] at first, then the detection head stake regional features as input and perform subsequent classification and localization respectively.
   
   �����׶�Ŀ���ⷽ���У�����ͨ��һЩ���ƣ���Region Proposal Network�����ɺ�ѡ����Ȼ����Щ��������������ݵ����ͷ���������յ�Ŀ�����ͱ߽��ع顣
   
   >Compared with two-stage algorithms, one-stage approaches [3], [44], [50] tile dense anchors on feature maps and predict the classification scores and coordinates directly. Benefiting from proposal-free setting,one-stage detectors enjoy high computational efficiency but often lag behind in accuracy.
   
   ����һ�׶η����У�����ֱ�����ܼ���ê���������ִ�м�⣬��ֱ��Ԥ�������������ꡣ**���ڲ���Ҫ���ɺ�ѡ����һ�׶μ�����ڼ���Ч���ϱ��ֳ����ƣ���ͨ����׼ȷ�����������.**

   Ϊ�˽��СĿ�����е���ս�����⣬���еķ���ͨ����ͨ��Ŀ�����ǿ��ʽ�����������ơ�
   
   **1. Sample-Oriented Methods ����ȡ������**
   
      >Such predicament originates from two aspects: __the targets with limited sizes only occupy a small portion__ in current datasets [6], [30], [31]; current overlap-based matching schemes [1], [3], [4], [47], [48] are __too  rigorous to sample sufficient positive anchors or points__ owing to the limited overlaps between priors and the regions of small objects.
   
   *������ǿ���ԣ�*

   - һЩ������Kisantal���˵Ĺ���ͨ����СĿ����и��ƣ�����ͬһͼ���еĲ�ͬλ�ý�������任������Сʵ����
   - RRNet������AdaResampling����������ָ�ͼ��ָ����Чλ�õĳ������̣���һ��������ճ������ĳ߶Ȳ��졣
   - Zhang���˺�Wang����ʹ���˻��ڷָ�Ĳ�������ø���СĿ���ѵ������������ָͼ���޸���ͼ���ںϵȡ�

   *�Ż��ı�ǩ���䣺*

   - ��һЩ����ͨ���Ż���ǩ��������������ص������ƥ����Դ��������Ų�������������ٻع�����е��Ŷ���
   - ���磬S3FDͨ����Ƶĳ߶Ȳ���ê��ƥ�����������΢С������ƥ��ê�㣬�Ӷ�������ٻ��ʡ�
   - Zhu���������Expected Max Overlapping��EMO��������������ê�㲽���ڼ����ص�ʱ��Ӱ�죬ΪС�����ṩ���õ�ê�����á�
   - ������������������DotD��RFLA�ȣ��������˲�ͬ�ķ�ʽ���Ľ���ǩ������ԣ���������������СĿ���ϵ����ܱ��֡�

   **2. Scale-Aware Methods �߶ȸ�֪����**
   > the following works mainly follow two paths. 
   One refers to construct scale-specific detectors by devising __multi-branch architecture or tailored training scheme__, and the other line of efforts intends to __fuse the hierarchical features__ for powerful representations of small objects.
   Both of these approaches actually minimize the information loss during feature extraction to a certain extent.


   *Scale-Specific Detectors:�߶��ض������*
   - The nature behind this line is simple: the features at different depths or levels were responsible for detecting the objects of corresponding scales only.
