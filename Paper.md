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
### A.Problem Definition
>Object detection aims to classify and locate instances. the terms tiny and small are typically defined by an area threshold  or length threshold 

Ŀ����Ŀ��Ϊ�����붨λʵ����
����ġ�С���������ֵ�򳤶���ֵ����
### B. Comparisons With Previous Reviews
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
#### 1. ���׶�Ŀ���ⷽ��
   
   �����׶�Ŀ���ⷽ���У�����ͨ��һЩ���ƣ���Region Proposal Network�����ɺ�ѡ����Ȼ����Щ��������������ݵ����ͷ���������յ�Ŀ�����ͱ߽��ع顣 
#### 2. һ�׶η���
   
   ��һ�׶η����У�����ֱ�����ܼ���ê���������ִ�м�⣬��ֱ��Ԥ�������������ꡣ**���ڲ���Ҫ���ɺ�ѡ����һ�׶μ�����ڼ���Ч���ϱ��ֳ����ƣ���ͨ����׼ȷ�����������.**

   Ϊ�˽��СĿ�����е���ս�����⣬���еķ���ͨ����ͨ��Ŀ�����ǿ��ʽ�����������ơ�
#### 3.1 Sample-Oriented Methods ����ȡ������
   
   > ������һ�ǵ�ǰ���ݼ��гߴ��С��Ŀ��ֻռ����һС�������򣻶��ǻ����ص���ƥ�䷽������������ȡ�����ϸ��޷��������㹻������ê���㣬��Ϊ�������СĿ������֮����ص����ޡ���Ҫ��Ϊ������������:
   
##### *������ǿ���ԣ�*

   - һЩ������Kisantal���˵Ĺ���ͨ����СĿ����и��ƣ�����ͬһͼ���еĲ�ͬλ�ý�������任������Сʵ����
   - RRNet������AdaResampling����������ָ�ͼ��ָ����Чλ�õĳ������̣���һ��������ճ������ĳ߶Ȳ��졣
   - Zhang���˺�Wang����ʹ���˻��ڷָ�Ĳ�������ø���СĿ���ѵ������������ָͼ���޸���ͼ���ںϵȡ�

##### *�Ż��ı�ǩ���䣺*

   - ��һЩ����ͨ���Ż���ǩ��������������ص������ƥ����Դ��������Ų�������������ٻع�����е��Ŷ���
   - ���磬S3FDͨ����Ƶĳ߶Ȳ���ê��ƥ�����������΢С������ƥ��ê�㣬�Ӷ�������ٻ��ʡ�
   - Zhu���������Expected Max Overlapping��EMO��������������ê�㲽���ڼ����ص�ʱ��Ӱ�죬ΪС�����ṩ���õ�ê�����á�
   - ������������������DotD��RFLA�ȣ��������˲�ͬ�ķ�ʽ���Ľ���ǩ������ԣ���������������СĿ���ϵ����ܱ��֡�

#### 3.2 Scale-Aware Methods �߶ȸ�֪����
   �߶ȱ仯���ܷǳ�����������ͬһ������������Ų�ͬ�ļ������
   > the following works mainly follow two paths. 
   - __multi-branch architecture or tailored training scheme__ ��ƶ�֧·�ܹ�����ѵ�������������ض��߶ȵļ����
   - __fuse the hierarchical features__ ���ֲ����������ں� for powerful representations of small objects.
   Both of these approaches actually minimize the information loss during feature extraction to a certain extent.

##### *Scale-Specific Detectors:�߶��ض������*
   - The nature behind this line is simple: ��ͬ��Ȼ򼶱��������������Ӧ�߶ȵ����塣
   - �����ó߶���سػ���SDP��ѡ���ʺ�СĿ�����������гػ������������ڲ�ͬ�м�������������飬ÿ����רע���ض��߶ȷ�Χ�ڵ����塣���ַ���ͨ���ڲ�ͬ�߶��ϲ�����߶�Ԥ�⣬ʹ�߷ֱ�����������СĿ�꣬�Ӷ������СĿ�����Ч����
  
   > **�߶���سػ�(SDP)** 
   ��һ����Բ�ͬ�߶�Ŀ����ض��ػ�������������ҪĿ����Ϊ��������ͼ�Ĳ�ͬ��Ȼ�㼶�Ͻ��гػ��������Ա����Ч�ش���ͬ�߶������������Ϣ��ͨ���߶���سػ�������ѡ���ԵضԲ�ͬ�߶ȵ�����ͼ���д���ʹ����Բ�ͬ�߶�Ŀ��������ܹ����õر�ģ�������á����ֳػ�������������߶�СĿ����Ŀ��Ȳ�ͬ�߶�Ŀ��ļ�⾫�ȡ�

   >**�ڲ�ͬ�м��������������**
   ָ����һ�������������飨ͨ���Ǻ�ѡ��������Ȥ���򣩵ķ������÷����ǻ�������������еĲ�ͬ�м�����������Բ�ͬ�߶ȵ��������顣ͨ���������������ڲ�ͬ��Ȳ�����ͬ���𡢲�ͬ�ֱ��ʵ�����ͼ����Щ����ͼ��Ӧ�Ų�ͬ��ε�������Ϣ��ͼ��ϸ�ڡ����������У�������Щ��ͬ��ε�����ͼ���Բ�����ͬ�߶ȵ��������飬ʹ��ģ���ܹ����õ���Ӧ��ͬ�߶�Ŀ��ļ�����������ķ�����������ģ�ͶԲ�ͬ�߶�����ĸ�֪����������߼���׼ȷ�ԡ�

##### *Hierarchical Feature Fusion���㼶�����ںϵķ���*
   - ��������СĿ���������У���������������Բ�׽��С�������Ӧ�������ڽ׶ε�����ͼ�������ܵ����ա����κ�������̬�����ص�Ӱ�죬ʹ�÷���������Ӿ�����ս�ԡ�
   - ���������ںϵķ����������ɲ�ͬ��ȵ��������Ի�ø��õ�СĿ��������ʾ��
   - ��PANet��BiFPN�ȣ�����ǿ���������������׼ȷ��λ���źŵ������ԡ�����������Zhang��������Ķ����RoI������ۡ�Woo���˵�StairNet��M2Det�еĲ��з�֧��IPG-Net��IPG�任ģ�顢Gong���˵Ļ���ͳ�Ƶ��ں����ӡ�SSPNet��ͻ���ض��߶������ȷ���.
   
   - **Ϊʲô�����ںϺ�Ϳ��Ը��õı�ʾСĿ�������ˣ�**
   >�����ں������ڸ��õر�ʾСĿ��������ԭ����Ҫ�����¼��㣺
   **1.��Ϣ�ḻ����ǿ��** �ںϲ�ͬ��Ȼ�ͬ�㼶�������ܹ��������ε���Ϣ����������Ӷ��ṩ���ḻ����ȫ����ӽǡ��Ͳ�������������ֲ�ϸ�ںͶ�λ��Ϣ�����߲������������ḻ��������Ϣ������Щ��ͬ��ε���Ϣ�ں���һ������ṩ��ȫ��ġ���������СĿ�����������ʾ��
   **2.��ǿ³���ԣ�** ����Թ��ձ仯����̬�仯����С������ʧ��Ӧ������ʱ����һ�㼶���������ܻ��ܵ��ϴ�Ӱ�졣�����ںϿ�����һ���̶������������³���ԣ�ʹ��ģ�͸��ܹ���Ӧ��ͬ�����µ�СĿ��������
   **3.����Ͷ�λ�Ľ�ϣ�** ͨ�����ײ�Ķ�λ��Ϣ��߲��������Ϣ���ϣ������ں��ܹ�ʵ�ָ�׼ȷ�Ķ�λ�͸��߲�ε�������⡣�����СĿ�����������Ҫ����ΪСĿ���������н��ٵ�������Ϣ���ںϲ�ͬ��ε���Ϣ���԰������õ�ʶ��Ͷ�λ��ЩĿ�ꡣ

�ܵ���˵�������ںϿ��Խ���ͬ��Ρ���ͬ�㼶����Ϣ���������ΪСĿ���ṩ��ȫ�桢��׼ȷ��������ʾ���Ӷ�����СĿ��ļ��Ч����

##### conclusion
   �ض��߶ȼܹ�ּ��������ʵĳ߶ȴ���СĿ�꣬���ںϷ���ּ���ֺ�**�������㼶**�в�ͬ��������֮��Ŀռ��������졣
   ���ߵ�Ŀ�����������Ĳ�ͬ������ṩ����������Ϣ��ͬʱ��ֹ���������Ӱ�������ڸ���СĿ���ԭʼ��Ӧ��
   ��ʵ��������֮���ƽ���Ǹ����⣬��ΪҪ��ǿ���Ͳ�������������Ϣ���ֱ����������ѹ��СĿ�����Ϣ�����������Ҫ��ϸȨ��ͽ����
   - **�������㼶**
   >�ڼ�����Ӿ��У�������ͨ��ָ�����ɲ�ͬ�ֱ��ʵ�ͼ����ɵĲ㼶�ṹ��������������ָ��������������ṹ�еĲ�ͬ��λ�ͬ�ֱ��ʵĲ㼶��
   ͼ���������һ�ֳ��õ�ͼ����������������ͬһͼ��Ķ���汾����Щ�汾���в�ͬ�ķֱ��ʡ��������ṹͨ����ԭʼͼ��ʼ��Ȼ��ͨ�����������ϲ����ȷ�ʽ����һϵ�о��в�ͬ�ֱ��ʵ�ͼ������Ľ����������Ǹ�˹��������������˹��������
   �����У��ᵽ�˲�ͬ�������������������ָ������һ��ͼ��������ṹ�У���ͬ�㼶��ֱ��ʵ�����ͼ�����������ͼ�����У���Щ��ͬ����������������ڶԲ�ͬ�߶ȵ�Ŀ����м��������
#### 3.3 Attention-Based Methods ����ע�������Ƶķ���
   Ϊ����ͼ�Ĳ�ͬ���ַ��䲻ͬ��Ȩ�أ�ǿ���м�ֵ�����������޹ؽ�Ҫ���������ַ�����������ͻ��СĿ�꣬��ΪСĿ���������ױ�����������ģʽ���ڸǣ��Ӷ���������ʾ�в��ּ����˸��š�
#### 3.4 Feature-Imitation Methods ����ģ�ⷽ��
��СĿ�����У�һ����Ҫ��ս������СĿ�����Ϣ���ٶ����µĵ�����������ʾ����ˣ�**�������ֵ���������**��ʾ�����ֱ�ӷ�ʽ��ͨ��**ģ��**�ϴ�Ŀ�������������**�ḻ**СĿ���������ʾ��

������Ϊ���ࣺ**������ѧϰ�ͻ��ڳ��ֱ��ʵ�����ģ���ܡ�**  

��Щ����ּ��ͨ��ģ�½ϴ�Ŀ�����������ǿСĿ���������ʾ���Ӷ����СĿ�����׼ȷ�Ժ�³���ԡ�
##### *Similarity Learning-Based Methods ����������ѧϰ�ķ���*
- ����ͨ�ü������ʩ�Ӷ����������Լ�������ֺ�СĿ��ʹ�Ŀ��֮���������ʾ��ࡣ
- �������Ӿ������Ƶļ������������ͨ����߶����˵ļ������Ż�����ܹ�������С�߶Ⱥʹ�߶����������������ԡ�
##### *Super-Resolution-Based Frameworks ���ڳ��ֱ��ʵĿ��*
- ּ�ڻָ�СĿ���Ť���ṹ�����������ǷŴ�����ģ������ۡ�
- 
## V. EXPERIMENTS


https://github.com/open-mmlab/mmdetection
https://github.com/open-mmlab/mmrotate
https://github.com/open-mmlab/OpenMMLabCourse