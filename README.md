# Awesome Scene Graph[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<p align="center">
  <img width="250" src="https://camo.githubusercontent.com/1131548cf666e1150ebd2a52f44776d539f06324/68747470733a2f2f63646e2e7261776769742e636f6d2f73696e647265736f726875732f617765736f6d652f6d61737465722f6d656469612f6c6f676f2e737667" "Awesome!">
</p>

A curated list of scene graph generation and related tasks, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision) and [awesome-action-recognition](https://github.com/jinwchoi/awesome-action-recognition/blob/master/README.md). :-)

For a list of papers on 2-D Scene Graph grouped in various methods, please visit [Methods]()

## Introduction

Please feel free to send me [pull requests]() or email (mqjyl2012@163.com) to add links.

**Markdown format of paper list items:**

```markdown
- [Paper Name](link) - Author 1 _et al_, `Conference Year`. [[code]](link)
```

## Table of Contents

- [Scene Graph Generation](#Scene-Graph-Generation)
  - [2-D Scene Graph](#2-D-Scene-Graph)
  - [Spatio-Temporal Scene Graph](#Spatio-Temporal-Scene-Graph)
  - [3-D Scene Graph](#3-D-Scene-Graph)
  - [Generate Scene Graph from Textual Description](#Generate-Scene-Graph-from-Textual-Description)
  - [Attribute Detection](#Attribute-Detection)
  - [Other Works](#Other-Works)
  - [Datasets](#Datasets)
  - [Evaluation Metrics](#Evaluation-Metrics)
- [Human-centric Relation](#Human-centric-Relation)
  - [Person in Centext](#Person-in-Centext(PIC))
  - [Human-Object Interaction](#Human-Object-Interaction(HOI)))
  - [HCR Datasets](#HCR-Datasets)
- [Object Recognition](#Object-Recognition)
- [Related High-level Scene Understanding Tasks](#Related-High-level-Scene-Understanding-Tasks)
  - [Image Caption](#Image-Caption)
  - [Referring Expression Comprehension](#Referring-Expression-Comprehension---Visual-Grounding)
  - [Visual Question Answering](#Visual-Question-Answering)
  - [Visual Reasoning](#Visual-Reasoning)
  - [Image Generation](#Image-Generation---Content-based-Image-Retrieval(CBIR))
  - [Image Retrieval](#Image-Retrieval)
  - [Other Applications](#Other-Applications)
- [Workshops](#Workshops)
- [Challenges](#Challenges)

## Scene Graph Generation
### 2-D Scene Graph
#### 2020
* [Contextual Translation Embedding for Visual Relationship Detection and Scene Graph Generation](https://arxiv.org/abs/1905.11624) - Zih-Siou Hung _et al_, `T-PAMI 2020`.
* [Leveraging Auxiliary Text for Deep Recognition of Unseen Visual Relationships](https://arxiv.org/abs/1910.12324v1) - Gal Sadeh Kenigsfield  _et al_, `ICLR 2020`.
* [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/pdf/2002.11949.pdf) - Kaihua Tang  _et al_, `CVPR 2020`.  [[code]](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
* [Weakly Supervised Visual Semantic Parsing](https://arxiv.org/abs/2001.02359) - Alireza Zareian  _et al_, `CVPR 2020`. 
* [GPS-Net: Graph Property Sensing Network for Scene Graph Generation](https://arxiv.org/abs/2003.12962) - Xin Lin  _et al_, `CVPR 2020`.  [[code]](https://github.com/taksau/GPS-Net)
* [Deep Generative Probabilistic Graph Neural Networks for Scene Graph Generation](https://www.semanticscholar.org/paper/Deep-Generative-Probabilistic-Graph-Neural-Networks-Khademi-Schulte/4a03b00fe590ef62966a0f74b9450ff62e82b9a5) - Mahmoud Khademi  _et al_, `AAAI 2020`. 
* [Unbiased Scene Graph Generation via Rich and Fair Semantic Extraction](https://arxiv.org/abs/2002.00176) - Bin Wen  _et al_, `ARXIV 2020`. 
* [Long-tail Visual Relationship Recognition with a Visiolinguistic Hubless Loss](https://arxiv.org/abs/2004.00436) - Sherif Abdelkarim  _et al_, `ARXIV 2020`. 
* [Bridging Knowledge Graphs to Generate Scene Graphs](https://arxiv.org/abs/2001.02314) - Alireza Zareian  _et al_, `ARXIV 2020`. 
* [NODIS: Neural Ordinary Differential Scene Understanding](https://arxiv.org/abs/2001.04735) - Cong Yuren  _et al_, `ARXIV 2020`. 
* [AVR: Attention based Salient Visual Relationship Detection](http://arxiv.org/abs/2003.07012) - Jianming Lv  _et al_, `ARXIV 2020`. 

#### 2019
* [Large-Scale Visual Relationship Understanding](https://arxiv.org/abs/1804.10660) - Ji Zhang  _et al_, `AAAI 2019`.  [[code]](https://github.com/facebookresearch/Large-Scale-VRD)
* [Learning to Compose Dynamic Tree Structures for Visual Contexts](https://arxiv.org/abs/1812.01880) - Kaihua Tang  _et al_, `CVPR 2019 Oral`.  [[code]](https://github.com/KaihuaTang/VCTree-Scene-Graph-Generation)
* [Counterfactual Critic Multi-Agent Training for Scene Graph Generation](https://arxiv.org/abs/1812.02347) - Long Chen  _et al_, `ICCV 2019 Oral`. 
* [On Exploring Undetermined Relationships for Visual Relationship Detection](https://arxiv.org/abs/1905.01595) - Yibing Zhan  _et al_, `CVPR 2019`. [[code]](https://github.com//Atmegal//MFURLN-CVPR-2019-relationship-detection-method)
* [Exploring Context and Visual Pattern of Relationship for Scene Graph Generation](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Exploring_Context_and_Visual_Pattern_of_Relationship_for_Scene_Graph_CVPR_2019_paper.html) - Wenbin Wang  _et al_, `CVPR 2019`. 
* [Relationship-Aware Spatial Perception Fusion for Realistic Scene Layout Generation](https://arxiv.xilesou.top/abs/1909.00640) - Hongdong Zheng  _et al_, `arXiv 2019`. 
* [The Limited Multi-Label Projection Layer](https://arxiv.org/abs/1906.08707) - Brandon Amos  _et al_, `arXiv 2019`.  [[code]](https://github.com/locuslab/lml)
* [Detecting Visual Relationships Using Box Attention](https://arxiv.org/abs/1807.02136) - Alexander Kolesnikov  _et al_, `ICCVW 2019`.
* [Visual Relationships as Functions: Enabling Few-Shot Scene Graph Prediction](https://arxiv.org/abs/1906.04876) - Apoorva Dornadula  _et al_, `ICCVW 2019`.
* [Attention-Translation-Relation Network for Scalable Scene Graph Generation](http://openaccess.thecvf.com/content_ICCVW_2019/papers/SGRL/Gkanatsios_Attention-Translation-Relation_Network_for_Scalable_Scene_Graph_Generation_ICCVW_2019_paper.pdf) - Nikolaos Gkanatsios  _et al_, `ICCVW 2019`.  [[code]](https://github.com/deeplab-ai/atr-net)
* [Attentive Relational Networks for Mapping Images to Scene Graphs](https://arxiv.org/abs/1811.10696v1) - Mengshi Qi  _et al_, `CVPR 2019`. 
* [Visual Spatial Attention Network for Relationship Detection](https://dl.acm.org/doi/10.1145/3240508.3240611) - Chaojun Han,  _et al_, `ACM MM 2019`.
* [Visual Relation Detection with Multi-Level Attention](https://dlacm.xilesou.top/doi/abs/10.1145/3343031.3350962) - Sipeng Zheng,  _et al_, `ACM MM 2019`.
* [Visual Relationship Recognition via Language and Position Guided Attention](https://ieeexplore.ieee.xilesou.top/abstract/document/8683464/) - Hao Zhou,  _et al_, `ICASSP 2019`.
* [Relationship Detection Based on Object Semantic Inference and Attention Mechanisms](https://www.researchgate.net/publication/333698036_Relationship_Detection_Based_on_Object_Semantic_Inference_and_Attention_Mechanisms) - Liang Zhang  _et al_, `ICMR 2019`.
* [Natural Language Guided Visual Relationship Detection](http://openaccess.thecvf.com/content_CVPRW_2019/html/MULA/Liao_Natural_Language_Guided_Visual_Relationship_Detection_CVPRW_2019_paper.html) - Wentong Liao _et al_, `CVPR 2019`.
* [Knowledge-Embedded Routing Network for Scene Graph Generation](https://arxiv.org/abs/1903.03326) - Tianshui Chen _et al_, `CVPR 2019`. [[code]](https://github.com/yuweihao/KERN)
* [Soft Transfer Learning via Gradient Diagnosis for Visual Relationship Detection](https://ieeexplore.ieee.xilesou.top/abstract/document/8658599) - Diqi Chen _et al_, `WACV 2019`.
* [Compensating Supervision Incompleteness with Prior Knowledge in Semantic Image Interpretation](https://arxiv.org/abs/1910.00462) - Ivan Donadello _et al_, `IJCNN 2019`. [[code]](https://github.com/ivanDonadello/Visual-Relationship-Detection-LTN)
* [Hierarchical Visual Relationship Detection](https://dl.acm.org/doi/10.1145/3343031.3350921) - Xu Sun _et al_, `ACM MM 2019`.
* [Visual Relationship Detection with Low Rank Non-Negative Tensor Decomposition](https://arxiv.xilesou.top/abs/1911.09895) - Mohammed Haroon Dupty _et al_, `arXiv 2019`.
* [Relational Reasoning using Prior Knowledge for Visual Captioning](https://arxiv.xilesou.top/abs/1906.01290) - Jingyi Hou _et al_, `arXiv 2019`.
* [Scene Graph Generation with External Knowledge and Image Reconstruction](https://arxiv.org/abs/1904.00560) - Jiuxiang Gu _et al_, `CVPR 2019`. [[code]](https://github.com/arxrean/SGG_Ex_RC)
* [Attention-Translation-Relation Network for Scalable Scene Graph Generation](http://openaccess.thecvf.com/content_ICCVW_2019/html/SGRL/Gkanatsios_Attention-Translation-Relation_Network_for_Scalable_Scene_Graph_Generation_ICCVW_2019_paper.html) - Nikolaos Gkanatsios _et al_, `ICCV 2019`.
* [Detecting Unseen Visual Relations Using Analogies](http://openaccess.thecvf.com/content_ICCV_2019/html/Peyre_Detecting_Unseen_Visual_Relations_Using_Analogies_ICCV_2019_paper.html) - Julia Peyre _et al_, `ICCV 2019`.
* [VrR-VG: Refocusing Visually-Relevant Relationships](https://arxiv.org/abs/1902.00313) - Yuanzhi Liang _et al_, `ICCV 2019`.
* [SpatialSense: An Adversarially Crowdsourced Benchmark for Spatial Relation Recognition](https://arxiv.org/abs/1908.02660) - Kaiyu Yang _et al_, `ICCV 2019`. [[code]](https://github.com/princeton-vl/SpatialSense)
* [BLOCK: Bilinear Superdiagonal Fusion for Visual Question Answering and Visual Relationship Detection](https://arxiv.org/abs/1902.00038) - Kaiyu Yang _et al_, `AAAI 2019`. [[code]](https://github.com/Cadene/block.bootstrap.pytorch)
* [On Class Imbalance and Background Filtering in Visual Relationship Detection](https://arxiv.org/abs/1903.08456) - Alessio Sarullo _et al_, `arXiv 2019`. 
* [Support Relation Analysis for Objects in Multiple View RGB-D Images](https://arxiv.org/abs/1905.04084) - Peng Zhang _et al_, `IJCAIW QR 2019`.
* [Improving Visual Relation Detection using Depth Maps](https://arxiv.org/abs/1905.00966) - Sahand Sharifzadeh _et al_, `arXiv 2019`. [[code]](https://github.com/Sina-Baharlou/Depth-VRD)
* [MR-NET: Exploiting Mutual Relation for Visual Relationship Detection](https://www.aaai.org/ojs/index.php/AAAI/article/view/4819) - Yi Bin _et al_, `AAAI 2019`.
* [Scene Graph Prediction with Limited Labels](https://arxiv.org/abs/1904.11622) - Vincent S. Chen _et al_, `ICCV 2019`. [[code]](https://github.com/vincentschen/limited-label-scene-graphs)
* [Differentiable Scene Graphs](https://arxiv.org/abs/1902.10200) - Moshiko Raboh _et al_, `ICCVW 2019`.
* [Graphical Contrastive Losses for Scene Graph Parsing](https://arxiv.org/abs/1903.02728) - Ji Zhang _et al_, `CVPR 2019`. [[code]](https://github.com/NVIDIA/ContrastiveLosses4VRD)
* [Generating Expensive Relationship Features from Cheap Objects](https://bmvc2019.org/wp-content/uploads/papers/0657-paper.pdf) - Xiaogang Wang _et al_, `BMVC 2019`.
* [Neural Message Passing for Visual Relationship Detection](https://users.ece.cmu.edu/~sihengc/paper/19_ICMLW_HuCCZ.pdf) - Yue Hu  _et al_, `ICML LRG Workshop 2019`.  [[code]](https://github.com/PhyllisH/NMP)
* [PANet: A Context Based Predicate Association Network for Scene Graph Generation](https://ieeexplore.ieee.xilesou.top/abstract/document/8784780) - Yunian Chen  _et al_, `ICME 2019`. 
* [Visual Relationship Detection with Relative Location Mining](https://arxiv.org/abs/1911.00713) - Hao Zhou  _et al_, `ACM MM 2019`. 
* [Visual relationship detection based on bidirectional recurrent neural network](https://link.springer.com/article/10.1007%2Fs11042-019-7732-z) - Yibo Dai  _et al_, `Multimedia Tools and Applications 2019`. 
* [Exploring the Semantics for Visual Relationship Detection](https://arxiv.org/abs/1904.02104) - Wentong Liao  _et al_, `arXiv 2019`. 
* [Optimising the Input Image to Improve Visual Relationship Detection](https://arxiv.org/abs/1903.11029) - Noel Mizzi _et al_, `arXiv 2019`.
* [Deeply Supervised Multimodal Attentional Translation Embeddings for Visual Relationship Detection](https://arxiv.org/abs/1902.05829) - Nikolaos Gkanatsios _et al_, `arXiv 2019`.
* [Learning Effective Visual Relationship Detector on 1 GPU](https://arxiv.org/abs/1912.06185) - Yichao Lu _et al_, `arXiv 2019`.

#### 2018
* [Graph R-CNN for Scene Graph Generation](https://arxiv.org/abs/1808.00191) - Jianwei Yang  _et al_, `ECCV 2018`.  [[code]](https://github.com/jwyang/graph-rcnn.pytorch)
* [LinkNet_Relational Embedding for Scene Graph](https://arxiv.org/abs/1811.06410v1) - Sanghyun Woo  _et al_, `NIPS 2018`.  [[code]](https://github.com/jiayan97/linknet-pytorch)
* [Generating Triples with Adversarial Networks for Scene Graph Construction](https://arxiv.org/abs/1802.02598) - Matthew Klawonn  _et al_, `AAAI 2018`.
* [Scene Graph Generation Based on Node-Relation Context Module](https://link.springer.xilesou.top/chapter/10.1007/978-3-030-04179-3_12) - Xin Lin  _et al_, `ICONIP 2018`.
* [Factorizable Net: An Efficient Subgraph-based Framework for Scene Graph Generation](https://arxiv.org/abs/1806.11538) - Yikang Li  _et al_, `ECCV 2018`.  [[code]](https://github.com/yikang-li/FactorizableNet)
* [Neural Motifs_Scene Graph Parsing with Global Context](https://arxiv.org/abs/1711.06640) - Rowan Zellers  _et al_, `CVPR 2018`.  [[code]](https://github.com/rowanz/neural-motifs)
* [Zoom-Net: Mining Deep Feature Interactions for Visual Relationship Recognition](https://arxiv.org/abs/1807.04979) - Guojun Yin  _et al_, `ECCV 2018`.  [[code]](https://github.com/gjyin91/ZoomNet)
* [Deep Structured Learning for Visual Relationship Detection](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16475) - Yaohui Zhu  _et al_, `AAAI 2018`.
* [Tensorize, Factorize and Regularize: Robust Visual Relationship Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hwang_Tensorize_Factorize_and_CVPR_2018_paper.pdf) - Seong Jae Hwang _et al_, `CVPR 2018`. [[code]](https://github.com/shwang54/visual-tensor-decomposition)
* [Representation Learning for Scene Graph Completion via Jointly Structural and Visual Embedding](https://www.ijcai.org/proceedings/2018/132) - Hai Wan _et al_, `IJCAI 2018`. [[code]](https://github.com/sysulic/RLSV)
* [Visual Relationship Detection Using Joint Visual-Semantic Embedding](http://www.cs.umanitoba.ca/~ywang/papers/icpr18.pdf) - Binglin Li _et al_, `ICPR 2018`.
* [Object Relation Detection Based on One-shot Learning](https://arxiv.org/abs/1807.05857) - Li Zhou _et al_, `arXiv 2018`.
* [A Problem Reduction Approach for Visual Relationships Detection](https://arxiv.org/abs/1809.09828) - Toshiyuki Fukuzawa _et al_, `ECCVW 2018`. 
* [An Interpretable Model for Scene Graph Generation](https://arxiv.org/abs/1811.09543) - Ji Zhang _et al_, `arXiv 2018`. 
* [Shuffle-Then-Assemble: Learning Object-Agnostic Visual Relationship Features](https://arxiv.org/abs/1808.00171) - Xu Yang _et al_, `ECCV 2018`. [[code]](https://github.com/yangxuntu/vrd)
* [Visual Relationship Detection with Deep Structural Ranking](http://www.jdl.link/doc/2011/20191720205091168_aaai18_small.pdf) - Kongming Liang _et al_, `AAAI 2018`. [[code]](https://github.com/GriffinLiang/vrd-dsr)
* [Mapping Images to Scene Graphs with Permutation-Invariant Structured Prediction](https://arxiv.org/abs/1802.05451) - Roei Herzig _et al_, `NIPS 2018`. [[code]](https://github.com/shikorab/SceneGraph)
* [Learning Prototypes for Visual Relationship Detection](https://ieeexplore.ieee.xilesou.top/abstract/document/8516557) - François Plesse _et al_, `CBMI 2018`.
* [Visual Relationship Detection with Language prior and Softmax](https://arxiv.org/abs/1904.07798) - Jaewon Jung _et al_, `IPAS 2018`. [[code]](https://github.com/pranoyr/visual-relationship-detection)
* [Visual Relationship Detection Based on Guided Proposals and Semantic Knowledge Distillation](https://arxiv.org/abs/1805.10802) - François Plesse _et al_, `ICME 2018`.
* [Context-Dependent Diffusion Network for Visual Relationship Detection](https://arxiv.org/abs/1809.06213) - Zhen Cui _et al_, `ACM MM 2018`. [[code]](https://github.com/pranoyr/visual-relationship-detection)
* [Region-Object Relevance-Guided Visual Relationship Detection](http://www.bmva.org/bmvc/2018/contents/papers/1020.pdf) - Yusuke Goutsu  _et al_, `BMVC 2018`.
* [Recurrent Visual Relationship Recognition with Triplet Unit for Diversity](https://www.worldscientific.com/doi/10.1142/S1793351X18400214) - Kento Masui  _et al_, `IJSC 2018`.
* [Deep Image Understanding Using Multilayered Contexts](https://www.hindawi.com/journals/mpe/2018/5847460/) - Donghyeop Shin  _et al_, `MPE 2018`.
* [Scene Graph Generation via Conditional Random Fields](https://arxiv.org/abs/1811.08075) - Weilin Cong _et al_, `arXiv 2018`.

#### 2017
* [Scene Graph Generation by Iterative Message Passing](https://arxiv.org/abs/1701.02426) - Danfei Xu _et al_, `CVPR 2017`.  [[code]](https://github.com/danfeiX/scene-graph-TF-release)
* [Scene Graph Generation from Objects, Phrases and Region Captions](https://arxiv.org/abs/1707.09700) - Yikang Li  _et al_, `ICCV 2017`.  [[code]](https://github.com/yikang-li/MSDN)
* [ViP-CNN: Visual Phrase Guided Convolutional Neural Network](https://arxiv.org/abs/1702.07191) - Yikang Li  _et al_, `CVPR 2017`. 
* [Detecting Visual Relationships with Deep Relational Networks](https://arxiv.org/abs/1704.03114) - Bo Dai  _et al_, `CVPR 2017`.  [[code]](https://github.com/doubledaibo/drnet_cvpr2017) 
* [Towards Context-Aware Interaction Recognition for Visual Relationship Detection](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhuang_Towards_Context-Aware_Interaction_ICCV_2017_paper.pdf) - Bohan Zhuang  _et al_, `ICCV 2017`.
* [Phrase Localization and Visual Relationship Detection with Comprehensive Image-Language Cues](https://arxiv.org/abs/1611.06641) - Bryan A _et al_, `ICCV 2017`. [[code]](https://github.com/BryanPlummer/pl-clc)
* [Deep Variation-Structured Reinforcement Learning for Visual Relationship and Attribute Detection](https://arxiv.org/abs/1703.03054) - Xiaodan Liang _et al_, `CVPR 2017`. [[code]](https://github.com/nexusapoorvacus/DeepVariationStructuredRL)
* [Visual Relationship Detection with Internal and External Linguistic Knowledge Distillation](https://arxiv.org/abs/1707.09423v1) - Ruichi Yu _et al_, `ICCV 2017`.
* [Visual Translation Embedding Network for Visual Relation Detection](https://arxiv.org/abs/1702.08319v1) - Hanwang Zhang _et al_, `CVPR 2017`. [[code]](https://github.com/YANYANYEAH/vtranse)
* [Detecting Visual Relationships with Deep Relational Networks](https://arxiv.org/abs/1704.03114) - Bo Dai _et al_, `CVPR 2017`. [[code]](https://github.com/doubledaibo/drnet_cvpr2017)
* [Pixels to Graphs by Associative Embedding](https://arxiv.org/abs/1706.07365) - Alejandro Newell _et al_, `NIPS 2017`. [[code]](https://github.com/princeton-vl/px2graph)
* [Relationship Proposal Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Relationship_Proposal_Networks_CVPR_2017_paper.pdf) - Ji Zhang _et al_, `CVPR 2017`. 
* [Weakly-Supervised Learning of Visual Relations](https://arxiv.org/abs/1707.09472) - Julia Peyre _et al_, `ICCV 2017`. [[code]](https://github.com/yjy941124/PPR-FCN)
* [PPR-FCN: Weakly Supervised Visual Relation Detection via Parallel Pairwise R-FCN](https://arxiv.org/abs/1708.01956) - Hanwang Zhang _et al_, `ICCV 2017`. [[code]](https://github.com/jpeyre/unrel)
* [Visual relationship detection with object spatial distribution](https://www.researchgate.net/publication/319566830_Visual_relationship_detection_with_object_spatial_distribution) - Yaohui Zhu _et al_, `ICME 2017`. 
* [On Support Relations and Semantic Scene Graphs](https://arxiv.org/abs/1609.05834) - Michael Ying Yang _et al_, `ISPRS 2017`.
* [Improving Visual Relationship Detection using Semantic Modeling of Scene Descriptions](https://arxiv.org/abs/1809.00204) - Bryan A _et al_, `ISWC 2017`.
* [Recurrent Visual Relationship Recognition with Triplet Unit](https://ieeexplore.ieee.org/document/8241583) - Kento Masui  _et al_, `ISM 2017`.

#### 2016 and before
* [Visual Relationship Detection with Language Priors](https://arxiv.org/abs/1608.00187) - Cewu Lu _et al_, `ECCV 2016 Oral`. [[code]](https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection)
* [Recognition using visual phrases](http://vision.cs.uiuc.edu/phrasal/recognition_using_visual_phrases.pdf) - Mohammad Amin Sadeghi _et al_, `CVPR 2011`.

### Spatio-Temporal Scene Graph
* [Video Relationship Reasoning using Gated Spatio-Temporal Energy Graph](https://arxiv.org/abs/1903.10547) - Yao-Hung Hubert Tsai _et al_, `CVPR 2019`. [[code]](https://github.com/yaohungt/Gated-Spatio-Temporal-Energy-Graph)
* [Video Relation Detection with Spatio-Temporal Graph](https://dl.acm.org/doi/10.1145/3343031.3351058) - Xufeng Qian _et al_, `ACM MM 2019`.
* [Video Visual Relation Detection via Multi-modal Feature Fusion](https://dlacm.xilesou.top/doi/abs/10.1145/3343031.3356076) - Xu Sun _et al_, `ACM MM 2019`.
* [Relation Understanding in Videos](https://dl.acm.org/doi/abs/10.1145/3343031.3356080) - Sipeng Zheng _et al_, `ACM MM 2019`.
* [Annotating Objects and Relations in User-Generated Videos](https://dl.acm.org/doi/10.1145/3323873.3325056) - Xindi Shang _et al_, `ICMR 2019`.
* [Relation Understanding in Videos: A Grand Challenge Overview](https://dl.acm.org/doi/10.1145/3343031.3356082) - Xindi Shang _et al_, `ACM MM 2019`.
* [Action Genome: Actions as Composition of Spatio-temporal Scene Graphs](https://arxiv.xilesou.top/abs/1912.06992) - Jingwei Ji _et al_, `arXiv 2019`.
* [Video Visual Relation Detection](https://dl.acm.org/doi/pdf/10.1145/3123266.3123380?download=true) - Xindi Shang _et al_, `ACM MM 2017`. [[code]](https://github.com/xdshang/VidVRD-helper)

### 3-D Scene Graph
* [Learning 3D Semantic Scene Graphs from 3D Indoor Reconstructions](https://arxiv.org/abs/2004.03967) - Johanna Wald _et al_, `CVPR 2020`.
* [3-D Scene Graph: A Sparse and Semantic Representation of Physical Environments for Intelligent Agents](https://arxiv.org/abs/1908.04929v1) - Ue-Hwan Kim _et al_, `IEEE transactions on cybernetics 2019`. [[code]](https://github.com/Uehwan/3-D-Scene-Graph)
* [3D Scene Graph: A Structure for Unified Semantics, 3D Space, and Camera](https://arxiv.org/abs/1910.02527) - Iro Armeni _et al_, `ICCV 2019`. [[code]](https://github.com/StanfordVL/3DSceneGraph)
* [Visual Graphs from Motion (VGfM): Scene understanding with object geometry reasoning](https://arxiv.org/abs/1807.05933) - Paul Gay _et al_, `ACCV 2018`. [[code]](https://github.com/paulgay/VGfM)

### Generate Scene Graph from Textual Description
* [Scene Graph Parsing as Dependency Parsing](https://arxiv.org/abs/1803.09189) - Yu-Siang Wang _et al_, `NAACL 2018`. [[code]](https://github.com/vacancy/SceneGraphParser)
* [Scene Graph Parsing by Attention Graph](https://arxiv.org/abs/1909.06273) - Martin Andrews  _et al_, `NIPS 2018`.

### Other Works
* [Relationship Prediction for Scene Graph Generation](http://cs229.stanford.edu/proj2019spr/report/8.pdf) - Uzair Navid Iftikhar _et al_, `2019`. 
* [Joint Learning of Scene Graph Generation and Reasoning for Visual Question Answering Mid-term report](http://ink-ron.usc.edu/xiangren/ml4know19spring/public/midterm/Arka_Sadhu_and_Xuefeng_Hu_Report.pdf) - Arka Sadhu _et al_, `2019`. 
* [Scene-Graph-Generation](https://github.com/HimangiM/Scene-Graph-Generation)
* [Joint Embeddings of Scene Graphs and Images](https://hal.inria.fr/hal-01667777)  - Eugene Belilovsky _et al_, `2020`. 

### Datasets
#### Image
* Visual Genome : [Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations](https://arxiv.org/abs/1602.07332) - Ranjay Krishna _et al_, `IJCV 2016`. [[download]](http://visualgenome.org/)
* VRD : [Visual Relationship Detection with Language Priors](https://arxiv.org/abs/1608.00187) - Cewu Lu _et al_, `ECCV 2016 Oral`. [[download]](https://cs.stanford.edu/people/ranjaykrishna/vrd/)
* Open Images : [The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale](https://arxiv.org/abs/1811.00982) - Alina Kuznetsova _et al_, `IJCV 2018`. [[download]](https://storage.googleapis.com/openimages/web/index.html)
* Scene Graph : [Image Retrieval using Scene Graphs](https://cs.stanford.edu/people/jcjohns/papers/cvpr2015/JohnsonCVPR2015.pdf) - Justin Johnson _et al_, `CVPR 2015`. [[download]](http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip)
* Visual Phrases : [Recognition Using Visual Phrases](http://vision.cs.uiuc.edu/phrasal/) - Ali Farhadi _et al_, `CVPR 2011`. [[download]](http://vision.cs.uiuc.edu/phrasal/)
* VrR-VG : [VrR-VG: Refocusing Visually-Relevant Relationships](https://arxiv.org/abs/1902.00313) - Yuanzhi Liang _et al_, `ICCV 2019`. [[download]](http://vrr-vg.com/)
* UnRel : [Weakly-Supervised Learning of Visual Relations](https://arxiv.org/abs/1707.09472) - Julia Peyre _et al_, `ICCV 2017`. [[download]](https://www.di.ens.fr/willow/research/unrel/)
* SpatialVOC2K : [SpatialVOC2K: A Multilingual Dataset of Images with Annotations and Features for Spatial Relations between Objects](https://www.aclweb.org/anthology/W18-6516/) - Anja Belz _et al_, `INLG 2018`. [[download]](https://github.com/muskata/SpatialVOC2K)
* SpatialSense : [SpatialSense: An Adversarially Crowdsourced Benchmark for Spatial Relation Recognition](https://arxiv.org/abs/1908.02660) - Kaiyu Yang _et al_, `ICCV 2019`. [[download]](https://drive.google.com/drive/folders/125fgCq-1YYfKOAxRxVEdmnyZ7sKWlyqZ?usp=sharing)
* Visual and Linguistic Treebank : [Image Description using Visual Dependency Representations](https://www.aclweb.org/anthology/D13-1128/) - Desmond Elliott _et al_, `EMNLP 2013`. [[download]]
* ViSen : [Combining geometric, textual and visual features for predicting prepositions in image descriptions](https://core.ac.uk/display/41826441) - Arnau Ramisa _et al_, `EMNLP 2015`. [[download]]
* SynthRel0 : [SynthRel0: Towards a Diagnostic Dataset for Relational Representation Learning](http://openaccess.thecvf.com/content_ICCVW_2019/papers/SGRL/Dorda_SynthRel0_Towards_a_Diagnostic_Dataset_for_Relational_Representation_Learning_ICCVW_2019_paper.pdf) - Daniel Dorda _et al_, `ICCVW 2019`. [[download]]

#### RGBD
* NYU Depth Dataset V2 : [Indoor Segmentation and Support Inference from RGBD Images](https://cs.nyu.edu/~silberman/papers/indoor_seg_support.pdf) - Nathan Silberman _et al_, `ECCV 2012`. [[download]](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

#### Video
* VidVRD : [Video Visual Relation Detection](https://dl.acm.org/doi/pdf/10.1145/3123266.3123380?download=true) - Xindi Shang _et al_, `ACM MM 2017`. [[download]](https://xdshang.github.io/docs/imagenet-vidvrd.html)
* VidOR dataset : [Annotating Objects and Relations in User-Generated Videos](https://dl.acm.org/doi/10.1145/3323873.3325056) - Xindi Shang _et al_, `ACM MM 2019`. [[download]](https://xdshang.github.io/docs/vidor.html)

#### 3-D
* 3D Scene Graph Dataset : [3D Scene Graph: A Structure for Unified Semantics, 3D Space, and Camera](https://3dscenegraph.stanford.edu/images/3DSceneGraph.pdf) - Iro Armeni _et al_, `ICCV 2019`. [[download]](https://3dscenegraph.stanford.edu/)
* [Learning 3D Semantic Scene Graphs from 3D Indoor Reconstructions](https://arxiv.org/abs/2004.03967) - Johanna Wald _et al_, `CVPR 2020`.  [[download]](https://3dssg.github.io/)

### Evaluation Metrics

## Human-centric Relation
### Person in Centext(PIC)
* [Visual Relationship Prediction via Label Clustering and Incorporation of Depth Information](https://tsujuifu.github.io/projs/eccv18_pic.html#) - Hsuan-Kung Yang _et al_, `ECCVW 2018`.

### Human-Object Interaction(HOI)
#### HOI Image
##### 2020
* [VSGNet: Spatial Attention Network for Detecting Human Object Interactions Using Graph Convolutions](https://arxiv.org/abs/2003.05541) - Oytun Ulutan _et al_, `CVPR 2020`. [[code]](https://github.com/ASMIftekhar/VSGNet)
* [Learning Human-Object Interaction Detection using Interaction Points](https://arxiv.org/pdf/2003.14023.pdf) - Tiancai Wang _et al_, `CVPR 2020`. [[code]](https://github.com/vaesl/IP-Net)
* [Detailed 2D-3D Joint Representation for Human-Object Interaction](https://arxiv.org/abs/2004.08154) - Yong-Lu Li _et al_, `CVPR 2020`. [[code]](https://github.com/DirtyHarryLYL/DJ-RN)
* [Cascaded Human-Object Interaction Recognition](https://arxiv.org/abs/2003.04262) - Tianfei Zhou _et al_, `CVPR 2020`. [[code]](https://github.com/tfzhou/C-HOI)
* [PPDM: Parallel Point Detection and Matching for Real-time Human-Object Interaction Detection](https://arxiv.org/abs/1912.12898v1) - Yue Liao _et al_, `CVPR 2020`. [[code]](https://github.com/YueLiao/PPDM)
* [Detecting Human-Object Interactions via Functional Generalization](https://arxiv.org/abs/1904.03181) - Ankan Bansal _et al_, `AAAI 2020`.
* [Classifying All Interacting Pairs in a Single Shot](https://arxiv.org/pdf/2001.04360.pdf) - Sanaa Chafik _et al_, `WACV 2020`.
* [Visual-Semantic Graph Attention Network for Human-Object Interaction Detection](https://arxiv.org/abs/2001.02302) - Zhijun Liang _et al_, `ARXIV 2020`.
* [Spatial Priming for Detecting Human-Object Interactions](https://arxiv.org/abs/2004.04851) - Ankan Bansal _et al_, `ARXIV 2020`.
* [GID-Net: Detecting Human-Object Interaction with Global and Instance Dependency](https://arxiv.org/abs/2003.05242) - Dongming Yang _et al_, `ARXIV 2020`.

##### 2019
* [Reasoning About Human-Object Interactions Through Dual Attention Networks](https://arxiv.org/abs/1909.04743) - Tete Xiao _et al_, `ICCV 2019`.
* [Relation Parsing Neural Network for Human-Object Interaction Detection](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Relation_Parsing_Neural_Network_for_Human-Object_Interaction_Detection_ICCV_2019_paper.pdf) - Penghao Zhou _et al_, `ICCV 2019`.
* [No-Frills Human-Object Interaction Detection: Factorization, Layout Encodings, and Training Techniques](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gupta_No-Frills_Human-Object_Interaction_Detection_Factorization_Layout_Encodings_and_Training_Techniques_ICCV_2019_paper.pdf) - Tanmay Gupta _et al_, `ICCV 2019`. [[code]](https://github.com/BigRedT/no_frills_hoi_det)
* [Pose-aware Multi-level Feature Network for Human Object Interaction Detection](https://arxiv.org/abs/1909.08453v1) - Bo Wan _et al_, `ICCV 2019`. [[code]](https://github.com/bobwan1995/PMFNet)
* [Deep Contextual Attention for Human-Object Interaction Detection](https://arxiv.org/abs/1910.07721) - Tiancai Wang _et al_, `ICCV 2019`. 
* [Understanding Human Gaze Communication by Spatio-Temporal Graph Reasoning](https://arxiv.org/abs/1909.02144) - Lifeng Fan _et al_, `ICCV 2019`. [[code]](https://github.com/LifengFan/Human-Gaze-Communication)
* [Holistic++ Scene Understanding: Single-view 3D Holistic Scene Parsing and Human Pose Estimation with Human-Object Interaction and Physical Commonsense](https://yixchen.github.io/holisticpp/file/holistic_scenehuman.pdf) - Yixin Chen _et al_, `ICCV 2019`. [[code]](https://github.com/yixchen/holistic_scene_human)
* [Transferable Interactiveness Knowledge for Human-Object Interaction Detection](https://arxiv.org/abs/1811.08264) - Yong-Lu Li _et al_, `CVPR 2019`. [[code]](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network)
* [Learning to Detect Human-Object Interactions with Knowledge](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Learning_to_Detect_Human-Object_Interactions_With_Knowledge_CVPR_2019_paper.pdf) - Bingjie Xu _et al_, `CVPR 2019`.
* [Do Deep Neural Networks Model Nonlinear Compositionality in the Neural Representation of Human-Object Interactions?](https://arxiv.org/abs/1904.00431) - Aditi Jha _et al_, `CCN 2019`.

##### 2018
* [Detecting and Recognizing Human-Object Interactions](https://arxiv.org/abs/1704.07333) - Georgia Gkioxari _et al_, `CVPR 2018`.
* [Learning Human-Object Interactions by Graph Parsing Neural Networks](http://web.cs.ucla.edu/~syqi/publications/eccv2018gpnn/eccv2018gpnn.pdf) - Siyuan Qi _et al_, `ECCV 2018`. [[code]](https://github.com/SiyuanQi/gpnn)
* [Pairwise Body-Part Attention for Recognizing Human-Object Interactions](https://arxiv.org/abs/1807.10889v1) - Hao-Shu Fang _et al_, `ECCV 2018`.
* [Compositional Learning for Human Object Interaction](http://openaccess.thecvf.com/content_ECCV_2018/papers/Keizo_Kato_Compositional_Learning_of_ECCV_2018_paper.pdf) - Keizo Kato _et al_, `ECCV 2018`. [[code]](https://github.com/kkatocmu/Compositional_Learning)
* [iCAN: Instance-Centric Attention Network for Human-Object Interaction Detection](https://arxiv.org/pdf/1808.10437.pdf) - Chen Gao _et al_, `BMVC 2018`. [[code]](https://github.com/vt-vl-lab/iCAN)
* [Interact as You Intend: Intention-Driven Human-Object Interaction Detection](https://arxiv.org/pdf/1808.09796.pdf) - Bingjie Xu _et al_, `TMM 2018`.
* [Scaling Human-Object Interaction Recognition through Zero-Shot Learning](http://vision.stanford.edu/pdf/shen2018wacv.pdf) - Liyue Shen _et al_, `WACV 2018`.
* [Learning to Detect Human-Object Interactions](http://vision.stanford.edu/pdf/shen2018wacv.pdf) - Yu-Wei Chao _et al_, `WACV 2018`. [[code]](https://github.com/ywchao/ho-rcnn)

##### 2017及以前
* [Fine-grained Event Learning of Human-Object Interaction with LSTM-CRF](https://arxiv.org/abs/1710.00262v1) - Tuan Do _et al_, `ESANN 2017`. 
* [Learning Models for Actions and Person-Object Interactions with Transfer to Question Answering](http://slazebni.cs.illinois.edu/publications/eccv16.pdf) - Arun Mallya _et al_, `ECCV 2016`. 
* [Human Centred Object Co-Segmentation](https://arxiv.org/abs/1606.03774) - Chenxia Wu _et al_, `ARXIV 2016`. 
* [HICO: A Benchmark for Recognizing Human-Object Interactions in Images](https://www.cs.princeton.edu/~jiadeng/paper/chao_iccv2015.pdf) - Yu-Wei Chao _et al_, `ICCV 2015`. 
* [Recognising Human-Object Interaction via Exemplar based Modelling](https://www.eecs.qmul.ac.uk/~sgg/papers/HuEtAl_ICCV2013.pdf) - Jian-Fang Hu _et al_, `ICCV 2013`. 
* [Learning person-object interactions for action recognition in still images](https://www.di.ens.fr/willow/pdfscurrent/delaitre_NIPS11.pdf) - Vincent Delaitre _et al_, `NIPS 2011`. 
* [Modeling Mutual Context of Object and Human Pose in Human-Object Interaction Activities](http://vision.stanford.edu/pdf/YaoFei-Fei_CVPR2010b.pdf) - Bangpeng Yao _et al_, `CVPR 2010`. 
* [Discriminative models for static human-object interactions](https://www.cs.cmu.edu/~deva/papers/action.pdf) - Chaitanya Desai _et al_, `CVPRW 2010`. 

#### HOI Video
* [Grounded Human-Object Interaction Hotspots from Video](https://arxiv.org/abs/1906.01963) - Tushar Nagarajan _et al_, `ICCV 2019`. [[code]](https://github.com/Tushar-N/interaction-hotspots)
* [iMapper: Interaction-guided Joint Scene and Human Motion Mapping from Monocular Videos](https://arxiv.org/abs/1806.07889) - Aron Monszpart _et al_, `Siggraph 2019`.
* [Causality Inspired Retrieval of Human-object Interactions from Video](https://www.semanticscholar.org/paper/Causality-Inspired-Retrieval-of-Human-object-from-Zhou-Liu/0ff0595b2f02b197b4adc63e1e7e60691f896752) - Liting Zhou _et al_, `CBMI 2019`.
* [Zero-Shot Generation of Human-Object Interaction Videos](https://arxiv.org/abs/1912.02401) - Megha Nawhal _et al_, `ARXIV 2019`.
* [Forecasting Human Object Interaction: Joint Prediction of Motor Attention and Egocentric Activity](https://arxiv.org/abs/1911.10967) - Miao Liu _et al_, `ARXIV 2019`.
* [Attend and Interact: Higher-Order Object Interactions for Video Understanding](https://arxiv.org/abs/1711.06330) - Chih-Yao Ma _et al_, `CVPR 2018`.
* [The "something something" video database for learning and evaluating visual common sense](https://arxiv.org/abs/1706.04261) - Raghav Goyal _et al_, `ICCV 2017`. [[code]](https://github.com/TwentyBN/smth-smth-v2-baseline-with-models) [[code_v2]](https://github.com/TwentyBN/something-something-v2-baseline)

#### Other Works
* [Detecting Human-Object Interactions in Real-Time](https://github.com/lmingyin/HOI-RT)

#### HOI Evaluation Metrics

### HCR Datasets
* PIC 1.0 / 2.0 : [[download]](http://picdataset.com/challenge/dataset/download/)
* HOI-W : [[download]](http://picdataset.com/challenge/dataset/download/)
* HCVRD: [HCVRD: A Benchmark for Large-Scale Human-Centered Visual Relationship Detection]() - Saurabh Gupta _et al_, `AAAI 2018`. [[download]]()
* Verbs in COCO (V-COCO) : [Visual Semantic Role Labeling](https://arxiv.org/abs/1505.04474) - Saurabh Gupta _et al_, `ARXIV 2015`. [[download]](https://github.com/s-gupta/v-coco)
* HICO : [A Benchmark for Recognizing Human-Object Interactions in Images](https://www.cs.princeton.edu/~jiadeng/paper/chao_iccv2015.pdf) - Yu-Wei Chao _et al_, `ICCV 2015`. [[download]](http://www-personal.umich.edu/~ywchao/hico/)
* TUHOI : [A Benchmark for Recognizing Human-Object Interactions in Images](https://www.aclweb.org/anthology/W14-5403.pdf) - Dieu-Thu Le _et al_, `ACL 2014`. [[download]](http://disi.unitn.it/~dle/dataset/TUHOI.html)
* 20BN-SOMETHING-SOMETHING : [The "something something" video database for learning and evaluating visual common sense](https://arxiv.org/abs/1706.04261) - Raghav Goyal _et al_, `ICCV 2017`. [[download]](https://20bn.com/datasets/something-something)

## Improve Object Recognition

## Related High-level Vision-and-Language Tasks
### Image Caption
#### Using Scene Graph
* [Learning visual relationship and context-aware attention for image captioning](https://www.sciencedirect.com/science/article/abs/pii/S0031320319303760) - Junbo Wang  _et al_, `Pattern Recognition 2020`. 
* [Object Relational Graph with Teacher-Recommended Learning for Video Captioning](https://arxiv.org/abs/2002.11566) - Ziqi Zhang  _et al_, `CVPR 2020`. 
* [Say As You Wish: Fine-grained Control of Image Caption Generation with Abstract Scene Graphs](https://arxiv.org/abs/2003.00387) - Shizhe Chen  _et al_, `CVPR 2020`.  [[code]](https://github.com/cshizhe/asg2cap)
* [Joint Commonsense and Relation Reasoning for Image and Video Captioning](https://wuxinxiao.github.io/assets/papers/2020/C-R_reasoning.pdf) - Jingyi Hou  _et al_, `AAAI 2020`. 
* [Auto-Encoding Scene Graphs for Image Captioning](https://arxiv.org/abs/1812.02378) - Xu Yang  _et al_, `CVPR 2019`.  [[code]](https://github.com/yangxuntu/SGAE)
* [Dense Relational Captioning: Triple-Stream Networks for Relationship-Based Captioning](https://arxiv.org/abs/1903.05942) - Dong-Jin Kim  _et al_, `CVPR 2019`.  [[code]](https://github.com/Dong-JinKim/DenseRelationalCaptioning)
* [Visual Semantic Reasoning for Image-Text Matching](https://arxiv.org/abs/1909.02701) - Kunpeng Li  _et al_, `ICCV 2019`.  [[code]](https://github.com/KunpengLi1994/VSRN)
* [Unpaired Image Captioning via Scene Graph Alignments](https://arxiv.org/abs/1903.10658) - Jiuxiang Gu  _et al_, `ICCV 2019`.  [[code]](https://github.com/gujiuxiang/unpaired_image_captioning)
* [Expressing Visual Relationships via Language](https://arxiv.org/abs/1906.07689) - Hao Tan  _et al_, `ACL 2019`.  [[code]](https://github.com/airsplay/VisualRelationships)
* [On the Role of Scene Graphs in Image Captioning](https://www.aclweb.org/anthology/D19-6405.pdf) - Dalin Wang  _et al_, `ACL 2019`. 
* [Adversarial Adaptation of Scene Graph Models for Understanding Civic Issues](https://arxiv.org/pdf/1901.10124.pdf) - Shanu Kumar  _et al_, `WWW 2019`.  [[code]](https://github.com/Sshanu/civic_issue_dataset)
* [Aligning Linguistic Words and Visual Semantic Units for Image Captioning](https://arxiv.org/abs/1908.02127v1) - Longteng Guo  _et al_, `ACM MM 2019`.  [[code]](https://github.com/ltguo19/VSUA-Captioning)
* [Better Understanding Hierarchical Visual Relationship for Image Caption](https://arxiv.org/abs/1912.01881) - Zheng-cong Fei  _et al_, `NeurIPS 2019 workshop on New In ML`. 
* [Visual Relationship Embedding Network for Image Paragraph Generation](https://ieeexplore.ieee.org/document/8907490) - Wenbin Che  _et al_, `TMM 2019`. 
* [Know More Say Less: Image Captioning Based on Scene Graphs](https://ieeexplore.ieee.org/document/8630068) - Xiangyang Li  _et al_, `TMM 2019`. 
* [Visual Relationship Attention for Image Captioning](https://ieeexplore.ieee.org/document/8851832) - Zongjian Zhang  _et al_, `IJCNN 2019`. 
* [Scene graph captioner: Image captioning based on structural visual representation](https://www.sciencedirect.com/science/article/pii/S1047320318303535) - Ning Xu  _et al_, `VCIR 2019`. 
* [Exploring Semantic Relationships for Image Captioning without Parallel Data](http://web.pkusz.edu.cn/adsp/files/2019/11/ICDM2019_camera_ready2.pdf) - Fenglin Liu  _et al_, `ICDM 2019`. 
* [TPsgtR: Neural-Symbolic Tensor Product Scene-Graph-Triplet Representation for Image Captioning](https://arxiv.org/abs/1911.10115) - Chiranjib Sur  _et al_, `ARXIV 2019`. 
* [Learning Visual Relation Priors for Image-Text Matching and Image Captioning with Neural Scene Graph Generators](https://arxiv.org/abs/1909.09953) - Kuang-Huei Lee  _et al_, `ARXIV 2019`. 
* [Relational Reasoning using Prior Knowledge for Visual Captioning](https://arxiv.org/abs/1906.01290) - Jingyi Hou  _et al_, `ARXIV 2019`. 
* [Exploring Visual Relationship for Image Captioning](https://arxiv.org/abs/1809.07041) - Ting Yao  _et al_, `ECCV 2018`. 
* [Paragraph Generation Network with Visual Relationship Detection](https://dl.acm.org/doi/10.1145/3240508.3240695) - Wenbin Che  _et al_, `ACM MM 2018`. 
* [Image Captioning with Scene-graph Based Semantic Concepts](https://dl.acm.org/doi/10.1145/3195106.3195114) - Lizhao Gao  _et al_, `ICMLC 2018`. 
* [Improved Image Description Via Embedded Object Structure Graph and Semantic Feature Matching](https://ieeexplore.ieee.org/abstract/document/8603262) - Li Ren  _et al_, `ISM 2018`. 
* [Sports Video Captioning by Attentive Motion Representation based Hierarchical Recurrent Neural Networks](https://jueduilingdu.github.io/data/qi2019sports.pdf) - Mengshi Qi  _et al_, `2018`. 

#### Classic Papers
* [Image Captioning and Visual Question Answering Based on Attributes and External Knowledge](https://ieeexplore.ieee.org/document/7934440) - Qi Wu  _et al_, `TPAMI 2017`. 
* [SPICE: Semantic Propositional Image Caption Evaluation](https://panderson.me/spice/) - Peter Anderson  _et al_, `ECCV 2016`.  [[code]](https://github.com/peteanderson80/SPICE)

#### Image Caption Datasets
* MS COCO : [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312) - Tsung-Yi Lin _et al_, `ECCV 2014`. [[download]](http://cocodataset.org/)
* Flickr30K : [Flickr30k Entities: Collecting Region-to-Phrase Correspondences for Richer Image-to-Sentence Models](https://arxiv.org/abs/1505.04870v2) - Bryan A. Plummer _et al_, `IJCV 2017`. [[download]](http://bryanplummer.com/Flickr30kEntities/)
* Flickr8K : [Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics](https://www.ijcai.org/Proceedings/15/Papers/593.pdf) - Micah Hodosh  _et al_, `IJCAI 2013`. [[download]](http://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b)
* Visual Genome : [Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations](https://arxiv.org/abs/1602.07332) - Ranjay Krishna _et al_, `IJCV 2016`. [[download]](http://visualgenome.org/)
* IAPR TC-12 : [The IAPR TC-12 Benchmark: A New Evaluation Resource for Visual Information Systems](http://thomas.deselaers.de/publications/papers/grubinger_lrec06.pdf) - Michael Grubinger _et al_, `International workshop onto Image 2006`. [[download]](https://www.imageclef.org/photodata)

### Referring Expression Comprehension - Visual Grounding
#### Using Scene Graph
* [Cross-Modal Relationship Inference for Grounding Referring Expressions](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Cross-Modal_Relationship_Inference_for_Grounding_Referring_Expressions_CVPR_2019_paper.pdf) - Sibei Yang  _et al_, `CVPR 2019`. [[code]](https://github.com/sibeiyang/sgmn/tree/master/lib/cmrin_models)
* [Relationship-Embedded Representation Learning for Grounding Referring Expressions](https://arxiv.org/abs/1906.04464) - Sibei Yang  _et al_, `TPAMI 2020`. [[code]](https://github.com/sibeiyang/sgmn/tree/master/lib/cmrin_models)
* [Referring Expression Comprehension with Semantic Visual Relationship and Word Mapping](https://www.semanticscholar.org/paper/Referring-Expression-Comprehension-with-Semantic-Zhang-Li/4dbc17138e7b610214359bb1659a30e01d183482) - Chao Zhang  _et al_, `ACM MM 2019`.
* [Learning to Relate from Captions and Bounding Boxes](https://www.aclweb.org/anthology/P19-1660/) - Sarthak Garg  _et al_, `ACL 2019`.
* [Joint Visual Grounding with Language Scene Graphs](https://arxiv.org/abs/1906.03561) - Daqing Liu  _et al_, `ARXIV 2019`. 
* [Modeling Relationships in Referential Expressions With Compositional Modular Networks](https://arxiv.org/abs/1611.09978) - Ronghang Hu  _et al_, `CVPR 2017`.  [[code]](https://github.com/ronghanghu/cmn)
* [Phrase Localization and Visual Relationship Detection With Comprehensive Image-Language Cues](http://openaccess.thecvf.com/content_ICCV_2017/papers/Plummer_Phrase_Localization_and_ICCV_2017_paper.pdf) - Bryan A. Plummer  _et al_, `ICCV 2017`.  [[code]](https://github.com/BryanPlummer/pl-clc)

#### Classic Papers
* [Graph-Structured Referring Expression Reasoning in The Wild](https://arxiv.org/abs/2004.08814) - Sibei Yang  _et al_, `CVPR 2020`.  [[code]](https://github.com/sibeiyang/sgmn)
* [Dynamic Graph Attention for Referring Expression Comprehension](https://arxiv.org/abs/1909.08164) - Sibei Yang  _et al_, `ICCV 2019`.  [[code]](https://github.com/sibeiyang/sgmn/tree/master/lib/dga_models)
* [Grounding Referring Expressions in Images by Variational Context](https://arxiv.org/abs/1712.01892) - Hanwang Zhang  _et al_, `CVPR 2018`.  [[code]](https://github.com/yuleiniu/vc)

#### Visual Grounding Datasets
* RefCOCO and RefCOCO+ : [Modeling Context in Referring Expressions](https://arxiv.org/abs/1608.00272) - Licheng Yu _et al_, `ECCV 2016`. [[download]](https://github.com/lichengunc/refer)

### Visual Question Answering
#### Using Scene Graph
* [DualVD: An Adaptive Dual Encoding Model for Deep Visual Understanding in Visual Dialogue](https://arxiv.org/abs/1911.07251) - Xiaoze Jiang  _et al_, `AAAI 2020`.  [[code]](https://github.com/JXZe/DualVD)
* [Visual Query Answering by Entity-Attribute Graph Matching and Reasoning](https://arxiv.org/abs/1903.06994) - Peixi Xiong  _et al_, `CVPR 2019`. 
* [Relation-Aware Graph Attention Network for Visual Question Answering](https://arxiv.org/abs/1903.12314) - Linjie Li  _et al_, `ICCV 2019`.  [[code]](https://github.com/linjieli222/VQA_ReGAT)
* [Multi-interaction Network with Object Relation for Video Question Answering](https://www.semanticscholar.org/paper/Multi-interaction-Network-with-Object-Relation-for-Jin-Zhao/7a162f189c9c553438b83a8a8ec7de4a6fa59069) - Weike Jin  _et al_, `ACM MM 2019`. 
* [CRA-Net: Composed Relation Attention Network for Visual Question Answering](https://dl.acm.org/doi/10.1145/3343031.3350925) - Liang Peng  _et al_, `ACM MM 2019`. 
* [An Empirical Study on Leveraging Scene Graphs for Visual Question Answering](https://arxiv.org/abs/1907.12133) - Cheng Zhangs  _et al_, `BMVC 2019`.  [[code]](https://github.com/czhang0528/scene-graphs-vqa)
* [Generating Natural Language Explanations for Visual Question Answering using Scene Graphs and Visual Attention](https://arxiv.org/abs/1902.05715) - Shalini Ghosh  _et al_, `ARXIV 2019`. 
* [R-VQA: Learning Visual Relation Facts with Semantic Attention for Visual Question Answering](https://arxiv.org/abs/1805.09701v1) - Pan Lu  _et al_, `SIGKDD 2018`.  [[code]](https://github.com/BierOne/relation-vqa)
* [Scene Graph Reasoning with Prior Visual Relationship for Visual Question Answering](https://arxiv.org/abs/1812.09681v2) - Zhuoqian Yang  _et al_, `ARXIV 2018`. 
* [VisKE: Visual Knowledge Extraction and Question Answering by Visual Verification of Relation Phrases](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Sadeghi_VisKE_Visual_Knowledge_2015_CVPR_paper.html) - Fereshteh Sadeghi  _et al_, `CVPR 2015`.  [[code]](https://github.com/fsadeghi/VisKE)

#### Classic Papers
* [Image Question Answering using Convolutional Neural Network with Dynamic Parameter Prediction](https://arxiv.org/abs/1511.05756) - Hyeonwoo Noh  _et al_, `CVPR 2016`.  [[code]](https://github.com/HyeonwooNoh/DPPnet)
* [Ask Your Neurons: A Neural-based Approach to Answering Questions about Images](https://arxiv.org/abs/1505.01121) - Mateusz Malinowski  _et al_, `ICCV 2015`. 

#### VQA Datasets
* VQAv1 : [VQA: Visual question answering](https://arxiv.org/abs/1505.00468) - Aishwarya Agrawal _et al_, `ICCV 2015`. [[download]](https://visualqa.org/vqa_v1_download.html)
* VQAv2 : [Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering](https://arxiv.org/abs/1612.00837) - Yash Goyal _et al_, `CVPR 2017`. [[download]](https://visualqa.org/download.html)
* COCO-QA : [Image Question Answering: A Visual Semantic Embedding Model and a New Dataset](https://arxiv.org/abs/1505.02074v1) - Mengye Ren _et al_, `ICML 2015`. or [Exploring Models and Data for Image Question Answering](https://arxiv.org/abs/1505.02074) - Mengye Ren _et al_, `NIPS 2015`. [[download]](http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/)

### Visual Reasoning
#### Using Scene Graph
* [Differentiable Scene Graphs](https://arxiv.org/abs/1902.10200) - Moshiko Raboh  _et al_, `WACV 2020`. 
* [Language-Conditioned Graph Networks for Relational Reasoning](https://arxiv.org/abs/1905.04405) - Ronghang Hu  _et al_, `ICCV 2019`. [[code]](https://github.com/ronghanghu/lcgn)
* [Explainable and Explicit Visual Reasoning over Scene Graphs](https://arxiv.org/abs/1812.01855) - Jiaxin Shi  _et al_, `CVPR 2019`.  [[code]](https://github.com/shijx12/XNM-Net)
* [Referring Relationships](https://cs.stanford.edu/people/ranjaykrishna/referringrelationships/index.html) - Ranjay Krishna  _et al_, `CVPR 2018`.  [[code]](https://github.com/StanfordVL/ReferringRelationships)
* [Broadcasting Convolutional Network for Visual Relational Reasoning](https://arxiv.org/abs/1712.02517) - Simyung Chang  _et al_, `ECCV 2018`.
* [A Simple Neural Network Module for Relational Reasoning](https://arxiv.org/abs/1706.01427) - Adam Santoro  _et al_, `ARXIV 2017`.  [[code]](https://github.com/siddk/relation-network)

#### Classic Papers
* [Object level Visual Reasoning in Videos](https://arxiv.org/abs/1806.06157) - Fabien Baradel _et al_, `ECCV 2018`. [[code]](https://github.com/fabienbaradel/object_level_visual_reasoning)
* [A simple neural network module for relational reasoning](https://arxiv.org/abs/1706.01427) - Adam Santoro _et al_, `NIPS 2017`. [[code]](https://github.com/clvrai/Relation-Network-Tensorflow)

#### Visual Reasoning Datasets
* GQA : [GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering](https://arxiv.org/abs/1902.09506) - Drew A. Hudson _et al_, `CVPR 2019`. [[download]](https://cs.stanford.edu/people/dorarad/gqa/index.html)
* CLEVR : [CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](http://vision.stanford.edu/pdf/johnson2017cvpr.pdf) - Justin Johnson _et al_, `CVPR 2017`. [[download]](https://cs.stanford.edu/people/jcjohns/clevr/) [[code]](https://github.com/facebookresearch/clevr-dataset-gen)

### Image Generation - Content-based Image Retrieval(CBIR)
#### Using Scene Graph
* [PasteGAN: A Semi-Parametric Method to Generate Image from Scene Graph](https://arxiv.org/abs/1905.01608) - Yikang Li _et al_, `NIPS 2019`. [[code]](https://github.com/yikang-li/PasteGAN)
* [Scene Graph Generation with External Knowledge and Image Reconstruction](https://arxiv.org/abs/1904.00560) - Jiuxiang Gu _et al_, `CVPR 2019`. [[code]](https://github.com/arxrean/SGG_Ex_RC)
* [Specifying Object Attributes and Relations in Interactive Scene Generation](https://arxiv.org/abs/1909.05379) - Oron Ashual _et al_, `ICCV 2019`. [[code]](https://github.com/ashual/scene_generation)
* [Triplet-Aware Scene Graph Embeddings](https://arxiv.org/abs/1909.09256) - Brigit Schroeder _et al_, `ICCVW 2019`.
* [Heuristics for Image Generation from Scene Graphs](https://openreview.net/forum?id=r1eqsNXgdV) - Subarna Tripathi _et al_, `ICLR 2019`. 
* [Interactive Image Generation Using Scene Graphs](https://arxiv.org/abs/1905.03743) - Gaurav Mittal _et al_, `ICLR 2019`. 
* [Visual-Relation Conscious Image Generation from Structured-Text](https://arxiv.org/abs/1908.01741v1) - Duc Minh Vo _et al_, `ARXIV 2019`. 
* [Using Scene Graph Context to Improve Image Generation](https://arxiv.org/abs/1901.03762) - Subarna Tripathi _et al_, `ARXIV 2019`. 
* [Learning Canonical Representations for Scene Graph to Image Generation](https://arxiv.org/abs/1912.07414) - Roei Herzig _et al_, `ARXIV 2019`. 
* [Relationship-Aware Spatial Perception Fusion for Realistic Scene Layout Generation](https://arxiv.org/abs/1909.00640) - Hongdong Zheng _et al_, `ARXIV 2019`. 
* [Image Generation from Scene Graphs](https://arxiv.org/abs/1804.01622) - Justin Johnson _et al_, `CVPR 2018`. [[code]](https://github.com/google/sg2im)

#### Classic Papers
* [Image Generation From Small Datasets via Batch Statistics Adaptation](https://arxiv.org/abs/1904.01774) - Atsuhiro Noguchi _et al_, `ICCV 2019`. [[code]](https://github.com/nogu-atsu/small-dataset-image-generation)
* [Text2Scene: Generating Compositional Scenes from Textual Descriptions](https://arxiv.org/abs/1809.01110) - Fuwen Tan _et al_, `CVPR 2019`. [[code]](https://github.com/uvavision/Text2Scene)
* [Unsupervised Cross-Domain Image Generation](https://arxiv.org/abs/1611.02200) - Yaniv Taigman _et al_, `ICLR 2017 conference submission`. [[code]](https://github.com/yunjey/domain-transfer-network)
* [Generative Visual Manipulation on the Natural Image Manifold](https://arxiv.org/abs/1609.03552) - Jun-Yan Zhu _et al_, `ECCV 2016`. [[code]](https://github.com/junyanz/iGAN)
* [Attribute2Image: Conditional Image Generation from Visual Attributes](https://arxiv.org/abs/1512.00570) - Xinchen Yan _et al_, `ECCV 2016`. [[code]](https://github.com/xcyan/eccv16_attr2img)

#### Image Generation Datasets
* COCO : [Microsoft COCO: Common objects in context](https://arxiv.org/abs/1405.0312) - Tsung-Yi Lin _et al_, `ECCV 2014`. [[download]](http://cocodataset.org/#home)

### Image Retrieval
#### Using Scene Graph
* [Cross-modal Scene Graph Matching for Relationship-aware Image-Text Retrieval](https://arxiv.org/abs/1910.05134) - Sijin Wang _et al_, `WACV 2020`.
* [Scene Graph based Image Retrieval -- A case study on the CLEVR Dataset](https://arxiv.org/abs/1911.00850) - Sahana Ramnath _et al_, `ICCVW 2019`. 
* [Compact Scene Graphs for Layout Composition and Patch Retrieval](https://arxiv.org/abs/1904.09348) - Subarna Tripathi _et al_, `CVPRW 2019`. 
* [Revisiting Visual Grounding](https://arxiv.org/abs/1904.02225) - Erik Conser _et al_, `ACL 2019`. 
* [Learning visual features for relational CBIR](https://link.springer.com/article/10.1007/s13735-019-00178-7) - Nicola Messina _et al_, `MIR 2019`. 
* [Learning Relationship-aware Visual Features](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11132/Messina_Learning_Relationship-aware_Visual_Features_ECCVW_2018_paper.pdf) - Nicola Messina _et al_, `ECCVW 2018`. [[code]](https://github.com/mesnico/learning-relationship-aware-visual-features)
* [Image retrieval by dense caption reasoning](http://qugank.github.io/papers/VCIP17.pdf) - Xinru Wei _et al_, `VCIP 2017`. 
* [Representation Learning for Visual-Relational Knowledge Graphs](https://arxiv.org/abs/1709.02314v5) - Daniel Oñoro-Rubio _et al_, `ARXIV 2017`. 
* [Image retrieval using scene graphs](https://ieeexplore.ieee.org/document/7298990) - Justin Johnson _et al_, `CVPR 2015`. 
* [Generating Semantically Precise Scene Graphs from Textual Descriptions for Improved Image Retrieval](https://www.semanticscholar.org/paper/Generating-Semantically-Precise-Scene-Graphs-from-Schuster-Krishna/2606e6a5759c030e259ebf3f4261b9c04a36a609) - Sebastian Schuster _et al_, `EMNLP 2015`. 

#### Classic Papers
* [Deep Learning of Binary Hash Codes for Fast Image Retrieval](https://www.iis.sinica.edu.tw/~kevinlin311.tw/cvprw15.pdf) - Kevin Lin _et al_, `CVPRW 2015`. [[code]](https://github.com/kevinlin311tw/caffe-cvprw15)
* [Beyond instance-level image retrieval: Leveraging captions to learn a global visual representation for semantic retrieval](http://openaccess.thecvf.com/content_cvpr_2017/papers/Gordo_Beyond_Instance-Level_Image_CVPR_2017_paper.pdf) - Albert Gordo _et al_, `CVPR 2017`.

#### Image Retrieval Datasets
* PatternNet : [PatternNet: A Benchmark Dataset for Performance Evaluation of Remote Sensing Image Retrieval](https://arxiv.org/abs/1706.03424) - Weixun Zhou _et al_, `ISPRS 2018`. [[download]](https://sites.google.com/view/zhouwx/dataset)
* Google landmark dataset (GLD) v1 : [Large-Scale Image Retrieval with Attentive Deep Local Features](https://arxiv.org/abs/1612.06321) - Hyeonwoo Noh _et al_, `ICCV 2017`. [[download]](https://github.com/tensorflow/models/tree/master/research/delf)
* Google landmark dataset (GLD) v2 : [Google Landmarks Dataset v2 -- A Large-Scale Benchmark for Instance-Level Recognition and Retrieval](https://arxiv.org/abs/2004.01804) - Tobias Weyand _et al_, `CVPR 2020`. [[download]](https://github.com/cvdfoundation/google-landmark)

### Other Applications
* [Semantic Image Manipulation Using Scene Graphs](https://arxiv.org/pdf/2004.03677.pdf) - Helisa Dhamo _et al_, `CVPR 2020`.
* [SOGNet: Scene Overlap Graph Network for Panoptic Segmentation](https://arxiv.org/abs/1911.07527) - Yibo Yang _et al_, `AAAI 2020`. [[code]](https://github.com/LaoYang1994/SOGNet)
* [ReLaText: Exploiting Visual Relationships for Arbitrary-Shaped Scene Text Detection with Graph Convolutional Networks](https://arxiv.org/abs/2003.06999) - Chixiang Ma _et al_, `ARXIV 2020`.
* [Event Detection with Relation-Aware Graph Convolutional Neural Networks](https://arxiv.org/abs/2002.10757) - Shiyao Cui _et al_, `ARXIV 2020`.
* [SceneGraphNet: Neural Message Passing for 3D Indoor Scene Augmentation](https://arxiv.org/abs/1907.11308) - Yang Zhou _et al_, `ICCV 2019`. [[code]](https://github.com/yzhou359/3DIndoor-SceneGraphNet)
* [Seq-SG2SL: Inferring Semantic Layout from Scene Graph Through Sequence to Sequence Learning](https://arxiv.org/abs/1908.06592) - Boren Li  _et al_, `ICCV 2019`.
* [PlanIT: planning and instantiating indoor scenes with relation graph and spatial prior networks](https://www.semanticscholar.org/paper/PlanIT%3A-planning-and-instantiating-indoor-scenes-Wang-Lin/7a6cf9c40e5dabd7c98a7731a9705fb8883024e9) - Kai Wang  _et al_, `TOGS 2019`. 
* [Hierarchical Relational Networks for Group Activity Recognition and Retrieval](http://openaccess.thecvf.com/content_ECCV_2018/papers/Mostafa_Ibrahim_Hierarchical_Relational_Networks_ECCV_2018_paper.pdf) - Mostafa S. Ibrahim  _et al_, `ECCV 2018`. [[code]](https://github.com/mostafa-saad/hierarchical-relational-network)
* [Scene Graphs for Interpretable Video Anomaly Classification](https://nips2018vigil.github.io/static/papers/accepted/30.pdf) - Nicholas F. Y. Chen _et al_, `NIPS 2018 ViGIL Workshop`.
* [Learning object interactions and descriptions for semantic image segmentation](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Learning_Object_Interactions_CVPR_2017_paper.pdf) - Guangrun Wang  _et al_, `CVPR 2017`.
* [Multi-Modal Knowledge Representation Learning via Webly-Supervised Relationships Mining](https://dl.acm.org/doi/10.1145/3123266.3123443) - Fudong Nian  _et al_, `ACM MM 2017`.
* [Towards a Domain Specific Language for a Scene Graph based Robotic World Model](https://arxiv.org/abs/1408.0200) - Sebastian Blumenthal  _et al_, `DSLRob 2013`.

## Workshops
* [ECCV PIC 2018 Workshop](http://www.picdataset.com/) : Person in Context Challenge
* [ICCV SGRL 2019 Workshop](https://cs.stanford.edu/people/ranjaykrishna/sgrl/index.html) : Scene Graph Representation and Learning
* [ICCV PIC 2019 Workshop](http://picdataset.com/challenge/index/) : Person in Context Challenge
* [ICML LRG 2019 Workshop](https://graphreason.github.io/) : Learning and Reasoning with Graph-Structured Representations

## Challenges
* VRU : [ACM MM 2019 Video Relation Understanding (VRU) Challenge](https://videorelation.nextcenter.org/mm19-gdc/) - [Dataset](https://xdshang.github.io/docs/vidor.html)
* PIC : [Person in Context Challenge](http://picdataset.com/challenge/index/) - [Dataset](http://picdataset.com/challenge/dataset/download/) - [Baseline](https://github.com/siliu-group/pic-challenge-baseline)

## Licenses

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Youliang Jiang](https://github.com/JLCreater2015) has waived all copyright and related or neighboring rights to this work.

## Contact Us

For additional questions of any kind, please feel free to ask away in the issues section or e-mail me at *mqjyl2012@163.com*!
