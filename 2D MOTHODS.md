# Awesome Scene Graph[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<p align="center">
  <img width="250" src="https://camo.githubusercontent.com/1131548cf666e1150ebd2a52f44776d539f06324/68747470733a2f2f63646e2e7261776769742e636f6d2f73696e647265736f726875732f617765736f6d652f6d61737465722f6d656469612f6c6f676f2e737667" "Awesome!">
</p>

A curated list of 2D scene graph generation grouped in different methods . :-)

## Introduction

**Markdown format of paper list items:**

```markdown
- [Paper Name](link) - Author 1 _et al_, `Conference Year`. [[code]](link)
```

## Table of Contents

- [Using Message Passing](#Using Message Passing)
- [Attention Mechanisms](#Attention Mechanisms)
- [Using Language Priors](#Using Language Priors)
- [Using External Knowledge Bases](#Using External Knowledge Bases)
- [Visual Embedding Approaches](#Visual Embedding Approaches)
- [Predict Object Pair-wise Relationship Categories Directly](#Predict Object Pair-wise Relationship Categories Directly)
- [Using Depth Information](#Using Depth Information)
- [Other Methods](#Other Methods)

### Using Message Passing
* [Scene Graph Generation by Iterative Message Passing](https://arxiv.org/abs/1701.02426) - Danfei Xu _et al_, `CVPR 2017`.  [[code]](https://github.com/danfeiX/scene-graph-TF-release)
* [Scene Graph Generation from Objects, Phrases and Region Captions](https://arxiv.org/abs/1707.09700) - Yikang Li  _et al_, `ICCV 2017`.  [[code]](https://github.com/yikang-li/MSDN)
* [ViP-CNN: Visual Phrase Guided Convolutional Neural Network](https://arxiv.org/abs/1702.07191) - Yikang Li  _et al_, `CVPR 2017`. 
* [Detecting Visual Relationships with Deep Relational Networks](https://arxiv.org/abs/1704.03114) - Bo Dai  _et al_, `CVPR 2017`.  [[code]](https://github.com/doubledaibo/drnet_cvpr2017) 
* [Recurrent Visual Relationship Recognition with Triplet Unit](https://ieeexplore.ieee.org/document/8241583) - Kento Masui  _et al_, `ISM 2017`.
* [Factorizable Net: An Efficient Subgraph-based Framework for Scene Graph Generation](https://arxiv.org/abs/1806.11538) - Yikang Li  _et al_, `ECCV 2018`.  [[code]](https://github.com/yikang-li/FactorizableNet)
* [Neural Motifs_Scene Graph Parsing with Global Context](https://arxiv.org/abs/1711.06640) - Rowan Zellers  _et al_, `CVPR 2018`.  [[code]](https://github.com/rowanz/neural-motifs)
* [Zoom-Net: Mining Deep Feature Interactions for Visual Relationship Recognition](https://arxiv.org/abs/1807.04979) - Guojun Yin  _et al_, `ECCV 2018`.  [[code]](https://github.com/gjyin91/ZoomNet)
* [Deep Structured Learning for Visual Relationship Detection](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16475) - Yaohui Zhu  _et al_, `AAAI 2018`.
* [Region-Object Relevance-Guided Visual Relationship Detection](http://www.bmva.org/bmvc/2018/contents/papers/1020.pdf) - Yusuke Goutsu  _et al_, `BMVC 2018`.
* [Recurrent Visual Relationship Recognition with Triplet Unit for Diversity](https://www.worldscientific.com/doi/10.1142/S1793351X18400214) - Kento Masui  _et al_, `IJSC 2018`.
* [Deep Image Understanding Using Multilayered Contexts](https://www.hindawi.com/journals/mpe/2018/5847460/) - Donghyeop Shin  _et al_, `MPE 2018`.
* [Large-Scale Visual Relationship Understanding](https://arxiv.org/abs/1804.10660) - Ji Zhang  _et al_, `AAAI 2019`.  [[code]](https://github.com/facebookresearch/Large-Scale-VRD)
* [Learning to Compose Dynamic Tree Structures for Visual Contexts](https://arxiv.org/abs/1812.01880) - Kaihua Tang  _et al_, `CVPR 2019 Oral`.  [[code]](https://github.com/KaihuaTang/VCTree-Scene-Graph-Generation)
* [Counterfactual Critic Multi-Agent Training for Scene Graph Generation](https://arxiv.org/abs/1812.02347) - Long Chen  _et al_, `ICCV 2019 Oral`. 
* [On Exploring Undetermined Relationships for Visual Relationship Detection](https://arxiv.org/abs/1905.01595) - Yibing Zhan  _et al_, `CVPR 2019`. [[code]](https://github.com//Atmegal//MFURLN-CVPR-2019-relationship-detection-method)
* [Exploring Context and Visual Pattern of Relationship for Scene Graph Generation](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Exploring_Context_and_Visual_Pattern_of_Relationship_for_Scene_Graph_CVPR_2019_paper.html) - Wenbin Wang  _et al_, `CVPR 2019`. 
* [Neural Message Passing for Visual Relationship Detection](https://users.ece.cmu.edu/~sihengc/paper/19_ICMLW_HuCCZ.pdf) - Yue Hu  _et al_, `ICML LRG Workshop 2019`.  [[code]](https://github.com/PhyllisH/NMP)
* [PANet: A Context Based Predicate Association Network for Scene Graph Generation](https://ieeexplore.ieee.xilesou.top/abstract/document/8784780) - Yunian Chen  _et al_, `ICME 2019`. 
* [Visual Relationship Detection with Relative Location Mining](https://arxiv.org/abs/1911.00713) - Hao Zhou  _et al_, `ACM MM 2019`. 
* [Visual relationship detection based on bidirectional recurrent neural network](https://link.springer.com/article/10.1007%2Fs11042-019-7732-z) - Yibo Dai  _et al_, `Multimedia Tools and Applications 2019`. 
* [Exploring the Semantics for Visual Relationship Detection](https://arxiv.org/abs/1904.02104) - Wentong Liao  _et al_, `arXiv 2019`. 
* [A hierarchical recurrent approach to predict scene graphs from a visual‐attention‐oriented perspective](https://onlinelibrary.wiley.com/doi/full/10.1111/coin.12202) - Wenjing Gao  _et al_, `Computational Intelligence 2019 SPECIAL ISSUE ARTICLE`. 
* [Relationship-Aware Spatial Perception Fusion for Realistic Scene Layout Generation](https://arxiv.xilesou.top/abs/1909.00640) - Hongdong Zheng  _et al_, `arXiv 2019`. 
* [The Limited Multi-Label Projection Layer](https://arxiv.org/abs/1906.08707) - Brandon Amos  _et al_, `arXiv 2019`.  [[code]](https://github.com/locuslab/lml)

### Attention Mechanisms
* [Towards Context-Aware Interaction Recognition for Visual Relationship Detection](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhuang_Towards_Context-Aware_Interaction_ICCV_2017_paper.pdf) - Bohan Zhuang  _et al_, `ICCV 2017`.
* [Graph R-CNN for Scene Graph Generation](https://arxiv.org/abs/1808.00191) - Jianwei Yang  _et al_, `ECCV 2018`.  [[code]](https://github.com/jwyang/graph-rcnn.pytorch)
* [LinkNet_Relational Embedding for Scene Graph](https://arxiv.org/abs/1811.06410v1) - Sanghyun Woo  _et al_, `NIPS 2018`.  [[code]](https://github.com/jiayan97/linknet-pytorch)
* [Generating Triples with Adversarial Networks for Scene Graph Construction](https://arxiv.org/abs/1802.02598) - Matthew Klawonn  _et al_, `AAAI 2018`.
* [Referring Relationships](https://cs.stanford.edu/people/ranjaykrishna/referringrelationships/index.html) - Ranjay Krishna  _et al_, `CVPR 2018`.  [[code]](https://github.com/StanfordVL/ReferringRelationships)
* [Scene Graph Generation Based on Node-Relation Context Module](https://link.springer.xilesou.top/chapter/10.1007/978-3-030-04179-3_12) - Xin Lin  _et al_, `ICONIP 2018`.
* [Detecting Visual Relationships Using Box Attention](https://arxiv.org/abs/1807.02136) - Alexander Kolesnikov  _et al_, `ICCVW 2019`.
* [Visual Relationships as Functions: Enabling Few-Shot Scene Graph Prediction](https://arxiv.org/abs/1906.04876) - Apoorva Dornadula  _et al_, `ICCVW 2019`.
* [Attention-Translation-Relation Network for Scalable Scene Graph Generation](http://openaccess.thecvf.com/content_ICCVW_2019/papers/SGRL/Gkanatsios_Attention-Translation-Relation_Network_for_Scalable_Scene_Graph_Generation_ICCVW_2019_paper.pdf) - Nikolaos Gkanatsios  _et al_, `ICCVW 2019`.
* [Attentive Relational Networks for Mapping Images to Scene Graphs](https://arxiv.org/abs/1811.10696v1) - Mengshi Qi  _et al_, `CVPR 2019`.
* [Expressing Visual Relationships via Language](https://arxiv.org/abs/1906.07689) - Hao Tan  _et al_, `ACL 2019`.  [[code]](https://github.com/airsplay/VisualRelationships)
* [Visual Spatial Attention Network for Relationship Detection](https://dl.acm.org/doi/10.1145/3240508.3240611) - Chaojun Han,  _et al_, `ACM MM 2019`.
* [Visual Relation Detection with Multi-Level Attention](https://dlacm.xilesou.top/doi/abs/10.1145/3343031.3350962) - Sipeng Zheng,  _et al_, `ACM MM 2019`.
* [Visual Relationship Recognition via Language and Position Guided Attention](https://ieeexplore.ieee.xilesou.top/abstract/document/8683464/) - Hao Zhou,  _et al_, `ICASSP 2019`.
* [Relationship Detection Based on Object Semantic Inference and Attention Mechanisms](https://www.researchgate.net/publication/333698036_Relationship_Detection_Based_on_Object_Semantic_Inference_and_Attention_Mechanisms) - Liang Zhang  _et al_, `ICMR 2019`.
* [Leveraging Auxiliary Text for Deep Recognition of Unseen Visual Relationships](https://arxiv.org/abs/1910.12324v1) - Gal Sadeh Kenigsfield  _et al_, `Under review as a conference paper at ICLR 2020`.

### Using Language Priors
* [Visual Relationship Detection with Language Priors](https://arxiv.org/abs/1608.00187) - Cewu Lu _et al_, `ECCV 2016 Oral`. [[code]](https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection)
* [Phrase Localization and Visual Relationship Detection with Comprehensive Image-Language Cues](https://arxiv.org/abs/1611.06641) - Bryan A _et al_, `ICCV 2017`. [[code]](https://github.com/BryanPlummer/pl-clc)
* [Deep Variation-Structured Reinforcement Learning for Visual Relationship and Attribute Detection](https://arxiv.org/abs/1703.03054) - Xiaodan Liang _et al_, `CVPR 2017`. [[code]](https://github.com/nexusapoorvacus/DeepVariationStructuredRL)
* [Improving Visual Relationship Detection using Semantic Modeling of Scene Descriptions](https://arxiv.org/abs/1809.00204) - Bryan A _et al_, `ISWC 2017`.
* [Tensorize, Factorize and Regularize: Robust Visual Relationship Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hwang_Tensorize_Factorize_and_CVPR_2018_paper.pdf) - Seong Jae Hwang _et al_, `CVPR 2018`. [[code]](https://github.com/shwang54/visual-tensor-decomposition)
* [Visual Relationship Detection with Language prior and Softmax](https://arxiv.org/abs/1904.07798) - Jaewon Jung _et al_, `IPAS 2018`. [[code]](https://github.com/pranoyr/visual-relationship-detection)
* [Visual Relationship Detection Based on Guided Proposals and Semantic Knowledge Distillation](https://arxiv.org/abs/1805.10802) - François Plesse _et al_, `ICME 2018`.
* [Context-Dependent Diffusion Network for Visual Relationship Detection](https://arxiv.org/abs/1809.06213) - Zhen Cui _et al_, `ACM MM 2018`. [[code]](https://github.com/pranoyr/visual-relationship-detection)
* [Natural Language Guided Visual Relationship Detection](http://openaccess.thecvf.com/content_CVPRW_2019/html/MULA/Liao_Natural_Language_Guided_Visual_Relationship_Detection_CVPRW_2019_paper.html) - Wentong Liao _et al_, `CVPR 2019`.
* [Knowledge-Embedded Routing Network for Scene Graph Generation](https://arxiv.org/abs/1903.03326) - Tianshui Chen _et al_, `CVPR 2019`. [[code]](https://github.com/yuweihao/KERN)
* [Soft Transfer Learning via Gradient Diagnosis for Visual Relationship Detection](https://ieeexplore.ieee.xilesou.top/abstract/document/8658599) - Diqi Chen _et al_, `WACV 2019`.
* [Compensating Supervision Incompleteness with Prior Knowledge in Semantic Image Interpretation](https://arxiv.org/abs/1910.00462) - Ivan Donadello _et al_, `IJCNN 2019`. [[code]](https://github.com/ivanDonadello/Visual-Relationship-Detection-LTN)
* [Hierarchical Visual Relationship Detection](https://dl.acm.org/doi/10.1145/3343031.3350921) - Xu Sun _et al_, `ACM MM 2019`.
* [Visual Relationship Detection with Low Rank Non-Negative Tensor Decomposition](https://arxiv.xilesou.top/abs/1911.09895) - Mohammed Haroon Dupty _et al_, `arXiv 2019`.
* [Relational Reasoning using Prior Knowledge for Visual Captioning](https://arxiv.xilesou.top/abs/1906.01290) - Jingyi Hou _et al_, `arXiv 2019`.

### Using External Knowledge Bases
* [Visual Relationship Detection with Internal and External Linguistic Knowledge Distillation](https://arxiv.org/abs/1707.09423v1) - Ruichi Yu _et al_, `ICCV 2017`.
* [Scene Graph Generation with External Knowledge and Image Reconstruction](https://arxiv.org/abs/1904.00560) - Jiuxiang Gu _et al_, `CVPR 2019`. [[code]](https://github.com/arxrean/SGG_Ex_RC)

### Visual Embedding Approaches
* [Visual Translation Embedding Network for Visual Relation Detection](https://arxiv.org/abs/1702.08319v1) - Hanwang Zhang _et al_, `CVPR 2017`. [[code]](https://github.com/YANYANYEAH/vtranse)
* [Representation Learning for Scene Graph Completion via Jointly Structural and Visual Embedding](https://www.ijcai.org/proceedings/2018/132) - Hai Wan _et al_, `IJCAI 2018`. [[code]](https://github.com/sysulic/RLSV)
* [Visual Relationship Detection Using Joint Visual-Semantic Embedding](http://www.cs.umanitoba.ca/~ywang/papers/icpr18.pdf) - Binglin Li _et al_, `ICPR 2018`.
* [Object Relation Detection Based on One-shot Learning](https://arxiv.org/abs/1807.05857) - Li Zhou _et al_, `arXiv 2018`.
* [Attention-Translation-Relation Network for Scalable Scene Graph Generation](http://openaccess.thecvf.com/content_ICCVW_2019/html/SGRL/Gkanatsios_Attention-Translation-Relation_Network_for_Scalable_Scene_Graph_Generation_ICCVW_2019_paper.html) - Nikolaos Gkanatsios _et al_, `ICCV 2019`.
* [Detecting Unseen Visual Relations Using Analogies](http://openaccess.thecvf.com/content_ICCV_2019/html/Peyre_Detecting_Unseen_Visual_Relations_Using_Analogies_ICCV_2019_paper.html) - Julia Peyre _et al_, `ICCV 2019`.
* [Contextual Translation Embedding for Visual Relationship Detection and Scene Graph Generation](https://arxiv.org/abs/1905.11624) - Zih-Siou Hung _et al_, `arXiv 2019`.
* [Deeply Supervised Multimodal Attentional Translation Embeddings for Visual Relationship Detection](https://arxiv.org/abs/1902.05829) - Nikolaos Gkanatsios _et al_, `arXiv 2019`.

### Predict Object Pair-wise Relationship Categories Directly
* [Relationship Proposal Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Relationship_Proposal_Networks_CVPR_2017_paper.pdf) - Ji Zhang _et al_, `CVPR 2017`. 
* [Visual relationship detection with object spatial distribution](https://www.researchgate.net/publication/319566830_Visual_relationship_detection_with_object_spatial_distribution) - Yaohui Zhu _et al_, `ICME 2017`. 
* [A Problem Reduction Approach for Visual Relationships Detection](https://arxiv.org/abs/1809.09828) - Toshiyuki Fukuzawa _et al_, `ECCVW 2018`. 
* [An Interpretable Model for Scene Graph Generation](https://arxiv.org/abs/1811.09543) - Ji Zhang _et al_, `arXiv 2018`. 
* [Shuffle-Then-Assemble: Learning Object-Agnostic Visual Relationship Features](https://arxiv.org/abs/1808.00171) - Xu Yang _et al_, `ECCV 2018`. [[code]](https://github.com/yangxuntu/vrd)
* [Visual Relationship Detection with Deep Structural Ranking](http://www.jdl.link/doc/2011/20191720205091168_aaai18_small.pdf) - Kongming Liang _et al_, `AAAI 2018`. [[code]](https://github.com/GriffinLiang/vrd-dsr)
* [Learning Prototypes for Visual Relationship Detection](https://ieeexplore.ieee.xilesou.top/abstract/document/8516557) - François Plesse _et al_, `CBMI 2018`.
* [Visual Relationship Detection Based on Local Feature and Context Feature](https://ieeexplore.ieee.xilesou.top/abstract/document/8525683) - Yuping Han _et al_, `ICNIDC 2018`.
* [VrR-VG: Refocusing Visually-Relevant Relationships](https://arxiv.org/abs/1902.00313) - Yuanzhi Liang _et al_, `ICCV 2019`.
* [SpatialSense: An Adversarially Crowdsourced Benchmark for Spatial Relation Recognition](https://arxiv.org/abs/1908.02660) - Kaiyu Yang _et al_, `ICCV 2019`. [[code]](https://github.com/princeton-vl/SpatialSense)
* [BLOCK: Bilinear Superdiagonal Fusion for Visual Question Answering and Visual Relationship Detection](https://arxiv.org/abs/1902.00038) - Kaiyu Yang _et al_, `AAAI 2019`. [[code]](https://github.com/Cadene/block.bootstrap.pytorch)
* [On Class Imbalance and Background Filtering in Visual Relationship Detection](https://arxiv.org/abs/1903.08456) - Alessio Sarullo _et al_, `arXiv 2019`. 

### Using Depth Information
* [On Support Relations and Semantic Scene Graphs](https://arxiv.org/abs/1609.05834) - Michael Ying Yang _et al_, `ISPRS 2017`.
* [Visual Relationship Prediction via Label Clustering and Incorporation of Depth Information](https://tsujuifu.github.io/projs/eccv18_pic.html#) - Hsuan-Kung Yang _et al_, `ECCVW 2018`.
* [Support Relation Analysis for Objects in Multiple View RGB-D Images](https://arxiv.org/abs/1905.04084) - Peng Zhang _et al_, `IJCAIW QR 2019`.
* [Improving Visual Relation Detection using Depth Maps](https://arxiv.org/abs/1905.00966) - Sahand Sharifzadeh _et al_, `arXiv 2019`.

### Other Methods
* [Weakly-Supervised Learning of Visual Relations](https://arxiv.org/abs/1707.09472) - Julia Peyre _et al_, `ICCV 2017`. [[code]](https://github.com/yjy941124/PPR-FCN)
* [PPR-FCN: Weakly Supervised Visual Relation Detection via Parallel Pairwise R-FCN](https://arxiv.org/abs/1708.01956) - Hanwang Zhang _et al_, `ICCV 2017`. [[code]](https://github.com/jpeyre/unrel)
* [Detecting Visual Relationships with Deep Relational Networks](https://arxiv.org/abs/1704.03114) - Bo Dai _et al_, `CVPR 2017`. [[code]](https://github.com/doubledaibo/drnet_cvpr2017)
* [Pixels to Graphs by Associative Embedding](https://arxiv.org/abs/1706.07365) - Alejandro Newell _et al_, `NIPS 2017`. [[code]](https://github.com/princeton-vl/px2graph)
* [Mapping Images to Scene Graphs with Permutation-Invariant Structured Prediction](https://arxiv.org/abs/1802.05451) - Roei Herzig _et al_, `NIPS 2018`. [[code]](https://github.com/shikorab/SceneGraph)
* [Scene Graph Parsing as Dependency Parsing](https://arxiv.org/abs/1803.09189) - Yu-Siang Wang _et al_, `NAACL 2018`. [[code]](https://github.com/vacancy/SceneGraphParser)
* [Scene Graph Generation via Conditional Random Fields](https://arxiv.org/abs/1811.08075) - Weilin Cong _et al_, `arXiv 2018`.
* [MR-NET: Exploiting Mutual Relation for Visual Relationship Detection](https://www.aaai.org/ojs/index.php/AAAI/article/view/4819) - Yi Bin _et al_, `AAAI 2019`.
* [Scene Graph Prediction with Limited Labels](https://arxiv.org/abs/1904.11622) - Vincent S. Chen _et al_, `ICCV 2019`. [[code]](https://github.com/vincentschen/limited-label-scene-graphs)
* [Differentiable Scene Graphs](https://arxiv.org/abs/1902.10200) - Moshiko Raboh _et al_, `ICCVW 2019`.
* [Graphical Contrastive Losses for Scene Graph Parsing](https://arxiv.org/abs/1903.02728) - Ji Zhang _et al_, `CVPR 2019`. [[code]](https://github.com/NVIDIA/ContrastiveLosses4VRD)
* [Generating Expensive Relationship Features from Cheap Objects](https://bmvc2019.org/wp-content/uploads/papers/0657-paper.pdf) - Xiaogang Wang _et al_, `BMVC 2019`.
* [Optimising the Input Image to Improve Visual Relationship Detection](https://arxiv.org/abs/1903.11029) - Noel Mizzi _et al_, `arXiv 2019`.