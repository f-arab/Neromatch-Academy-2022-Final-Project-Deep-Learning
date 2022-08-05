# Overview
This package is a Python code for our final project of Deep Learning course offered by Neuromatch Academy on Summer 2022. 

#  Waste detection on TACO dataset using Mask R-CNN
In recent years, increased production of disposable goods has led to the production of massive amounts of garbage which current methods are unable to manage well. Authorities predict that waste production will reach 3 billion tons per year by 2050. Due to insufficient manpower, an automated process of waste management is critical. Such a process would be possible to implement by AI and robotic systems.

 In this project, we aim to understand how well state-of-the-art object detection and segmentation methods can identify waste in different environments  (i.e. beach, street, water, indoor, outdoor, white background, etc.) based on the TACO dataset. Although these models can perform well on popular and large datasets such as COCO, specific challenges remain for waste datasets such as having small objects of different sizes/shapes, non-uniform luminance, and partial visibility. We proposed to use Mask R-CNN model was trained and tuned on the TACO dataset with different backbones such as ResNet-50 and MobileNets. Experimental results showed a reasonable performance on the waste detection dataset, which is comparable to other object detection datasets. 
 
 The experimental results showed that the Mask R-CNN with simple and efficient backbones, such as MobileNets, can detect and segment trash objects reasonably well. The performance is comparable to more complicated backbones such as the ResNet-50, which can be computationally more expensive. In the future, we plan to compare different architectures and additional backbones and evaluate the models on different waste datasets. Additionally, advanced augmentation techniques can be applied to overcome the imbalance in the data for individual classes. 


## Installation

Open your terminal and run these lines of code

```
git clone https://github.com/congvmit/trashdetect_engine.git
cd trashdetect_engine
pip install -e .
```

or you can install this package from pypi

```
pip install -U trashdetect_engine
```


## Dataset & Checkpoints

Dataset: https://drive.google.com/drive/folders/1GNqEOhNi_arOYmWSPGwKV5sibtwSSSyb?usp=sharing

Best checkpoint: https://drive.google.com/file/d/1m-j_weoxZ5rZjlKGShTGG0PiFgZ0pi0H/view?usp=sharing

# Contact
This is a joint work with my teamates at NMA 2022 Summer School:

- Minh-Cong Vo
- Fahimeh Arab
- Unaizah Obaidellah 
- Sarah Nagasawa

Email me (Fahimeh) at farab002@ucr.edu for further questions or comments.

