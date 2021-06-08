# Global Wheat Detection
The implementation of deep learning approach for wheat detection.

The full problem was presented here:
https://www.kaggle.com/c/global-wheat-detection

## Approach
### Goal
The competition is already over, so the goal of this pet project is to 
get a relatively fast solution which can be actually applied in the real
software products.

### Detector choice
#### Architecture
R-CNN models won't be considered due to their 
2-step nature leading to higher processing times.

There are advanced architectures like DE⫶TR that may produce high quality
results, but there is a problem with them:
> A single epoch takes 28 minutes, so 300 epoch training takes around 6 days on a single machine with 8 V100 cards
> ([link](https://github.com/facebookresearch/detr))

I don't have 8 V100 cards to experiment with :-( The true SOTA on COCO
dataset is [SWIN](https://github.com/microsoft/Swin-Transformer). The smallest
version of this architecture requires 267 GFLOPs while YOLOv5s requires 17 GFLOPs.

Among single shot detection architectures the relatively fast ones are
YOLO and EfficientDet families. SqueezeDet is not considered due to its low 
accuracy comparing to other detectors.
I've decided to utilize YOLOv5 since is may fit the task in terms of
training and inference time. YOLOv5 became a source of controversy and turned out to be
worse than YOLOv4 in some tasks. I wanted to check how it performs at this
specific task.

#### Dataset size
Another reason of choosing smaller detector is the amount of available data. Unlike COCO with its 200K+
labeled images, current dataset contains ~3.5K images. Complex models
benefit from large datasets, but they struggle in generalizing in fewer samples.


### Dataset
Wheat Detection Dataset consists of photos of wheat of different types
and colors on complex backgrounds.
- train set: 3422 images
- test set: 10 images
Images size: 1024⨯1024