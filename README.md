# AI_miniproject
Implementation of VQA

50.021 Artificial Intelligence Project

```
conda install mkl -c anaconda
conda install pytorch cuda90 -c pytorch
conda install scikit-image (not sure if should be here. Check pillow?)
pip install torchvision
conda install matplotlib
conda install nltk
```

`vqaTools` and `vqaEvaluation` are modified from from https://github.com/GT-Vision-Lab/VQA

A signficiant part of the code/structure of the code is based on https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

The scripts should expect the data to be in the following folders: (Data downloaded from http://www.visualqa.org/download.html)

  * `annotations`
    * `v2_mscoco_train2014_annotations.json`
    * `v2_mscoco_val2014_annotations.json`
  * `images`
    * `train2014 - vqa training images`
    * `val2014 - vqa validation images`
  * `questions`
    *  `v2_OpenEnded_mscoco_train2014_questions.json`
    *  `v2_OpenEnded_mscoco_val2014_questions.json`
