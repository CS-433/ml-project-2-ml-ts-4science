
### Preprocessing

40x: source
20x: 0.42 or 0.5 mpp?
10x: 1 mpp
5x: 2 mpp

#### BACH
https://iciar2018-challenge.grand-challenge.org/Dataset/
- Microscopy images, NOT WSIs. Whole-slide images are high resolution images containing the entire sampled tissue. In this sense, microscopy images are just details of the whole-slide images.
- 20x, 0.42 mpp

#### BRACS
https://www.bracs.icar.cnr.it/details/
- The Regions of Interest are provided in .png file format. The filename of a RoI includes the filename of the corresponding WSI as well as the subtype of RoI (e.g. BRACS_010_PB_32.png is the RoI number 32, extracted from the WSI named BRACS_010.svs and labeled as Pathological Benign). The resolution of each RoI is 40× and its dimension can easily exceed 4,000 by 4,000 pixels.
- 40x, 0.25 mpp

#### MIDOG
https://imig.science/midog2021/download-dataset/
- cropped regions of interest from 200 individual tumor cases
- The assignment of the scanners to the files is as follows:
    001.tiff to 050.tiff: Hamamatsu XR
    051.tiff to 100.tiff: Hamamatsu S360 (with 0.5 numerical aperture)
    101.tiff to 150.tiff: Aperio ScanScope CS2
    151.tiff to 200.tiff: Leica GT450 (only images, no annotations provided for this scanner)
- 0-100 0.23 mpp, 101-200 0.25 mpp
- Should we crop based on the MIDOG.json file?

#### BCNB
https://bupt-ai-cz.github.io/BCNB/

#### BreakHis
https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

#### PanopTILs
https://sites.google.com/view/panoptils
The PanopTILs has a region-level task (i.e. prediction is a segmentation), and a cell-level task (for which you will need to add Cell-ViT on top of the UNI embeddings). You could look at this dataset last, after you process the other 4.

#### MIDOG - discarded

# MODEL TRAINNING

After getting the embeddings for each dataset, start trainning with the following pipeline: 

### Mean approach                

| MEAN-POOLING PATCH-LEVEL | -> |Linear clasifier / Multi Layer Perceptron|

### Attention Mechanism approach

| Attention $\sum_k p_k \text{patch}_k$ | -> |Linear clasifier / Multi Layer Perceptron|

### Overview 
- Attention should work better than Mean pooling approach.
- Linear classifier, generally, works better when the task is easier.
- MLP, generally, works better when the task is harder.
- As UNI is trained with 20x, there should be a pump in the metrics of this resolution.
  'easier' or 'harder' are related to distinguishability of tissue patterns.



[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)
