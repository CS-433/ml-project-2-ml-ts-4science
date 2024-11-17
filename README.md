
### Preprocessing

#### BACH
https://iciar2018-challenge.grand-challenge.org/Dataset/
- Microscopy images, NOT WSIs. Whole-slide images are high resolution images containing the entire sampled tissue. In this sense, microscopy images are just details of the whole-slide images.

#### BRACS
https://www.bracs.icar.cnr.it/details/
- The Regions of Interest are provided in .png file format. The filename of a RoI includes the filename of the corresponding WSI as well as the subtype of RoI (e.g. BRACS_010_PB_32.png is the RoI number 32, extracted from the WSI named BRACS_010.svs and labeled as Pathological Benign). The resolution of each RoI is 40× and its dimension can easily exceed 4,000 by 4,000 pixels.

#### MIDOG
https://imig.science/midog2021/download-dataset/
- cropped regions of interest from 200 individual tumor cases
- The assignment of the scanners to the files is as follows:
    001.tiff to 050.tiff: Hamamatsu XR
    051.tiff to 100.tiff: Hamamatsu S360 (with 0.5 numerical aperture)
    101.tiff to 150.tiff: Aperio ScanScope CS2
    151.tiff to 200.tiff: Leica GT450 (only images, no annotations provided for this scanner)

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)
