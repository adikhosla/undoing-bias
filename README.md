Undoing the Damage of Dataset Bias
==================================

This software package contains code for the discriminative framework described in our ECCV 2012 paper, "Undoing the Damage of Dataset Bias".

Installation
------------

Before you can use the code, you need to download this repository and compile the learning code:

    $ git clone http://github.com/adikhosla/undoing-bias
    $ cd undoing-bias
    $ make

Demo
----

Features for classification experiments
---------------------------------------

We used 4 datasets in our experiments, namely Caltech-101, LabelMe, PASCAL VOC 2007 and SUN09 (described in the paper). The features used in the classification experiments are available for download from the <a href="http://undoingbias.csail.mit.edu">project website</a> in Matlab format. 

The <a href="http://undoingbias.csail.mit.edu/features.tar">archive</a> contains 4 mat files with features for the train and test images and the labels for the set of images for the five object categories used in our experiments (car, cat, chair, dog, person).

Reference
---------

Please cite our paper if you use this code:

    Undoing the Damage of Dataset Bias
    Aditya Khosla, Tinghui Zhou, Tomasz Malisiewicz, Alexei A. Efros, Antonio Torralba. 
    European Conference on Computer Vision (ECCV), 2012

<a href="http://people.csail.mit.edu/khosla/papers/eccv2012_khosla.pdf">[paper]</a> <a href="http://people.csail.mit.edu/khosla/bibtex/eccv2012.bib">[bibtex]</a> <a href="http://undoingbias.csail.mit.edu/">[project page]</a>

