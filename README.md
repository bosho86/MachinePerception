# MachinePerception
Neuronal Networks models used for the class of Machine Perception
In this repository you will find the models implemented for the project 3 of Gaze Estimation of the class of Machine Perception 
at ETH Zurich at 2019.

We implemented 4:

1. A simple convolutional NN
It was the simplest Net we used for training and to have an starting
point to start. Table 1 shows the structure of this primitive Net.
After that the signal was pass through a flatten layer. Then dense
layers were used to obtain the output.

2. The AxelNet
In a next step we decided to implement the AlexNet[1] model for the
eye gaze estimation. 

3. VGGNet
In a second model we implemented a variant of the VGG Net as
described in [2].

4. DenseNet

The rest of the framework, is distributed by the Assistants of the class and belongs to them.

References:
[1]http://papers.nips.cc/paper/
4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
[2] Karen Simonyan and Andrew Zisserman. 2015. Very Deep Convolutional Net-
works for Large-Scale Image Recognition. CoRR abs/1409.1556 (2015).
