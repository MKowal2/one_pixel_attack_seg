# One pixel attack for dense prediction tasks 

In this repository, I experiement with the characteristic found in the paper [One pixel attack for fooling deep neural networks ](https://arxiv.org/abs/1710.08864). Code for classification: https://github.com/nitarshan/one-pixel-attack.

I apply a similar attack to experiment with how many pixels are required to significantly reduce the accuracy for a deeplabv2 semantic segmentation network. 
It takes more than 1 pixel to reduce the accuracy more than 0.01%, but with 6+ pixels, the accuracy can drop on a single image by a percent.
