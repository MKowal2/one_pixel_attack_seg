# One pixel attack for dense prediction tasks 

In this repository, I experiement with the characteristic found in the paper [One pixel attack for fooling deep neural networks ](https://arxiv.org/abs/1710.08864). Code for classification: https://github.com/nitarshan/one-pixel-attack.

I apply a similar attack to experiment with how many pixels are required to significantly reduce the accuracy for a deeplabv2 semantic segmentation network. 
It takes more than 1 pixel to reduce the accuracy more than 0.01%, but with 6+ pixels, the accuracy can drop on a single image by a percent.

Here is an example with 6 pixels, 75 iterations, and population size of 250. You can see some of the pixels that were changed are the white sky pixels turned dark, above the airplane. You can see the segmentation mask, after the attack, misses some of the background planes in the lower right corner of the image.

<p align="center">
  <img src="/experiments/num_pix = 6, iters=75, pop_size=250,img.png" height="400">
</p>

<p align="center">
  <img src="/experiments/num_pix = 6, iters=75, pop_size=250,seg.png" height="700">
</p>
