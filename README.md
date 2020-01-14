# styleGAN2
PyTorch implementation of styleGAN2 from Karras et al 

# Main differences with the original version
There is no path length regularization. If anyone is up to the task, please be my guess. It just seems
computationally expensive, and transitions without it are smooth enough in my humble opinion. Because of this,
the output for the critic is substantially different: I return a scalar, which gives a metric of the
quality of the image. Though this was the original approach in PGGAN, now, in order to increase linear separability,
the authors train the critic to classify the image by using 40 labels, comming from the 40 attributes available
in the original CelebA dataset, and then calculate the cost function as the exponential of the conditional
entropy of the classification.

Again, since I do not have the resources of NVIDIA corporation, I altered the size of the latent vector,
the depth of the mapping subnetwork, and the number of channels in each convolutional block. Also the 
number of batches before lazy regularization. I pretty much downscaled everything. All of these
can be changed in the options, so feel free to if you need. 

The normalization of the weights for linear networks is somehow different. The authors use the number of input
channels, whereas I use the sum of input and output channels.

Kept the pulling away term from my previous implementation of PGGAN, since it gave me good results when
I was struggling with model collapse.

Added the option 'asRanker' in the critic constructor, to skip the application of the standard deviation
layer when using a pretrained critic as a ranker of good and bad images.