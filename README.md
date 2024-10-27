#   GAN
fuuuuuck

This is my first implementation of a generative adversarial network, these networks 
don't learn directly from the Dataset. Actually there are two subnetworks in GAN: 
Generator & Discriminator, the Discriminator is a simple binary classifier that learns
the distinction between real data from Dataset and fake data which Generator generated 
from noise, for example in an image generative ai Generator will make fake data from  
Gaussian noise.
The better these two networks get at their job the better results will be generated so 
Generator eventually should win the game with tricking the Discriminator.
