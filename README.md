# neuralnet

UPDATE: 
* have added slides to my kx meetup presentation here [kx meetup slides](https://github.com/ryantorsparks/neuralnet/blob/master/kx%20meetup%20convnets.pdf), they are probably a good start for reading

I've tried to do/translate the Stanford cs231n assignment into kdb/q. Currently, I've completed assignments 1 and 2, and started on the third and final. The first 2 look at
* 2 layer neural networks
* n-layer fully connected neural networks (10 layer example)
* different update functions:
  * sgd
  * sgd momentum
  * rms prop
  * adam 
* batch normalization in fully connected nets
* dropout in fully connected nets
* simple convolutional neural networks:
  * max pooling layers
  * convolutional layers
  * convolutional relu/pool layers (sandwich layers)
  * three layer convnet
* deeper convnets
  * n layer convnet model
  * dropout in the fully connected layers of an n layer convnet
  * training a good model for cifar data using a 7 layer convnet

TODO:
* implement ensembles for convnets
* convert the convnet stuff to use flat list matrixes instead of actual nested lists of lists (bit of a huge task though)
* reduce some of the hard coding (e.g. around filtersizes etc.)

Source is http://cs231n.github.io/assignments2016/

This looks at classifying the CIFAR 10 image dataset (https://www.cs.toronto.edu/~kriz/cifar.html). 

NOTE: 
* without qml (see here for installing https://github.com/zholos/qml), the matrix multiplication will likely be too slow to get anywhere
* Haven't necessarily done things the most optimized way, am more focused on learning the concepts myself, will hopefully improve over time
  * UPDATE: have refactored the fully connected neural networks to be able to run storing things as "flat" matrixes, i.e. as (shape;list). At some stage would be good to do the same for the convnets.
* best accuracy I've found is only around 51% for a two layer net, and 55% for a 4 layer net - this is expected to increase when convolution nets are looked at
  * Update: with deeper convnet already have validation 80% accuracy, but with overfitting, hoping to improve with hyperparamter experimentation (probably want to speed it up first before I try and do that)
* I've written this all on 32 bit kdb, and as such there were many problems trying to stay under the memory limits, so therefore many methods are sub-optimal in terms of speed (would write them differenlty if done on 64 bit)
  * UPDATE: kx have kindly let me use 64 bit temporarily for this project - I've left the data loading stuff the way it is though in case people want to try run it with 32 bit q
