# neuralnet
I've tried to do/translate the Stanford cs231n assignment into kdb/q. Currently, I've only looked at assignments 1 and 2, which cover:
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
  * training a good model for cifar data using a 7 layer convnet

TODO:
* implement dropout for n layer convnets
* implement a good mnist handwriting convnet

Source is http://cs231n.github.io/assignments2016/

This looks at classifying the CIFAR 10 image dataset (https://www.cs.toronto.edu/~kriz/cifar.html). 

NOTE: 
* without qml (see here for installing https://github.com/zholos/qml), the matrix multiplication will likely be too slow to get anywhere
* Haven't necessarily done things the most optimized way, am more focused on learning the concepts myself, will hopefully improve over time
* best accuracy I've found is only around 51% for a two layer net, and 55% for a 4 layer net - this is expected to increase when convolution nets are looked at
  * Update: with deeper convnet already have 78% accuracy, but with overfitting, hoping to improve with dropout and hyperparamter experimentation
* I've written this all on 32 bit kdb, and as such there were many problems trying to stay under the memory limits, so therefore many methods are sub-optimal in terms of speed (would write them differenlty if done on 64 bit)
