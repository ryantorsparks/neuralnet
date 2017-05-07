# neuralnet
I've tried to do/translate the Stanford cs231n assignment into kdb/q. Currently, I've only looked at assignments 1 and 2, which cover:
* 2 layer neural networks
* n-layer fully connected neural networks
* different update functions:
  * sgd
  * sgd momentum
  * rms prop
  * adam 
* batch normalization in fully connected nets

Source is http://cs231n.github.io/assignments2016/assignment1/ 

This looks at classifying the CIFAR 10 image dataset (https://www.cs.toronto.edu/~kriz/cifar.html). 

NOTE: 
* without qml (see here for installing https://github.com/zholos/qml), the matrix multiplication will likely be too slow to get anywhere
* Haven't necessarily done things the most optimized way, am more focused on learning the concepts myself, will hopefully improve over time
* best accuracy I've found is only around 51% for a two layer net, and 55% for a 4 layer net - this is expected to increase when convolution nets are looked at
* I've written this all on 32 bit kdb, and as such there were many problems trying to stay under the memory limits, so therefore many methods are sub-optimal in terms of speed (would write them differenlty if done on 64 bit)
