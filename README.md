# neuralnet
I've tried to do/translate the Stanford cs231n assignment (currently only assignment 1, two layer neural net) into kdb/q. 

Source is http://cs231n.github.io/assignments2016/assignment1/ 

This looks at classifying the CIFAR 10 image dataset (https://www.cs.toronto.edu/~kriz/cifar.html). 

NOTE: 
* without qml (see here for installing https://github.com/zholos/qml), the matrix multiplication will likely be too slow to get anywhere
* Haven't necessarily done things the most optimized way, might try that later
* best accuracy I've found is only around 51% (which seems in line with what's expected for a simple, 2 layer neural net)
* I've written this all on 32 bit kdb, and as such there were many problems trying to stay under the memory limits, so as such many methods are sub-optimal in terms of speed (would write them differenlty if done on 64 bit)
