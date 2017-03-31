# neuralnet
I've tried to do/translate the Stanford cs231n assignment (currently only assignment 1, two layer neural net) into kdb/q. 

Source is http://cs231n.github.io/assignments2016/assignment1/ 

This looks at classifying the CIFAR 10 image dataset (https://www.cs.toronto.edu/~kriz/cifar.html). 

NOTE: 
* without qml (see here for installing https://github.com/zholos/qml), the matrix multiplication will likely be too slow to get anywhere
* Haven't necessarily done things the most optimized way, might try that later
* best accuracy I've found is only around 51% (which seems in line with what's expected for a simple, 2 layer neural net)
* In order to get the CIFAR 10 data into kdb as a readable format, I loaded the images into python, then exported to csvs (if there is a better way of doing this, please let me know)
