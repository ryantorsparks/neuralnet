// kick of a cnn run in the background, and log to logs/[hostname]_cnn_dropout_0.05_learnRate_0.01.log 
// for example. Arrange the params in alphabetical order
// usage: nohup q run_cnn.q -reg 0.05 -learnRate 0.0001 -learnRateDecay 0.9 -dropout 0.5 &
logName:(first "." vs string[.z.h]),"_cnn_","_" sv {raze asc 0N 2#x}.z.x except \:"-";
system"1 logs/",logName,".log";
-1"##################  starting run_cnn.q ####################";
 
-1"inputs are "," " sv .z.X;
inputd:(first "F"$)each .Q.opt .z.x
/ add in optimConfig if learnRate is specified
if[`learnRate in key inputd;inputd:`learnRate _ inputd,enlist[`optimConfig]!enlist(enlist[`learnRate]!enlist inputd`learnRate)]
 
\l load_all.q
\l load_cifar_data.q
 
defaultStartd:(!). flip (`useBatchNorm,1b;(`numFilters;16 32 64 128);`batchSize,50;`updateRule`adam;`filterSize,3;`printEvery,50;(`dimHidden;500 500);(`dimInput;3 32 32);(`numEpochs;100);`wScale,.05;`learnRateDecay,0.9;`nClass,10;(`xTrain;xTrain);(`yTrain;yTrain);(`xVal;xVal);(`yVal;yVal);`model`nLayerConvNet;(`optimConfig;(enlist `learnRate)!enlist 5e-4);`reg,0.05;`dropout,0.5)
 
startd:defaultStartd,inputd
 
-1"starting cnn with input d \n",.Q.s `xTrain`yTrain`xVal`yVal _ startd;
 
res:solver.train startd;
