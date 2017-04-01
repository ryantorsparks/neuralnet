\l nn_util.q

lg "loading in CIFAR binary data";
allData:raze {0N 3073#read1 hsym `$"CIFAR_binary/cifar-10-batches-bin/data_batch_",string[x],".bin"}each 1+til 5

lg " get training and validation data"
yTrain:`int$49000#allData[;0]
xTrain:{flip 0N 1024#x}each `real$1_'49000#allData

yVal:`int$-1000#allData[;0]
xVal:{flip 0N 1024#x}each `real$1_'-1000#allData

lg "clearing data, for 32 bit memory issues"
delete allData from `.;
.Q.gc[];

lg "subtracting means from xTraining data"
lg "getting averages for training data"
avgRes:{`real$avg xTrain[;x]}each til 1024;
{if[0=x mod 100;lg "subtracting mean for index ",string[x]," of 1024"];xTrain[;x]:xTrain[;x]-\:avgRes x;}each til 1024;

lg "same for validation data"
lg "getting averages for validation data"
{if[0=x mod 100;lg "subtracting mean for index ",string[x]," of 1024"];xVal[;x]:xVal[;x]-\:avgRes x;}each til 1024;

lg "same for testing data"
allData:0N 3073#read1 hsym `$"CIFAR_binary/cifar-10-batches-bin/test_batch.bin"
yTest:`int$allData[;0]
xTest:{flip 0N 1024#x}each `real$1_'allData
delete allData from `.;
.Q.gc[];

lg "same for validation data"
lg "getting averages for validation data"
{if[0=x mod 100;lg "subtracting mean for index ",string[x]," of 1024"];xTest[;x]:xTest[;x]-\:avgRes x;}each til 1024;
delete avgRes from `.;
