5 layer neural net that implements gpu for mmu, vs a 5 layer net using .qml.mm. We can see that most functions have become significantly faster now - as they have been converted to input/output lists. Note this was for a fairly large batchsize example, I haven't had time to "optimise the hyperparameters" yet, using the same params as a smaller batchsize has lead to lower accuracy

| nc                              | timepc      | timepc_old  | pct         | pct_old      | faster | 
|---------------------------------|-------------|-------------|-------------|--------------|--------| 
| dot                             | 4.197397    | 189.8416    | 67.0992     | 81.91524     | 1      | 
| affineBackward                  | 1.570905    | 17.00013    | 7.874734    | 1.294487     | 1      | 
| adam                            | 0.717085    | 17.21669    | 7.189301    | 2.621956     | 1      | 
| solver.xTrainParser             | 5.728308    | 79.70751    | 5.851577    | 1.537578     | 1      | 
| fullyConnectedNet.loss          | 4.648077    | 11.18403    | 5.540673    | 0.6245188    | 1      | 
| solver.genBatch                 | 2.53165     | 35.29912    | 2.538164    | 0.5375756    | 1      | 
| solver.step                     | 1.79778     | 26.36362    | 1.802405    | 0.4014955    | 1      | 
| softmaxLoss                     | 0.4852669   | 6.1343      | 0.4865156   | 0.09342017   | 1      | 
| reluBackward                    | 0.1197203   | 16.11033    | 0.4801133   | 0.9813863    | 1      | 
| solver.checkAccuracy            | 8.143833    | 8.682292    | 0.1542952   | 0.03525971   | 1      | 
| fullyConnectedForwardPassLoop   | 0.02378636  | 0.02285152  | 0.1417709   | 0.006380167  | 0      | 
| affineForward                   | 0.02106927  | 1.279903    | 0.1255766   | 0.3573503    | 1      | 
| fullyConnectedBackwardPassLoop| | 0.0248452   | 0.03345556  | 0.1245456   | 0.002547498  | 1      | 
| solver.i.step                   | 0.1208094   | 3.405189    | 0.1211203   | 0.05185813   | 1      | 
| reluForward                     | 0.02489603  | 1.217006    | 0.1187078   | 0.2718315    | 1      | 
| symi                            | 0.001768841 | 0.001820141 | 0.1097967   | 0.003501858  | 1      | 
| randArrayFlat                   | 9.9008      | 0.07815965  | 0           |              | 0      | 
| affineReluForward               | 0.008504636 | 0.009325    | 0.04055132  | 0.00208284   | 1      | 
| getModelValue                   | 0.01049556  | 0.01791197  | 0.03170025  | 0.0008607854 | 1      | 
| affineReluBackward              | 0.006216142 | 0.3998278   | 0.02492854  | 0.02435615   | 1      | 
| solver.i.train                  | 0.02069055  | 0.04227778  | 0.02074379  | 0.0006438546 | 1      | 
| fullyConnectedNet.params        | 0.00486688  | 0.006442529 | 0.01561409  | 0.0005690625 | 1      | 
| shape                           | 0.004523022 | 0.00355     | 0.009926263 | 0.001009185  | 0      | 
| solver.train                    | 4.849       | 27.402      | 0.007655869 | 0.009273537  | 1      | 
| removeDFromDictKey              | 0.007250394 | 0.01258889  | 0.007269049 | 0.0001917181 | 1      | 

