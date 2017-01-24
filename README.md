# DeepLearning-MachineLearning

It's a Matlab Tool for DeepLearning/MachineLearning.
I don't integrate them yet,so each folder (method) works individually.

# Folder (DeepNeuralNetwork)
   This tool is Deep Neural Network.
   
   The detail of how to used, you can find in example file (Main_DNN.m)

   I write the regression and classification example for XOR problem.
   
   The exmple data is sampledata.m, actually you can generate your own data by yourself (example code is marked).
   
   
   # Regression case:
        
        SizeOutputLayer: please set to 1.
   
   # Calssification case:
        
        SizeOutputLayer: please set to number of class you want to classify.
   
   # How to design your own DNN structure?
   
   SizeInputLayer: please set to dimension of input data plus one (i.e. dim+1).
   
   It' very easy to set your 'number of hidden layer' and 'number of hidden node'
   
   For instance, 2 hidden layers, first one has 20 node, second one has 10 node:  
     
     DesignDNNLayersize=[SizeInputLayer 20 10 SizeOutputLayer];
    
   Your own DNN structure is
   
   InputLayer(dim+1 nodes)→HiddenLayer1(20 nodes)→HiddenLayer2(10 nodes)→OutputLayer(Regression: 1 node, Classification: NumClass nodes)
   
   # Aactivation Function
   
   I provide four Aactivation Function(sigmoid,tanh ,ReLU, linear). 
   
   How to set the Aactivation Function?
   
   Following the above example:
   AactivationFunction={'sigmoid','sigmoid','linear'}; 
  
  Note: last one must be 'linear', because it's for output result, we don't need to activate it.
   
   # Other parameter for DNN
   
   maxIter=1000; %% maximum iteration 
   
   LearningRate=0.01; %% Learning rate
   
   % which type of optimal search approach
   
   LearningApproach='Adam'; %% 'SGD','Momentum','AdaGrad','RMSProp','Adam'
