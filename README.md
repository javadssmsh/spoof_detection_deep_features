# spoof_detection_deep_features
## Keras implementation of the paper "Spoofing Detection on the ASVspoof2015 Challenge Corpus Employing Deep Neural Networks" 
### Requirements:
- keras 
- librosa
- matplotlib
- numpy
- tensorflow
- sklearn
- pandas
  
### Description:
In order to discriminate between the human and spoofed speech signals:
- We train a DNN on the spoofing challenge training data for each input feature. The input feature to the DNN are delta + double delta Mel-frequency cepstral coefficients (DMCC) 

- Only global mean and variance normalization is applied to the input features.

- The DNN has 5 hidden layers and the 5-th layer is the bottleneck layer.

- Each hidden layer has 1000 neurons and uses sigmoid activation with the exception of 5-th layer which is linear and has 64 nodes.

- The output layer, which is the classification layer, is a softmax of dimension 2 i.e., one output for human speech signal and one     for spoof signal by considering the five spoofing attacks in the challenge training data as one class. 

- We extract the deep features from the 5th layer of the DNN.
- And then finally fit the GMM on those features.

The results im getting are not at all comparable to the results stated in the paper as i have missed a few crucial step that were mentioned in the paper but were out of my understanding,
