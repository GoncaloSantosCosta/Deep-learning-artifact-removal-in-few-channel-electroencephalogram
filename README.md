# Deep learning artifact removal in few-channel electroencephalogram

This is the full code for applying the 1D-SEResNet model to preprocess 2-channel EEG, as well as the code to train the model from new 19-channel EEG data.

Two folders are available, one for each of the uses we mentioned above.

#### Main files
- [DataProcessing/main.py]: execute it to perform preprocess 2-channel EEG data.
- [Model/Training/main_training.py]: execute it to train a new 1D-SEResNet model.
- [1DseResNet.pth]: pre-trained 1d-SEResNet model
