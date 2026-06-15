## Deep learning artifact removal in few-channel electroencephalogram

This repository contains the full code for applying a 1D-SEResNet model to preprocess 2-channel EEG data, as well as the code required to train the model from new 19-channel EEG data.

Two folders are available, one for each use case described above.

#### Main files

- `DataProcessing/main.py`: execute this script to preprocess 2-channel EEG data.
- `ModelTraining/main_training.py`: execute this script to train a new 1D-SEResNet model.
- `1DseResNet.pth`: pre-trained 1D-SEResNet model.
