import os
# path to the dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__),"dataset\filename.txt")

# Path to the convert input text corpus to relevant dictionary
SAVED_DICTIONARY_PATH = os.path.join(os.path.dirname(__file__),"models\dictionary")

# Path to model directory
MODELS_DIR = os.path.join(os.path.dirname(__file__),"models")

# Length of look back for the next back
SEQUENCE_LENGTH = 50

# Number of sequence to skip 
SKIP = 1 

# Number of Layers in RNN 
RNN_LAYERS = 500

# Activation function for the neural net
ACTIVATION = "relu"

# the amount of randomness to include
DROPOUT = 0.3

# Learning rate for the network
LEARNING_RATE = 0.001

# Number of Epoch for the model to run
EPOCH = 20

# the batch size of the input data given to the neural network
BATCH_SIZE = 32

# the percentage of the data to be used as a validation metrics
VALIDATION = 0.25