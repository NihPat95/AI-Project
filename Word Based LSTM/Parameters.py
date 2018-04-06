import os
# path to the dataset
# Small piece of dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__),"dataset\Game Of Throne Book 1 Prologue.txt")
# Complete dataset
# DATASET_PATH = os.path.join(os.path.dirname(__file__),"dataset\Alice in wonderland.txt")

# Path to the convert input text corpus to relevant dictionary
SAVED_DICTIONARY_PATH = os.path.join(os.path.dirname(__file__),"models\dictionary")

# Path to model directory
MODELS_DIR = os.path.join(os.path.dirname(__file__),"models")

# Length of look back for the next back
SEQUENCE_LENGTH = 10

# Number of sequence to skip 
SKIP = 1 

# Number of Layers in RNN 
RNN_LAYERS = 500

# Activation function for the neural net
ACTIVATION = "relu"
DROPOUT = 0.3

# Learning rate for the network
LEARNING_RATE = 0.001

# Parameter for running the model
EPOCH = 30
BATCH_SIZE = 32
VALIDATION = 0.25