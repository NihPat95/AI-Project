import Parameters as p
from nltk import character_tokenize
from collections import Counter
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy


class TrainModel:
    
    def InputFormat(self):
        with open(p.DATASET_PATH, encoding="utf8") as file:
            # read the complete file and convert to lowercase
            rawData = file.read().lower()
        
        # get a list of all the characters from the data set file
        self.characterList = list(rawData) 
        
        # get the list of unique characters 
        self.uniqueCharacters = Counter(self.characterList)
        
        # create a dictionary with dict[index] -> character
        self.characterFromIndex = [x for x in self.uniqueCharacters]
        
        # create a dictionary with dict[character] -> index
        self.indexFromCharacter = {x: i for i,x in enumerate(self.uniqueCharacters)}
        
        self.TOTAL_CHARACTERS = len(self.characterList)
        self.UNIQUE_CHARACTER_COUNT = len(self.characterFromIndex)
        
        print("Total characters in Corpus ", self.TOTAL_CHARACTERS)
        print("Number of Unique characters ", self.UNIQUE_CHARACTER_COUNT)
        
        # save the dictionary and character list
        with open(p.SAVED_DICTIONARY_PATH,'wb') as file:
            pickle.dump((self.characterList,self.characterFromIndex,self.indexFromCharacter), file)
    
    def FormatInputForModel(self):
        
        self.InputFormat()
        
        # list of sequences 
        self.sequence = []
        # list of next character for the sequence in self.sequence
        self.nextcharacter = []
        
        # for each sequence of p.SEQUENCE_LENGTH characters in the dataset
        # set the next character to be the character that follow p.SEQUENCE_LENGHT characters
        for i in range(0, self.TOTAL_CHARACTERS - p.SEQUENCE_LENGTH, p.SKIP):
            self.sequence.append(self.characterList[i: i+p.SEQUENCE_LENGTH])
            self.nextcharacter.append(self.characterList[i+p.SEQUENCE_LENGTH])
        
        # set the total number of sequence
        self.NUMBER_OF_SEQUENCE = len(self.sequence)
        
        print("Total Number of Sequence ", self.NUMBER_OF_SEQUENCE)
        
        # create matrix 
        # Input Sequence Data
        # X[number of sequence, sequence length, unique character count]
        # Corresponding Label Data that is essentially the next character
        # y[number of sequence, unique character count]
        X = np.zeros((self.NUMBER_OF_SEQUENCE, p.SEQUENCE_LENGTH, self.UNIQUE_CHARACTER_COUNT), dtype=np.bool)
        y = np.zeros((self.NUMBER_OF_SEQUENCE, self.UNIQUE_CHARACTER_COUNT), dtype=np.bool)
        for i, sentence in enumerate(self.sequence):
            for t, character in enumerate(sentence):
                X[i,t,self.indexFromCharacter[character]] = 1
            y[i, self.indexFromCharacter[self.nextCharacter[i]]] = 1
        return X,y
    
    def BidirectionalLSTM(self):
        # Build the neural network with the following configurations
        # A bidirectional RNN network with LSTM cells
        # with a dropout and an activation layer of softmax and rectified units
        model = Sequential()
        model.add(Bidirectional(LSTM(p.RNN_LAYERS, activation=p.ACTIVATION), input_shape=(p.SEQUENCE_LENGTH, self.UNIQUE_CHARACTER_COUNT)))
        model.add(Dropout(p.DROPOUT))
        model.add(Dense(self.UNIQUE_CHARACTER_COUNT))
        model.add(Activation('softmax'))
        optimizer = Adam(lr = p.LEARNING_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
        return model
    
    def BuiltModel(self):
        print("Getting the Input For Model")
        X,y = self.FormatInputForModel()
        print("Starting To Built Model")
        model = self.BidirectionalLSTM()
        print("Training The Model") 
        train = model.fit(X, y,\
                          batch_size=p.BATCH_SIZE, \
                          shuffle=True, \
                          epochs=p.EPOCH, 
                          validation_split=p.VALIDATION)
        print("Saving The Model")
        model.save(p.MODELS_DIR + "\TrainedModel("+str(p.EPOCH)+","+str(p.RNN_LAYERS)+","+str(p.SEQUENCE_LENGTH)+")")
        return model