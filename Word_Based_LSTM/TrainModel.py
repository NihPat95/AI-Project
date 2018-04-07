import Parameters as p
from nltk import word_tokenize
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
        
        # get a list of all the words from the data set file
        self.wordList = word_tokenize(rawData)
        
        print(self.wordList)
        
        # get the list of unique words 
        self.uniqueWords = Counter(self.wordList)
        
        print(self.uniqueWords)
        
        # create a dictionary with dict[index] -> word
        
        self.wordFromIndex = [x for x in self.uniqueWords]
        
        print(self.wordFromIndex)
        
        # create a dictionary with dict[word] -> index
        self.indexFromWord = {x: i for i,x in enumerate(self.uniqueWords)}
        
        print(self.indexFromWord)
        
        self.TOTAL_WORDS = len(self.wordList)
        self.UNIQUE_WORD_COUNT = len(self.wordFromIndex)
        
        print("Total Words in Corpus ", self.TOTAL_WORDS)
        print("Number of Unique Words ", self.UNIQUE_WORD_COUNT)
        
        # save the dictionary and word list
        with open(p.SAVED_DICTIONARY_PATH,'wb') as file:
            pickle.dump((self.wordList,self.wordFromIndex,self.indexFromWord), file)
    
    def FormatInputForModel(self):
        
        self.InputFormat()
        
        # list of sequences 
        self.sequence = []
        
        # list of next word for the sequence in self.sequence
        self.nextWord = []
        
        for i in range(0, self.TOTAL_WORDS - p.SEQUENCE_LENGTH, p.SKIP):
            self.sequence.append(self.wordList[i: i+p.SEQUENCE_LENGTH])
            self.nextWord.append(self.wordList[i+p.SEQUENCE_LENGTH])
        
        # set the total number of sequence
        self.NUMBER_OF_SEQUENCE = len(self.sequence)
        
        print("Total Number of Sequence ", self.NUMBER_OF_SEQUENCE)
        
#        print(self.sequence)
 #       print(self.nextWord)
        
        # create matrix 
        # X[number of sequence, sequence length, unique word count]
        # y[number of sequence, unique word count]
        
        X = np.zeros((self.NUMBER_OF_SEQUENCE, p.SEQUENCE_LENGTH, self.UNIQUE_WORD_COUNT), dtype=np.bool)
        y = np.zeros((self.NUMBER_OF_SEQUENCE, self.UNIQUE_WORD_COUNT), dtype=np.bool)
        for i, sentence in enumerate(self.sequence):
            for t, word in enumerate(sentence):
                X[i,t,self.indexFromWord[word]] = 1
            y[i, self.indexFromWord[self.nextWord[i]]] = 1

        print (X)
        print (y) 
        
        return X,y
    
    def BidirectionalLSTM(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(p.RNN_LAYERS, activation=p.ACTIVATION), input_shape=(p.SEQUENCE_LENGTH, self.UNIQUE_WORD_COUNT)))
        model.add(Dropout(p.DROPOUT))
        model.add(Dense(self.UNIQUE_WORD_COUNT))
        model.add(Activation('softmax'))
        optimizer = Adam(lr = p.LEARNING_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
        return model
    
    def ContinueModelTrain(self, path, epochDone):
        with open(p.SAVED_DICTIONARY_PATH,'rb') as file:
            self.worldList, self.wordFromIndex, self.indexFromWord = pickle.load(file)
        print("Loaded The Dictionary")
        print("Getting the Input For Model")
        X,y = self.FormatInputForModel()
        model = self.BidirectionalLSTM()
        model.load_weights(path)
        train = model.fit(X, y,\
                          batch_size=p.BATCH_SIZE, \
                          shuffle=True, \
                          initial_epoch = epochDone, \
                          epochs=p.EPOCH, 
                          validation_split=p.VALIDATION)
        print("Saving The Model")
        model.save(p.MODELS_DIR + "\TrainedModel("+str(p.EPOCH)+","+str(p.RNN_LAYERS)+","+str(p.SEQUENCE_LENGTH)+")")
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
