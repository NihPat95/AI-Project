import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from keras.models import load_model
from TrainModel import TrainModel
import Parameters as p
import pickle

class GenerateText:
    
    def getModel(self, path=None):
        # train the model from scratch
        if path == None:
            self.model = TrainModel().BuiltModel()
        else:
        # load the model from the path given
            self.model = load_model(path)
        
        print("Model Loaded")
        
    def getDictionary(self):
        # load the dictionary
        with open(p.SAVED_DICTIONARY_PATH,'rb') as file:
            self.worldList, self.wordFromIndex, self.indexFromWord = pickle.load(file)
        
        print("Loaded The Dictionary")
        
    def getText(self, textLength, inputSentence, path=None, model=None):
        
        if model != None:
            self.model = model 
        else:
            self.getModel(path)
        self.getDictionary()
        self.loadedNecessary = True
        
        
        inputSentence = inputSentence.lower()
        sentence = []
        # tokenize the given input sentence
        sentence = word_tokenize(inputSentence)
        
        # if the length of the sentence is less than the SEQUENCE_LENGTH 
        # ignore the given sentence
        if(len(sentence) < p.SEQUENCE_LENGTH):
            print("Give A Long Input Sentence Of Size Atleast " , p.SEQUENCE_LENGTH)
            return ""
        
        # only care about the last n sequence length words 
        sentence = sentence[-p.SEQUENCE_LENGTH:]
        
        # check if the word is in the corpus or not
        for i, word in enumerate(sentence):
            try:
                check = self.indexFromWord[word]
            except:
                print("Word " + word +"\n Not in the dataset")
                return ""
        
        output = inputSentence
        
        for i in range(textLength):
            # format the single sequence to get the next character
            x = np.zeros((1, p.SEQUENCE_LENGTH, len(self.indexFromWord)))
            for t, word in enumerate(sentence):
                    x[0,t,self.indexFromWord[word]] = 1
        
            # get prediction matrix for the next character
            predict = self.model.predict(x)[0]
            
            # pick the index of the next word with maximum prediction
            nextIndex = np.argmax(predict)
            
            # get the actual word from the index
            nextWord = self.wordFromIndex[nextIndex]
            
            output = output + " " + nextWord
            sentence = sentence[1:] + [nextWord]
            
        return output
                    