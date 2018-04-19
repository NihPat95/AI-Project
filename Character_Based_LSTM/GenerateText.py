import numpy as np
from nltk.tokenize import sent_tokenize, character_tokenize
from keras.models import load_model
from TrainModel import TrainModel
import Parameters as p
import pickle

class GenerateText:
           
    def getModel(self, path=None):
        # built the model from scratch
        if path == None:
            self.model = TrainModel().BuiltModel()
        else:
        # if the given path then load the model from the path
            self.model = load_model(path)
        
        print("Model Loaded")
        
    def getDictionary(self):
        # load the dictionary from the saved file
        with open(p.SAVED_DICTIONARY_PATH,'rb') as file:
            self.worldList, self.characterFromIndex, self.indexFromCharacter = pickle.load(file)
        
        print("Loaded The Dictionary")
        
    def getText(self, textLength, inputSentence, path=None):
        
        self.getModel(path)
        self.getDictionary()
        
        inputSentence = inputSentence.lower()
        sentence = []
        # tokenize the given input sentence
        sentence = list(inputSentence)
        
        # if the length of the sentence is less than the SEQUENCE_LENGTH 
        # ignore the given sentence
        if(len(sentence) < p.SEQUENCE_LENGTH):
            print("Give A Long Input Sentence Of Size Atleast " , p.SEQUENCE_LENGTH)
            return ""
        
        # only care about the last n sequence length characters 
        sentence = sentence[-p.SEQUENCE_LENGTH:]
        
        # check if the character is in the corpus or not
        for i, character in enumerate(sentence):
            try:
                check = self.indexFromCharacter[character]
            except:
                print("character " + character +"\n Not in the dataset")
                return ""
        
        output = inputSentence
        
        for i in range(textLength):
            
            # format the single sequence to get the next character
            x = np.zeros((1, p.SEQUENCE_LENGTH, len(self.indexFromCharacter)))
            for t, character in enumerate(sentence):
                    x[0,t,self.indexFromCharacter[character]] = 1
        
            # get prediction matrix for the next character
            predict = self.model.predict(x)[0]
            
            # pick the index of the next character with maximum prediction
            nextIndex = np.argmax(predict)
            
            # get the actual character from the index
            nextCharacter = self.characterFromIndex[nextIndex]
            
            output = output + "" + nextCharacter
            sentence = sentence[1:] + [nextCharacter]
            
        return output
                    