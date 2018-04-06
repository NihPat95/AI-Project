import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from keras.models import load_model
from TrainModel import TrainModel
import Parameters as p
import pickle

class GenerateText:
    
    loadedNecessary = False
    
    def pick(self, prediciton, threshold=1.0):
        prediciton = np.asarray(prediciton).astype('float64') 
        prediciton = np.log(prediciton) / threshold
        predicitonE = np.exp(prediciton)
        prediciton = predicitonE / np.sum(predicitonE)
        return np.argmax(prediciton)
           
    def getModel(self, path=None):
        if path == None:
            self.model = TrainModel().BuiltModel()
        else:
            self.model = load_model(path)
        
        print("Model Loaded")
        
    def getDictionary(self):
        with open(p.SAVED_DICTIONARY_PATH,'rb') as file:
            self.worldList, self.wordFromIndex, self.indexFromWord = pickle.load(file)
        
        print("Loaded The Dictionary")
        
    def getText(self, textLength, inputSentence, path=None, model=None):
        
        if self.loadedNecessary == False:
            if model != None:
                self.model = model 
            else:
                self.getModel(path)
            self.getDictionary()
            self.loadedNecessary = True
        
        inputSentence = inputSentence.lower()
        sentence = []
        sentence = word_tokenize(inputSentence)
        
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
            x = np.zeros((1, p.SEQUENCE_LENGTH, len(self.indexFromWord)))
            for t, word in enumerate(sentence):
                    x[0,t,self.indexFromWord[word]] = 1
        
            predict = self.model.predict(x)[0]
            nextIndex = np.argmax(predict)
            # nextIndex = self.pick(predict,0.4)
            
            nextWord = self.wordFromIndex[nextIndex]
            
            output = output + " " + nextWord
            sentence = sentence[1:] + [nextWord]
            
        return output
                    