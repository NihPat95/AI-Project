from GenerateText import GenerateText
import sys
from pyparsing import unicodeString
import os
import Parameters
from TrainModel import TrainModel

sentence = 'There were doors all round the hall, but they were all locked; and when Alice had been all the'
textLength = 10
path = None 

path = os.path.join(os.path.dirname(__file__),"models/TrainedModel(5,500,10)")
m = TrainModel().ContinueModelTrain(path, 1)

obj = GenerateText()
text = obj.getText(textLength, sentence, path=None, model=m)

print("Generated Output:\n" + text)

