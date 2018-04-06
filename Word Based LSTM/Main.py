from GenerateText import GenerateText
import sys
from pyparsing import unicodeString
import os
import Parameters
from TrainModel import TrainModel

sentence = 'be no mistake about it: it was neither more nor less than a pig, and she felt that it would be quit'

# sentence = 'I saw men freeze last winter, and the one before, when I was half a boy. Everyone talks'
textLength = 10
path = None 

#path = os.path.join(os.path.dirname(__file__),"models\TrainedModel(10,500,10)")


obj = GenerateText()
text = obj.getText(textLength, sentence, path)

print("Generated Output:\n" + text)

