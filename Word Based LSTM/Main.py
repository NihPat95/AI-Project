from GenerateText import GenerateText
import sys
from pyparsing import unicodeString
import os
import Parameters
from TrainModel import TrainModel

# sentence = 'be no mistake about it: it was neither more nor less than a pig, and she felt that it would be quit'

sentence = 'Will shared his unease. He had been four years'
textLength = 10
path = None 

# path = os.path.join(os.path.dirname(__file__),"models\TrainedModel(10,500,10)")

path = os.path.join(os.path.dirname(__file__),"models\TrainedModel(1,500,10)")
m = TrainModel().ContinueModelTrain(path, 1)

obj = GenerateText()
text = obj.getText(textLength, sentence, path=None, model=m)

print("Generated Output:\n" + text)

