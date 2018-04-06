from GenerateText import GenerateText
import sys
from pyparsing import unicodeString
import os
import Parameters
from TrainModel import TrainModel

sentence = 'a nervous tension that came perilous close to fear. Will shared his unease. He had been four years'   
textLength = 30
path = None 

path = os.path.join(os.path.dirname(__file__),"models\TrainedModel(20,500,50)")

# TrainModel().BuiltModel()

obj = GenerateText()
text = obj.getText(textLength, sentence, path)

print("Generated Output:\n" + text)

