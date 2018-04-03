from GenerateText import GenerateText
import sys
from pyparsing import unicodeString
import os
import Parameters

sentence = 'If it snows, we could be a fortnight getting'
textLength = 50
path = None 

# path = os.path.join(os.path.dirname(__file__),"models\TrainedModel(10,256,25)")

obj = GenerateText()
text = obj.getText(textLength, sentence, path)
print("Generated Output:\n" + text)

