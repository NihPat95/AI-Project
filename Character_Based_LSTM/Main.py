from GenerateText import GenerateText
import sys
from pyparsing import unicodeString
import os
import Parameters
from TrainModel import TrainModel

# add the seed sentence of length atleast equal 
# to the sequence length set in the parameters
sentence = ''   

# the length of the text you would like to generate
textLength = 30

# set the path to the last save model
# make sure that the saved model and the save dictionary 
# were generated on the same corpus with the same parameters
path = None 

# an example of path would look like this
# path = os.path.join(os.path.dirname(__file__),"models\TrainedModel(20,500,50)")

obj = GenerateText()
text = obj.getText(textLength, sentence, path)

print("Generated Output:\n" + text)

