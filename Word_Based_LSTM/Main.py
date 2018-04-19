from GenerateText import GenerateText
import sys
from pyparsing import unicodeString
import os
import Parameters
from TrainModel import TrainModel

# the seed sentence provided to the model to complete
sentence = ''

# length of the generated output sentence
textLength = 10

# path to a saved model
# make sure the save model and saved dictionary are generated of the
# parameters and the text dataset
path = None 

# an example of how the save path might look like
# path = os.path.join(os.path.dirname(__file__),"models/TrainedModel(5,500,10)")

obj = GenerateText()

# model is the path of the save checkpoint model from where it should
# continue running the epochs for
# to continue training a checkpoint make sure set the initial epoch
# to for how many epoch that checkpoint ran  
text = obj.getText(textLength, sentence, path=None, model=None)

print("Generated Output:\n" + text)

