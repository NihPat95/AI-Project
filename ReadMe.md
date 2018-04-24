# Text Generation Model

This project includes three models for text generation:  
	1. Markov model  
	2. Randomized model\
	3. Recurrent neural network model (LSTM)\

## Markov model
To run this model consider the below file as main file to execute:  
~\Text-Generative-Model\Markov_Chain_Based_Model\MarkovChain.py    

Modify the below variables in above mentioned file:  
filename = <provide the path of input file / text corpus>  
sequenceLength =< positive Integer value to specify the sequence length (before the generated text) to consider>  
op = gen(<provide the seed text or none>)  
