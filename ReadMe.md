# Text Generation Model

This project includes three models for text generation:  
	1. Markov model  
	2. Randomized model  
	3. Word based Recurrent neural network model (LSTM)  
	4. Character based Recurrent neural network model (LSTM)

## Markov model
To run this model execute the below file :  
~\Text-Generative-Model\Markov_Chain_Based_Model\MarkovChain.py    

Modify the below variables in above mentioned file:  
* filename = <provide the path of input file / text corpus>  
* sequenceLength =<positive Integer value to specify the sequence length (before the generated text) to consider>  
```python
* op = gen( <provide the seed text or none> )  

## Randomized model


## Word based Recurrent neural network model (LSTM)
To run this model execute the below file:
~\Text-Generative-Model\Word_Based_LSTM\Main.py

Modify the below variables in ~\Text-Generative-Model\Word_Based_LSTM\Parameters.py file
* DATASET_PATH = <provide the path of input file / text corpus>

## Character based Recurrent neural network model (LSTM)  
To run this model execute the below file:  
~\Text-Generative-Model\Character_Based_LSTM\Main.py  

Modify the below variables in ~\Text-Generative-Model\Word_Based_LSTM\Parameters.py file  
* DATASET_PATH = <provide the path of input file / text corpus>  

There are other parameters related to Recurrent Neural Network which can be configured in Parameters.py




