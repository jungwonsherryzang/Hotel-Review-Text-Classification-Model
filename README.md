# Hotel-Review-Text-Classification-Model
Our team's final model code is available here: https://colab.research.google.com/drive/1T_Wf9TuFoCEqt7ef9bvXEg0pKDJ4DYx_?usp=sharing

## INTRODUCTION
As the world’s largest online travel platform, TripAdvisor contains hundreds of millions of travel and hotel reviews that could serve as a thorough source to benefit hotel businesses and travel sectors. The aim of this project is to analyze customer reviews extracted from the world’s most popular online hospitality platform, TripAdvisor. Given the heterogenous and massive amount of data that exist on the platform, the purpose of the project was to utilize DN based Recurrent Neural Networks (RNN) architecture to analyze and classify customer reviews.

This project aims to evaluate the data wrangling, exploring and preprocessing and performance of a multi-class text classification project.
Each of team members selected RNN model with different class ratings. Having the best accuracy final model of this project was bidirectional LSTM RNN model with three class ratings. 

## DATA PREPROCESSING
Text preprocessing was required to format the raw text in such a way that the RNN model could consume it as input. All punctuation was removed from the text because it added no linguistic value or context for the model to learn from. Furthermore, a collection of frequently used words, known as stopwords in the nltk (Natural Language Toolkit), were removed. Such words include, “the”, “a”, “an” and “in”, which once more are removed from the text because they add no sentiment value to the input. Removing these words helps to reduce the computational load by reducing the size of the corpus to be analyzed by the model. The text was also processed using stemming, which reduces each word down to its root with the purpose of improving computational performance. Following this, the text was tokenized to format each review in a list format. The tokenized reviews were then converted to sequences of integers. Each index integer corresponds with a word in the corpus.

<img width="369" alt="class rating" src="https://user-images.githubusercontent.com/91277856/162642145-c4119002-6499-4a36-b040-0bdabedb4534.png">

For an RNN model the input data was required to be in a sequence format of the same length. The tokenized data were firstly padded to size 200. This meant that each review was either extended or shrunk to a consistent length of 200 words.

Summary of data preprocessing:
1. Clean text to remove stop words/punctuation, stemming, etc. techniques to simplify the model data input.
2. Split data into training, validation, and testing data sets.
3. Tokenize to clean text to create a list of words for each review.
4. Convert tokenized words to a sequence of integers with an index for each word in the
corpus.
5. Pad the sequences to ensure all input sequences are of the same length.

## MODEL SELECTION
Recurrent Neural Network (RNN) is a class of artificial neural network where connections between nodes form a directed graph along a sequence. This architecture allows RNN to exhibit temporal behavior and capture sequential data which makes it a more ‘natural’ approach when dealing with textual data since text is naturally sequential.

<img width="391" alt="rnn" src="https://user-images.githubusercontent.com/91277856/162642159-62c08aee-7cf3-4892-ba5a-7a968812a4c6.png">


RNN is an ideal model for NLP text classification/generation and Sequence labelling. Natural Language (NL) is the language human uses for communication, which is a contrast to constructed language that is made artificially by human such as programming language. Natural Language Processing (NLP) is a main process of Artificial Intelligence (AI) that machines like computer can analyze and process human’s language phenomenon.

RNN is well-suited to multi-class text classification because it exhibits temporal behavior and captures sequential data which makes it well-suited for multi-class text classification as the text is naturally sequential. And RNN uses layers that give that model a short-term memory which helps in predicting the next data accurately and it captures the statistical structure of the text and makes easy to classify the data between different classes that makes RNN well-suited for multi-class text classification.

## OUTCOME

<img width="349" alt="lstm" src="https://user-images.githubusercontent.com/91277856/162642165-21620545-cc6d-4e85-a3c3-34cf60ece548.png">

Embedding Layer
The embedding layer was initialized with random weights and then learns an embedding code for each word in the training corpus. The input dimension defines the total vocabulary size + 1 of the training data, i.e., the total number of unique words that an embedding will be created for. The output dimension defines the vector space in which the words will be embedded, for each word. The input length for this layer corresponds with the maximum sequence length defined earlier to be 200 as a default value.


Bidirectional LSTM layers
LSTM layers comprise of numerous LSTM cells that process input sequentially. The first cell processes the input, determines a hidden state and output then passes it on to the next cell as input. The return_sequences configuration was set to True for the first LSTM layer. This is because if the layer was configured to False, then only the last hidden layer state would be passed on to the next LSTM layer. Inherently, with the setting configured to True, more information is passed to the next layer because the next layers have access to all the hidden states. Since the LSTM output is in tensor form, it was necessary to introduce a flatten layer in the next layer to prepare the output for the dense layer.

As per best practices, each LSTM layer was accompanied with dropout. Dropout layers reduce the risk of overfitting the training data by randomly bypassing certain neurons. This helped to reduce the sensitivity to unique weights belonging to specific neurons. A widely accepted range for dropout was 0.2 through 0.5. Although 0.2 was used as the default dropout rate, it was still necessary to tune and find the optimal value.


Flatten Layer
Flatten layers simply flatten the input, transforming it from a multidirectional tensor to a single dimension. The flatten layer essentially prepared the output of the LSTM such that it could be processed by the following dense layers.


Dense Layer (ReLu)
The ReLu activation layer applied the rectified linear unit function max(x, 0) to the input. Essentially, if any negative values are passed to the function, then they are transformed to zeros, otherwise all positive values remain the same. The threshold was held at the default value of zero. The dense layer compressed the input to a default value of 65 units. However, this hyperparameter was tuned to find the optimal value.


Output Dense Layer
For this model, there were three possible classes to categorize the reviews. Therefore, the softmax activation function was required. The softmax function outputs a vector of three real values that sum to one. Each of the three values represents the probability that the review belongs to a particular class type. The highest probability is thus selected as the model prediction. The softmax can was appropriate because the outputs were mutually exclusive.


Model Performance
Without tuning, the model performed reasonably well achieving a test accuracy and loss of 0.7729 and 0.8078, respectively.

<img width="346" alt="accuracy" src="https://user-images.githubusercontent.com/91277856/162642225-f28364c3-aa59-421c-a1f6-131705c3c30c.png">
<img width="346" alt="loss" src="https://user-images.githubusercontent.com/91277856/162642222-6460d55f-7f0a-44da-b8a0-d3f51e5c464a.png">
<img width="346" alt="outcome" src="https://user-images.githubusercontent.com/91277856/162642228-77509f14-bbca-4a87-ab4a-e6f5ff5be8c4.png">
