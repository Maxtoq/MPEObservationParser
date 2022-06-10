import matplotlib.pyplot as plt
from textwrap import wrap
import numpy as np


def sentences_generated(sentences):
    # Creates the vectors for the chart
    unique_sentences = []
    unique_count = []

    # Check all the sentences 
    for sentence in sentences:
        # Join them (to be prettier on the graph)
        s = ' '.join(map(str,sentence))
        # If it is already in the vector of unique sentences
        if s in unique_sentences:
            # Find index
            for i in range(len(unique_sentences)):
                if s == unique_sentences[i]:
                    # Add 1 to the count
                    unique_count[i] = int(unique_count[i]) +1
        else:
            # If it is a new sentence
            # Ass it to the vectors
            unique_sentences.append(s)
            unique_count.append(1)

    # Wrap the sentences (to be prettier on the graph)
    unique_sentences = [ '\n'.join(wrap(s, 15)) for s in unique_sentences ] 
    #Transform the vectors into arrays
    x = np.array(unique_sentences)
    y = np.array(unique_count)
    
    
    plt.figure(figsize=(10, 9), dpi=80)
    #plt.title("Sentences Count")
    plt.barh(x,y,0.6)
    plt.show()

def type_generated(sentences):
    # Creates the vectors for the chart
    unique_types = ["Located","Object","Landmark","You","I","Not"]
    unique_count = [0,0,0,0,0,0]

    # Check all the sentences 
    for sentence in sentences:
        for word in sentence:
            if word in unique_types:
                i = unique_types.index(word)
                unique_count[i] = int(unique_count[i]) + 1

    #Transform the vectors into arrays
    x = np.array(unique_types)
    y = np.array(unique_count)
    
    
    plt.figure(figsize=(10, 9), dpi=80)
    #plt.title("Word Count")
    plt.barh(x,y,0.6)
    plt.show()

def words_generated(sentences):
    # Creates the vectors for the chart
    unique_words = []
    unique_count = []

    # Check all the sentences 
    for sentence in sentences:
        for word in sentence:
            if word in unique_words:
            # Find index
                for i in range(len(unique_words)):
                    if word == unique_words[i]:
                        # Add 1 to the count
                        unique_count[i] = int(unique_count[i]) +1
            else:
                # If it is a new sentence
                # Ass it to the vectors
                unique_words.append(word)
                unique_count.append(1)

    #Transform the vectors into arrays
    x = np.array(unique_words)
    y = np.array(unique_count)
    
    plt.figure(figsize=(10, 9), dpi=80)
    #plt.title("Sentences Count")
    plt.barh(x,y,0.6)
    plt.show()

def analyze(sentences):
    for i in range(2):
        # Count all the sentences generated
        sentences_generated(sentences[i])
        # Count the type of sentence generated
        type_generated(sentences[i])
        # Count words
        words_generated(sentences[i])