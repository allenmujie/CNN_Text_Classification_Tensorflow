# -*- coding: utf-8 -*-
import numpy as np
import re
import os
import logging
import itertools
import nltk
from   collections      import Counter
from   nltk.tag         import StanfordNERTagger
from   nltk.tokenize    import word_tokenize
from   nltk.corpus      import stopwords
from   nltk.stem.porter import PorterStemmer
from   sklearn.cross_validation import train_test_split
# nltk.download()

logging.basicConfig(format = '%(levelname)s:%(message)s', level = logging.DEBUG)

# Cache Stop Words (For Performance optimization)
cachedStopWords    = stopwords.words("english")
cleanedDataDirPath = 'data/cleanedData/'
downloadDataDir    = 'data/downloaded/'

# Train/Test Split Config
trainSize    = 0.8
trainTestDirPath   = 'data/train-test-split/'
trainDir           = trainTestDirPath + 'train/'
testDir            = trainTestDirPath + 'test/'

# NLTK Porter Stemmer
porter_stemmer = PorterStemmer()

def clean_str(string):
    """
    Cleaning operations for the String
    """
    # Regex Based Data Cleaning
    string = re.sub(r'\d+', '', string) # Remove numbers
    string = re.sub('[^A-Za-z0-9\.]+', ' ', string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    # Remove words with length < 3 chars
    string = ' '.join([word for word in string.split() if len(word) > 3])

    # Stop Word Removal
    string = ' '.join([word for word in string.split() if word not in stopwords.words("english")])

    # Stemming
    # string = ' '.join([porter_stemmer.stem(word) for word in string.split()])

    return string.strip().lower()

def cleanAndWriteListToFile(fileName, theList):    
    if not os.path.exists(cleanedDataDirPath):
        os.makedirs(cleanedDataDirPath)
    
    with open(cleanedDataDirPath + fileName, 'w') as f:
        for item in theList:
            f.write(clean_str(item) + '\n')

def writeListToFile(outDirPath, fileName, theList):    
    if not os.path.exists(outDirPath):
        os.makedirs(outDirPath)
    
    with open(outDirPath + fileName, 'w') as f:
        for item in theList:
            f.write(item + '\n')

def loadAndCleanData(dataDirPath):
    """
    Loads the datasets to be cleaned
    """
    #load data from files
    for fileName in os.listdir(dataDirPath):
        print('Cleaning File: ', fileName)
        logging.debug(dataDirPath + fileName)
        lines = list(open(dataDirPath + fileName).readlines())
        lines = [s.strip() for s in lines]

        cleanAndWriteListToFile(fileName, lines)
    return 

def generateTrainTestSplit(data):
    print("\n")
    print("Generating Train/Test Split..")
    
    train, test = train_test_split(data, train_size = trainSize)
    return train, test

def loadAndSplitData(dataDirPath):
    for fileName in os.listdir(dataDirPath):
        print('Loading File: ', fileName)
        logging.debug(dataDirPath + fileName)

        # Read
        lines = list(open(dataDirPath + fileName).readlines())
        lines = [s.strip() for s in lines]

        # Generate Train/Test Split
        trainData, testData = generateTrainTestSplit(lines)

        # Write to File
        writeListToFile(trainDir, fileName, trainData)
        writeListToFile(testDir , fileName, testData)
    return 

loadAndCleanData(downloadDataDir)
loadAndSplitData(cleanedDataDirPath)