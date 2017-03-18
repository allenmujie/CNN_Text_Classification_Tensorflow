import logging
import os
from sklearn.cross_validation import train_test_split

logging.basicConfig(format = '%(levelname)s:%(message)s', level = logging.DEBUG)

trainSize    = 0.8
dataDirPath  = './cleanedData/'
ouputDirPath = './train-test-split/'
trainDir     = ouputDirPath + 'train/'
testDir      = ouputDirPath + 'test/'

def writeListToFile(outDirPath, fileName, theList):    
    if not os.path.exists(outDirPath):
        os.makedirs(outDirPath)
    
    with open(outDirPath + fileName, 'w') as f:
        for item in theList:
            f.write(item + '\n')

def generateTrainTestSplit(data):
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

loadAndSplitData(dataDirPath)