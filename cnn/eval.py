#! /usr/bin/env python3

import tensorflow as tf
import numpy      as np
import os
import time
import datetime
import data_helpers
from   text_cnn           import TextCNN
from   tensorflow.contrib import learn
from   sklearn.metrics    import confusion_matrix
from   sklearn.metrics    import precision_score
from   sklearn.metrics    import recall_score
from   sklearn.metrics    import f1_score
from   sklearn.metrics    import roc_auc_score
from   sklearn.metrics    import classification_report
from   sklearn.metrics    import cohen_kappa_score

# Parameters
# ==================================================
# python eval.py --eval_train --checkpoint_dir="./runs/1476421626/checkpoints/"

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Data
testDataDirPath = "../data/train-test-split/test/"

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(testDataDirPath)
    y_test        = np.argmax(y_test, axis=1)
else:
    x_raw  = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path      = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test          = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():

    session_conf = tf.ConfigProto(
        allow_soft_placement = FLAGS.allow_soft_placement,
        log_device_placement = FLAGS.log_device_placement)

    sess = tf.Session(config = session_conf)

    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver             = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x           = graph.get_operation_by_name("input_x").outputs[0]

        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions       = graph.get_operation_by_name("output/predictions").outputs[0]
        softmaxScores     = graph.get_operation_by_name("output/softmaxScores").outputs[0]

        # Generate batches for one epoch
        batches           = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions   = []
        all_softMaxScores = []

        for x_test_batch in batches:
            batch_predictions    = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions      = np.concatenate([all_predictions, batch_predictions])

            batch_softMax_scores = sess.run(softmaxScores, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_softMaxScores    = np.append(all_softMaxScores, batch_softMax_scores)
            # all_softMaxScores    = np.append(all_softMaxScores, [[smxScore] for smxScore in batch_softMax_scores])

def analyzeConfusnMtrx(actuals, predicted, predProbs):
    # print("Actuals: ", actuals)
    # print("Predicted: ", predicted)

    num_samples  = actuals.shape[0]
    uniq_classes = np.unique(actuals)
    num_classes  = len(np.unique(actuals))
    predProbs    = predProbs.reshape((num_samples, num_classes))
    predProbs    = [max(probs) for probs in predProbs]
    predProbs    = np.asarray(predProbs)
    numRows      = len(actuals)

    # Find max values for Prediction Probabilities

    print("Shape of Actuals: " , num_samples)
    print("Number of Classes: ", num_classes)
    print("Unique Classes:"    , uniq_classes)
    print("Number of Rows: ", numRows)
    print("\n")

    for eachClass in range(num_classes):
        
        # Find indices of 'eachClass' in 'Actuals'
        print("-------------------")
        print("Class: ", eachClass)
        actIndices          = np.where(actuals == eachClass)
        actIndices          = np.asarray(actIndices)[0]
        lkdUpActuals        = actuals[actIndices]
        predsFromActIndices = predicted[actIndices]
        probsFromActIndices = predProbs[actIndices]
        
        print("\n")
        
        # Get Prediction stats
        for eachClassLkp in range(num_classes):
            print("..............")
            print("Looking up cell: ", (eachClass, eachClassLkp))
            probDist = []
            
            # Find indices of 'predsFromActIndices' where class = eachClassLkp
            eachClsLkpIndices = np.where(predsFromActIndices == eachClassLkp)
            
            # Get probabilities
            filteredProbs = probsFromActIndices[eachClsLkpIndices]
            print("\n")
            print(filteredProbs)
            print("*****")
            for val in filteredProbs:
                print(val)
            
            if (len(filteredProbs)):
                print("Min:", np.min(filteredProbs))
                print("Max:", np.max(filteredProbs))
                print("Std:", np.std(filteredProbs))
                print("Mean:", np.mean(filteredProbs))
                print("Median:", np.median(filteredProbs))
                print("10th Percentile:", np.percentile(filteredProbs, 10))
                print("25th Percentile:", np.percentile(filteredProbs, 25))
                print("75th Percentile:", np.percentile(filteredProbs, 75))
                print("95th Percentile:", np.percentile(filteredProbs, 95))
                print("\n")

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Correct Predictions:")
    print(correct_predictions)
    print("Total number of test examples: {}".format(len(y_test)))
    print("Precision", precision_score(y_test, all_predictions, average='micro'))
    print("Recall", recall_score(y_test, all_predictions, average='micro'))
    print("f1_score", f1_score(y_test, all_predictions, average='micro'))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    print("confusion_matrix")
    confusionMatrix = confusion_matrix(y_test, all_predictions)
    print(confusionMatrix)
    print("Normalized Confusion Matrix:")
    print(confusionMatrix / confusionMatrix.astype(np.float).sum(axis=1))
    print("Classification Report:")
    print(classification_report(y_test, all_predictions))
    print("Cohen's Kappa Score:")
    print(cohen_kappa_score(y_test, all_predictions))

    print("Predictions:")
    print(all_predictions)
    print(len(all_predictions))

    # print("All Softmax Scores:")
    # np.set_printoptions(threshold = np.nan)
    # print(all_softMaxScores)

    print("type y_test:", type(y_test))
    print("Analyzing results from the Confusion Matrix:")
    analyzeConfusnMtrx(y_test, all_predictions, all_softMaxScores)
