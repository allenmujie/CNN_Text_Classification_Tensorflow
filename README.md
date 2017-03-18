## Multi-Class Text Classification Using Convolutional Neural Networks (Tensorflow)
- Train, Validate & Test a Convolutional Neural Network (CNN) based model to adapt to a text-classfication task
- Supports adding multiple classes dynamically
- Supports adding pretrained word2vec embeddings from Google News Dataset. Also, learns new word2vec embeddings from the training datasets
- Exposes a REST Webservice & an API to make new predictions (TODO)
- Containerizable docker image for easy deployment of the prediction service on any docker enabled server (TODO)
- Reusable Vagrant Base Box image file for out of the box deployment using Vagrant (TODO)
- Runs parallely on a cluster in Distributed Tensorflow Mode (TODO)

![CNN](https://github.cerner.com/sw029693/Tensorflow_Auto_triage/blob/master/cnn_auto_triage/images/network.png)

## Requirements

- Python 3
- Tensorflow > 0.8
- Numpy

## Training

Print parameters:

```bash
python train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement

```

## Collect, Clean, Split data (Train/test split):

Execute teh datagen.sh script as follows:
```bash
bash datagen.sh
```

## Train:

The training step uses pre-trained word2vec word embeddings. New word embeddings are also learnt from the training data in addition to the pre-trained vectors. The pre-trained Google news word embeddings can be downloaded from [here](https://github.com/mmihaltz/word2vec-GoogleNews-vectors). This repository hosts the word2vec pre-trained Google News corpus (3 billion running words) word vector model (3 million 300-dimension English word vectors)

```bash
python train.py --embedding_dim=300 --word2vec=./word2vec/GoogleNews-vectors-negative300.bin
```

## Evaluating

```bash
python eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
```


## TensorBoard

```bash
tensorboard --logdir ./runs/1459637919/summaries/
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
- [Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
