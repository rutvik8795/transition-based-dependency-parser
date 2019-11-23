

# Base Configuration:

python train.py data/train.conll \
                data/dev.conll \
                --pretrained-embedding-file data/glove.6B.50d.txt \
                --cache-processed-data \
                --experiment-name basic

# Tanh and Sigmoid Activations
python train.py data/train.conll \
                data/dev.conll \
                --use-cached-data \
                --pretrained-embedding-file data/glove.6B.50d.txt \
                --activation-name tanh \
                --experiment-name tanh

python train.py data/train.conll \
                data/dev.conll \
                --use-cached-data \
                --pretrained-embedding-file data/glove.6B.50d.txt \
                --activation-name sigmoid \
                --experiment-name sigmoid

# Without Pretrained embeddings
python train.py data/train.conll \
                data/dev.conll \
                --use-cached-data \
                --experiment-name wo_glove

# Without tunable embeddings
python train.py data/train.conll \
                data/dev.conll \
                --pretrained-embedding-file data/glove.6B.50d.txt \
                --use-cached-data \
                --trainable-embeddings \
                --experiment-name wo_emb_tune

# NOTE:
#
# 1. You will not need predict.py and evaluate.py as shown in the readme.
#    to obtain Dev results. They will be printed at the end of the training.
#
# 2. After training your dev results will also be stored in
#    serialization directory of the experiment with name `metric.txt`.
