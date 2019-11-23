# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np

# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """
    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        # Comment the next line after implementing call.
        return tf.pow(vector, 3)
        #raise NotImplementedError
        # TODO(Students) End


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # TODO(Students) Start
        # Calcualate weight1 from a random normal distribution
        shapeTuple = (num_tokens*embedding_dim, hidden_dim)
        sqRoot = np.sqrt(num_tokens*embedding_dim*hidden_dim)
        standardNormal = np.random.standard_normal(shapeTuple)/sqRoot
        self._W1 = tf.Variable(standardNormal, dtype="float32")

        # Calculate weight2 from a random normal distribution
        shapeTuple = (hidden_dim, num_transitions)
        sqRoot = np.sqrt(hidden_dim*num_transitions)
        standardNormal = np.random.standard_normal(shapeTuple)/sqRoot
        self._W2 = tf.Variable(standardNormal, dtype="float32")

        # Calculate the embeddings from a uniform distribution with a minimum value of -0.01 and maximum value of 0.01
        MIN_VAL = -0.01
        MAX_VAL = 0.01
        uniformDistribution = tf.random.uniform(shape=[vocab_size, embedding_dim], minval=MIN_VAL, maxval=MAX_VAL)
        self.embeddings = tf.Variable(uniformDistribution, trainable=trainable_embeddings)
        # TODO(Students) End

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        # Find the embeddings of the input
        inputTf = tf.nn.embedding_lookup(self.embeddings, inputs)
        # Get the shape of the input
        shape = inputTf.get_shape()
        # Reshape the input embedding tensor so that it can be multiplied with W1
        inputTf = tf.reshape(inputTf, [shape[0], shape[1] * shape[2]])
        # Apply the activation function from cubic, tanh or sigmoid based on the choice to get the hidden layer
        hidden_layer = self._activation(tf.matmul(inputTf, self._W1))
        # Get the logits from the hidden layer by multiplying it with the weight matrix W2
        logits = tf.matmul(hidden_layer, self._W2)
        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        # apply the mask where the labels are -1 i.e. they are infeasible and calculate the exponential
        # for calculating the softmax
        exp = tf.where(tf.equal(labels, -1), tf.zeros_like(tf.exp(logits)), tf.exp(logits))
        # apply the softmax
        sm = tf.math.divide_no_nan(exp, tf.reduce_sum(exp, axis=1, keepdims=True))
        # Take the log
        logSmLogits = tf.where(tf.equal(sm, 0), tf.zeros_like(sm), tf.math.log(tf.clip_by_value(sm, 1e-10, 1.0)))
        # Calculate the cross entropy loss
        loss = tf.scalar_mul(-1, tf.reduce_mean(tf.reduce_sum(tf.multiply(labels, logSmLogits), axis=1)))
        # Calculate the regularization term
        regularization=self._regularization_lambda*tf.add(tf.nn.l2_loss(self._W1),tf.nn.l2_loss(self._W2))
        # TODO(Students) End
        return loss + regularization
