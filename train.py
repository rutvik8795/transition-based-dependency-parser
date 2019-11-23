#pylint: disable = redefined-outer-name, invalid-name
# inbuilt lib imports:
from typing import List, Dict, Union
import os
import argparse
import random
import json

# external lib imports:
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, optimizers

# project imports
from lib.dependency_tree import DependencyTree
from lib.parsing_system import ParsingSystem
from lib.vocabulary import Vocabulary
from lib.data import (
    read_conll_data,
    generate_batches,
    load_embeddings,
    generate_training_instances,
)
from lib.model import DependencyParser
from predict import predict
from evaluate import evaluate


def train(model: models.Model,
          optimizer: optimizers.Optimizer,
          train_instances: List[Dict[str, np.ndarray]],
          validation_sentences: List[List[str]],
          validation_trees: List[DependencyTree],
          parsing_system: ParsingSystem,
          vocabulary: Vocabulary,
          num_epochs: int,
          batch_size: int) -> Dict[str, Union[models.Model, str]]:
    """
    Trains a model on the given training instances as
    configured and returns the trained model.
    """
    print("\nGenerating Training batches:")
    train_batches = generate_batches(train_instances, batch_size)
    train_batches = [(batch["inputs"], batch["labels"]) for batch in train_batches]

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")
        # Training Epoch
        total_training_loss = 0
        generator_tqdm = tqdm(train_batches)
        for index, (batch_inputs, batch_labels) in enumerate(generator_tqdm):
            with tf.GradientTape() as tape:
                model_outputs = model(inputs=batch_inputs, labels=batch_labels)
                loss_value = model_outputs["loss"]
                grads = tape.gradient(loss_value, model.trainable_variables)

            clipped_grads = [tf.clip_by_norm(grad, 5) for grad in grads]
            optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
            total_training_loss += loss_value
            description = ("Average training loss: %.2f " % (total_training_loss/(index+1)))
            generator_tqdm.set_description(description, refresh=False)

        # Validation evaluation
        print("Evaluating validation performance:")
        predicted_trees = predict(model, validation_sentences, parsing_system, vocabulary)
        evaluation_report = evaluate(validation_sentences, parsing_system,
                                     predicted_trees, validation_trees)
        print("\n"+evaluation_report)

    training_outputs = {"model": model, "evaluation_report": evaluation_report}
    return training_outputs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Dependency Parsing Model')

    # General training arguments
    parser.add_argument('train_data_file_path', type=str, help='training data file path')
    parser.add_argument('validation_data_file_path', type=str, help='validation data file path')
    parser.add_argument('--batch-size', type=int, default=10000, help='batch size')
    parser.add_argument('--num-epochs', type=int, default=5, help='max num epochs to train for')
    parser.add_argument('--cache-processed-data', action="store_true", default=False,
                        help='if passed, it will cache generated training instances '
                             'at the same path with with extension .jsonl '
                             'You can use --use-cache next time to used the cached datasets. '
                             'Do not use it until you have finalized feature generation code.')
    parser.add_argument('--use-cached-data', action="store_true", default=False,
                        help='if passed, use the cached preproced data instead of connl files.')
    parser.add_argument('--pretrained-embedding-file', type=str,
                        help='if passed, use glove embeddings to initialize. '
                        'the embedding matrix')
    parser.add_argument('--experiment-name', type=str, default="default",
                        help='optional experiment name which determines where to store '
                             'the model training outputs.')

    # Model specific arguments
    parser.add_argument('--num-tokens', type=int, help='num_tokens ', default=48)
    parser.add_argument('--hidden-dim', type=int, help='hidden_dim of neural network', default=200)
    parser.add_argument('--embedding-dim', type=int, help='embedding_dim of word embeddings', default=50)
    parser.add_argument('--activation-name', type=str, choices=("cubic", "tanh", "sigmoid"),
                        help='activation-name', default="cubic")
    parser.add_argument('--trainable-embeddings', action="store_false",
                        help='are embeddings trainable', default=True)
    parser.add_argument('--regularization-lambda', type=float,
                        help='regularization_lambda ', default=1e-8)


    args = parser.parse_args()

    # Set numpy, tensorflow and python seeds for reproducibility.
    tf.random.set_seed(1337)
    np.random.seed(1337)
    random.seed(13370)

    # Print if GPU is available or not.
    device_name = tf.test.gpu_device_name()
    print(f"GPU found: {device_name == '/device:GPU:0'}")

    # Setup Serialization dir
    save_serialization_dir = os.path.join("serialization_dirs", args.experiment_name)
    if not os.path.exists(save_serialization_dir):
        os.makedirs(save_serialization_dir)

    # Setup Training / Validation data
    print("Reading training data")
    train_sentences, train_trees = read_conll_data(args.train_data_file_path)

    print("Reading validation data")
    validation_sentences, validation_trees = read_conll_data(args.validation_data_file_path)

    vocabulary = Vocabulary(train_sentences, train_trees)

    sorted_labels = [item[0] for item in
                     sorted(vocabulary.label_token_to_id.items(), key=lambda e: e[1])]
    non_null_sorted_labels = sorted_labels[1:]

    parsing_system = ParsingSystem(non_null_sorted_labels)

    # Generating training instances takes ~20 minutes everytime. So once you finalize the
    # feature generation and want to try different configs for experiments, you can use caching.
    if args.use_cached_data:
        print("Loading cached training instances")
        cache_processed_data_path = args.train_data_file_path.replace("conll", "jsonl")
        if not os.path.exists(cache_processed_data_path):
            raise Exception(f"You asked to use cached data but {cache_processed_data_path} "
                            f"is not available.")
        with open(cache_processed_data_path, "r") as file:
            train_instances = [json.loads(line)
                               for line in tqdm(file.readlines()) if line.strip()]
    else:
        print("Generating training instances")
        train_instances = generate_training_instances(parsing_system,
                                                      train_sentences,
                                                      vocabulary, train_trees)

    # If cached training data is asked for.
    if args.cache_processed_data:
        print("Caching training instances for later use")
        cache_processed_data_path = args.train_data_file_path.replace("conll", "jsonl")
        with open(cache_processed_data_path, "w") as file:
            for instance in tqdm(train_instances):
                file.write(json.dumps(instance) + "\n")

    # Setup Model
    config_dict = {"vocab_size": len(vocabulary.id_to_token),
                   "embedding_dim": args.embedding_dim,
                   "num_tokens": args.num_tokens,
                   "hidden_dim": args.hidden_dim,
                   "num_transitions": parsing_system.num_transitions(),
                   "regularization_lambda": args.regularization_lambda,
                   "trainable_embeddings": args.trainable_embeddings,
                   "activation_name": args.activation_name}
    model = DependencyParser(**config_dict)

    if args.pretrained_embedding_file:
        embedding_matrix = load_embeddings(args.pretrained_embedding_file,
                                           vocabulary, args.embedding_dim)
        model.embeddings.assign(embedding_matrix)

    # Setup Optimizer
    optimizer = optimizers.Adam()

    # Train
    training_outputs = train(model, optimizer, train_instances,
                             validation_sentences, validation_trees,
                             parsing_system, vocabulary, args.num_epochs,
                             args.batch_size)

    # Save the trained model
    trained_model = training_outputs["model"]
    trained_model.save_weights(os.path.join(save_serialization_dir, f'model.ckpt'))

    # Save the last epoch dev metrics
    evaluation_report = training_outputs["evaluation_report"]
    metrics_path = os.path.join(save_serialization_dir, "metrics.txt")
    with open(metrics_path, "w") as file:
        file.write(evaluation_report)

    # Save the used vocab
    vocab_path = os.path.join(save_serialization_dir, "vocab.pickle")
    vocabulary.save(vocab_path)

    # Save the used config
    config_path = os.path.join(save_serialization_dir, "config.json")
    with open(config_path, "w") as file:
        json.dump(config_dict, file)

    print(f"\nFinal model stored in serialization directory: {save_serialization_dir}")
