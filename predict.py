#pylint: disable = redefined-outer-name, invalid-name
# inbuilt lib imports:
from typing import List
import os
import argparse

# external lib imports:
from tqdm import tqdm
import numpy as np
from tensorflow.keras import models

# project imports
from lib.dependency_tree import DependencyTree
from lib.parsing_system import ParsingSystem
from lib.vocabulary import Vocabulary
from lib.data import (
    read_conll_data,
    get_configuration_features,
    write_conll_data,
    Sentence
)
from lib.util import load_pretrained_model


def predict(model: models.Model,
            sentences: List[Sentence],
            parsing_system: ParsingSystem,
            vocabulary: Vocabulary) -> List[DependencyTree]:
    """
    Predicts the dependency tree for a given sentence by greedy decoding.
    We generate a initial configuration (features) for ``sentence`` using
    ``parsing_system`` and ``vocabulary``. Then we apply the ``model`` to predict
    what's the best transition for this configuration and apply this transition
    (greedily) with ``parsing_system`` to get the next configuration. We do
    this till the terminal configuration is reached.
    """
    predicted_trees = []
    num_transitions = parsing_system.num_transitions()
    for sentence in tqdm(sentences):
        configuration = parsing_system.initial_configuration(sentence)
        while not parsing_system.is_terminal(configuration):
            features = get_configuration_features(configuration, vocabulary)
            features = np.array(features).reshape((1, -1))
            logits = model(features)["logits"].numpy()
            opt_score = -float('inf')
            opt_trans = ""
            for j in range(num_transitions):
                if (logits[0, j] > opt_score and
                        parsing_system.can_apply(configuration, parsing_system.transitions[j])):
                    opt_score = logits[0, j]
                    opt_trans = parsing_system.transitions[j]
            configuration = parsing_system.apply(configuration, opt_trans)
        predicted_trees.append(configuration.tree)
    return predicted_trees


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict with trained model')

    parser.add_argument('load_serialization_dir', type=str,
                        help='serialization directory from which to load the trained model.')
    parser.add_argument('data_file_path', type=str, help='data file path to predict on.')
    parser.add_argument('--predictions-file', type=str, help='output predictions file.')

    args = parser.parse_args()

    print("Reading data")
    sentences, _ = read_conll_data(args.data_file_path)

    vocabulary_path = os.path.join(args.load_serialization_dir, "vocab.pickle")
    vocabulary = Vocabulary.load(vocabulary_path)

    sorted_labels = [item[0] for item in
                     sorted(vocabulary.label_token_to_id.items(), key=lambda e: e[1])]
    non_null_sorted_labels = sorted_labels[1:]

    parsing_system = ParsingSystem(non_null_sorted_labels)

    # Load Model
    model = load_pretrained_model(args.load_serialization_dir)

    predicted_trees = predict(model, sentences, parsing_system, vocabulary)

    write_conll_data(args.predictions_file, sentences, predicted_trees)

    print(f"Written predictions to {args.predictions_file}")
