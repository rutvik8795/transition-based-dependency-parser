#pylint: disable = redefined-outer-name, invalid-name
# inbuilt lib imports:
from typing import List
import os
import argparse

# project imports
from lib.dependency_tree import DependencyTree
from lib.parsing_system import ParsingSystem
from lib.vocabulary import Vocabulary
from lib.data import read_conll_data, Sentence


def evaluate(sentences: List[Sentence],
             parsing_system: ParsingSystem,
             predicted_trees: List[DependencyTree],
             label_trees: List[DependencyTree]) -> str:
    """
    Predict the dependency trees and evaluate them comparing with gold trees.
    """
    return parsing_system.evaluate(sentences, predicted_trees, label_trees)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate dependency tree predictions.')
    parser.add_argument('load_serialization_dir', type=str,
                        help='serialization directory of the trained model. Used only for vocab.')
    parser.add_argument('gold_data_path', type=str, help='gold data file path.')
    parser.add_argument('prediction_data_path', type=str,
                        help='predictions data file path.')

    args = parser.parse_args()

    print("Reading data")
    sentences, label_trees = read_conll_data(args.gold_data_path)
    _, predicted_trees = read_conll_data(args.prediction_data_path)

    print("Reading vocabulary")
    vocabulary_path = os.path.join(args.load_serialization_dir, "vocab.pickle")
    vocabulary = Vocabulary.load(vocabulary_path)
    sorted_labels = [item[0] for item in
                     sorted(vocabulary.label_token_to_id.items(), key=lambda e: e[1])]
    non_null_sorted_labels = sorted_labels[1:]
    parsing_system = ParsingSystem(non_null_sorted_labels)

    print("Evaluating")
    report = evaluate(sentences, parsing_system, predicted_trees, label_trees)

    print(report)
