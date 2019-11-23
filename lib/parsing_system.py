from typing import List

from lib.dependency_tree import DependencyTree
from lib.configuration import Configuration
import constants


class ParsingSystem:
    """
    Main class to maintain the state of parsing system
    and operate on it.
    """

    def __init__(self, labels: List[str]) -> None:
        self.single_root = True
        self.labels = labels
        self.transitions = []
        self.root_label = labels[0]
        self.make_transitions()

    def make_transitions(self) -> None:
        """
        Generate all possible transitions which this parsing system can
        take for any given configuration.
        """
        for label in self.labels:
            self.transitions.append("L(" + label + ")")
        for label in self.labels:
            self.transitions.append("R(" + label + ")")

        self.transitions.append("S")

    def initial_configuration(self, sentence) -> Configuration:
        configuration = Configuration(sentence)
        length = len(sentence)

        # For each token, add dummy elements to the configuration's tree
        # and add the words onto the buffer
        for i in range(1, length+1):
            configuration.tree.add(constants.NONEXIST, constants.UNKNOWN)
            configuration.buffer.append(i)

        # Put the ROOT node on the stack
        configuration.stack.append(0)

        return configuration

    def is_terminal(self, configuration: Configuration) -> bool:
        return configuration.get_stack_size() == 1 and configuration.get_buffer_size() == 0

    def get_oracle(self,
                   configuration: Configuration,
                   tree: DependencyTree) -> str:
        """
        Provide a static-oracle recommendation for the next parsing step to take
        """
        word1 = configuration.get_stack(1)
        word2 = configuration.get_stack(0)
        if word1 > 0 and tree.get_head(word1) == word2:
            return "L(" + tree.get_label(word1) + ")"
        elif word1 >= 0 and tree.get_head(word2) == word1 and not configuration.has_other_child(word2, tree):
            return "R(" + tree.get_label(word2) + ")"
        return "S"

    def can_apply(self, configuration: Configuration, transition: str) -> bool:
        """
        Determine whether the given transition is legal for this
        configuration.
        """
        if transition.startswith("L") or transition.startswith("R"):
            label = transition[2:-1]
            if transition.startswith("L"):
                h = configuration.get_stack(0)
            else:
                h = configuration.get_stack(1)
            if h < 0:
                return False
            if h == 0 and label != self.root_label:
                return False

        n_stack = configuration.get_stack_size()
        n_buffer = configuration.get_buffer_size()

        if transition.startswith("L"):
            return n_stack > 2
        elif transition.startswith("R"):
            if self.single_root:
                return (n_stack > 2) or (n_stack == 2 and n_buffer == 0)
            else:
                return n_stack >= 2
        return n_buffer > 0

    def apply(self, configuration: Configuration, transition: str) -> Configuration:
        """
        =================================================================

        Implement arc standard algorithm based on
        Incrementality in Deterministic Dependency Parsing(Nirve, 2004):
        Left-reduce
        Right-reduce
        Shift

        =================================================================
        """
        # TODO(Students) Start
        # Get the second top element
        stack1 = configuration.get_stack(1)
        # Get the top element
        stack2 = configuration.get_stack(0)

        # If the transition is right-arc, then add the arc from the second top element of the stack to the top element
        # and remove the top element of the stack.
        if transition.startswith("R"):
            configuration.add_arc(stack1, stack2, transition[2:-1])
            configuration.remove_top_stack()
        # If the transition is left-arc, then add the arc from the top element of the stack to the second top element
        # and remove the second top element of the stack.
        elif transition.startswith("L"):
            configuration.add_arc(stack2, stack1, transition[2:-1])
            configuration.remove_second_top_stack()
        # If the transition is not left-arc or right-arc, perform shift action from the input buffer to the stack.
        else:
            configuration.shift()
        return configuration
        # TODO(Students) End

    def num_transitions(self) -> int:
        return len(self.transitions)

    def print_transitions(self) -> None:
        for transition in self.transitions:
            print(transition)

    def get_punctuation_tags(self) -> List[str]:
        return ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]

    def evaluate(self, sentences, trees, gold_trees) -> str:
        """
        Evaluate performance on a list of sentences, predicted parses, and gold parses
        """
        result = []
        punctuation_tags = self.get_punctuation_tags()

        if len(trees) != len(gold_trees):
            print("Incorrect number of trees.")
            return None

        correct_arcs = 0
        correct_arcs_no_punc = 0
        correct_heads = 0
        correct_heads_no_punc = 0

        correct_trees = 0
        correct_trees_no_punc = 0
        correct_root = 0

        sum_arcs = 0
        sum_arcs_no_punc = 0

        for i in range(len(trees)):
            tree = trees[i]
            gold_tree = gold_trees[i]
            tokens = sentences[i]

            if tree.n != gold_tree.n:
                print("Tree", i+1, ": incorrect number of nodes.")
                return None

            if not tree.is_tree():
                print("Tree", i+1, ": illegal.")
                return None

            n_correct_head = 0
            n_correct_head_no_punc = 0
            n_no_punc = 0

            for j in range(1, tree.n+1):
                if tree.get_head(j) == gold_tree.get_head(j):
                    correct_heads += 1
                    n_correct_head += 1
                    if tree.get_label(j) == gold_tree.get_label(j):
                        correct_arcs += 1
                sum_arcs += 1

                tag = tokens[j-1].pos
                if tag not in punctuation_tags:
                    sum_arcs_no_punc += 1
                    n_no_punc += 1
                    if tree.get_head(j) == gold_tree.get_head(j):
                        correct_heads_no_punc += 1
                        n_correct_head_no_punc += 1
                        if tree.get_label(j) == gold_tree.get_label(j):
                            correct_arcs_no_punc += 1

            if n_correct_head == tree.n:
                correct_trees += 1
            if n_correct_head_no_punc == n_no_punc:
                correct_trees_no_punc += 1
            if tree.get_root() == gold_tree.get_root():
                correct_root += 1

        result = ""
        result += "UAS: " + str(correct_heads * 100.0 / sum_arcs) + "\n"
        result += "UASnoPunc: " + str(correct_heads_no_punc * 100.0 / sum_arcs_no_punc) + "\n"
        result += "LAS: " + str(correct_arcs * 100.0 / sum_arcs) + "\n"
        result += "LASnoPunc: " + str(correct_arcs_no_punc * 100.0 / sum_arcs_no_punc) + "\n\n"

        result += "UEM: " + str(correct_trees * 100.0 / len(trees)) + "\n"
        result += "UEMnoPunc: " + str(correct_trees_no_punc * 100.0 / len(trees)) + "\n"
        result += "ROOT: " + str(correct_root * 100.0 / len(trees)) + "\n"

        return result
