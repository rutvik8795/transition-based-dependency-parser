import pickle

from collections import Counter
import constants

class Vocabulary:

    def __init__(self,
                 sentences,
                 trees) -> None:
        self.word_token_to_id = {}
        self.pos_token_to_id = {}
        self.label_token_to_id = {}
        self.id_to_token = {}

        word = []
        pos = []
        label = []
        for sentence in sentences:
            for token in sentence:
                word.append(token.word)
                pos.append(token.pos)

        root_label = None
        for tree in trees:
            for k in range(1, tree.n + 1):
                if tree.get_head(k) == 0:
                    root_label = tree.get_label(k)
                else:
                    label.append(tree.get_label(k))

        if root_label in label:
            label.remove(root_label)

        index = 0
        word_count = [constants.UNKNOWN, constants.NULL, constants.ROOT]
        word_count.extend(Counter(word))
        for word in word_count:
            self.word_token_to_id[word] = index
            self.id_to_token[index] = word
            index += 1

        pos_count = [constants.UNKNOWN, constants.NULL, constants.ROOT]
        pos_count.extend(Counter(pos))
        for pos in pos_count:
            self.pos_token_to_id[pos] = index
            self.id_to_token[index] = pos
            index += 1

        label_count = [constants.NULL, root_label]
        label_count.extend(Counter(label))
        for label in label_count:
            self.label_token_to_id[label] = index
            self.id_to_token[index] = label
            index += 1

    def get_word_id(self, token: str) -> int:
        if token in self.word_token_to_id:
            return self.word_token_to_id[token]
        return self.word_token_to_id[constants.UNKNOWN]

    def get_pos_id(self, token: str):
        if token in self.pos_token_to_id:
            return self.pos_token_to_id[token]
        return self.pos_token_to_id[constants.UNKNOWN]

    def get_label_id(self, token: str):
        if token in self.label_token_to_id:
            return self.label_token_to_id[token]
        return self.label_token_to_id[constants.UNKNOWN]

    def save(self, pickle_file_path: str) -> None:
        with open(pickle_file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, pickle_file_path: str) -> 'Vocabulary':
        with open(pickle_file_path, "rb") as file:
            vocabulary = pickle.load(file)
        return vocabulary
