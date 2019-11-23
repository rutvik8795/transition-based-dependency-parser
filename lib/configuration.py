from lib.dependency_tree import DependencyTree
import constants


class Configuration:

    def __init__(self, sentence):
        self.stack = []
        self.buffer = []
        self.tree = DependencyTree()
        self.sentence = sentence

    def shift(self):
        k = self.get_buffer(0)
        if k == constants.NONEXIST:
            return False
        self.buffer.pop(0)
        self.stack.append(k)
        return True

    def remove_second_top_stack(self):
        n_stack = self.get_stack_size()
        if n_stack < 2:
            return False
        self.stack.pop(-2)
        return True

    def remove_top_stack(self):
        n_stack = self.get_stack_size()
        if n_stack <= 1:
            return False
        self.stack.pop()
        return True

    def get_stack_size(self):
        return len(self.stack)

    def get_buffer_size(self):
        return len(self.buffer)

    def getSentenceSize(self):
        return len(self.sentence)

    def get_head(self, k):
        return self.tree.get_head(k)

    def get_label(self, k):
        return self.tree.get_label(k)

    def get_stack(self, k):
        """
            Get the token index of the kth word on the stack.
            If stack doesn't have an element at this index, return constants.NONEXIST
        """
        n_stack = self.get_stack_size()
        if k >= 0 and k < n_stack:
            return self.stack[n_stack-1-k]
        return constants.NONEXIST

    def get_buffer(self, k):
        """
        Get the token index of the kth word on the buffer.
        If buffer doesn't have an element at this index, return constants.NONEXIST
        """
        if k >= 0 and k < self.get_buffer_size():
            return self.buffer[k]
        return constants.NONEXIST

    def get_word(self, k):
        """
        Get the word at index k
        """
        if k == 0:
            return constants.ROOT
        else:
            k -= 1

        if k < 0 or k >= len(self.sentence):
            return constants.NULL
        return self.sentence[k].word

    def get_pos(self, k):
        """
        Get the pos at index k
        """
        if k == 0:
            return constants.ROOT
        else:
            k -= 1

        if k < 0 or k >= len(self.sentence):
            return constants.NULL
        return self.sentence[k].pos

    def add_arc(self, h, t, l):
        """
        Add an arc with the label l from the head node h to the dependent node t.
        """
        self.tree.set(t, h, l)

    def get_left_child(self, k, cnt):
        """
            Get cnt-th leftmost child of k.
            (i.e., if cnt = 1, the leftmost child of k will be returned,
                   if cnt = 2, the 2nd leftmost child of k will be returned.)
        """
        if k < 0 or k > self.tree.n:
            return constants.NONEXIST

        c = 0
        for i in range(1, k):
            if self.tree.get_head(i) == k:
                c += 1
                if c == cnt:
                    return i
        return constants.NONEXIST

    def get_right_child(self, k, cnt):
        """
        Get cnt-th rightmost child of k.
        (i.e., if cnt = 1, the rightmost child of k will be returned,
               if cnt = 2, the 2nd rightmost child of k will be returned.)
        """
        if k < 0 or k > self.tree.n:
            return constants.NONEXIST

        c = 0
        for i in range(self.tree.n, k, -1):
            if self.tree.get_head(i) == k:
                c += 1
                if c == cnt:
                    return i
        return constants.NONEXIST

    def has_other_child(self, k, goldTree):
        for i in range(1, self.tree.n+1):
            if goldTree.get_head(i) == k and self.tree.get_head(i) != k:
                return True
        return False


    def get_str(self):
        """
            Returns a string that concatenates all elements on the stack and buffer, and head / label
        """
        s = "[S]"
        for i in range(self.get_stack_size()):
            if i > 0:
                s += ","
            s += self.stack[i]

        s += "[B]"
        for i in range(self.get_buffer_size()):
            if i > 0:
                s += ","
            s += self.buffer[i]

        s += "[H]"
        for i in range(1, self.tree.n+1):
            if i > 1:
                s += ","
            s += self.get_head(i) + "(" + self.get_label(i) + ")"

        return s
