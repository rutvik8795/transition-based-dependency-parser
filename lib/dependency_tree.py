import constants


class DependencyTree:
    """
    Main class to maintain the state of dependency tree
    and operate on it.
    """

    def __init__(self) -> None:
        self.n = 0
        self.head = [constants.NONEXIST]
        self.label = [constants.UNKNOWN]
        self.counter = -1

    def add(self, head: str, label: str) -> None:
        """
        Add the next token to the parse.
        h: Head of the next token
        l: Dependency relation label between this node and its head
        """
        self.n += 1
        self.head.append(head)
        self.label.append(label)

    def set(self, k, h, l):
        """
        Establish a labeled dependency relation between the two given nodes.
        k: Index of the dependent node
        h: Index of the head node
        l: Label of the dependency relation
        """
        self.head[k] = h
        self.label[k] = l

    def get_head(self, k) -> int:
        if k <= 0 or k > self.n:
            return constants.NONEXIST
        return self.head[k]

    def get_label(self, k) -> int:
        if k <= 0 or k > self.n:
            return constants.NULL
        return self.label[k]

    def get_root(self) -> int:
        """
        Get the index of the node which is the root of the parse
        (i.e., the node which has the ROOT node as its head).
        """
        for k in range(1, self.n+1):
            if self.get_head(k) == 0:
                return k
        return 0

    def is_single_root(self) -> bool:
        """
        Check if this parse has only one root.
        """
        roots = 0
        for k in range(1, self.n+1):
            if self.get_head(k) == 0:
                roots += 1
        return roots == 1

    def is_tree(self) -> bool:
        """
        Check if the tree is legal.
        """
        h = []
        h.append(-1)
        for i in range(1, self.n+1):
            if self.get_head(i) < 0 or self.get_head(i) > self.n:
                return False
            h.append(-1)

        for i in range(1, self.n+1):
            k = i
            while k > 0:
                if h[k] >= 0 and h[k] < i:
                    break
                if h[k] == i:
                    return False
                h[k] = i
                k = self.get_head(k)

        return True

    """
    Check if the tree is projective
    """
    def is_projective(self) -> bool:
        if not self.is_tree():
            return False
        self.counter = -1
        return self.visit_tree(0)

    def visit_tree(self, w) -> bool:
        """
        Inner recursive function for checking projective of tree
        """
        for i in range(1, w):
            if self.get_head(i) == w and not self.visit_tree(i):
                return False
        self.counter += 1
        if w != self.counter:
            return False
        for i in range(w+1, self.n+1):
            if self.get_head(i) == w and not self.visit_tree(i):
                return False
        return True

    def equal(self, t) -> bool:
        if t.n != self.n:
            return False
        for i in range(1, self.n+1):
            if self.get_head(i) != t.get_head(i):
                return False
            if self.get_label(i) != t.get_label(i):
                return False
        return True

    def print_tree(self) -> None:
        for i in range(1, self.n+1):
             print(str(i) + " " + str(self.get_head(i)) + " " + self.get_label(i))
        print("\n")
