"""Reading and writing trees in Newick format.

Supported: nested parentheses, leaf and internal labels, branch lengths
(including scientific notation), quoted labels with single quotes, and
square-bracket comments (which are skipped). That covers the trees written
by most phylogenetics software.
"""

from __future__ import annotations

from typing import List, Optional

from .tree import Node, Tree, TreeError

_SPECIAL = set("(),:;[]'")


class NewickError(TreeError):
    pass


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def error(self, msg: str) -> NewickError:
        return NewickError(f"{msg} at position {self.pos} in Newick string")

    def skip(self) -> None:
        """Skip whitespace and [comments]."""
        text = self.text
        while self.pos < len(text):
            ch = text[self.pos]
            if ch.isspace():
                self.pos += 1
            elif ch == "[":
                end = text.find("]", self.pos)
                if end == -1:
                    raise self.error("unclosed comment")
                self.pos = end + 1
            else:
                break

    def peek(self) -> str:
        self.skip()
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def label(self) -> Optional[str]:
        self.skip()
        text = self.text
        if self.pos < len(text) and text[self.pos] == "'":
            # quoted label; '' inside quotes is an escaped quote
            self.pos += 1
            chars: List[str] = []
            while True:
                if self.pos >= len(text):
                    raise self.error("unclosed quote")
                ch = text[self.pos]
                if ch == "'":
                    if self.pos + 1 < len(text) and text[self.pos + 1] == "'":
                        chars.append("'")
                        self.pos += 2
                        continue
                    self.pos += 1
                    break
                chars.append(ch)
                self.pos += 1
            return "".join(chars)
        start = self.pos
        while self.pos < len(text) and text[self.pos] not in _SPECIAL and not text[self.pos].isspace():
            self.pos += 1
        raw = text[start:self.pos]
        # unquoted underscores mean spaces in the Newick standard; we keep them
        # as-is because sequence names in FASTA files usually contain them
        return raw or None

    def length(self) -> float:
        if self.peek() != ":":
            return 0.0
        self.pos += 1
        self.skip()
        start = self.pos
        text = self.text
        while self.pos < len(text) and (text[self.pos].isdigit() or text[self.pos] in "+-.eE"):
            self.pos += 1
        token = text[start:self.pos]
        try:
            return float(token)
        except ValueError:
            raise self.error(f"bad branch length {token!r}") from None

    def subtree(self) -> Node:
        if self.peek() == "(":
            self.pos += 1
            node = Node()
            while True:
                node.add_child(self.subtree())
                ch = self.peek()
                if ch == ",":
                    self.pos += 1
                elif ch == ")":
                    self.pos += 1
                    break
                else:
                    raise self.error("expected ',' or ')'")
            node.name = self.label()
        else:
            node = Node(self.label())
        node.length = self.length()
        return node


def parse_newick(text: str) -> Tree:
    """Parse a single Newick tree."""
    if not text or not text.strip():
        raise NewickError("empty Newick string")
    parser = _Parser(text.strip())
    root = parser.subtree()
    if parser.peek() == ";":
        parser.pos += 1
    if parser.peek() != "":
        raise parser.error("unexpected trailing characters")
    tree = Tree(root)
    tree.validate()
    return tree


def _quote(name: str) -> str:
    if any(ch in _SPECIAL or ch.isspace() for ch in name):
        return "'" + name.replace("'", "''") + "'"
    return name


def write_newick(tree: Tree, precision: int = 6, lengths: bool = True) -> str:
    """Write a tree as a Newick string ending in ';'."""
    parts: dict = {}
    for node in tree.postorder():
        label = _quote(node.name) if node.name else ""
        if node.is_leaf:
            text = label
        else:
            text = "(" + ",".join(parts.pop(id(c)) for c in node.children) + ")" + label
        if lengths and not node.is_root:
            text += ":" + format(node.length, f".{precision}g")
        parts[id(node)] = text
    return parts[id(tree.root)] + ";"


def read_newick_file(path: str) -> Tree:
    with open(path, encoding="utf-8") as handle:
        return parse_newick(handle.read())
