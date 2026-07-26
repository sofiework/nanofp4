import ast
import torch


class SpecTree:
    """Static speculative token tree, encoded vLLM-style as a list of paths.

    Each path is a tuple of ranked-child indices from the root, e.g.
      (0,)  -> top-1 child of the root
      (1,)  -> 2nd-ranked child of the root (sibling of (0,))
      (0,1) -> 2nd child of node (0,)
    The root is the implicit empty path (). Nodes are stored in BFS/level order
    with the root at flat index 0. A pure chain of length k is
      [(0,), (0,0), (0,0,0), ...] -> the degenerate tree == plain chain drafting.
    """

    def __init__(self, paths: list[tuple[int, ...]]):
        # close the set under prefixes so every path's ancestors exist as nodes
        node_set = {()}
        for p in paths:
            for d in range(len(p) + 1):
                node_set.add(tuple(p[:d]))
        nodes = sorted(node_set, key=lambda p: (len(p), p))    # BFS/level order, root first
        index = {p: i for i, p in enumerate(nodes)}

        self.num_nodes = len(nodes)
        self.parent = [(-1 if p == () else index[p[:-1]]) for p in nodes]
        self.depth = [len(p) for p in nodes]
        self.max_depth = max(self.depth)
        self.children = [[] for _ in nodes]
        for i, p in enumerate(nodes):
            if p != ():
                self.children[self.parent[i]].append(i)
        # nodes grouped by depth (root = level 0); drives the proposer's level-order expansion
        self.levels = [[i for i in range(self.num_nodes) if self.depth[i] == d]
                       for d in range(self.max_depth + 1)]

        # attention mask [N, N]: node i attends to itself + its proper ancestors (nothing else)
        mask = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.bool)
        for i in range(self.num_nodes):
            j = i
            while j != -1:
                mask[i, j] = True
                j = self.parent[j]
        self.mask = mask    # cpu bool; ModelRunner materializes the additive cuda bias

    @property
    def num_draft(self) -> int:
        return self.num_nodes - 1    # non-root nodes = speculative tokens per step

    def additive_bias(self, device, dtype=torch.float32) -> torch.Tensor:
        # 0 where a query may attend, -inf otherwise
        return torch.where(self.mask.to(device), 0.0, float("-inf")).to(dtype)


def walk_accept(tree: SpecTree, target_argmax: list[int], draft_tokens: list[int]):
    """Greedy tree verify: from the root, follow the branch whose drafted token matches the target's
    argmax at each node. Returns (accepted_path, committed) where accepted_path is the flat node
    indices [0, ...] of the matched branch and committed is the target's argmax along it + 1 bonus.
    Reduces to "longest matching prefix + 1 bonus" on a chain."""
    node, path, committed = 0, [0], []
    while True:
        t = target_argmax[node]                # target's greedy next-token at this node
        committed.append(t)                    # always committed (it's the target's own pick)
        child = next((c for c in tree.children[node] if draft_tokens[c] == t), None)
        if child is None:
            break                              # t is the bonus token; branch ends
        path.append(child)
        node = child
    return path, committed


def chain_paths(k: int) -> list[tuple[int, ...]]:
    return [tuple([0] * d) for d in range(1, k + 1)]


def build_spec_tree(spec: str, num_spec_tokens: int) -> SpecTree:
    # spec == "" -> chain of num_spec_tokens; else a vLLM-style path-list literal
    if spec:
        paths = [tuple(p) for p in ast.literal_eval(spec)]
    else:
        paths = chain_paths(num_spec_tokens)
    return SpecTree(paths)
