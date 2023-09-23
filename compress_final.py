"""
A2
"""
from __future__ import annotations

import time
from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    # ....
    freq_dict = {}
    # Iterate through the bytes in the text
    for byte in text:
        # If the byte is already in the dictionary, increment its frequency
        if byte in freq_dict:
            freq_dict[byte] += 1
        # Otherwise, add it to the dictionary with a frequency of 1
        else:
            freq_dict[byte] = 1
    # Return the dictionary
    return freq_dict


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    # https://www.educative.io/answers/how-huffmans-algorithm-works
    # Used the above link to help me understand the algorithm
    if len(freq_dict) == 1:
        symbol = next(iter(freq_dict.items()))[0]
        return HuffmanTree(None, HuffmanTree(symbol))

    symbol_list = [(f, HuffmanTree(s)) for
                   s, f in freq_dict.items()]

    while len(symbol_list) > 1:
        # Sort the list in ascending order by frequency
        symbol_list.sort()

        # Combine the two trees with the lowest frequencies into a new tree
        freq1, tree1 = symbol_list.pop(0)
        freq2, tree2 = symbol_list.pop(0)
        new_freq = freq1 + freq2
        new_tree = HuffmanTree(None, tree1, tree2)
        symbol_list.append((new_freq, new_tree))

    # There should be only one tree left in the list
    huffman_tree = symbol_list[0][1]
    return huffman_tree


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    codes = {}

    def dfs(node: HuffmanTree, code: str) -> None:
        """ Performs dfs search to get the code from each node """
        if node is None:
            return
        # If the node is a leaf, add it to the dictionary
        if node.is_leaf():
            codes[node.symbol] = code
        # Otherwise, recursively call dfs on the left and right subtrees
        else:
            dfs(node.left, code + "0")
            dfs(node.right, code + "1")

    # Call dfs on the tree
    dfs(tree, "")
    return codes


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """

    def _number_nodes_helper(node: HuffmanTree, count: int) -> int:
        """
        Helper function for the number_nodes function that
        recursively numbers the internal nodes of a Huffman
        tree according to postorder traversal.
        """
        if node is None:
            return count

        # Recursively number the left and right subtrees
        count = _number_nodes_helper(node.left, count)
        count = _number_nodes_helper(node.right, count)

        # Number this node if it is internal
        if node.left is not None and node.right is not None:
            node.number = count
            count += 1

        return count

    _number_nodes_helper(tree, 0)


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    # We start by getting the codes for each symbol
    total_weight = sum(freq_dict.values())
    # Get the codes for each symbol
    symbol_codes = get_codes(tree)
    # Calculate the weighted sum of the length of each symbol's code
    weighted_sum = sum(freq_dict[symbol] * len(symbol_codes[symbol])
                       for symbol in freq_dict)
    # Return the weighted sum divided by the total weight
    return weighted_sum / total_weight


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    # Convert the given text to its binary representation
    bits = "".join(codes[_byte] for _byte in text)

    # Pad the bits with zeroes to the next multiple of 8
    padding = (8 - len(bits) % 8) % 8
    bits += "0" * padding

    # Convert the string of bits to bytes
    result = bytearray()

    # Recursively traverse Huffman tree to generate the codes for each symbol.
    def dfs(node: HuffmanTree, code: str) -> None:
        """ Performs dfs search to get the code from each node """
        if node is None:
            return
        if node.is_leaf():
            codes[node.symbol] = code
            return
        dfs(node.left, code + "0")
        dfs(node.right, code + "1")

    # Break the bitstring into bytes
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i + 8]
        byte = int(byte_bits, 2)
        result.append(byte)

    # return the bytes representing the compressed text
    return bytes(result)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    result = []
    if not tree.is_leaf():
        # Traverse the left subtree recursively and append the result
        result += tree_to_bytes(tree.left)
        # Traverse the right subtree recursively and append the result
        result += tree_to_bytes(tree.right)
        # Check if the left subtree is a leaf
        if tree.left.is_leaf():
            result += [0, tree.left.symbol]
        else:
            result += [1, tree.left.number]
        # Check if the right subtree is a leaf
        if tree.right.is_leaf():
            result += [0, tree.right.symbol]
        else:
            result += [1, tree.right.number]
    # Return the bytes
    return bytes(result)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    # Read the input file and build the frequency dictionary
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    # Build the Huffman tree and get the codes for each symbol
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    # Number the nodes in the tree
    number_nodes(tree)
    # Print the average number of bits per symbol
    print("Bits per symbol:", avg_length(tree, freq))
    # Convert the huffman tree to bytes and compress the text
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    # Write the compressed text to the output file
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    trees = {}

    # Iterate over the list of ReadNodes and create a HuffmanTree for each
    for i, node in enumerate(node_lst):
        if node.l_type == 0:
            left = HuffmanTree(node.l_data, None, None)
        else:
            if node.l_data not in trees:
                trees[node.l_data] = HuffmanTree(None, None, None)
            left = trees[node.l_data]

        if node.r_type == 0:
            right = HuffmanTree(node.r_data, None, None)
        else:
            if node.r_data not in trees:
                trees[node.r_data] = HuffmanTree(None, None, None)
            right = trees[node.r_data]

        # Create a new HuffmanTree for this node and add it to the trees dict
        trees[i] = HuffmanTree(None, left, right)

    # Return the root of the Huffman tree
    return trees[root_index]


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    stack = []
    for node in node_lst[:root_index + 1]:
        parent = None
        if node.l_type == 0 and node.r_type == 0:
            # both children are leaves
            left = HuffmanTree(node.l_data, None, None)
            right = HuffmanTree(node.r_data, None, None)
            parent = HuffmanTree(None, left, right)
        elif node.l_type == 0 and node.r_type == 1:
            # left child is a leaf, right child is a subtree
            left = HuffmanTree(node.l_data, None, None)
            right = stack.pop() if stack else None
            parent = HuffmanTree(None, left, right)
        elif node.l_type == 1 and node.r_type == 0:
            # left child is a subtree, right child is a leaf
            left = stack.pop() if stack else None
            right = HuffmanTree(node.r_data, None, None)
            parent = HuffmanTree(None, left, right)
        elif node.l_type == 1 and node.r_type == 1:
            # both children are subtrees
            right = stack.pop() if stack else None
            left = stack.pop() if stack else None
            parent = HuffmanTree(None, left, right)

        stack.append(parent)

    return stack[-1]


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    # Convert the bytes to a binary string representation
    bits = ''.join(f'{byte:08b}' for byte in text)

    # Traverse the Huffman tree using the binary string
    result = bytearray()
    node = tree
    # Move to the left child if the current bit is 0
    for bit in bits:
        if bit == '0':
            node = node.left
        # Move to the right child if the current bit is 1
        else:
            node = node.right
        # If a leaf node is reached, output the symbol and start again
        if node.is_leaf():
            result.append(node.symbol)
            node = tree
            # Stop if the desired number of bytes has been decompressed
            if len(result) == size:
                break
    # Finally, return the decompressed bytes
    return bytes(result)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    # Create a list of symbol-frequency pairs and sort it by frequency
    freq_pairs = [(symbol, freq_dict[symbol]) for symbol in freq_dict]
    freq_pairs.sort(key=lambda pair: pair[1])

    freq_pairs2 = [(symbol, freq_dict[symbol]) for symbol in freq_dict]
    if freq_pairs == freq_pairs2:
        return

    nodes_to_process = [tree]
    # Traverse the tree in level order using a for loop
    for current_node in nodes_to_process:
        # If the current node is a leaf, assign it the symbol with the
        # lowest frequency
        if current_node.is_leaf():
            current_node.symbol = freq_pairs.pop()[0]
        # Add the children of the current node to the list of nodes to process
        nodes_to_process.extend(
            child for child in [current_node.left, current_node.right] if
            child is not None)
        # If there are no more nodes to process, we exit the loop
        if not nodes_to_process:
            break


if __name__ == "__main__":

    import doctest

    doctest.testmod()

    # import python_ta
    # commenting out these lines allows the file to run without python_ta :)
    # python_ta.check_all(config={
    #     'allowed-io': ['compress_file', 'decompress_file'],
    #     'allowed-import-modules': [
    #         'python_ta', 'doctest', 'typing', '__future__',
    #         'time', 'utils', 'huffman', 'random'
    #     ],
    #     'disable': ['W0401']
    # })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
