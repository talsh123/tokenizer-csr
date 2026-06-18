##################### LOGIC FUNCTION SECTION #####################

# load_vocab creates a list of all the tokens as strings
# gets a filepath of a vocabulary .txt file
# returns a list of tokens strings
def load_vocab(filepath):
    # open the file in read mode
    f = open(filepath, 'r', encoding='utf-8')

    # create an empty list to hold our tokens
    vocab_list = []

    # loop through each line in the file, line by line
    for line in f:
        # clean up the \n character at the end of each line
        clean_line = line.strip('\n')
        # puts the cleaned string in our list
        vocab_list.append(clean_line)

    f.close()

    return vocab_list

# build_alphabet creates a sorted dictionary of all the possible characters in the vocab and their indexes.
# gets the list of token strings
# returns a dictionary of all the possible characters and their indexes
def build_alphabet(vocab_list):
    # creates a set (in python, set does not include duplicates)
    unique_chars = set()

    # loop through every word in the vocab
    for word in vocab_list:
        # loop through every character in the word
        for char in word:
            # add each unique letter to the set
            unique_chars.add(char)
    
    # sort the characters
    # if do not sort - every time we run the scripts different .mem files will generate
    # this causes inconsistencies in the index assignments, and makes it hard for the code to be checked at each run
    sorted_chars = sorted(list(unique_chars))

    # create a dictionary to map each character to a number
    char_to_index = {}

    # index starts at 0
    index = 0
    for char in sorted_chars:
        char_to_index[char] = index
        index += 1

    return char_to_index

# split_vocab splits the original vocabulary to 2: root and continuation vocabularies
# gets the list of all the token strings
# returns the vocabularies for root and continuation in (token, token_id) form
def split_vocab(vocab_list):
    # splits the vocabulary into 2 lists:
    # root_vocab: subword pieces that appear in the start of the word, and their token id
    # cont_vocab: subword pieces that appear in the middle or end of a word, and their token id

    # each list is tuple (token, original_token_id)
    root_vocab = []
    cont_vocab = []

    # iterates over the tuple (token_id, token_string)
    for token_id, word in enumerate(vocab_list):
        if word.startswith('##'): # if the token starts with "##" prefix
            # strip the prefix
            stripped = word[2:] # stripped word
            cont_vocab.append((stripped, token_id))
        else: # a normal word with no "##"
            root_vocab.append((word, token_id))
    
    # some prints
    print(f"vocabulary split:")
    print(f"root tokens: {len(root_vocab)}")
    print(f"continuation tokens: {len(cont_vocab)}")
    print(f"total: {len(root_vocab) + len(cont_vocab)}")
    
    return root_vocab, cont_vocab

# build_trie build the dense, massive trie sparse 2D array that needs CSR compression
# gets a list of (token, token_id) tuples and the character to index mapping
# returns the transition dense 2D array, the is_terminal array and the token_ids array
def build_trie(vocab_with_ids, char_to_index):
    # get the number of unique characters in the vocabulary
    alphabet_size = len(char_to_index)

    # this is the root node of the trie
    # we start be setting -1 * alphabet_size => [-1, -1, ....]
    # -1 means no path yet
    root_node_transitions = [-1] * alphabet_size

    # transitions - list of lists. Each inner list contains alphabet_size entries. The 2D representation.
    # is_terminal - list of booleans. if is_terminal[node_id] is True - a valid token ends at this node.
    # token_ids - list of integers. token_ids[node_id] is the BERT token ID for terminal nodes, -1 for non-terminal nodes.
    transitions = [root_node_transitions]
    is_terminal = [False]
    token_ids = [-1]

    # this loop builds the trie from root
    # loop through every (token, token_id) pair
    for token, token_id in vocab_with_ids:

        # start at the root node for a new word
        current_node = 0

        # iterate over each character in the token string
        for char in token:
            # find the index for this character
            char_index = char_to_index[char]

            # looks at the current node and find the next node (next character)
            next_node = transitions[current_node][char_index]

            if next_node == -1: # -1 no path yet
                new_node_id = len(transitions) # sets the new node ID to the length of the transitions array. So it always counts sequentially.

                # update current node to point to our new node
                transitions[current_node][char_index] = new_node_id

                # expand our 3 lists to make room for the new node
                new_node_transitions = [-1] * alphabet_size # this new node start with no children, so we [-1, -1, ...] at this size of alphabet_size
                transitions.append(new_node_transitions) # increase transitions by 1
                is_terminal.append(False) # increase is_terminal by 1
                token_ids.append(-1) # increase token_ids by 1

                # Move forward to the new node we just created
                current_node = new_node_id
            else:
                # the path already exists, just move to that node
                current_node = next_node

        # we finished iterating the characters of this word
        # we set this final node as the end, meaning this node is terminal
        is_terminal[current_node] = True
        token_ids[current_node] = token_id

    # at this point we finished iterating over all of the characters over all of the token in the vocabulary, so we retuns the arrays
    return transitions, is_terminal, token_ids

# compresses a trie using CSR and exports to .mem file
# gets the transitions dense 2D array, is_terminal, token_ids and a filename prefix we can choose (we chose "root" and "cont")
# writes the .mem files and returns row_ptr and edges arrays of the trie
def flatten_csr(transitions, is_terminal, token_ids, prefix):
    # initialize the row_ptr and edges arrays, sets offset to 0
    row_ptr = []
    edges = []
    current_offset = 0

    # for each element inside transitions (row)
    for row in transitions:
        edge_count = 0
        # enumerate(row) returns (index, item)
        for char_idx, next_node in enumerate(row):
            if next_node != -1: # we skip all -1s, meaning where there is no path
                edges.append((char_idx, next_node)) # we append the connection to the edges array
                edge_count += 1 # we track valid connections for this specific node
        row_ptr.append((current_offset, edge_count)) # we document in row_ptr where the connections for this specific row start and how many connection are there
        current_offset += edge_count # we update the offset

    # export row pointers
    with open(f'{prefix}_csr_row_ptr.mem', 'w') as f_ptr:
        for offset, count in row_ptr:
            # shift offset by 16 bits to the left, and then OR with count
            # X - format the integer as uppercase Hexadecimal
            # 08 - pad with leading zeros so it is exactly 8 characters long
            f_ptr.write(f"{(offset << 16) | (count):08X}\n")

    # export edges
    with open(f'{prefix}_csr_edges.mem', 'w') as f_edges:
        for char_idx, dest in edges:
            # shift char_idx by 17 bits to the left, and then OR with dest
            # X - format the integer as uppercase Hexadecimal
            # 08 - pad with leading zeros so it is exactly 8 characters long
            f_edges.write(f"{(char_idx << 17) | (dest):08X}\n")

    # export terminal flags
    with open(f'{prefix}_is_terminal.mem', 'w') as f_term:
        for flag in is_terminal:
            if flag: # if flag = True
                f_term.write("01\n")
            else: # if flag = False
                f_term.write("00\n")
                
    # export token IDs
    with open(f'{prefix}_token_ids.mem', 'w') as f_ids:
        for tid in token_ids:
            if tid == -1: # -1 meaning no path exists, no token
                f_ids.write("00000000\n")
            else: # if a token is found
                # X - format the integer as uppercase Hexadecimal
                # 08 - pad with leading zeros so it is exactly 8 characters long
                f_ids.write(f"{tid:08X}\n")

    # some prints
    total_edges = sum(count for _, count in row_ptr)
    print(f"{prefix} trie CSR stats:")
    print(f"total nodes: {len(row_ptr)}")
    print(f"total edges: {total_edges}\n")
            
    return row_ptr, edges

# creates the file the pre-tokenizer uses to convert ascii bytes to index of the character
# gets the char to index mapping
# creates the ASCII-to-alphabet-index lookup table .mem file and doesn't return anything
def export_char_map(char_to_index):
    with open('char_to_index_map.mem', 'w') as f:
        # loops through the ascii code range of 0 - 128
        for ascii_code in range(128):
            # gets the string of this ascii code
            ch = chr(ascii_code)
            # checks if the character exists in the character to index mapping
            if ch in char_to_index:
                # if it exists, writes the HEX value to the file
                # X - format the integer as uppercase Hexadecimal
                # 04 - pad with leading zeros so it is exactly 4 characters long
                f.write(f"{char_to_index[ch]:04X}\n")
            else: # if it does not exist, write FFFF
                f.write("FFFF\n")

##################### DEBUG FUNCTION SECTION #####################

# print_edge_stats is a debug function which prints useful information to the user about row_ptr
# gets the row_ptr
# prints some useful information about the row_ptr
def print_edge_stats(row_ptr):
    counts = [count for _, count in row_ptr]
    max_edges = max(counts)
    avg_edges = sum(counts) / len(counts)
    
    # how many nodes have each edge count
    from collections import Counter
    dist = Counter(counts)
    
    print(f"edge distribution:")
    print(f"total nodes: {len(row_ptr)}")
    print(f"max edges: {max_edges}")
    print(f"avg edges: {avg_edges:.2f}")
    print(f"nodes with 1 edge: {dist[1]}")
    print(f"nodes with 2 edges: {dist[2]}")
    print(f"nodes with > 10: {sum(v for k,v in dist.items() if k > 10)}")
    print(f"nodes with > 50: {sum(v for k,v in dist.items() if k > 50)}\n")

##################### MAIN SECTION #####################

# step 1: loads the vocabulary
vocab = load_vocab("vocab.txt")
print(f"total vocabulary entries: {len(vocab)}")

# step 2: builds the shared alphabet
char_to_index = build_alphabet(vocab)
print(f"alphabet size: {len(char_to_index)} unique characters\n")

# step 3: splits the vocabulary into root and continuation vocabularies
root_vocab, cont_vocab = split_vocab(vocab)

# step 4: builds separate tries - the root and the continuation tries 
root_transitions, root_is_terminal, root_token_ids = build_trie(root_vocab, char_to_index)
print(f"root trie nodes: {len(root_transitions)}")

cont_transitions, cont_is_terminal, cont_token_ids = build_trie(cont_vocab, char_to_index)
print(f"continuation trie nodes: {len(cont_transitions)}\n")

# step 5: compresses both tries using CSR and exports .mem files
root_row_ptr, root_edges = flatten_csr(root_transitions, root_is_terminal, root_token_ids, "root")
cont_row_ptr, cont_edges = flatten_csr(cont_transitions, cont_is_terminal, cont_token_ids, "cont")

# step 6: print edge stats for both tries
print_edge_stats(root_row_ptr)
print_edge_stats(cont_row_ptr)

# step 7: exports the shared character map
export_char_map(char_to_index)