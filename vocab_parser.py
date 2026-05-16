##################### LOGIC FUNCTION SECTION #####################

# gets a filepath of a .txt file
# returns a list of strings
def load_vocab(filepath):
    # we open the file in read mode
    f = open(filepath, 'r', encoding='utf-8')

    # create an empty list to hold our tokens
    vocab_list = []

    # loop through each line in the file, line by line
    for line in f:
        # clean up the \n character at the end of each line
        clean_line = line.strip('\n')
        # add the index and the cleaned string to our list
        vocab_list.append(clean_line)

    f.close()

    return vocab_list

def build_alphabet(vocab_list):
    # ignore duplicates
    unique_chars = set()

    # loop through every word in the vocab
    for word in vocab_list:
        # loop through every character in the word
        for char in word:
            unique_chars.add(char)
    
    # sort the characters if we won't sort 
    # will give you different indices every time you run the script.
    sorted_chars = sorted(list(unique_chars))

    # create a dictionary to map each character to a number
    char_to_index = {}

    index = 0
    for char in sorted_chars:
        char_to_index[char] = index
        index += 1

    return char_to_index

def split_vocab(vocab_list):
    # splitrs the vocabulary into 2 lists:
    # root_vocab: first-piece subwords (no ## prefix) with their original token IDs
    # cont_vocab: continuation subwords (## prefix stripped) with their original token IDs

    # each list is tuple (word, original_token_id)
    root_vocab = []
    cont_vocab = []

    # iterates over the original vocabulary
    for token_id, word in enumerate(vocab_list):
        if word.startswith('##'): # if we find a word which starts with "##" prefix
            # Strip the prefix
            stripped = word[2:] # stripped word
            cont_vocab.append((stripped, token_id))
        else: # a normal word with no "##"
            root_vocab.append((word, token_id))
    
    print(f"Vocabulary split:")
    print(f"  Root tokens:         {len(root_vocab)}")
    print(f"  Continuation tokens: {len(cont_vocab)}")
    print(f"  Total:               {len(root_vocab) + len(cont_vocab)}")
    
    return root_vocab, cont_vocab

# gets a list of (word, token_id) tuples and the character to index mapping
# returns the transition table, the terminal flags and the token IDs
def build_trie(vocab_with_ids, char_to_index):
    # get the number of unique characters in the vocabulary
    alphabet_size = len(char_to_index)

    # initialize the root node (Node 0)
    # -1 means no path yet
    root_node_transitions = [-1] * alphabet_size

    # These 3 lists will grow together.
    # Their index is the Node ID.
    transitions = [root_node_transitions]
    is_terminal = [False]
    token_ids = [-1] # -1 means no token ends here

    # loop through every (word, token_id) pair
    for word, tid in vocab_with_ids:

        # always start at the root node for a new word
        current_node = 0

        # walk through each character in the word 
        for char in word:
            # find the column index for this character
            char_idx = char_to_index[char]

            # look at the current node to see where this character goes
            next_node = transitions[current_node][char_idx]

            if next_node == -1: # path doesn't exist
                new_node_id = len(transitions)

                # update the current node to point to our new node
                transitions[current_node][char_idx] = new_node_id

                # expand our 3 lists to make room for the new node
                new_node_transitions = [-1] * alphabet_size
                transitions.append(new_node_transitions)
                is_terminal.append(False)
                token_ids.append(-1)

                # Move forward to the new node we just created
                current_node = new_node_id
            else:
                # the path already exists, just move to that node
                current_node = next_node

        # we finished walking the characters of the word.
        # this node is now the end
        is_terminal[current_node] = True
        token_ids[current_node] = tid

    return transitions, is_terminal, token_ids

# this functions gets the transition table, the terminal flags and the token IDs
# and exports those 3 lists to 3 .mem files
def export_to_mem(transitions, is_terminal, token_ids):
    # export transitions table
    f_trans = open('trie_transitions.mem', 'w')
    for row in transitions:
        row_strings = []
        for val in row:
            # convert -1 to 0 for Verilog - $readmemh expects positive hexadecimal values.
            # To keep Verilog simple, we are going to convert those -1 values into 00000000
            if val == -1:
                hex_val = "00000000"
            else:
                hex_val = f"{val:08X}"
        
            row_strings.append(hex_val)

        # join the row with spaces and write to file
        line_to_write = " ".join(row_strings) + "\n"
        f_trans.write(line_to_write)
    f_trans.close()

    # export terminal flags
    f_term = open('trie_is_terminal.mem', 'w')
    for flag in is_terminal:
        if flag == True:
            f_term.write("01\n")
        else:
            f_term.write("00\n")
    f_term.close()

    # export token IDs
    f_ids = open('trie_token_ids.mem', 'w')
    for tid in token_ids:
        # if it's -1 (not a terminal node), just write 0
        if tid == -1:
            hex_tid = "00000000"
        else:
            hex_tid = f"{tid:08X}"

        f_ids.write(hex_tid + "\n")
    f_ids.close()

# compresses a trie into CSR format and exports to .mem file
# gets the transitions, is_terminal, token_ids and a custom filename prefix
# returns row_ptr and edges arrays that contain and allow to traverse the compressed vocab
def flatten_csr(transitions, is_terminal, token_ids, prefix):
    # initialize sparse structure containers
    row_ptr = []
    edges = []
    current_offset = 0

    # Traverse dense graph
    for row in transitions:
        edge_count = 0
        for char_idx, next_node in enumerate(row):
            if next_node != -1:
                edges.append((char_idx, next_node))
                edge_count += 1
        row_ptr.append((current_offset, edge_count))
        current_offset += edge_count

    # export row pointers
    with open(f'{prefix}_csr_row_ptr.mem', 'w') as f_ptr:
        for offset, count in row_ptr:
            f_ptr.write(f"{(offset << 16) | (count & 0xFFFF):08X}\n")

    # export edges
    with open(f'{prefix}_csr_edges.mem', 'w') as f_edges:
        for char_idx, dest in edges:
            f_edges.write(f"{(char_idx << 17) | (dest & 0x1FFFF):08X}\n")

    # export edges
    with open(f'{prefix}_is_terminal.mem', 'w') as f_term:
        for flag in is_terminal:
            f_term.write("01\n" if flag else "00\n")

    # export token IDs
    with open(f'{prefix}_token_ids.mem', 'w') as f_ids:
        for tid in token_ids:
            if tid == -1:
                f_ids.write("00000000\n")
            else:
                f_ids.write(f"{tid:08X}\n")

    total_edges = sum(count for _, count in row_ptr)
    print(f"\n{prefix} trie CSR stats:")
    print(f"  Total nodes: {len(row_ptr)}")
    print(f"  Total edges: {total_edges}")
            
    return row_ptr, edges

# The core CSR lookup logic
def lookup_csr(current_node, char_idx, row_ptr, edges):
    # Get the start index and how many edges we need to check
    offset, count = row_ptr[current_node]
    
    # Scan only the edges that belong to this node
    for i in range(offset, offset + count):
        edge_char, dest_node = edges[i]
        
        if edge_char == char_idx:
            # Found a matching path
            return dest_node
            
    # Loop finished without finding the character, dead end
    return -1

def export_char_map(char_to_index):
    with open('char_to_index_map.mem', 'w') as f:
        for ascii_code in range(128):
            ch = chr(ascii_code)
            if ch in char_to_index:
                f.write(f"{char_to_index[ch]:04X}\n")
            else:
                f.write("FFFF\n")

##################### DEBUG FUNCTION SECTION #####################

def print_edge_stats(row_ptr):
    counts = [count for _, count in row_ptr]
    max_edges = max(counts)
    avg_edges = sum(counts) / len(counts)
    
    # how many nodes have each edge count
    from collections import Counter
    dist = Counter(counts)
    
    print(f"\nEdge distribution:")
    print(f"  Total nodes:  {len(row_ptr)}")
    print(f"  Max edges:    {max_edges}")
    print(f"  Avg edges:    {avg_edges:.2f}")
    print(f"  Nodes with 1 edge:  {dist[1]}")
    print(f"  Nodes with 2 edges: {dist[2]}")
    print(f"  Nodes with >10:     {sum(v for k,v in dist.items() if k > 10)}")
    print(f"  Nodes with >50:     {sum(v for k,v in dist.items() if k > 50)}")

##################### VERIFICATION FUNCTION SECTION #####################

def verify_dual_trie(word, root_row_ptr, root_edges, root_is_terminal, root_token_ids,
                           cont_row_ptr, cont_edges, cont_is_terminal, cont_token_ids,
                           char_to_index):
    """
    Simulates the hardware dual-trie tokenization.
    
    - First piece: search in root trie
    - Subsequent pieces: search in continuation trie (no ## prefix needed)
    
    This mirrors the hardware behavior where use_root starts at 1,
    switches to 0 after the first token is emitted, and resets on word_done.
    """
    print(f"\nTesting word (dual-trie): '{word}'")
    tokens = []
    found_tids = []
    start_idx = 0
    use_root = True  # mirrors the hardware use_root flag

    while start_idx < len(word):
        # Select which trie to search
        if use_root:
            row_ptr = root_row_ptr
            edges = root_edges
            is_terminal = root_is_terminal
            token_ids = root_token_ids
        else:
            row_ptr = cont_row_ptr
            edges = cont_edges
            is_terminal = cont_is_terminal
            token_ids = cont_token_ids

        current_node = 0
        best_match_end = -1
        best_match_tid = -1
        chars_consumed = 0
        longest_chars_consumed = 0

        # Search string is always just the remaining characters
        # No ## prefix needed - the continuation trie already has them stripped
        search_string = word[start_idx:]

        for i, char in enumerate(search_string):
            if char not in char_to_index:
                break

            char_idx = char_to_index[char]
            next_node = lookup_csr(current_node, char_idx, row_ptr, edges)

            if next_node == -1:
                break

            current_node = next_node
            chars_consumed += 1

            if is_terminal[current_node]:
                best_match_end = i
                best_match_tid = token_ids[current_node]
                longest_chars_consumed = chars_consumed

        # Check if we failed
        if best_match_tid == -1 or longest_chars_consumed == 0:
            print("  -> [UNK] encountered.")
            return ["[UNK]"], [-1]

        # Record the match
        matched_str = search_string[:best_match_end + 1]
        if not use_root:
            matched_str = "##" + matched_str  # for display only
        tokens.append(matched_str)
        found_tids.append(best_match_tid)

        # Advance pointer
        start_idx += longest_chars_consumed

        # After first token, switch to continuation trie
        use_root = False

    print(f"  Subwords:  {tokens}")
    print(f"  Token IDs: {found_tids}")
    return tokens, found_tids

##################### MAIN SECTION #####################
# Step 1: Load the vocabulary
vocab = load_vocab("vocab.txt")
print(f"Total vocabulary entries: {len(vocab)}")

# Step 2: Build the shared alphabet (one mapping for both tries)
char_to_index = build_alphabet(vocab)
print(f"Alphabet size: {len(char_to_index)} unique characters")

# Step 3: Split vocabulary into root and continuation
root_vocab, cont_vocab = split_vocab(vocab)

# Step 4: Build separate tries
print("\nBuilding root trie...")
root_transitions, root_is_terminal, root_token_ids = build_trie(root_vocab, char_to_index)
print(f"  Root trie nodes: {len(root_transitions)}")

print("\nBuilding continuation trie...")
cont_transitions, cont_is_terminal, cont_token_ids = build_trie(cont_vocab, char_to_index)
print(f"  Continuation trie nodes: {len(cont_transitions)}")

# Step 5: Compress both tries to CSR and export .mem files
root_row_ptr, root_edges = flatten_csr(root_transitions, root_is_terminal, root_token_ids, "root")
cont_row_ptr, cont_edges = flatten_csr(cont_transitions, cont_is_terminal, cont_token_ids, "cont")

# Step 6: Print edge stats for both tries
print_edge_stats(root_row_ptr)
print_edge_stats(cont_row_ptr)

# Step 7: Export the shared character map
export_char_map(char_to_index)

# Step 8: Verify with dual-trie system
print("\n" + "=" * 50)
print("DUAL-TRIE VERIFICATION")
print("=" * 50)

test_words = ["embedding", "unquestionably", "verilog", "hello", "hardware"]
for w in test_words:
    verify_dual_trie(w,
                     root_row_ptr, root_edges, root_is_terminal, root_token_ids,
                     cont_row_ptr, cont_edges, cont_is_terminal, cont_token_ids,
                     char_to_index)