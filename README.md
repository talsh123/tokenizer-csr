# tokenizer-csr — BERT vocabulary → CSR trie compiler

The **offline compiler** that turns the 30,522-entry `bert-base-uncased` vocabulary into the
compact **CSR (Compressed-Sparse-Row) trie** memory images that the FPGA WordPiece tokenizer
loads into on-chip BRAM.

This is the Python half of a three-repository project:

| Repo | What it holds |
|------|---------------|
| **[tokenizer-vivado](https://github.com/talsh123/tokenizer-vivado)** | Vivado project + custom RTL + testbenches (the hardware tokenizer) |
| **[tokenizer-vitis](https://github.com/talsh123/tokenizer-vitis)** | MicroBlaze firmware: lwIP TCP server, AXI DMA driver, PHY patches |
| **tokenizer-csr** (this repo) | compiles the BERT vocab → the eight `.mem` CSR trie images |

> The tokenizer runs entirely in FPGA logic and reproduces HuggingFace `bert-base-uncased`
> token IDs exactly (100 %, 66/66 on the evaluation corpus). Full write-up, results, and the
> defense deck live in **[tokenizer-vivado](https://github.com/talsh123/tokenizer-vivado)**.

---

## Why compress the vocabulary?

WordPiece tokenization is a **greedy longest-match** walk over a prefix tree (trie) of the
vocabulary. A trie is naturally a 2-D table `transitions[node][char]`, but that table is
overwhelmingly empty — most nodes have only a handful of outgoing edges out of the whole
alphabet. Stored densely it is on the order of **~261 MB**, far more than the Artix-7's
on-chip BRAM.

**CSR** stores only the edges that actually exist, shrinking the two tries to roughly
**~705 KB** of packed BRAM (about a **370×** reduction) — small enough to live entirely in
fabric, so every trie step is a single-cycle BRAM read with no external memory.

---

## What the generator produces

`vocab_parser.py` reads `vocab.txt` (one token per line; the line number **is** the BERT
token ID) and writes nine `.mem` files (`$readmemh` format, consumed by the RTL):

| File | Meaning |
|------|---------|
| `root_csr_row_ptr.mem` / `cont_csr_row_ptr.mem` | per-node `(offset, count)` into the edge list |
| `root_csr_edges.mem`  / `cont_csr_edges.mem`  | the sorted outgoing edges `(char, dest)` |
| `root_is_terminal.mem`/ `cont_is_terminal.mem`| `01` if a token ends at this node, else `00` |
| `root_token_ids.mem`  / `cont_token_ids.mem`  | the 16-bit BERT token ID at each terminal node |
| `char_to_index_map.mem` | ASCII byte (0–127) → alphabet index, or `FFFF` if unused |

**Two tries, because WordPiece has two kinds of piece:**

- **root** — word-initial pieces (e.g. `token`), built from vocab entries *without* `##`.
- **continuation** — mid-word pieces (e.g. `##izing`), built from the `##`-prefixed entries
  with the `##` stripped.

Approximate sizes: the root trie is ~56,719 nodes / 56,718 edges and the continuation trie
~7,864 / 7,863.

---

## The CSR encoding (bit layout)

Each `.mem` line is one 32-bit word, big-endian hex. The packing matches the RTL's register
widths exactly, so `$readmemh` loads them without truncation:

```
row_ptr word :  offset[31:16]  |  count[15:0]        →  (offset << 16) | count
edges  word  :  char[31:17]    |  dest[16:0]         →  (char   << 17) | dest
token_ids    :  16-bit BERT token ID                 →  4 hex digits (0000 if none)
is_terminal  :  01 = terminal, 00 = not
char map     :  4 hex digits   (index, or FFFF)
```

To walk a node: read its `row_ptr` to get `(offset, count)`, then binary-search the `count`
edges starting at `edges[offset]` for the current character — `O(log K)` per node.

> The alphabet is built from the **sorted** set of characters in the vocabulary, so the
> generator is **deterministic**: the same `vocab.txt` always yields byte-identical `.mem`
> files. (Sorting is deliberate — without it, index assignments would shuffle every run.)

---

## Running it

```bash
python vocab_parser.py      # reads ./vocab.txt, writes the nine .mem files here
```

No third-party dependencies (standard library only). It prints the vocab split, the trie
node/edge counts, and an edge-count distribution for each trie.

To use a **different vocabulary**: replace `vocab.txt`, re-run, copy the regenerated `.mem`
files into `tokenizer-vivado/uart.srcs/sources_1/new/`, and re-synthesize. Nothing about the
run-time hardware datapath changes.

---

## Files

```
flat_trie_compression/
├── vocab_parser.py     the compiler (load → split → build tries → CSR → emit)
├── vocab.txt           bert-base-uncased vocabulary (30,522 lines; line no. = token ID)
└── *.mem               the generated CSR images (also mirrored into the Vivado repo)
```

## License

[MIT](LICENSE) © 2026 Tal-Shalom Ben Ovadia and Rafi Erez.
