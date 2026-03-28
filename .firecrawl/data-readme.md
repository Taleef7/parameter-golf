\# Data Workflows

This directory contains the dataset download helpers and export scripts used for the challenge.

Canonical local layout:
\- \`data/datasets//\`
\- \`data/tokenizers/\`
\- \`data/manifest.json\`
\- \`data/docs\_selected.jsonl\`
\- \`data/docs\_selected.source\_manifest.json\`

\## Downloading Published Data

Download the cached FineWeb export for a tokenizer variant with:

\`\`\`bash
python3 data/cached\_challenge\_fineweb.py --variant sp1024
\`\`\`

This populates \`./data/datasets/fineweb10B\_sp1024/\` and \`./data/tokenizers/\`.
By default it downloads the full validation split and 8B training tokens (80 train shards).

To fetch more training shards, pass \`--train-shards\`:

\`\`\`bash
python3 data/cached\_challenge\_fineweb.py --variant sp1024 --train-shards 180
\`\`\`

The downloader is manifest-driven and can fetch only a prefix of train shards from a larger published export. With the current shard size of \`100\_000\_000\` tokens, \`10B\` retokenized training tokens is \`100\` train shards:

\`\`\`bash
MATCHED\_FINEWEB\_REPO\_ID=your-hf-username/your-dataset-repo \
MATCHED\_FINEWEB\_REMOTE\_ROOT\_PREFIX=your\_50B\_export\_root \
python3 data/cached\_challenge\_fineweb.py --variant sp1024 --train-shards 100
\`\`\`

Validation is always downloaded in full from the fixed \`fineweb\_val\_\*\` split. Training on the first \`N\` train shards means training on the prefix of the same frozen shuffled export, so the data order stays aligned with the baseline for that tokenizer family.

The default published repo is \`willdepueoai/parameter-golf\`, with the export rooted under the repo subdirectory \`datasets/\`.

\## Rebuilding Tokenizers From Published Docs

To retrain a tokenizer or re-export shards from exactly the same selected documents, run the standalone retokenizer against the published docs cache:

\`\`\`bash
python3 data/download\_hf\_docs\_and\_tokenize.py \
 --repo-id your-hf-username/your-dataset-repo \
 --remote-root your\_50B\_export\_root \
 --output-root /tmp/my\_custom\_tokenizer\_export \
 --tokenizer-config ./data/tokenizer\_specs.json
\`\`\`

The sidecar \`docs\_selected.source\_manifest.json\` includes \`docs\_sha256\`, so users can verify they are rebuilding from the exact same document list and order as the baseline export.

\## Useful Knobs

For CPU-heavy exports, useful knobs are:

\`\`\`bash
MATCHED\_FINEWEB\_SP\_BATCH\_SIZE=2048
MATCHED\_FINEWEB\_TOKENIZER\_THREADS=16
MATCHED\_FINEWEB\_TIKTOKEN\_THREADS=16
MATCHED\_FINEWEB\_GPT2\_DECODE\_BATCH\_SIZE=512
\`\`\`

These control batched tokenizer encoding during shard export, tokenizer thread count, tiktoken thread count, and batched GPT-2 decode for the blobstore docs-cache path.