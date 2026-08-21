# Corpus Format

Each owned corpus lives in its own directory and starts from `documents.jsonl`.

`documents.jsonl`
- One JSON object per line.
- Required fields:
  - `doc_id`: stable document id owned by this project
  - `source`: source family such as `textbooks` or `pubmed`
  - `title`: cleaned title string
  - `text`: cleaned body text without duplicating the title
  - `meta`: import metadata, quality metrics, and source-specific fields

`rejected.jsonl`
- Rejected raw candidates with:
  - `doc_id`
  - `source`
  - `title`
  - `text`
  - `reason`
  - `meta`

`stats.json`
- Import summary:
  - raw row counts
  - kept and rejected document counts
  - reject reasons
  - duplicate counts
  - basic quality aggregates

`manifest.json`
- Corpus-level metadata:
  - corpus name and version
  - created timestamp
  - document count
  - chunk count if built later
  - source list
  - import settings

Recommended workflow:
1. import raw source into `documents.jsonl`
2. inspect `stats.json` and `rejected.jsonl`
3. build `chunks.jsonl`
4. build BM25 and dense indexes
