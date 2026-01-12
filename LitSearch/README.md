## Sample Dataset Folder

For saving original data and intermediate files.

We directly access LitSarch data through the datasets package. For other datasets:

### Bring your own data
We provide the code snippet for loading local corpus. Be default, the corpus is expected to be in ```.jsonl``` format. Each entry contains two fields
```
{
    "corpus_id": # a unique identifier,
    "text": # The textual content, e.g., title+abstract
}
```