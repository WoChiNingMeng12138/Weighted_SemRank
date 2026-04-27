## Task 1: Importance-Aware Scoring (Global Aspect Weighting, GAW)

### Background

Scientific paper retrieval is a critical task for literature discovery and research.  
Unlike general web search, scientific retrieval is often driven by specialized terminology and multi-faceted intent, where a query may depend on several fine-grained technical concepts rather than a single broad topic.

### Motivation

Recent semantic retrieval methods improve over purely dense retrieval, but they still have two major limitations:

- **Uniform weighting**: extracted concepts are typically treated as equally important, even though some concepts are much more informative than others.
- **Open-loop execution**: these systems usually perform a single retrieval-and-rerank pass without explicitly modeling concept salience at the corpus level.

In scientific literature, many high-frequency concepts are broad and non-discriminative, while rare concepts often carry stronger technical meaning.  
This motivates a corpus-aware weighting strategy that can emphasize informative concepts and suppress generic ones.

### Related Work
We have made improvements based on the research presented in the following two papers:

- **[SemRank (2025)](https://arxiv.org/abs/2505.21815)** combines LLM-guided query understanding with a concept-based semantic index built from topics and key phrases.
- **[PairSem (2026)](https://arxiv.org/abs/2510.09897)** extends this idea by modeling structured entity-aspect relations for more fine-grained scientific matching.

Our work focuses on the **importance-aware scoring** part of this direction and implements a lightweight extension on top of the SemRank pipeline.

### Our Idea: Global Aspect Weighting (GAW)

We replace the uniform concept aggregation in SemRank with a corpus-aware weighting scheme.

For each extracted concept \( p_i \), we compute a global rarity-based weight:

$$
w_i = \log\left(\frac{N+1}{df(p_i)+1}\right)
$$

where:

- \( N \) is the total number of documents in the corpus
- \( df(p_i) \) is the number of documents containing concept \( p_i \)

The final semantic score becomes a weighted aggregation instead of uniform averaging:

$$
\text{Score}(q, d) = \sum_i w_i \cdot \text{Sim}(p_i, d)
$$

This design gives higher influence to rare, high-information concepts and lower influence to common background terms.

---

## Experimental Setup

### Datasets

We evaluate our method on two scientific retrieval benchmarks:

- **CSFCube**
  - Corpus size: 4,207 papers
  - Focus: faceted query-by-example retrieval in computer science

- **DORISMAE**
  - Corpus size: 8,482 papers
  - Focus: multi-faceted scientific document retrieval with complex relevance relations

### Compared Methods

- **semrank+u (Baseline)**: original SemRank with uniform concept weighting
- **semrank_gaw (Proposed)**: our GAW-based variant with corpus-aware concept weighting

### Evaluation Metrics

We use the below metrics to evaluate our works:

- **Recall@50**
- **Recall@100**

---

## Results

| Dataset | Method | Recall@50 | Recall@100 |
|---------|--------|-----------|------------|
| CSFCube | semrank+u (Baseline) | 0.5415 | 0.6979 |
| CSFCube | semrank_gaw (Proposed) | 0.5393 | 0.6995 |
| DORISMAE | semrank+u (Baseline) | 0.5651 | 0.7228 |
| DORISMAE | semrank_gaw (Proposed) | 0.5642 | 0.7277 |

---

## Discussion

The results show that **GAW slightly improves Recall@100 on both datasets**, while Recall@50 remains comparable to the original SemRank baseline.

- On **CSFCube**, GAW improves Recall@100 from **0.6979** to **0.6995**
- On **DORISMAE**, GAW improves Recall@100 from **0.7228** to **0.7277**

Although the gains are modest, they are consistent at deeper recall levels.  
This suggests that corpus-aware weighting helps recover additional relevant documents by emphasizing more discriminative technical concepts.

At the same time, Recall@50 does not improve in the current setup, indicating that the weighting strategy may be more beneficial for broader recall than for very early ranking positions.