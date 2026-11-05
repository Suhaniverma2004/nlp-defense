# Similarity-Preservation Diagnostics

This repository's generated text variants and embedding comparisons are useful
for studying how a representation changes under a transformation. They are
**not adversarial attack detection**.

## What the output means

- `transformation_changed` (or `changed` in the chatbot output) is exact string
  inequality: the transformed text differs from the original.
- `cosine_similarity` is the cosine between the two text embedding vectors.
  It ranges from -1 to 1; larger values mean closer vectors under the selected
  embedding model and pooling method.
- `changed_variant_count` is the number of exact-text changes in a
  transformation group.
- `unchanged_variant_count` is the number of generated outputs equal to the
  original. These rows have no transformation similarity score and are excluded
  from the group's mean.
- `mean_cosine_similarity_changed_variants` is the arithmetic mean of the
  non-missing cosine scores for changed rows only. It is undefined when there
  are no changed rows.

The chatbot CLI requires a caller-supplied model and a model/checkpoint
identifier. It does not initialize a random model. The caller must load and
verify a trained checkpoint before using the model-based diagnostic. The
Streamlit dashboard instead identifies its baseline explicitly as
`distilbert-base-uncased`; it computes cosine similarity from mean-pooled last
hidden states. That baseline is not a detector either.

## Interpretation limits

Similarity preservation is not equivalent to adversarial attack detection. Low
similarity does not prove an attack was detected. High similarity does not
prove an input is safe. An unchanged transformation is not a changed example
and is counted separately rather than included in transformation-impact
averages.

Genuine attack detection requires a trained classifier, explicit ground-truth
labels, and a properly designed evaluation protocol with suitable metrics and
held-out data. Similarity-only results must not be reported as attack-detection
accuracy, robustness, or classification performance.
