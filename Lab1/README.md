# Exercise

The exercise includes the following steps:

1. Select the protein (HNRNPC/HNRNPA2B1) analyzed during the experiments, which uniquely identifies the data set used.

2. Familiarize oneself with the expected motif, extracted from the experimentally determined 3D structure of the complex, formed by the analyzed RNA-binding protein, in order to identify the number of nucleotides (`len`) included in it. Let's assume that our expected motif is as follows `TTTT`, then `len = 4`.

3. Determine the length of potentially promising motifs `w = {len, len + 1, len + 2}`. Suppose our expected motif is `TTTT` (`len = 4`), then for `w = len + 1`, motifs could be like `NTTTT` or `TTTTN`, and for `w = len + 2`, like `NTTTTN`, where `N` is any nucleotide.

4. Extract all promising continuous motifs (represented by the corresponding fSHAPE data profiles) of length `w` for each `w` in `{len, len + 1, len + 2}` satisfying the assumed binding site characteristics, based on the reactivity values obtained by the fSHAPE method (i.e., reactivity value for at least one nucleotide within the motif must exceed a value of `1.0`), from among short fragments of transcripts contained in the archive `identifier_selected_protein_binding_sites_fshape.zip`, which are likely to contain previously undiscovered RNA binding sites to the protein under consideration.

5. Perform cluster analysis using at least two methods (e.g., `KMEANS++`, `DBSCAN`, etc.) for the set of motifs extracted in the previous step for each motif length (`w`) analyzed independently.

6. Determine a promising consensus motif(s) (e.g., using the `STUMPY` library) based on the fSHAPE data profiles contained in the three most abundant clusters identified in the previous step (the minimum power of the cluster for which the consensus motif must be determined should not be less than `3`).

7. Perform a transcripts search (e.g., using the `STUMPY` library), stored in the archive `identifier_selected_protein_search_fshape.zip`, based on the fSHAPE data profiles extracted for the promising consensus motifs identified in the previous step and the expected motif. Compile the results in tabular form, where each motif will be described by its sequence, ranges of nucleotide numbers, the name of the transcript file in which it was identified, and the values of the following measures, namely `znEd`, `ssf`, `aS` determined in the context of the expected motif. The records should be ordered in non-decreasing order according to the last column (`aS`).