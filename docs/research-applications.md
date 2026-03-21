# Research Applications

## Possible Research Direction: Synthetic Bootstrapped Pretraining

Recent work by Zitong Yang et al., *Synthetic Bootstrapped Pretraining* (arXiv:2509.15248), argues that inter-document relationships can be used to generate synthetic pretraining data rather than relying only on standard next-token prediction.

One plausible extension for this repo is to replace pairwise similarity retrieval with an explicit HypergraphRAG representation during synthetic data generation.

With that framing, this substrate could help:

- organize documents into explicit graph or hypergraph structures instead of flat nearest-neighbor pairs
- capture higher-order, multi-document relationships around shared concepts
- provide structured knowledge neighborhoods for synthetic data generation
- preserve reusable provenance-aware knowledge structure across repeated generation runs

Potential upside:

- better synthetic data diversity and less repeated nearest-neighbor redundancy
- stronger higher-order reasoning signals during pretraining
- better modeling of latent concepts shared across multiple documents

This is a possible application and research direction for the substrate, not a claim that such a pipeline is already implemented in this repo.

## Reference

Yang, Z., Zhang, A., Liu, H., Hashimoto, T., Candes, E., Wang, C., & Pang, R. (2025).  
*Synthetic Bootstrapped Pretraining*. arXiv:2509.15248.
