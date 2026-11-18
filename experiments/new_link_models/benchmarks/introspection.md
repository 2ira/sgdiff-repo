# Learned model components

`learned_parameters.csv` contains registered NCNC beta/JK scalars and Refined-GAE softmax weights for propagation depths 1–4. CN0 registers a beta parameter that its forward path does not use. Scalar magnitudes are not causal importance because learned representations can rescale.

`attention_per_pair.csv` evaluates the first 64 positive and 64 negative test candidates from each seeded split on the clean inference graph. Entropy is normalized by log(candidate count); counts 0/1 have entropy 0. The uniform-attention variant is independently checked for normalized entropy 1 when count > 1, and the no-PE variant for zero encoded PE. Attention describes aggregation behavior, not causal feature attribution.
