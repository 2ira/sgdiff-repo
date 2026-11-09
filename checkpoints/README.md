# Checkpoints

Trained weights are written by the trainer to `runs/<run_name>/best.pt`
(git-ignored). Copy the ones you want to share into this directory.

A checkpoint is a plain dictionary saved with `torch.save` and loaded with
`weights_only=True` (no pickle of arbitrary objects):

```python
{
    "model_state": state_dict,
    "config": {...},          # ExperimentConfig.to_dict()
    "feature_dim": int,       # node-feature width the model expects
    "num_labels": int,        # DRNL vocabulary size
    "epoch": int,
    "metrics": {...},
}
```

Reload with:

```python
from sgdiff.engine.trainer import load_checkpoint
model, cfg = load_checkpoint("checkpoints/sgdiff_cora.pt", device="cpu")
```
