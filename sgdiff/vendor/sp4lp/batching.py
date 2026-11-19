"""
sp4lp/batching.py
==============================================================================
Da cammini ragged a tensori padded.

Supporta piu' cammini per coppia: `pair_of_path` dice a quale coppia del batch
appartiene ogni cammino, cosi' il modello puo' codificarli tutti insieme e poi
aggregarli per coppia (la AGG dell'Eq. 8).
"""

from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["PathBatch", "make_path_batch"]


@dataclass
class PathBatch:
    nodes: torch.Tensor         # [T] nodi di tutti i cammini, concatenati
    row: torch.Tensor           # [T] a quale CAMMINO appartiene ogni nodo
    pos: torch.Tensor           # [T] posizione dentro il proprio cammino
    lengths: torch.Tensor       # [P] lunghezza di ogni cammino
    reach: torch.Tensor         # [B] per coppia
    pair_of_path: torch.Tensor  # [P] a quale COPPIA appartiene ogni cammino
    n_pairs: int                # B
    pair_lengths: torch.Tensor  # [B] lunghezza del primo cammino di ogni coppia
    feat: torch.Tensor = None   # [T, 3] (grado, cn con u, cn con v) o None
    pair_feat: torch.Tensor = None  # [B, 8] statistiche strutturali simmetriche

    def __len__(self):
        return self.n_pairs


def make_path_batch(store, pairs_np, device):
    rows = store.rows_for(pairs_np)
    out = store.gather(pairs_np)
    feat = None
    if len(out) == 3:                       # PathStore, senza feature nodi
        flat, lengths, reach = out
        n_paths = np.ones(len(lengths), dtype=np.int64)
    elif len(out) == 4:
        flat, lengths, n_paths, reach = out
    else:
        flat, lengths, n_paths, reach, feat = out

    P = len(lengths)
    B = len(n_paths)
    row = np.repeat(np.arange(P, dtype=np.int64), lengths)
    starts = (np.concatenate(([0], np.cumsum(lengths)[:-1])) if P
              else np.zeros(0, dtype=np.int64))
    pos = np.arange(len(flat), dtype=np.int64) - np.repeat(starts, lengths)
    pair_of_path = np.repeat(np.arange(B, dtype=np.int64), n_paths)
    first = (np.concatenate(([0], np.cumsum(n_paths)[:-1])) if B
             else np.zeros(0, dtype=np.int64))
    pair_lengths = lengths[first] if P else np.zeros(B, dtype=np.int64)

    t = lambda a, d=torch.long: torch.as_tensor(a, dtype=d, device=device)
    return PathBatch(
        nodes=t(flat), row=t(row), pos=t(pos), lengths=t(lengths),
        reach=t(reach, torch.bool), pair_of_path=t(pair_of_path),
        n_pairs=int(B), pair_lengths=t(pair_lengths),
        feat=(t(feat) if feat is not None else None),
        pair_feat=(t(store.pair_feat[rows], torch.float32)
                   if getattr(store, 'pair_feat', None) is not None else None))
