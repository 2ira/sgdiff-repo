# Data directory

```
data/
├── raw/         # untouched downloads (git-ignored)
├── processed/   # cached graphs as .npz (git-ignored, rebuild with `sgdiff prepare`)
└── dataset_report.json   # statistics of the last `sgdiff prepare` run
```

Nothing in `data/raw` or `data/processed` is committed: every benchmark is
re-fetchable with `python -m sgdiff prepare`.

## Benchmarks and sources

| name | source | nodes | undirected edges | features | auto |
|------|--------|------:|-----------------:|:--------:|:----:|
| `cora` | [Planetoid](https://github.com/kimiyoung/planetoid) via raw.githubusercontent | 2708 | 5278 (paper: 10556 directed) | yes | yes |
| `citeseer` | Planetoid | 3327 | 4552 (paper: 4614) | yes | yes |
| `pubmed` | Planetoid | 19717 | 44324 (paper: 44327) | yes | yes |
| `usair` | [networkrepository USAir97](https://networkrepository.com/USAir97.php) | 332 | 2126 (paper: 4252 directed) | no | yes |
| `ns` | [KONECT dimacs10-netscience](http://konect.cc/networks/dimacs10-netscience/) | 1589 | 2742 | no | yes |
| `router` | SEAL data bundle (not redistributable) | 5022 | 12516 | no | **no** |
| `synthetic` | generated (`make_synthetic`) | 400 | ~1700 | yes | built-in |

Small differences on `citeseer` (62 edges) and `pubmed` (3 edges) come from
de-duplicating multi-edges and dropping self-loops, which the paper's counts
include.

### Router

`Router` is distributed with the SEAL release and cannot be mirrored here.
Download it, then place the file at `data/raw/router/Router.mat` (MATLAB key
`net`) or pass a path:

```bash
python -m sgdiff train --config configs/sgdiff_cora.yaml \
  --set data.name=router --set data.root=data
# or convert a TUDataset / edge list:
python -c "from sgdiff.data.datasets import load_edgelist; g=load_edgelist('router.edges', name='router'); print(g.num_nodes, g.num_edges)"
```

`load_mat`, `load_edgelist` and `load_tu` accept `.mat`, whitespace edge lists
and `<name>_A.txt` TUDataset files respectively.

## Mirror fallback

`sgdiff.data.download` first requests the original URL and then retries through
mirrors (`https://ghproxy.net/`), which is what makes the Planetoid files
downloadable behind restrictive networks. Set `SGDIFF_LOG_LEVEL=DEBUG` to see
each attempt.
