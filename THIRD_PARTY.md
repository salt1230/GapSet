# Third-party sources

The repository MIT license covers our code, wrappers and documentation. It
does not relicense other authors' code or dataset contents.

Included unchanged under their original MIT notices:

|Work|Pinned commit|Included files|
|---|---|---|
|[SenseFi](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark)|15fd872433a46935f1be706e57395d5c57181c6d|NTU_Fi_model.py, widar_model.py; `third_party/sensefi/LICENSE`|
|[MDA/DFS-CSI](https://github.com/mestrelan/MDA-CSI)|7b119ee6eb1c7735b831af826fdf56cc7089bfdb|DFS-CSI/Run/test4amplitude48canais.py; `third_party/dfs/LICENSE`|

These optional sources were publicly available at the pinned snapshots but no
explicit license file was located there. Their author files are not included
or relicensed. Consult the upstream authors' current terms before reuse or
redistribution. Version and SHA-256 are in `third_party/sources.lock.json`.

|Component|Original repository|Pinned commit|
|---|---|---|
|WiFlexFormer|https://github.com/StrohmayerJ/WiFlexFormer|71b20bb39ce3b35a17e0f0e66274141ff9e31a0b|
|CSI-BERT2|https://github.com/RS2002/CSI-BERT2|4336889f8311c4ffcde4e65fe8103f3f15d1af83|
|HMS-RWKV|https://github.com/yiliu97/HMS-RWKV|d90de9c180ec42e1ab51dedaa8191270fa89174e|
|AMCS-TCN|https://github.com/boyuanzhang759-rgb/AMCS-TCN-by-BoYuan|cc36e78fc2533c2ec4fbb5cfb8c450cdcf8e98c5|

For permitted local use, the fetch helper retrieves only the listed source
files from the pinned commits, verifies every SHA-256, and never executes
upstream training or installs dependencies:

```bash
python tools/fetch_baselines.py --component hms
```

Other choices are `wiflexformer`, `csibert2`, `amcs`, `all`. Fetched folders are
ignored by Git. Retrieval capability does not grant a redistribution license.

All comparison methods use the common8 interface and clean-training recipe.
SenseFi input stems/head shapes are adapted; unused author top-level training
imports are avoided through selected class ASTs. CSI-BERT2 is a clean classifier
backbone without pretraining; AMCS excludes original specialized filtering and
augmentation; DFS is an amplitude classification backbone, not its full binary
specialist system. HMS retains both public branches, with dropout0.15. GRU-D
is our explicitly disclosed equation adaptation. Active versus registered
parameter differences for CSI-BERT2/DFS are reported in the paper table data.

Dependencies such as PyTorch, NumPy, transformers and Mamba are installed
separately and retain their respective upstream licenses. No dataset, paper PDF,
upstream pretrained checkpoint or vendor archive is redistributed here. The
three checkpoints in `checkpoints/` belong to our final GapSet-R model and retain
exactly the model tensors used for the paper's three-seed results.
