# HARP — Hamiltonian Read-out for Properties

HARP is an E(3)-equivariant network that turns one cheap
B3LYP/def2-SVP single point into coupled-cluster-quality molecular properties.
It predicts a local correction to the B3LYP Kohn–Sham Hamiltonian and evaluates
every property from the corrected Hamiltonian with its quantum-mechanical
operator:

- total energy
- optical gap
- dipole and quadrupole moments
- polarizability tensor
- Mulliken atomic charges
- Mayer bond orders

Supported elements: **H, C, N, O, F, Si, P, S, Cl**.

Because properties come from the eigenstates of the predicted Hamiltonian
rather than from pooled atomic features, the size dependence is computed, not
learned. HARP extrapolates beyond the training size range, for example from
≤ 24-atom training molecules to 58-atom octithiophene.

> W. He, X. Chen, N. Song, H. Xu, T. S. Hindges, B. Li, Z. Lin, Y. Yao,
> A. R. Harutyunyan, F. Liu, Y. Wang, H. Tang, J. Li,
> *Coupled-cluster molecular properties across the main group that
> extrapolate beyond training size* (2026).

HARP builds on the effective-Hamiltonian approach of MEHnet for hydrocarbons
(Tang *et al.*, *Nat. Comput. Sci.* **5**, 144–154, 2025). HARP was called
MEHnet-MG in arXiv v1 of the paper.

## Repository layout

| path | contents |
|---|---|
| `src/` | model, data loader, training and inference code |
| `models/HARP/` | released configuration, `test.py` (evaluation), `run_ham_chain.py` (arbitrary geometries), training launchers |
| `orca_templates/` | the ORCA 6.0 inputs used to generate every reference label and DFT baseline (see below) |
| `dataset_filter/` | model-in-the-loop outlier screen used to clean the dataset, and the list of removed molecules |
| `scripts/` | accuracy tables versus DFT, SI statistics, GPU inference timing |
| `examples/oligothiophene/` | geometry builder for the out-of-distribution oligothiophene test |

## Installation

You need Python ≥ 3.9 and PyTorch ≥ 2.x. Then install the dependencies:

```bash
pip install e3nn numpy scipy sympy tqdm pyscf torch_cluster
```

The `torch_cluster` wheel must match your torch/CUDA build; see
[pytorch_cluster](https://github.com/rusty1s/pytorch_cluster). You do not
need to install this repo: the entry scripts add `src/` to the import path.
On NERSC Perlmutter, `module load pytorch/2.6.0-1 && source ./env.sh` is
enough.

Two things need [ORCA 6.0](https://www.faccts.de/orca/): inference on new
geometries (for the B3LYP baseline) and regenerating labels. Put `orca` on
your `PATH`, or set `ORCA_BIN`.

## Trained weights

The released checkpoint is `1265_model.pt` (epoch 1265, the validation
minimum; 36 MB). It is attached to the
[v1.0 release](https://github.com/He-Wenhao/HARP/releases/tag/v1.0):

```bash
mkdir -p models/HARP/output/model
curl -L -o models/HARP/output/model/1265_model.pt \
  https://github.com/He-Wenhao/HARP/releases/download/v1.0/1265_model.pt
```

`models/HARP/config.ini` already sets `[test] model_file = 1265_model.pt`.

## Evaluate on the test set

1. Place the 959-molecule test set under
   `dataset/P_S_Cl_dataset_v3/P_S_Cl_data_1_1000/`. The test set contains the
   structures and every coupled-cluster label. It will be archived on Zenodo
   upon publication; until then it is available from the corresponding
   author.
2. Run:

   ```bash
   python models/HARP/test.py                      # -> models/HARP/output/test/test.json
   python scripts/compare_methods.py --model-dir models/HARP   # MAE/RMSE vs BP86, B3LYP, DSD-PBEP86
   python scripts/si_tables.py --model-dir models/HARP         # error distributions, tensor channels, per element
   ```

`test.json` holds `pred` and `label` lists for each property over the test
molecules, and `file_names` gives the molecule order. The config uses
`device = cuda:0`. For CPU, set `[general] device = cpu`, which is much slower.

## Inference on new geometries

The Hamiltonian read-out needs AO operator matrices from a B3LYP/def2-SVP
single point. `run_ham_chain.py` runs that single point in ORCA and then
applies the model:

```bash
# out-of-distribution oligothiophenes T1..T8 (paper Fig. 5)
python models/HARP/run_ham_chain.py --series oligothiophene --n-max 8 --device cpu
# frontier-orbital diagnostics of the predicted Hamiltonian (SI Fig. S2)
python models/HARP/run_ham_chain.py --series oligothiophene --orbitals --n-max 8 --device cpu
# re-run a dataset molecule from its stored geometry
python models/HARP/run_ham_chain.py --validate 143
```

The neural part of inference takes about 0.1 s per molecule on an A100
(`scripts/measure_inference_time.py`). The B3LYP single point dominates the
total cost.

## Reference data protocol

`orca_templates/` holds the ORCA inputs used for the dataset, extracted from the
production runs (identical for every molecule). Only the geometry file name
was replaced by `molecule.xyz`.

| label | template |
|---|---|
| energy, multipoles, charges, bond orders (composite CCSD(T)/cc-pVTZ) | `composite_1_CCSD_T_cc-pVDZ.inp` + DLPNO-CCSD(T) cc-pVDZ → cc-pVTZ basis correction (`composite_2_*`, `composite_3_*`) |
| polarizability | `polarizability_finite-field_CCSD_cc-pVDZ.inp` (ORCA compound script, dipole finite differences) |
| optical gap | `optical_gap_EOM-CCSD_cc-pVDZ.inp` (lowest of 3 roots) |
| IP / EA | `ionization_*.inp`, `electron_affinity_*.inp` |
| Δ-learning baselines | `baseline_B3LYP_def2-SVP.inp` (released model), `baseline_BP86_def2-SVP.inp` (ablation) |
| DFT comparator | `comparator_DSD-PBEP86.inp` |

The `%pal nprocs` and `%maxcore` settings are those of the production
runs; adjust them to your machine.

## Training

```bash
sbatch models/HARP/submit.sh   # set #SBATCH -A to your allocation first
```

`submit.sh` detects the node count, writes `world_size` into `config.ini`,
and launches `torchrun` across nodes. To train inside an existing allocation,
run `bash models/HARP/submit.sh` instead. Checkpoints go to
`models/HARP/output/model/`. To resume, set `load_model = True` and
`load_model_path` in `config.ini`.

## License

MIT; see [LICENSE](LICENSE). Code adapted from third-party projects under
`src/model/from_*` keeps its original license; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
