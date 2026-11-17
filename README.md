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
| `scripts/` | test-input builder (`build_basic_b3lyp.py`), accuracy tables versus DFT, SI statistics, GPU inference timing |
| `examples/oligothiophene/` | geometry builder for the out-of-distribution oligothiophene test |

## Installation

You need Python ≥ 3.9 and PyTorch ≥ 2.x. Then install the dependencies:

```bash
pip install e3nn numpy scipy sympy tqdm pyscf periodictable matplotlib \
    "torch_geometric<2.8" torch_scatter torch_cluster huggingface_hub pydantic
pip install "fairchem-core==1.10.0" --no-deps   # eSEN/EquiformerV2 backbone
```

The `torch_scatter` and `torch_cluster` wheels must match your torch/CUDA
build; see [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster).
`fairchem-core` is installed without its dependencies; `huggingface_hub` and
`pydantic` are the ones it needs here. `torch_geometric` 2.6.1 and 2.7.0 are
tested; 2.8 routes `radius_graph` through `pyg-lib>=0.6`, which older torch
builds lack. You do not need to install this repo: the entry scripts add
`src/` to the import path.
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

1. Download the test-set inputs from Zenodo
   ([10.5281/zenodo.22006187](https://doi.org/10.5281/zenodo.22006187)) and
   unpack them into the test group directory:

   ```bash
   G=dataset/P_S_Cl_dataset_v3/P_S_Cl_data_1_1000
   mkdir -p $G
   curl -L -o main-group-cc_test-set-full-inputs.tar.gz \
     https://zenodo.org/records/22006187/files/main-group-cc_test-set-full-inputs.tar.gz
   tar -xzf main-group-cc_test-set-full-inputs.tar.gz --no-same-owner -C $G
   ```

   This gives `basic/` (BP86/def2-SVP Fock, overlap and polarizability),
   `obs/` (coupled-cluster labels) and `obs_B3LYP/`, `obs_bp86/`,
   `obs_DSD_PBEP86/` (DFT baselines and comparators) for the 959 test
   molecules.
2. Build the B3LYP model inputs:

   ```bash
   python scripts/build_basic_b3lyp.py $G      # -> $G/basic_B3LYP/
   ```

   The released model starts from B3LYP, so `config.ini` reads
   `basic_B3LYP/` (`basic_subdir = basic_B3LYP`). The script assembles it
   from `basic/` (geometry) and `obs_B3LYP/` (B3LYP Fock, overlap,
   polarizability); no ORCA run is needed. **Do not point the model at
   `basic/` or symlink `basic` to `basic_B3LYP`.** The BP86 inputs run
   without error, and the energy still comes out right, but the gap,
   multipoles, polarizability, charges and bond orders are wrong.
   `test.py` checks the inputs against `obs_B3LYP/` and stops if they do
   not match.

   The loader caches preprocessed molecules in `$G/tmp_basic_b3lyp/` and
   reuses them without re-reading `basic_B3LYP/`. If you ran `test.py` before
   building `basic_B3LYP/` correctly, delete that directory first.
3. Run:

   ```bash
   python models/HARP/test.py                      # -> models/HARP/output/test/test.json
   python scripts/compare_methods.py --model-dir models/HARP   # MAE/RMSE vs BP86, B3LYP, DSD-PBEP86
   python scripts/si_tables.py --model-dir models/HARP         # error distributions, tensor channels, per element
   ```

`test.json` holds `pred` and `label` lists for each property over the test
molecules, and `file_names` gives the molecule order. The config uses
`device = cuda:0`. For CPU, set `[general] device = cpu`, which is much slower.

The DFT columns of `compare_methods.py` use the observables shipped with the
test set: BP86 and B3LYP at def2-SVP, DSD-PBEP86 at def2-TZVP. They are
larger than the DFT columns of Table 1, where each functional was recomputed
at the basis of the property's coupled-cluster reference (cc-pVTZ for energy,
multipoles, charges and bond orders; cc-pVDZ for polarizability and gap).
The model column is unaffected. `compare_methods.py` reports dipole and
quadrupole errors per Cartesian component in a.u., whereas Table 1 reports
|μ| in Debye and ‖Q‖.

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
