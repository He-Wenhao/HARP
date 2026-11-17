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
| `scripts/` | test-input builder (`build_basic_b3lyp.py`), Table 1 (`table1.py`), accuracy tables versus DFT, SI statistics, GPU inference timing |
| `data/` | `table1_dft_recompute.tar.gz`: the DFT comparator data behind Table 1 (also on Zenodo) |
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
   ([10.5281/zenodo.22006186](https://doi.org/10.5281/zenodo.22006186), all versions) and
   unpack them into the test group directory:

   ```bash
   G=dataset/P_S_Cl_dataset_v3/P_S_Cl_data_1_1000
   mkdir -p $G
   curl -L -o main-group-cc_test-set-full-inputs.tar.gz \
     https://zenodo.org/records/22006187/files/main-group-cc_test-set-full-inputs.tar.gz
   tar -xzf main-group-cc_test-set-full-inputs.tar.gz --no-same-owner -C $G
   ```

   This gives `basic/` (BP86/def2-SVP Fock, overlap and polarizability),
   `obs/` (coupled-cluster labels), `obs_B3LYP/` and `obs_bp86/`
   (def2-SVP Δ-learning baselines) and `obs_DSD_PBEP86/` (def2-TZVP) for the
   959 test molecules. The Table 1 DFT comparators are a separate recompute;
   see "Reproduce Table 1" below.
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
   python scripts/table1.py --model-dir models/HARP            # Table 1 (see below)
   python scripts/si_tables.py --model-dir models/HARP         # SI accuracy tables
   python scripts/compare_methods.py --model-dir models/HARP   # MAE/RMSE vs the def2-SVP/def2-TZVP baselines
   ```

`test.json` holds `pred` and `label` lists for each property over the test
molecules, and `file_names` gives the molecule order. The config uses
`device = cuda:0`. For CPU, set `[general] device = cpu`, which is much slower.

### Reproduce Table 1

`scripts/table1.py` prints Table 1 and compares every cell with the printed
value. The HARP column comes from `test.json`. The DFT columns come from
`data/table1_dft_recompute.tar.gz` (also on Zenodo), in which each functional
was recomputed at the basis of the property's coupled-cluster reference:
cc-pVTZ for the energy (compared as an atomization energy, with cc-pVTZ
single-atom energies), dipole, quadrupole, charges and bond orders; cc-pVDZ
for the polarizability and the optical gap. The inputs are
`orca_templates/table1_*.inp`. Dipole errors are the norm of the error vector
in Debye and quadrupole errors the Frobenius norm of the error tensor, as in
the paper.

The printed DFT columns were aggregated over 962 molecules: the 959 test
molecules plus three molecules (325, 772, 889) that the outlier screen
removed from the test set (`dataset_filter/outliers_summary.txt`). The HARP
column is over the 959 test molecules. The data package includes the
coupled-cluster labels and geometries of the three molecules.

```bash
python scripts/table1.py --model-dir models/HARP              # --set paper (default): 962 for DFT, as printed
python scripts/table1.py --model-dir models/HARP --set test   # all columns over the 959 test molecules
```

| property | BP86 | B3LYP | DSD-PBEP86 | HARP |
|---|---|---|---|---|
| E (kcal/mol) | 63.0 (62.9) | 13.2 | 15.1 | 0.23 |
| E_gap (eV) | 1.32 | 0.61 | 4.84 | 0.067 |
| \|μ\| (Debye) | 0.160 (0.157) | 0.159 (0.158) | 0.109 | 0.022 |
| ‖Q‖ (a.u.) | 0.389 (0.385) | 0.347 (0.342) | 0.233 (0.230) | 0.044 |
| α_iso (a.u.) | 4.71 (4.68) | 2.79 (2.76) | 1.18 | 0.121 |
| q (e) | 0.0205 | 0.0256 | 0.0220 | 0.0053 |
| bond order | 0.0556 (0.0555) | 0.0588 | 0.0251 | 0.0066 |

Printed values; in parentheses the 959-molecule value where it differs at the
printed precision. The HARP column can vary in the fourth significant figure
between GPU runs.

`compare_methods.py` compares against the DFT observables shipped with the
test set instead (BP86 and B3LYP at def2-SVP, DSD-PBEP86 at def2-TZVP), with
per-component dipole and quadrupole errors in a.u. Its DFT columns are
therefore larger than in Table 1; its model column agrees with Table 1 for
E, E_gap, α_iso, q and bond order.

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
was replaced by `molecule.xyz`. The dataset labels and baselines were computed
with ORCA 6.0, the Table 1 DFT comparators with ORCA 6.1.1.

| label | template |
|---|---|
| energy, multipoles, charges, bond orders (composite CCSD(T)/cc-pVTZ) | `composite_1_CCSD_T_cc-pVDZ.inp` + DLPNO-CCSD(T) cc-pVDZ → cc-pVTZ basis correction (`composite_2_*`, `composite_3_*`) |
| polarizability | `polarizability_finite-field_CCSD_cc-pVDZ.inp` (ORCA compound script, dipole finite differences) |
| optical gap | `optical_gap_EOM-CCSD_cc-pVDZ.inp` (lowest of 3 roots) |
| IP / EA | `ionization_*.inp`, `electron_affinity_*.inp` |
| Δ-learning baselines | `baseline_B3LYP_def2-SVP.inp` (released model), `baseline_BP86_def2-SVP.inp` (ablation) |
| DSD-PBEP86/def2-TZVP observables (`obs_DSD_PBEP86/` in the test-set archive) | `comparator_DSD-PBEP86.inp` |
| Table 1 DFT comparators, cc-pVTZ (E, μ, Q, q, bond order) | `table1_BP86_cc-pVTZ.inp`, `table1_B3LYP_cc-pVTZ.inp`, `table1_DSD-PBEP86_cc-pVTZ_relaxed.inp` (MP2-relaxed density) |
| Table 1 DFT comparators, cc-pVDZ (α, gap) | `table1_{BP86,B3LYP,DSD-PBEP86}_cc-pVDZ_polar.inp` |
| Table 1 atomization-energy reference | `table1_single-atom_{BP86,B3LYP,DSD-PBEP86}_cc-pVTZ.inp` |

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
