import os
import re
import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional


_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


@dataclass(frozen=True)
class OrcaRunConfig:
    orca_bin: str = "orca"
    nprocs: int = 1
    maxcore_mb: int = 3000
    charge: int = 0
    multiplicity: int = 1
    method: str = "B3LYP"
    basis: str = "def2-SVP"
    tight_scf: bool = True
    grid: Optional[str] = None  # e.g. "Grid5 FinalGrid6"
    # Optional: align total energy by replacing single-atom B3LYP energies
    # with single-atom CCSD(T) energies (as used in the provided dataset).
    align_energy_to_ccsdt: bool = True
    e_single_json: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "E_single.json"
    )


def _atomic_symbol_ok(sym: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][a-z]?", sym))


def write_orca_input(
    inp_path: str,
    elements: List[str],
    coordinates_ang: List[List[float]],
    cfg: OrcaRunConfig,
) -> None:
    if len(elements) != len(coordinates_ang):
        raise ValueError("elements and coordinates length mismatch")
    for s in elements:
        if not _atomic_symbol_ok(s):
            raise ValueError(f"Invalid element symbol: {s!r}")

    # Match the reference template as closely as possible:
    #
    #   ! B3LYP def2-SVP
    #   %maxcore 2500
    #   %pal
    #   nprocs 7
    #   end
    #   %scf
    #   AutoTRAH false
    #   maxiter 1000
    #   end
    #   %ELPROP
    #   Origin CenterOfNucCharge
    #   dipole true
    #   quadrupole true
    #   Polar true
    #   END
    #   % OUTPUT
    #   Print[ P_Overlap ] 1
    #   Print[ P_Iter_F ] 1
    #   END
    #
    scf_line = f"! {cfg.method} {cfg.basis}"
    lines = []
    lines.append(scf_line)
    lines.append(f"%maxcore {int(cfg.maxcore_mb)}")
    lines.append("%pal")
    lines.append(f"nprocs {int(cfg.nprocs)}")
    lines.append("end")
    lines.append("%scf")
    lines.append("AutoTRAH false")
    lines.append("maxiter 1000")
    lines.append("end")
    lines.append("%ELPROP")
    lines.append("Origin CenterOfNucCharge")
    lines.append("dipole true")
    lines.append("quadrupole true")
    lines.append("Polar true")
    lines.append("END")
    lines.append("%OUTPUT")
    lines.append("Print[ P_Overlap ] 1")
    lines.append("Print[ P_Iter_F ] 1")
    lines.append("END")
    lines.append(f"* xyz {int(cfg.charge)} {int(cfg.multiplicity)}")
    for el, (x, y, z) in zip(elements, coordinates_ang):
        lines.append(f"  {el}  {x:.10f}  {y:.10f}  {z:.10f}")
    lines.append("*")

    with open(inp_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_orca(
    workdir: str,
    inp_filename: str = "run.inp",
    stdout_filename: str = "log",
    cfg: OrcaRunConfig = OrcaRunConfig(),
    timeout_s: Optional[int] = None,
) -> None:
    os.makedirs(workdir, exist_ok=True)
    inp_path = os.path.join(workdir, inp_filename)
    if not os.path.exists(inp_path):
        raise FileNotFoundError(inp_path)

    orca_bin = cfg.orca_bin
    if not os.path.isabs(orca_bin):
        resolved = shutil.which(orca_bin)
        if resolved is None:
            raise FileNotFoundError(f"ORCA binary not found in PATH: {orca_bin!r}")
        orca_bin = resolved

    out_path = os.path.join(workdir, stdout_filename)
    with open(out_path, "w") as out:
        subprocess.run(
            [orca_bin, inp_filename],
            cwd=workdir,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout_s,
        )

    # ORCA versions differ on the separator used for the property file name.
    # We accept both.
    prop_path_us = os.path.join(workdir, "run_property.txt")
    prop_path_dot = os.path.join(workdir, "run.property.txt")
    if not (os.path.exists(prop_path_us) or os.path.exists(prop_path_dot)):
        raise RuntimeError(
            f"Expected ORCA property file missing: {prop_path_us} (or {prop_path_dot}). "
            f"Check ORCA output in {out_path}."
        )


def _get_property_file_path(workdir: str) -> str:
    p_us = os.path.join(workdir, "run_property.txt")
    if os.path.exists(p_us):
        return p_us
    p_dot = os.path.join(workdir, "run.property.txt")
    if os.path.exists(p_dot):
        return p_dot
    raise FileNotFoundError(f"No ORCA property file found in {workdir}")


def _read_text(path: str) -> str:
    with open(path, "r", errors="replace") as f:
        return f.read()


def parse_property_dft_energy_hartree(prop_text: str) -> float:
    # ORCA 6 property format (run.property.txt)
    m = re.search(r"&FINALEN\s+\[&Type \"Double\"\]\s+(" + _FLOAT_RE + r")", prop_text)
    if m:
        return float(m.group(1))
    m = re.search(r"&TOTALENERGY\s+\[&Type \"Double\"\]\s+(" + _FLOAT_RE + r")", prop_text)
    if m:
        return float(m.group(1))
    m = re.search(r"&SCF_ENERGY\s+\[&Type \"Double\"\]\s+(" + _FLOAT_RE + r")", prop_text)
    if m:
        return float(m.group(1))
    # Legacy ORCA 5-ish plain-text format
    m = re.search(r"Total DFT Energy .*?\s+(" + _FLOAT_RE + r")\s*$", prop_text, flags=re.M)
    if m:
        return float(m.group(1))
    m = re.search(r"Total Energy\s+(" + _FLOAT_RE + r")\s*$", prop_text, flags=re.M)
    if m:
        return float(m.group(1))
    raise ValueError("Failed to parse DFT energy from ORCA property file")


def parse_property_electronic_dipole_au(prop_text: str) -> Tuple[float, float, float]:
    # ORCA 6 property format:
    #   &DIPOLETOTAL [&Type "ArrayOfDoubles", &Dim (3,1)] "Total"
    #   0  <x>
    #   1  <y>
    #   2  <z>
    m = re.search(
        r"&DIPOLETOTAL\s+\[&Type \"ArrayOfDoubles\", &Dim \(3,1\)\].*?\n"
        r"\s*0\s*\n\s*\n"
        r"0\s+(" + _FLOAT_RE + r")\s*\n"
        r"1\s+(" + _FLOAT_RE + r")\s*\n"
        r"2\s+(" + _FLOAT_RE + r")\s*\n",
        prop_text,
        flags=re.M,
    )
    if m:
        # Match the historical dataset convention for dipoles:
        # obs_B3LYP stores the electronic dipole (negative of total).
        return (-float(m.group(1)), -float(m.group(2)), -float(m.group(3)))

    # Legacy ORCA 5-ish format (run_property.txt)
    m = re.search(
        r"Electronic Contribution:\s*\n"
        r"\s*0\s*\n"
        r"\s*0\s+(" + _FLOAT_RE + r")\s*\n"
        r"\s*1\s+(" + _FLOAT_RE + r")\s*\n"
        r"\s*2\s+(" + _FLOAT_RE + r")\s*\n",
        prop_text,
        flags=re.M,
    )
    if m:
        return (-float(m.group(1)), -float(m.group(2)), -float(m.group(3)))

    raise ValueError("Failed to parse dipole from ORCA property file")


def parse_property_electronic_quadrupole_au(prop_text: str) -> Dict[str, float]:
    # ORCA 6 property format:
    #   &QUADELECCONTRIB ... Order: XX, YY, ZZ, XY, XZ, YZ
    m = re.search(
        r"&QUADELECCONTRIB\s+\[&Type \"ArrayOfDoubles\", &Dim \(6,1\)\].*?\n"
        r"\s*0\s*\n\s*\n"
        r"0\s+(" + _FLOAT_RE + r")\s*\n"
        r"1\s+(" + _FLOAT_RE + r")\s*\n"
        r"2\s+(" + _FLOAT_RE + r")\s*\n"
        r"3\s+(" + _FLOAT_RE + r")\s*\n"
        r"4\s+(" + _FLOAT_RE + r")\s*\n"
        r"5\s+(" + _FLOAT_RE + r")\s*\n",
        prop_text,
        flags=re.M,
    )
    if m:
        # NOTE: Despite the inline comment in some ORCA outputs, the ordering in
        # our stored dataset (`obs_B3LYP/*.json`) matches: XX, YY, ZZ, XY, YZ, XZ.
        # (Verified by comparing against `dataset/.../obs_B3LYP/1.json`.)
        xx, yy, zz, xy, yz, xz = [float(m.group(i)) for i in range(1, 7)]
        # Historical dataset convention: store - (electronic contribution)
        return {"xx": -xx, "yy": -yy, "zz": -zz, "xy": -xy, "xz": -xz, "yz": -yz}

    # Legacy ORCA 5-ish format (run_property.txt)
    m = re.search(
        r"Electronic part.*?\n"
        r"\s*0\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s*\n"
        r"\s*1\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s*\n"
        r"\s*2\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s*\n",
        prop_text,
        flags=re.M,
    )
    if m:
        q = [[float(m.group(i)) for i in range(1, 4)],
             [float(m.group(i)) for i in range(4, 7)],
             [float(m.group(i)) for i in range(7, 10)]]
        xx, xy, xz = (-q[0][0], -q[0][1], -q[0][2])
        yx, yy, yz = (-q[1][0], -q[1][1], -q[1][2])
        zx, zy, zz = (-q[2][0], -q[2][1], -q[2][2])
        return {"xx": xx, "yy": yy, "zz": zz, "xy": 0.5 * (xy + yx), "xz": 0.5 * (xz + zx), "yz": 0.5 * (yz + zy)}

    raise ValueError("Failed to parse quadrupole from ORCA property file")


def parse_log_quadrupole_electronic_row_au(log_text: str) -> Dict[str, float]:
    # Parse the EL row from the QUADRUPOLE MOMENT table in the ORCA main log.
    #
    # Example (from user generation log):
    #   XX  YY  ZZ  XY  XZ  YZ
    #   NUC ...
    #   EL  -575.41 ...  -0.206066162  0.000194381
    # Dataset convention stores - (EL row), so that xz in obs_B3LYP is +0.206...
    m = re.search(
        r"QUADRUPOLE MOMENT[\s\S]*?\n\s*XX\s+YY\s+ZZ\s+XY\s+XZ\s+YZ\s*\n"
        r"[\s\S]*?\n\s*EL\s+("
        + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+("
        + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")",
        log_text,
        flags=re.M,
    )
    if not m:
        raise ValueError("Failed to parse quadrupole EL row from ORCA log")
    xx, yy, zz, xy, xz, yz = [float(m.group(i)) for i in range(1, 7)]
    return {"xx": -xx, "yy": -yy, "zz": -zz, "xy": -xy, "xz": -xz, "yz": -yz}


def _maybe_align_energy_to_ccsdt(raw_energy_h: float, elements: List[str], cfg: OrcaRunConfig) -> float:
    if not cfg.align_energy_to_ccsdt:
        return raw_energy_h
    if not cfg.e_single_json or not os.path.exists(cfg.e_single_json):
        return raw_energy_h
    with open(cfg.e_single_json, "r") as f:
        es = json.load(f)
    if "B3LYP" not in es or "CCSDt" not in es:
        return raw_energy_h
    shift = 0.0
    for e in elements:
        shift += float(es["CCSDt"][e]) - float(es["B3LYP"][e])
    # Match dataset JSON float formatting: they store energies with ~10-13 decimals.
    # Round to 13 decimals to ensure bit-identical JSON values in unit tests.
    return round(raw_energy_h + shift, 13)

def parse_property_polarizability_au(prop_text: str) -> List[List[float]]:
    # ORCA 6 property format:
    m = re.search(
        r"&RAWCARTESIAN\s+\[&Type \"ArrayOfDoubles\", &Units \"a\.u\".*?&Dim \(3,3\)\]\s*\n"
        r"\s*0\s+1\s+2\s*\n\s*\n"
        r"0\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s*\n"
        r"1\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s*\n"
        r"2\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s*\n",
        prop_text,
        flags=re.M,
    )
    if m:
        vals = [float(m.group(i)) for i in range(1, 10)]
        return [vals[0:3], vals[3:6], vals[6:9]]

    # Legacy ORCA formats
    m = re.search(
        r"The raw cartesian tensor.*?\n"
        r"\s*(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s*\n"
        r"\s*(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s*\n"
        r"\s*(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s*\n",
        prop_text,
        flags=re.M,
    )
    if m:
        vals = [float(m.group(i)) for i in range(1, 10)]
        return [vals[0:3], vals[3:6], vals[6:9]]

    raise ValueError("Failed to parse polarizability from ORCA property file")


def parse_property_mulliken_QA(prop_text: str, natom: int) -> List[float]:
    m = re.search(
        r"\$Mayer_Population_Analysis[\s\S]*?&QA\s+\[&Type \"ArrayOfDoubles\", &Dim \(" + str(natom) + r",1\)\][\s\S]*?\n\s*0\s*\n\s*\n",
        prop_text,
        flags=re.M,
    )
    if not m:
        # Fallback: locate QA block without natom in regex.
        idx = prop_text.find("&QA")
        if idx == -1:
            raise ValueError("Failed to find QA block in ORCA property file")
        tail = prop_text[idx:].splitlines()
        vals = []
        for line in tail:
            line = line.strip()
            if re.match(r"^\d+\s+" + _FLOAT_RE + r"$", line):
                vals.append(float(line.split()[1]))
            if len(vals) >= natom:
                return vals[:natom]
        raise ValueError("Failed to parse QA charges")

    # Parse natom lines of the form: "<i>  <val>"
    start = m.end()
    tail = prop_text[start:].splitlines()
    vals = []
    for line in tail:
        line = line.strip()
        if not line:
            continue
        if line.startswith("$End"):
            break
        if re.match(r"^\d+\s+" + _FLOAT_RE + r"$", line):
            vals.append(float(line.split()[1]))
            if len(vals) == natom:
                break
    if len(vals) != natom:
        raise ValueError(f"Parsed {len(vals)} QA charges, expected {natom}")
    # Dataset stores Mulliken charges with 6 decimal places.
    return [round(float(v), 6) for v in vals]


def parse_property_mayer_bond_orders(prop_text: str) -> List[List[Any]]:
    # Parse &BONDORDERS (values) and &COMPONENTS (idxA, ZA, idxB, ZB)
    # We assume both arrays have length N and appear within $Mayer_Population_Analysis.
    idx = prop_text.find("$Mayer_Population_Analysis")
    if idx == -1:
        raise ValueError("Missing Mayer population analysis in ORCA property file")
    block = prop_text[idx:]

    m_n = re.search(r"&NBONDORDERSPRINT\s+\[&Type \"Integer\"\]\s+(\d+)", block)
    if not m_n:
        raise ValueError("Failed to parse number of bond orders")
    n = int(m_n.group(1))

    # Bond order values
    m_vals = re.search(r"&BONDORDERS\s+\[&Type \"ArrayOfDoubles\", &Dim \(" + str(n) + r",1\)\][\s\S]*?\n\s*0\s*\n\s*\n", block, flags=re.M)
    if not m_vals:
        raise ValueError("Failed to locate BONDORDERS block")
    tail_vals = block[m_vals.end():].splitlines()
    bos = []
    for line in tail_vals:
        line = line.strip()
        if re.match(r"^\d+\s+" + _FLOAT_RE + r"$", line):
            bos.append(float(line.split()[1]))
            if len(bos) == n:
                break
    if len(bos) != n:
        raise ValueError("Failed to parse all bond order values")

    # Components
    m_comp = re.search(r"&COMPONENTS\s+\[&Type \"ArrayOfIntegers\", &Dim \(" + str(n) + r",4\)\][\s\S]*?\n\s*0\s+1\s+2\s+3\s*\n\s*\n", block, flags=re.M)
    if not m_comp:
        raise ValueError("Failed to locate COMPONENTS block")
    tail_comp = block[m_comp.end():].splitlines()
    comps = []
    for line in tail_comp:
        line = line.strip()
        if re.match(r"^\d+\s+\d+\s+\d+\s+\d+\s+\d+$", line):
            parts = line.split()
            # parts[0] is bond index; then idxA, ZA, idxB, ZB
            comps.append((int(parts[1]), int(parts[3])))
            if len(comps) == n:
                break
    if len(comps) != n:
        raise ValueError("Failed to parse bond components")

    # Dataset stores Mayer bond orders rounded to 4 decimals.
    return [[a, b, round(float(bo), 4)] for (a, b), bo in zip(comps, bos)]


def parse_log_mulliken_charges(log_text: str, natom: int) -> List[float]:
    # Find the LAST Mulliken atomic charges block; parse the final natom lines.
    # We match a typical ORCA formatting:
    #   MULLIKEN ATOMIC CHARGES
    #   0  C   -0.123
    #   ...
    idx = log_text.rfind("MULLIKEN ATOMIC CHARGES")
    if idx == -1:
        idx = log_text.rfind("MULLIKEN ATOMIC CHARGE")
    if idx == -1:
        raise ValueError("Failed to locate Mulliken charges block in log")
    tail = log_text[idx:].splitlines()
    # Collect float at end of lines that start with an integer index.
    vals = []
    for line in tail:
        line = line.strip()
        if not line:
            continue
        if not re.match(r"^\d+\s+", line):
            continue
        parts = line.split()
        try:
            vals.append(float(parts[-1]))
        except Exception:
            continue
    if len(vals) < natom:
        raise ValueError(f"Parsed only {len(vals)} Mulliken charges, expected {natom}")
    return vals[-natom:]


def parse_log_mayer_bond_orders(log_text: str) -> List[List[Any]]:
    # Parse the LAST "Mayer bond orders" section.
    idx = log_text.rfind("Mayer bond orders")
    if idx == -1:
        raise ValueError("Failed to locate Mayer bond orders in log")
    lines = log_text[idx:].splitlines()

    bond_order = []
    # The historical parser (script/read/b3lyp_read.py) expects lines that contain 'B'
    # and then pack 7 fields per bond.
    # We'll implement the same logic over the raw split tokens.
    i = 0
    while i < len(lines) and "Mayer bond orders" not in lines[i]:
        i += 1
    i += 1
    while i < len(lines):
        if "B(" not in lines[i] and " B" not in lines[i] and not lines[i].lstrip().startswith("B"):
            if bond_order and lines[i].strip() == "":
                break
            i += 1
            continue
        tokens = lines[i].strip().split()
        j = 0
        while 7 * j + 7 <= len(tokens):
            a = int(tokens[7 * j + 1].rstrip(":").rstrip(")"))
            b = int(tokens[7 * j + 3].rstrip(":").rstrip(")"))
            bo = float(tokens[7 * j + 6])
            bond_order.append([a, b, bo])
            j += 1
        i += 1
    return bond_order


def parse_log_homo_lumo_gap_ev(log_text: str) -> float:
    # Prefer explicit HOMO-LUMO GAP lines, if present.
    m = re.findall(r"HOMO-LUMO GAP\s+(" + _FLOAT_RE + r")\s+eV", log_text)
    if m:
        return float(m[-1])
    m = re.findall(r"HOMO-LUMO GAP\s*\(eV\)\s*:\s*(" + _FLOAT_RE + r")", log_text)
    if m:
        return float(m[-1])

    # ORCA 6: reconstruct from ORBITAL ENERGIES table (in eV).
    m = re.search(r"ORBITAL ENERGIES[\s\S]*?NO\s+OCC\s+E\(Eh\)\s+E\(eV\)([\s\S]*?)(?:\n\s*\n|\Z)", log_text)
    if not m:
        raise ValueError("Failed to parse HOMO-LUMO gap (eV) from log")
    block = m.group(1)
    occ = []
    virt = []
    for line in block.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            occ_val = float(parts[1])
            e_ev = float(parts[3])
        except Exception:
            continue
        if occ_val > 1e-6:
            occ.append(e_ev)
        else:
            virt.append(e_ev)
    if not occ or not virt:
        raise ValueError("Failed to locate occupied/virtual orbitals for gap")
    homo = max(occ)
    lumo = min(virt)
    return lumo - homo


def parse_log_nuclear_repulsion_hartree(log_text: str) -> float:
    m = re.findall(r"Nuclear Repulsion Energy\s*:\s*(" + _FLOAT_RE + r")", log_text)
    if m:
        return float(m[-1])
    m = re.findall(r"Nuclear Repulsion\s+(" + _FLOAT_RE + r")", log_text)
    if m:
        return float(m[-1])
    # ORCA variants
    m = re.findall(r"ENN=\s*(" + _FLOAT_RE + r")\s*Eh", log_text)
    if m:
        return float(m[-1])
    m = re.findall(r"ENuc=\s*(" + _FLOAT_RE + r")\s*Eh", log_text)
    if m:
        return float(m[-1])
    raise ValueError("Failed to parse nuclear repulsion energy from log")


def compute_b3lyp_baseline_from_orca_run(workdir: str, natom: int) -> Dict[str, Any]:
    log_text = _read_text(os.path.join(workdir, "log"))
    prop_text = _read_text(_get_property_file_path(workdir))

    energy_raw = parse_property_dft_energy_hartree(prop_text)
    x, y, z = parse_property_electronic_dipole_au(prop_text)
    # Quadrupole is taken from the main ORCA log (matches dataset generation log).
    quad = parse_log_quadrupole_electronic_row_au(log_text)
    alpha = parse_property_polarizability_au(prop_text)
    atomic_charge = parse_property_mulliken_QA(prop_text, natom)
    bond_order = parse_property_mayer_bond_orders(prop_text)
    Ee = parse_log_homo_lumo_gap_ev(log_text)
    E_nn = parse_log_nuclear_repulsion_hartree(log_text)

    return {
        "energy": energy_raw,  # may be aligned by caller via cfg if desired
        "E_nn": E_nn,
        "Ee": Ee,
        "x": x,
        "y": y,
        "z": z,
        **quad,
        "alpha": alpha,
        "atomic_charge": atomic_charge,
        "bond_order": bond_order,
    }


def run_orca_b3lyp_and_parse(
    elements: List[str],
    coordinates_ang: List[List[float]],
    workdir: str,
    cfg: OrcaRunConfig = OrcaRunConfig(),
    timeout_s: Optional[int] = None,
) -> Dict[str, Any]:
    write_orca_input(os.path.join(workdir, "run.inp"), elements, coordinates_ang, cfg)
    run_orca(workdir=workdir, cfg=cfg, timeout_s=timeout_s)
    out = compute_b3lyp_baseline_from_orca_run(workdir=workdir, natom=len(elements))
    out["energy"] = _maybe_align_energy_to_ccsdt(out["energy"], elements, cfg)
    return out


def save_baseline_json(path: str, baseline: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(baseline, f)

