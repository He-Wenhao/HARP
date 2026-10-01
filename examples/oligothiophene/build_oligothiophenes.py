"""Oligothiophene geometry builder for the size-extrapolation test.

Generates **α,α'-linked planar s-trans oligothiophenes** T_n:

    H–(C4H2S)_n–H        # n thiophene rings, α-linked, planar, s-trans,
                         # alternating S up / S down.

Why this molecule:
  * Sulfur-rich π-conjugated chain — exercises v8's sulfur coverage,
    which is rare for ML interatomic potentials.
  * Iconic in organic electronics (P3HT-style donor, OLED / OPV / OFET
    semiconductor building block). Long oligothiophenes are the standard
    model system for π-conjugation-driven gap closing in chemistry.
  * Extended π-conjugation → α grows super-linearly with n, HOMO–LUMO
    gap closes with 1/n, both well-studied in the literature.
  * Existing CCSD-level benchmarks on T_1–T_3 in the literature; T_4+
    rapidly become CCSD-prohibitive, which is exactly the regime where
    v8 demonstrates its value.

Atom count grows from 9 (T_1) to 58 (T_8) — straddling the training
distribution and well past it, with the heavy-atom count (5n) growing
linearly.

Geometry conventions (planar in xy-plane, chain along +x):

    Thiophene ring parameters (gas-phase MW geometry):
        d(S–Cα)      = 1.714 Å
        d(Cα–Cβ)     = 1.370 Å        (C=C)
        d(Cβ–Cβ')    = 1.423 Å        (C–C, partial double in aromatic)
        d(Cα–Cα')    = 1.448 Å        (inter-ring single bond)
        d(C–H)       = 1.080 Å
        ∠C–S–C       = 92.2°
        ∠S–Cα–Cβ     = 111.5°
        ∠Cα–Cβ–Cβ'   = 112.5°

    The s-trans configuration alternates sulfur up/down between adjacent
    rings (lowest-energy planar conformer for unsubstituted T_n in gas
    phase). Inter-ring C–H protons are removed at α positions where the
    rings link.
"""
import math
from typing import List, Tuple

# Bond lengths (Å)
R_SC      = 1.714
R_CC_DB   = 1.370
R_CC_SB   = 1.423
R_LINK    = 1.448      # inter-ring α–α' single bond
R_CH      = 1.080

# Internal angles (degrees)
ANG_CSC   = 92.2       # ∠C–S–C
ANG_SCC   = 111.5      # ∠S–Cα–Cβ
ANG_CCC   = 112.5      # ∠Cα–Cβ–Cβ'


def _thiophene_ring(s_up: bool) -> Tuple[List[str], List[List[float]]]:
    """Return one thiophene ring with C2 at the left, C5 at the right,
    both on the x-axis. If `s_up` is True, the sulfur is above (+y);
    otherwise it is below.

    The returned list always has 9 atoms in this order:
        S, C2, C3, C4, C5, H2, H3, H4, H5
    Caller deletes H2 / H5 at link positions as needed."""
    # C2-C5 bisector at origin: place C2 at (-dx,0), C5 at (+dx,0) with
    # S above the midpoint at (0, dy).
    half = math.radians(ANG_CSC / 2.0)
    dx = R_SC * math.sin(half)
    dy = R_SC * math.cos(half)
    sgn = 1.0 if s_up else -1.0

    C2 = [-dx, 0.0,         0.0]
    C5 = [+dx, 0.0,         0.0]
    S  = [0.0, sgn * dy,    0.0]

    # C3 is bonded to C2 with C2-C3 = R_CC_DB; angle ∠S-C2-C3 = ANG_SCC.
    # Vector from C2 to S:
    cs = (S[0] - C2[0], S[1] - C2[1])
    csn = math.hypot(*cs)
    csx, csy = cs[0] / csn, cs[1] / csn
    # Rotate (csx, csy) by ANG_SCC in the direction that places C3 on
    # the same y-side as the ring base (away from S). The correct
    # rotation sign depends on s_up; pick sign such that C3 ends up on
    # the opposite side of the x-axis to S.
    theta = math.radians(ANG_SCC)
    # Try one sign first; flip if C3 ends up on the wrong y-side.
    for sign in (-1.0, +1.0):
        ct = math.cos(sign * theta); st = math.sin(sign * theta)
        rx, ry = csx * ct - csy * st, csx * st + csy * ct
        cand = [C2[0] + R_CC_DB * rx, C2[1] + R_CC_DB * ry, 0.0]
        if (sgn > 0 and cand[1] < 0) or (sgn < 0 and cand[1] > 0):
            C3 = cand
            break
    # C4 is the mirror of C3 across the y-axis (ring is C2v symmetric).
    C4 = [-C3[0], C3[1], 0.0]

    # Hydrogens: each on its C atom, opposite the bisector of the two
    # adjacent ring atoms. _h_outward returns a position at R_CH.
    def _h_outward(Cn, nb1, nb2):
        v1 = [nb1[0] - Cn[0], nb1[1] - Cn[1]]
        v2 = [nb2[0] - Cn[0], nb2[1] - Cn[1]]
        n1 = math.hypot(*v1); n2 = math.hypot(*v2)
        v1 = [v1[0] / n1, v1[1] / n1]
        v2 = [v2[0] / n2, v2[1] / n2]
        inward = [v1[0] + v2[0], v1[1] + v2[1]]
        ni = math.hypot(*inward)
        outward = [-inward[0] / ni, -inward[1] / ni]
        return [Cn[0] + R_CH * outward[0],
                Cn[1] + R_CH * outward[1],
                0.0]

    H2 = _h_outward(C2, S, C3)
    H3 = _h_outward(C3, C2, C4)
    H4 = _h_outward(C4, C3, C5)
    H5 = _h_outward(C5, S, C4)

    elements = ["S", "C", "C", "C", "C", "H", "H", "H", "H"]
    coords = [S, C2, C3, C4, C5, H2, H3, H4, H5]
    return elements, coords


def OLIGOTHIOPHENE(n: int) -> Tuple[List[str], List[List[float]]]:
    """Build the n-ring planar s-trans oligothiophene T_n.

    α-linked, all-planar, alternating sulfur up/down. Connecting
    α-hydrogens (the ring H attached to C2 or C5 at the link position)
    are deleted so the inter-ring single bond C5–C2' is the only thing
    connecting adjacent rings.

    Returns (elements, coords_ang).

    Atom counts:
      T1:  9 atoms (C4H4S)
      Tn (n ≥ 2):  9n − 2(n−1) − 2 = 7n + 0... let me redo —
        Each ring contributes 9 atoms. Internal rings lose both
        terminal H's (2 H per internal ring). Terminal rings lose one
        H each (the H on the linking α-carbon).
        Total = 9n − 2 (internal × n−2) − 2 (terminals × 1 each)
              = 9n − 2(n−2) − 2 = 9n − 2n + 4 − 2 = 7n + 2.
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1; got {n}")

    # x_offset between successive ring origins (C2-C5 midpoint).
    # half-C5-to-next-C2 = (R_link + 2·dx_within_ring) / 2, but the
    # rings are centered, so the per-ring period is 2·dx + R_link
    # (intra-ring C2→C5 distance + inter-ring linker).
    half = math.radians(ANG_CSC / 2.0)
    dx_ring = R_SC * math.sin(half)
    period  = 2.0 * dx_ring + R_LINK

    elements_all: List[str]      = []
    coords_all:   List[List[float]] = []

    for i in range(n):
        s_up = (i % 2 == 0)  # ring 0 has S up; ring 1 S down; etc.
        elements, coords = _thiophene_ring(s_up)
        x_shift = i * period

        # Translate every atom along +x by x_shift.
        coords = [[c[0] + x_shift, c[1], c[2]] for c in coords]

        # Indices within the ring (order: S, C2, C3, C4, C5, H2, H3, H4, H5)
        # Linking H's:
        #   H2 (index 5) is the H attached to C2 (leftmost α C, links to
        #                  ring i-1 when i > 0)
        #   H5 (index 8) is the H attached to C5 (rightmost α C, links
        #                  to ring i+1 when i < n-1)
        drop = set()
        if i > 0:
            drop.add(5)        # H2 — connected to previous ring
        if i < n - 1:
            drop.add(8)        # H5 — connects to next ring

        for j, (el, co) in enumerate(zip(elements, coords)):
            if j in drop:
                continue
            elements_all.append(el)
            coords_all.append(co)

    return elements_all, coords_all


def formula(elements: List[str]) -> str:
    c = {}
    for el in elements:
        c[el] = c.get(el, 0) + 1
    order = ("C", "H", "N", "O", "F", "Si", "P", "S", "Cl")
    parts = [f"{el}{c[el]}" for el in order if el in c]
    return "".join(parts)


def chain_axis_length(coords: List[List[float]]) -> float:
    xs = [p[0] for p in coords]
    return max(xs) - min(xs)


if __name__ == "__main__":
    # Quick sanity check / demo
    for n in range(1, 9):
        els, crd = OLIGOTHIOPHENE(n)
        print(f"T{n}: N_atoms={len(els):3d}  formula={formula(els):14s}  "
              f"L_x={chain_axis_length(crd):6.2f} Å")
