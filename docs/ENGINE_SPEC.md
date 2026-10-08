# ReaxMetal Engine Specification (M0 — functional form as implemented by the pinned reference)

**Status: specification only. Nothing in this document is implemented or tested yet** (one exception: the element/mass compatibility predicates of Q-06
exist as `compat_flags.hpp` with boundary tests). Revised in M0.5 for the LAMMPS-first architecture (`LAMMPS_INTEGRATION.md`): §3 ghost semantics,
§7 charge model (now named EEM), D-1/D-5, new quirks Q-26…Q-31.
Reference: LAMMPS `stable_30Sep2026`, commit `8de817dd79bfe4525d5d39246a212d833e6dee07`
(`src/REAXFF/`, `src/KOKKOS/*reaxff*`). Line numbers below refer to that commit.
Where this document and the pinned source disagree, **the source wins** and this document is a bug.

The project rule (mission rule 3) is to reproduce the *reference equations and numerical conventions*, not
the textbook ones. LAMMPS' own documentation says the same: "where the printed equations and the reference
implementation disagree, pair style reaxff follows the reference implementation" (`doc/src/pair_reaxff.rst`).
Sections 9–10 list every place where reference behaviour is surprising, with the engine's disposition.

Notation: `i,j,k,l` atoms (indices into the *periodic-image-expanded* environment, ADR-004); `r_ij` distance;
`BO'` uncorrected, `BO` corrected bond order; `Δ'`, `Δ` uncorrected/corrected deviations from valence;
`gp[n]` = n-th general (global) ffield parameter, 0-based as in `gp.l[n]`.

## 0. Scope of the supported model

Supported (planned): standard ReaxFF energy (13 terms, §5) with analytic forces; the standard **EEM** charge model — defined as the
LAMMPS-compatible model of `fix qeq/reaxff` and `fix qeq/shielded` (§7), *not* a different physics; the three vdW variants selected by ffield
contents plus the `lgvdw` correction; `enobonds`; `checkqeq no`; control-file cutoffs; global virial for NPT once validated. Deferred with
explicit errors: ACKS2, QTPIE, `qeq/rel`, external electric fields, tabulated interactions, non-neutral groups, multi-rank (FEATURE_MATRIX.md).
Time integration, thermostats, minimisation and all trajectory/atom management are **LAMMPS'** (the engine is a force/energy evaluator behind
`pair_style reaxff/metal`). No element, ffield, or chemical system is built into the engine (ADR-003).

## 1. Units, sign conventions, constants

Energy kcal/mol, length Å, charge e (LAMMPS `real` units). The reference stores **gradients** in its force
workspace (`workspace->f` = +∂E/∂x) and adds `-f` to the LAMMPS force (`pair_reaxff.cpp:685-697`).
The engine's public force is the physical force `F = −∂E/∂x`.

Constants as written in `reaxff_defs.h:34-37,39,59` — **replicated exactly, including their imprecision**:

| Symbol | Value | Used for | Remark |
|---|---|---|---|
| `constPI` | `3.14159265` | `DEG2RAD(a) = a*constPI/180` (valence angle) | not π (absolute error 3.6e-9, relative 1.1e-9) |
| `C_ele` | `332.06371` | Coulomb energy/force | kcal·Å/(mol·e²) |
| `EV_to_KCALpMOL` | `14.4` | QEq matrix elements; `LR.H` | really e²/Å in eV; name is misleading |
| `KCALpMOL_to_EV` | `23.02` | polarization energy `e_pol` | really eV→kcal/mol; exact value is 23.0605… |
| `HB_THRESHOLD` | `1e-2` | minimum `BO` for an H-bond donor bond | hard, discontinuous |
| `MIN_SINE` | `1e-10` | sin(θ) floor in torsion | |
| `SMALL` (nonbonded) | `1e-4` | tie-break when `orig_id(i)==orig_id(j)` (`reaxff_nonbonded.cpp:74`) | |
| `EPSILON` (QEq H) | `1e-4` | same tie-break in `fix_qeq_reaxff.cpp:687` | |

Further literals that are part of the model: `exp(-75Δlp)` (lone pair), `25.0` (triple-bond stabilisation),
`0.040`, `4.`, `0.16` (C2 correction), `+1e-8` (over-coordination denominator), BO floor `1e-10`,
`sin θ ≥ 1e-5` in the valence-angle derivative, `pow(x, 0.33333333333333)` in the CPU Coulomb cube root
(`reaxff_nonbonded.cpp:194,558`) but `pow(x, 1.0/3.0)` in the QEq matrix (`fix_qeq_reaxff.cpp:757`) and
`cbrt` in Kokkos (`pair_reaxff_kokkos.cpp:696,1288`).

## 2. Parameter model (ffield)

Parser reference: `reaxff_ffield.cpp:58-628`. The pinned parser **uses fixed column positions**; the
header comments in real files are not authoritative (see Q-04).

### 2.1 General parameters (`gp[0..n_global-1]`, normally 39)
Index → role, **taken from the code's use sites** (cross-checked against the label comments of
`potentials/ffield.reax.cho`; where they differ the code is used and the difference noted):

| gp | role in code | gp | role in code |
|---|---|---|---|
| 0 | `p_boc1` | 20 | `p_pen3` |
| 1 | `p_boc2` | 21 | `p_pen4` |
| 2 | `p_coa2` | 23 | `p_tor2` |
| 3 | `gp3` (triple-bond stab., label `p(trip4)`) | 24 | `p_tor3` |
| 4 | `gp4` (label `p(trip3)`) | 25 | `p_tor4` |
| 5 | `p_lp3` (C2 correction, label `kc2`) | 27 | `p_cot2` |
| 6 | `p_ovun6` | 28 | `p_vdW1` |
| 7 | `gp7` (label `p(trip2)`) | 29 | bond-order cutoff ×100 → `bo_cut = 0.01*gp[29]` |
| 8 | `p_ovun7` | 30 | `p_coa4` |
| 9 | `p_ovun8` | 31 | `p_ovun4` |
| 10 | `gp10` (label `p(trip1)`) | 32 | `p_ovun3` |
| 11 | taper lower radius `swa` (`nonb_low`) | 33 | `p_val8` |
| 12 | taper upper radius `swb` (`nonb_cut`) | 34 | ACKS2 bond softness only (unused: ACKS2 rejected) |
| 14 | `p_val6` (**file label says `p(val7)`**) | 35 | `lgre` scale (only `lgvdw`) |
| 15 | `p_lp1` | 37 | `gp37` — `(int)` cast; `==2` forces triple-bond stabilisation for all pairs |
| 16 | `p_val9` | 38 | `p_coa3` |
| 17 | `p_val10` | 13,18,22,26,36 | not used |
| 19 | `p_pen2` | | |

Unused indices are still stored (and a file may carry fewer/more than 39: `n_global` is read, `gp.l` is
sized from it). Indices ≥ `n_global` must be rejected if the engine reads them.

### 2.2 Per-element ("single body") parameters
Four lines per element, five when `lgvdw` (`reaxff_ffield.cpp:164-248`). Columns (after the symbol) →
fields. **C++ field names are used here only to cross-reference the source**; engine names are in brackets.

| Line | Columns (0-based after symbol) |
|---|---|
| 1 | `r_s`, `valency`, `mass`, `r_vdw`, `epsilon`, `gamma`, `r_pi`, `valency_e` (9 columns incl. symbol) |
| 2 | `alpha`, `gamma_w`, `valency_boc` **[val_angle]**, `p_ovun5`, *(skip)*, `chi`, `eta` **(stored ×2)**, `p_hbond` |
| 3 | `r_pi_pi`, `p_lp2`, *(skip, heat increment)*, `b_o_131`(=p_boc4), `b_o_132`(=p_boc3), `b_o_133`(=p_boc5), `bcut_acks2` |
| 4 | `p_ovun2`, `p_val3`, *(skip)*, `valency_val` **[val_boc]**, `p_val5`, `rcore2`, `ecore2`, `acore2` |
| 5 (lgvdw only) | `lgcij`, `lgre` |

Derived at parse time: `nlp_opt = 0.5*(valency_e - valency)`; `eta ← 2*eta_file`;
symbol uppercased and truncated to 3 chars; `p_hbond ∈ {0,1,2}` truncated from a double (1 = H-bond
donor hydrogen, 2 = acceptor).
**Light-element override** (`:298-305`): if `mass < 21` and `valency_val != valency_boc`, then
`valency_val ← valency_boc` (with a warning).

### 2.3 Global vdW variant (`:250-295`)
Per element: `inner = (rcore2 > 0.01 && acore2 > 0.01)`, `shield = (gamma_w > 0.5)`.
`inner&&shield → 3`, `inner&&!shield → 2`, `!inner&&shield → 1`, neither → **error**. The system-wide
`vdw_type` is set by the *first* element and conflicting later elements only produce a warning ("keeping
earlier"). Engine disposition: treat a conflict as an error (the LAMMPS result is a latent division by zero).

### 2.4 Pair ("two body") parameters
Bond block (`:307-358`): line 1 `i j De_s De_p De_pp p_be1 p_bo5 v13cor p_bo6 p_ovun1`;
line 2 `p_be2 p_bo3 p_bo4 (skip) p_bo1 p_bo2 [ovc]` (`ovc` defaults to 0). Indices are 1-based in the file.
Mixing for all `i≤j` (`:360-385`), performed **after** the bond block and **before** the off-diagonal block:

```
r_s = ½(r_s_i + r_s_j)      r_p = ½(r_pi_i + r_pi_j)        r_pp = ½(r_pipi_i + r_pipi_j)
p_boc3 = √(b132_i b132_j)   p_boc4 = √(b131_i b131_j)       p_boc5 = √(b133_i b133_j)
D = √(ε_i ε_j)              α = √(α_i α_j)                  r_vdW = 2√(rvdw_i rvdw_j)
γ_w = √(γw_i γw_j)          γ = (γ_i γ_j)^(-3/2)            rcore,ecore,acore = √(core2_i core2_j)
lgcij = √(lgcij_i lgcij_j)  lgre = 2·gp[35]·√(lgre_i lgre_j)
```
Off-diagonal block (`:387-429`): `i j D r_vdW/2 α r_s r_p r_pp [lgcij]`; a value overrides the mixed one
only if `> 0.0` (`lgcij`: `>= 0.0`); `r_vdW` is stored as `2*value`.

### 2.5 Angle, torsion, H-bond tables
* **3-body** (`:431-484`): `j k l θ00 p_val1 p_val2 p_coa1 p_val7 p_pen1 p_val4`. Stored in `thbp[j][k][l]`
  *and mirrored* into `thbp[l][k][j]`; multiple lines for the same unordered triple are **appended and
  all summed** in the energy loop (`reaxff_valence_angles.cpp:225-368`). See Q-09 for the `j==l` double
  increment and the missing bound check (`REAX_MAX_3BODY_PARAM = 5`).
* **4-body** (`:486-552`): `j k l m V1 V2 V3 p_tor1 p_cot1`. Explicit entry (`j,m ≥ 1`) sets
  `fbp[j][k][l][m]` and `fbp[m][l][k][j]`, `cnt = 1` — **only `prm[0]` is ever used**. Compact entry
  (`0 k l 0`) fills `fbp[p][k][l][o]` and `fbp[o][l][k][p]` for all `p,o` that have not been set by an
  explicit entry (`tor_flag`). Compact entries are processed in file order and *later compact entries
  overwrite earlier ones* where their (k,l)/(l,k) footprints overlap (Q-10).
* **H-bond** (`:554-586`): `i j k r0_hb p_hb1 p_hb2 p_hb3`, stored **directionally** in `hbp[i][j][k]`
  (no symmetrisation). `r0_hb` is pre-set to `-1`; `r0_hb ≤ 0` means "no hydrogen bond for this triple".
  The block may be missing (EOF → warning, hydrogen bonds then simply absent).

## 3. Geometry, cutoffs and interaction lists

Cutoffs (`reaxff_ffield.cpp:620-624`, `pair_reaxff.cpp:189-274`, `reaxff_control.cpp`):

| Name | Source | Default |
|---|---|---|
| `nonb_low` (swa), `nonb_cut` (swb) | `gp[11]`, `gp[12]` | typically 0, 10 Å |
| `bo_cut` | `0.01*gp[29]` | |
| `bond_cut` | control `nbrhood_cutoff` | 5.0 Å |
| `hbond_cut` | control `hbond_cutoff` (`0` disables H-bonds) | 7.5 Å |
| `thb_cut`, `thb_cutsq` | control `thb_cutoff`, `thb_cutoff_sq` | 0.001, 1e-5 |
| `bg_cut` | control `bond_graph_cutoff` | 0.3 (does not enter CPU energies) |

LAMMPS builds them from a half neighbor list with ghost neighbors. **Reference semantics for the engine (revised in M0.5, ADR-013):**
the engine is **ghost-native**: its input is LAMMPS' *expanded atom set* — `nlocal` owned atoms followed by `nghost` ghost atoms, each ghost an
owner plus an integer lattice shift (verified on every step, LAMMPS_INTEGRATION §4) — and it reproduces the reference's owner-computes rules
(only owned atoms are centres of angle/torsion/H-bond/vdW terms; ordering of equal-image pairs by `tag` with the coordinate tie-break;
ghost rows limited to `bond_cut`; `bond_mark`), so the results are those of LAMMPS by construction. The standalone harness (fixtures, tests, later Python)
builds the same expanded set with an **image expander** — periodic images are always a *host* concern; no kernel does minimum-image or shift arithmetic
(minimum-image is demonstrably wrong: LAMMPS_INTEGRATION §7.3). Correctness for owned atoms requires a **sufficient ghost shell**; the
engine requires `shell ≥ max(nonb_cut, hbond_cut, 2·bond_cut)` (the reference only warns, `pair_reaxff.cpp:372-375`; we error) and
experiment REF-GHOST (VALIDATION.md) determines whether this is also sufficient. Forces are returned for **all** owned+ghost atoms; the host folds ghost forces onto owners.

Bond creation (`reaxff_forces.cpp:130-252`, `BOp` `reaxff_bond_orders.cpp:148-243`): a bond i–j exists iff
`r_ij ≤ bond_cut` and `BO'_ij ≥ bo_cut`. H-bond candidates (`:204-223`): donor `i` with `p_hbond==1`,
partner `j` with `p_hbond==2`, `r ≤ hbond_cut`, stored per donor hydrogen.

## 4. Bond order

For every candidate pair with parameter set `tbp[type_i][type_j]` (`BOp`, `:148-243`):

```
BO'_s  = (1+bo_cut)·exp(p_bo1·(r/r_s)^p_bo2)    if r_s_i>0 and r_s_j>0   else 0
BO'_π  =            exp(p_bo3·(r/r_p)^p_bo4)    if r_pi_i>0 and r_pi_j>0 else 0
BO'_ππ =            exp(p_bo5·(r/r_pp)^p_bo6)   if r_pipi_i>0 and r_pipi_j>0 else 0
BO' = BO'_s + BO'_π + BO'_ππ ;  bond exists iff BO' ≥ bo_cut
then:  BO'_s ← BO'_s − bo_cut ;  BO' ← BO' − bo_cut        (only the σ part carries the offset)
```
`Δ'_i = Σ_j BO'_ij − val_i`; `Δ'boc_i = Σ_j BO'_ij − val_boc_i` (sums over the *offset-reduced* BO').

Correction (`BO`, `:245-468`) per bond, computed for `i<j` and copied for the mirror bond. If
`ovc < 0.001 && v13cor < 0.001` the correction is the identity (all `f=1`). Otherwise:

```
f1 = ½[ (val_i+f2)/(val_i+f2+f3) + (val_j+f2)/(val_j+f2+f3) ],
        f2 = e^{-p_boc1 Δ'_i} + e^{-p_boc1 Δ'_j},  f3 = -(1/p_boc2)·ln(½(e^{-p_boc2 Δ'_i}+e^{-p_boc2 Δ'_j}))   (if ovc ≥ 0.001; else f1=1)
f4 = 1/(1+exp(−(p_boc4·BO'² − Δ'boc_i)·p_boc3 + p_boc5)),  f5 likewise with Δ'boc_j                         (if v13cor ≥ 0.001; else f4=f5=1)
BO   = BO'·f1·f4·f5        BO_π = BO'_π·f1²·f4·f5       BO_ππ = BO'_ππ·f1²·f4·f5       BO_σ = BO − BO_π − BO_ππ
```
`BO'` in `f4/f5` and in the products is the **offset-reduced** value (Q-13). Values `< 1e-10` are flushed to 0.
The derivative machinery stores, per directed bond, coefficients `C1dbo,C2dbo,C3dbo` (∂BO/∂BO', ∂/∂Δ'_i, ∂/∂Δ'_j),
`C1..4dbopi`, `C1..4dbopi2` (`:387-404`) plus `dBOp`, `dln_BOp_{s,π,ππ}` (∂BO'/∂x_i) and
`dDeltap_self[i] = Σ_j dBOp_ij`.

Atomic quantities after correction (`:436-467`), with `Σ` the corrected total `total_bond_order`:
```
Δ = Σ − val          Δe = Σ − val_e        Δboc = Σ − val_angle (file "Val(angle)")        Δval = Σ − val_boc (file "Val(boc)")
vlpex = Δe − 2·trunc(Δe/2)      nlp = exp(−p_lp1·(2+vlpex)²) − trunc(Δe/2)        Δlp = nlp_opt − nlp
Clp = 2·p_lp1·exp(−p_lp1(2+vlpex)²)·(2+vlpex)
mass > 21 :  nlp_temp = nlp_opt, Δlp_temp = 0, dΔlp_temp = 0          else :  nlp_temp = nlp, Δlp_temp = Δlp, dΔlp_temp = Clp
```
`trunc` is C `(int)` truncation toward zero (Q-14), **not** floor.

## 5. Energy terms

Per-term "reference" = the function in `src/REAXFF` that defines it. Each bullet gives the energy; the
gradient bookkeeping is §6.

### 5.1 Bond energy `E_bond` (`reaxff_bonds.cpp:35-138`)
Counted once per undirected bond. With `p_be1,p_be2,De_*` from `tbp`:
```
e = −De_σ·BO_σ·exp(p_be1(1 − BO_σ^p_be2)) − De_π·BO_π − De_ππ·BO_ππ                 (BO_σ = 0 → pow term 0)
```
**Terminal triple-bond stabilisation**, added to `E_bond` when `BO ≥ 1` and (`gp[37]==2` **or** the pair is
the mass pair 12.0000/15.9990 by exact double equality, Q-06):
`E = gp10·exp(−gp7(BO−2.5)²)·[exp(−gp3(Σ_i−BO)) + exp(−gp3(Σ_j−BO))] / (1 + 25·exp(gp4(Δ_i+Δ_j)))`.

### 5.2 Lone pair `E_lp` and C2 correction (`reaxff_multi_body.cpp:68-129`)
`E_lp,i = p_lp2·Δlp_i / (1 + exp(−75·Δlp_i))` — evaluated if `numbonds>0 || enobonds`.
**C2 correction** (applies if `gp[5] > 0.001` and the central atom's symbol is `C`; Q-06): for every
*directed* bond i→j to another `C` (each C–C bond is therefore visited twice, once per end, Q-15):
`vov3 = BO − Δ_i − 0.04·Δ_i⁴`; if `vov3 > 3`: `E += gp5·(vov3−3)²`. Added to `E_lp`.

### 5.3 Over- and under-coordination (`:131-240`)
`dfvl = 0` if `mass>21` else `1`.
```
sum1 = Σ_j p_ovun1·De_σ·BO_ij              sum2 = Σ_j (Δ_j − dfvl·Δlp_temp_j)·(BO_π,ij + BO_ππ,ij)
exp1 = p_ovun3·exp(p_ovun4·sum2)           Δlpcorr = Δ_i − dfvl·Δlp_temp_i/(1+exp1)
E_ov = sum1 · Δlpcorr/(Δlpcorr + val_i + 1e-8) / (1 + exp(p_ovun2_i·Δlpcorr))
E_un = −p_ovun5_i·(1 − exp(p_ovun6·Δlpcorr)) / (1 + exp(−p_ovun2_i·Δlpcorr)) / (1 + p_ovun7·exp(p_ovun8·sum2))   (if numbonds>0 || enobonds)
```
(`p_ovun3=gp32, p_ovun4=gp31, p_ovun6=gp6, p_ovun7=gp8, p_ovun8=gp9`.)

### 5.4 Valence angle, penalty, 3-body conjugation (`reaxff_valence_angles.cpp:70-386`)
Central atom `j`, bonded neighbors `i,k` (each unordered pair once). Gate: `BOA = BO − thb_cut`;
`BO_ij,BO_jk > thb_cut`, `BOA_jk > 0`, `BO_ij·BO_jk > thb_cutsq`; parameters `thbp[type_i][type_j][type_k]`,
summed over all stored sets with `|p_val1| > 0.001`.
```
SBOp = Σ_t (BO_π,jt + BO_ππ,jt) ;  prod = Π_t exp(−BO_jt⁸)
vlpex ≥ 0 : vlpadj = 0                else vlpadj = nlp_j
SBO = SBOp + (1−prod)(−Δboc_j − p_val8·vlpadj)
SBO2 = 0 (SBO≤0) | SBO^p_val9 (0<SBO≤1) | 2 − (2−SBO)^p_val9 (1<SBO<2) | 2 (SBO≥2)
f7(BOA) = 1 − exp(−p_val3·BOA^p_val4)                         f8(Δboc_j) = p_val5 − (p_val5−1)(2+e6)/(1+e6+e7),  e6=exp(p_val6Δboc_j), e7=exp(−p_val7Δboc_j)
θ0 = DEG2RAD(180 − θ00·(1 − exp(−p_val10(2−SBO2))))
E_val = f7(BOA_ij)·f7(BOA_jk)·f8·[ p_val1·(1 − exp(−p_val2(θ0−θ)²)) ]      (p_val1 < 0 : p_val1·(−exp(...)), to avoid linear Me–H–Me)
E_pen = p_pen1·f9(Δ_j)·exp(−p_pen2(BOA_ij−2)²)·exp(−p_pen2(BOA_jk−2)²),   f9 = (2+e3)/(1+e3+e4), e3=exp(−p_pen3Δ_j), e4=exp(p_pen4Δ_j)
E_coa = p_coa1/(1+exp(p_coa2·Δval_j)) · exp(−p_coa3(Σ_i−BOA_ij)²)·exp(−p_coa3(Σ_k−BOA_jk)²)·exp(−p_coa4(BOA_ij−1.5)²)·exp(−p_coa4(BOA_jk−1.5)²)
```
`cos θ` is clamped to [−1,1] before `acos`; the θ-gradient uses `sin θ ≥ 1e-5` (Q-18).

### 5.5 Torsion and 4-body conjugation (`reaxff_torsion_angles.cpp:114-399`)
Bond `j–k` (each undirected bond once), neighbors `i` of `j` and `l` of `k`, `i ≠ l` **by atom identity as
indexed in the expanded environment**, parameters `fbp[type_i][type_j][type_k][type_l].prm[0]` (skipped if `cnt==0`).
Gate: `BO_jk, BO_ij, BO_kl > thb_cut` and `BO_ij·BO_jk·BO_kl > thb_cut` (Q-19). With
`ω = atan2(unnorm_sin, unnorm_cos)` from the four bond vectors:
```
f10 = (1−e^{−p_tor2·BOA_ij})(1−e^{−p_tor2·BOA_jk})(1−e^{−p_tor2·BOA_kl})
f11 = (2+e^{−p_tor3(Δboc_j+Δboc_k)})/(1+e^{−p_tor3(Δboc_j+Δboc_k)}+e^{p_tor4(Δboc_j+Δboc_k)})
exp_tor1 = exp(p_tor1·(2 − BO_π,jk − f11)²)
CV = ½[V1(1+cos ω) + V2·exp_tor1(1−cos 2ω) + V3(1+cos 3ω)]
E_tor = f10·sin θ_ijk·sin θ_jkl·CV
E_con = p_cot1·fn12·(1 + (cos²ω − 1)·sin θ_ijk·sin θ_jkl),   fn12 = Π e^{−p_cot2(BOA−1.5)²} over ij, jk, kl
```
(`p_tor2=gp23, p_tor3=gp24, p_tor4=gp25, p_cot2=gp27`; sin θ floored at `MIN_SINE` in the derivative.)

### 5.6 Hydrogen bond (`reaxff_hydrogen_bonds.cpp:35-158`)
Hydrogen `j` (`p_hbond==1`); `i` bonded to `j` with `p_hbond(i)==2` and `BO_ij ≥ HB_THRESHOLD`; `k` from the
H-bond list of `j` (`p_hbond(k)==2`, `r_jk ≤ hbond_cut`); skip if `i` and `k` are the **same atom** (`orig_id`);
`hbp = hbp[type_i][type_j][type_k]`, skip if `r0_hb ≤ 0`:
`E_hb = p_hb1·(1−e^{−p_hb2·BO_ij})·exp(−p_hb3(r0/r_jk + r_jk/r0 − 2))·sin⁴(θ_ijk/2)` with θ the angle at `j` between `ji` and `jk`.

### 5.7 van der Waals (`reaxff_nonbonded.cpp:62-215`)
For each unordered pair with `r ≤ nonb_cut`; `Tap(r)` the 7th-order taper (§5.9). With
`p_vdW1 = gp[28]`, pair parameters `D, α, r_vdW, γ_w, ecore, acore, rcore, lgcij, lgre`:
```
shielded (type 1,3): fn13 = (r^p_vdW1 + (1/γ_w)^p_vdW1)^(1/p_vdW1);  e1 = exp(α(1−fn13/r_vdW)), e2 = exp(½α(1−fn13/r_vdW))
unshielded (type 2): same with fn13 = r
E_vdW = Tap·D·(e1 − 2e2)
inner wall (type 2,3): E += Tap·ecore·exp(acore(1 − r/rcore))
lg (lgvdw): E += −Tap·lgcij/(r⁶ + lgre⁶)
```
**The lg term is nested inside the inner-wall branch** (`reaxff_nonbonded.cpp:168-188`): it is evaluated only
when the global `vdw_type` is 2 or 3. With a 4-line ffield LAMMPS rejects `lgvdw yes` at parse time
(`reaxff_ffield.cpp:240-243`), but a 5-line ffield of `vdw_type` 1 would have its lg parameters parsed and then
silently never used. The engine reproduces the nesting (energy identical to LAMMPS) **and** raises an
explicit error for that combination, since an unused parameter block is exactly the "silently omitted term"
rule 5 forbids. (Not yet verified against a crafted file; M2 test.)
### 5.8 Coulomb and polarization
`E_ele = C_ele·q_i·q_j·Tap / (r³ + γ)^(1/3)` (γ = pair `gamma`, i.e. (γ_iγ_j)^(−3/2)), per unordered pair.
`E_pol = Σ_i 23.02·(χ_i q_i + ½η_i q_i²)` (`η` = stored ×2 value).

### 5.9 Taper (`reaxff_init_md.cpp:72-106`)
`Tap(r) = Σ_{n=0..7} Tap[n]·rⁿ` with the 7th-order coefficients listed there for `swa=gp[11]`, `swb=gp[12]`
(`Tap[7]=20/d⁷`, …); evaluated by Horner in the order written (§NUMERICAL_POLICY). `dTap` is evaluated as the
polynomial derivative **divided by r** (`dTap += Tap[1]/r`), i.e. the code's `CEvd`/`CEclmb` are `(1/r)·dE/dr`.

## 6. Force assembly (gradient bookkeeping)

All bonded terms depend on positions only through bond orders and (for angle-like terms) explicit geometry.
The reference therefore does **not** differentiate BO inside each term; it accumulates *coefficients* and
makes one final pass (`Compute_Total_Force` → `Add_dBond_to_Forces`, `reaxff_bond_orders.cpp:36-146`):

1. Each term adds to per-directed-bond coefficients `Cdbo, Cdbopi, Cdbopi2` and to per-atom `CdDelta[i]`
   (= ∂E/∂Δ_i), and applies its *explicit-geometry* gradient straight to `f` (angle `dcos_θ`, torsion `dcos_ω`,
   H-bond `dr`, vdW/Coulomb pair).
2. After all terms, for every undirected bond (i<j): with `c = Cdbo_ij + Cdbo_ji`, etc., the gradient on
   `i`, `j`, **and every other bond partner `k` of `i` and of `j`** is added from `dBOp`, `dln_BOp_π/ππ`,
   `dDeltap_self[i|j]` and the stored `C*dbo*` coefficients. This is the same "delay the BO-derivative to the end
   of the step" scheme described in the PuReMD-GPU paper (§4.2).

For the engine this means force correctness requires the **complete coefficient set to exist before the final
pass**, which determines the kernel ordering on the GPU (ADR-006): term kernels → coefficient reductions →
bond-derivative gather kernel.

## 7. Charge equilibration (`fix_qeq_reaxff.cpp`, `fix qeq/reaxff nevery swa swb tol reaxff [maxiter N]`)

Solve `H s = −χ`, `H t = −1` with
`H_ii = η_i` (stored ×2), `H_ij = 14.4·Tap_qeq(r_ij) / (r_ij³ + (γ_iγ_j)^(−3/2))^(1/3)` for `r_ij ≤ swb`
(`:744-760`, taper from the **fix** arguments `swa,swb`, not from the ffield; Q-21), then
`q = s − (Σs/Σt)·t` (`:851-880`). Solver (`:764-816`): Jacobi-preconditioned CG, `M⁻¹ = 1/η_i`; loop while
`√(r·M⁻¹r)/‖b‖ > tol` and `i < imax` (default 200, i.e. at most 199 iterations); non-convergence is a **warning**
unless `nowarn` (Q-22). Initial guess: `s = 4(s₀+s₂) − (6s₁+s₃)`, `t = t₂ + 3(t₀−t₁)` from the last four
solutions (Q-23); histories are zero at start. Per-type `χ,η,γ` come from the ffield (`reaxff`) or a 4-column
file `type χ η γ` (ffield eta values enter ×2).

### 7.1 Naming and model definition (M0.5, owner instruction)
The internal abstraction is named **EEM** and is *defined as* the model above (the one LAMMPS runs with the bundled ffields). It is not a physically
different model from `fix qeq/reaxff`; `fix qeq/shielded` is the same kernel with a different solver/list (Q-26). The relation to AMS-ReaxFF's EEM is
**documented only as far as verified** (LAMMPS_INTEGRATION §7.2): the LAMMPS side is established from source; the AMS details (kernel, taper, tolerance
definition, constraints) could only be seen as search-result summaries because `scm.com` pages were not retrievable — equivalence is **not claimed**.

### 7.2 Parameters and conventions exposed to the host
`chi` (eV) as in the ffield, `eta = 2×ffield` (Q-29), `gamma` as in the ffield; all indexed by LAMMPS type. Units are Å, eV, e inside the solve and kcal/mol outside
(constants Q-01/Q-03).

### 7.3 Convergence policy (owner decision: strict by default)
`EemStatus ∈ {Converged, NotConverged, Failed}` is returned with every solve. **Default: `NotConverged`/`Failed` is an error** (adapter: `error->one`).
A separate, explicitly requested compatibility mode (`eem.compat_warn_continue`) warns and continues like LAMMPS. The convergence test itself follows the
reference (`√(rᵀM⁻¹r)/‖b‖ ≤ tol`, `maxiter`) so that "converged" means the same thing as in LAMMPS. Additional strict-mode checks (candidates, each to be justified before
adoption): `Σq` neutrality within tolerance, finite charges, and an AMS-style stability screen `eta > 7.2·gamma` **only if** confirmed from the primary AMS source.

### 7.4 Strictness and the stock fix
The unmodified stock fix cannot report non-convergence to the pair style (Q-28). Strictness therefore requires either a **derived fix**
(`fix qeq/reaxff/metal` subclassing `FixQEqReaxFF`, verifying the true residual with the protected `H`, `s`, `t`, `Hdia_inv`, `tolerance`) or an own GPU-resident fix;
see LAMMPS_INTEGRATION §7.4. Runs with the stock fix are in compat-warn mode by definition.

## 8. Discontinuities in the reference (must be reproduced, will affect NVE drift)

| Source | Where | Nature |
|---|---|---|
| `BO' ≥ bo_cut`, `r ≤ bond_cut` | bond creation | `BO_σ` carries the offset so BO→0 continuously; the `bond_cut` hard edge is a jump if `BO'(bond_cut)>bo_cut` |
| `BO > thb_cut`, products vs `thb_cutsq`/`thb_cut` | valence, torsion gates | jump in E_val/E_pen/E_coa/E_tor/E_con |
| `BO ≥ 0.01` (HB_THRESHOLD) and `r ≤ hbond_cut` | H-bond | jump (E_hb nonzero at both edges) |
| `|p_val1| > 0.001`, `vlpex ≥ 0` branch, SBO piecewise | valence | piecewise definition; `trunc` in `nlp` → kinks at integer `Δe/2` |
| `mass > 21`, `numbonds>0` (enobonds=no) | atom terms | discrete switches |
| QEq tolerance | charges | non-conservative noise (and Q-03) |
| Taper | vdW/Coulomb | smooth to 3rd derivative at `swb` — **not** a discontinuity |

## 9. What the engine computes differently by design (and why)

| # | Reference behaviour | Engine behaviour | Rationale |
|---|---|---|---|
| D-1 | *(superseded in M0.5)* ghost bookkeeping | **replicated** (ghost-native engine, owner-computes, `tag` ordering) — now disposition **R**, not a deviation; the explicit-image alternative is dropped | ADR-013; remaining risk = ghost-shell sufficiency (REF-GHOST) |
| D-2 | Missing bond parameters zero-filled (Q-12) | error (`ffield.strict_missing_pairs`) | silent garbage |
| D-3 | `thbp` array overrun for >5 sets (Q-09) | error | undefined behaviour |
| D-4 | `vdw_type` conflict between elements = warning (§2.3) | error | latent division by zero |
| D-5 | Non-converged QEq = warning, continue (`fix qeq/reaxff` default) | **strict by default**: non-convergence is an error / explicit failure status; LAMMPS behaviour only via an explicitly requested compatibility mode (`eem.compat_warn_continue`); never silently accepted. The unmodified stock fix cannot be made strict (§7.4) | owner decision M0.5 #6 |

## 10. Quirk catalogue (reference behaviours that surprised the audit)

Disposition: **R** replicate, **D** deviate by design (§9), **F** flag only.

| ID | Quirk | Source | Disp. |
|---|---|---|---|
| Q-01 | Truncated constants (`constPI`, `C_ele`, 14.4, 23.02) | `reaxff_defs.h:34-37` | R |
| Q-02 | Cube root as `pow(x,0.33333333333333)` (CPU nonbonded), `pow(x,1/3)` (QEq), `cbrt` (Kokkos) | `reaxff_nonbonded.cpp:194`, `fix_qeq_reaxff.cpp:757`, `pair_reaxff_kokkos.cpp:696` | R (CPU form in FP64 reference) |
| Q-03 | `14.4·23.02 = 331.488 ≠ C_ele = 332.06371` (0.17 %): QEq minimises a functional whose Coulomb:self-energy ratio differs from the reported energy, so charges are *not* an exact stationary point of the reported E | constants above | R; consequence for FD tests, VALIDATION.md §4 |
| Q-04 | C++ `valency_boc` = file column "Val(angle)", `valency_val` = file column "Val(boc)" (names swapped relative to header labels); gp[14] labelled `p(val7)` but used as `p_val6` | `reaxff_ffield.cpp:198,228`; `reaxff_valence_angles.cpp:108` | R (engine uses semantic names) |
| Q-05 | `mass<21` forces `valency_val := valency_boc` at parse time | `reaxff_ffield.cpp:298-305` | R |
| Q-06 | Element knowledge inside kernels: `strcmp(name,"C")` (C2 correction), exact `mass==12.0000 && mass==15.9990` (C–O stabilisation), `mass>21` first/second-row split | `reaxff_multi_body.cpp:101,107,137`; `reaxff_bonds.cpp:108-110`; `reaxff_bond_orders.cpp:457` | R **as load-time per-type/pair flags** (ADR-003) |
| Q-07 | Ghost bookkeeping: `bond_mark` (ghosts start at 1000; a bond's corrected BO is recomputed instead of copied when `bond_mark[j]>3`), `j<n \|\| nbr<n`, `orig_id` ordering with coordinate tie-breaks | `reaxff_forces.cpp:150-154,233-237`; `reaxff_bond_orders.cpp:295`; `reaxff_bonds.cpp:64-73` | R (ghost-native, ADR-013) |
| Q-08 | Neighbor rows: local atoms use `nonb_cut`, ghost rows only `bond_cut` | `pair_reaxff.cpp:659-662` | R (ghost-native, ADR-013) |
| Q-09 | `j==l` three-body mirror increments `cnt` twice (second slot is all-zero and skipped by `|p_val1|>0.001`); >5 sets overrun `prm[5]` unchecked; torsion keeps only `prm[0]` | `reaxff_ffield.cpp:452-454`; `reaxff_types.h:149` | R / D-3 |
| Q-10 | Compact torsion `0-X-Y-0` is order-dependent among compact entries; explicit entries always win | `reaxff_ffield.cpp:525-550` | R |
| Q-11 | H-bond table is directional `[i][j][k]`; `r0_hb≤0` = absent; `hbond_cut>0.1` (allocation estimate) vs `>0` (physics) | `reaxff_ffield.cpp:556-586`; `reaxff_forces.cpp:319` vs `:204` | R |
| Q-12 | Absent bond pair ⇒ zero `p_bo1,p_bo2,p_bo3,…` ⇒ `BO'_s=(1+bo_cut)`, `BO'_π=BO'_ππ=1` at *every* `r ≤ bond_cut` (a phantom triple bond; derived by reading, to be confirmed on a crafted file in M1/M2) | `reaxff_ffield.cpp:148-149,360-385`; `reaxff_bond_orders.cpp:162-175` | D-2 |
| Q-13 | Corrected-BO formulas consume the **offset-reduced** `BO'`; σ-offset `(1+bo_cut)…−bo_cut`; 1e-10 flush | `reaxff_bond_orders.cpp:164,231-236,360,408-415` | R |
| Q-14 | `(int)` truncation in `vlpex`/`nlp` | `reaxff_bond_orders.cpp:449-452` | R (`trunc`, not `floor`) |
| Q-15 | C2 correction visits each C–C bond twice (once per end, using that end's Δ) | `reaxff_multi_body.cpp:102-128` | R |
| Q-16 | Each bond energy counted once via `orig_id`/coordinate ordering | `reaxff_bonds.cpp:64-73` | D-1 |
| Q-17 | CPU uses `control->thb_cutsq`; **Kokkos hard-codes 1e-5** | `reaxff_valence_angles.cpp:222`; `pair_reaxff_kokkos.cpp:366` | R (CPU) / F |
| Q-18 | θ gradient floors sin θ at 1e-5, torsion at `MIN_SINE`=1e-10; `acos` of clamped cos | `reaxff_valence_angles.cpp:213-215`; `reaxff_torsion_angles.cpp:220-224` | R |
| Q-19 | Torsion gate uses `thb_cut` for the *triple product*, valence uses `thb_cutsq` for the product of two | `reaxff_torsion_angles.cpp:241-243` vs `reaxff_valence_angles.cpp:222` | R |
| Q-20 | H-bond excluded when `i`,`k` are the same *atom* but different image is allowed; `HB_THRESHOLD` and `hbond_cut` are hard cuts | `reaxff_hydrogen_bonds.cpp:98`, `:80` | R |
| Q-21 | QEq taper (`swa,swb` fix arguments) is independent of the ffield taper (`gp[11],gp[12]`) used for Coulomb | `fix_qeq_reaxff.cpp:509-531` vs `reaxff_init_md.cpp:79-80` | R |
| Q-22 | QEq non-convergence only warns; stopping criterion is `√(r·M⁻¹r)/‖b‖`, not `‖r‖/‖b‖` | `fix_qeq_reaxff.cpp:787,812-814` | R + status flag |
| Q-23 | QEq initial guess comes from 4-step history (cubic `s`, quadratic `t`); result below tolerance depends on history | `fix_qeq_reaxff.cpp:666-670` | R option; fixtures use tight tolerance |
| Q-24 | Doc says tabulation uses "linear interpolation"; code builds cubic splines | `doc/src/pair_reaxff.rst` vs `reaxff_lookup.cpp:52,100` | F (tabulation rejected) |
| Q-25 | LAMMPS skips pure-single (FP32) Kokkos runs of its ReaxFF regression tests (`skip_tests: kokkos_*_single`), and for the `mixed` (FP32 compute / FP64 accumulate, `kokkos_type.h:392-407`) runs the harness multiplies the test epsilon by 2e9 (`test_pair_style.cpp:825-830`): 5×2e-10×2e9 = a **relative error of 2.0**, i.e. a vacuous criterion | `unittest/force-styles/tests/atomic-pair-reaxff*.yaml`; `test_pair_style.cpp:823-830` | F — no upstream FP32 accuracy reference exists; we must build our own (NUMERICAL_POLICY §5) |
| Q-26 | `fix qeq/shielded` vs `fix qeq/reaxff`: same kernel/constants (14.4, shielding), but full list with H×0.5, history 5 deep initialised from `q` (vs 4 deep, zero), taper lower bound fixed `swa=0`, argument list `(cutoff tol maxiter)` | `fix_qeq_shielded.cpp:238,253-268`; `fix_qeq.cpp:84,95,134-135` | R (both accepted; agree to 3.3e-15 on the test system, LAMMPS_INTEGRATION §7) |
| Q-27 | LAMMPS documents that `fix qeq/reaxff` mishandles multiple images of one atom when a cell dimension < non-bonded cutoff; **not reproduced** in 4 cubic cases (dense explicit-image EEM agrees to ≤2.7e-14) | `doc/src/fix_qeq_reaxff.rst`; `tools/eem_dense_check.py` | F — open for triclinic (REF-QEQ-CELL) |
| Q-28 | Stock `fix qeq/reaxff` exposes no convergence status and its parameters are `protected` (no `extract`): a pair style cannot detect non-convergence | `fix_qeq_reaxff.h:30-80`; `.cpp:812-814` | D-5 → derived-fix plan (§7.4) |
| Q-29 | `extract()` convention: arrays indexed by LAMMPS type 1..ntypes; `eta` = 2×file value; `gamma` raw file value; NULL-mapped types all 0 | `pair_reaxff.cpp:701-731`; `fix_qeq_reaxff.cpp:226-248` | R |
| Q-30 | `fix qeq/*` charge solve requires and enforces `Σq = 0` for its group (warning if the initial charges are non-neutral); no per-region constraints | `fix_qeq_reaxff.cpp:429-430,851-880`; `fix_qeq_reaxff.rst` | R; non-neutral = Deferred |
| Q-31 | `e_pol` is tallied into LAMMPS' **Coulomb** energy (`ecoul`), the rest of the 13 terms into `evdwl`; `pvector` slot 13 `eqeq` | `reaxff_nonbonded.cpp:58`; `pair_reaxff.cpp:501-514` | R |

## 11. Open specification items (to be closed in the named milestone, not guessed now)

* Exact derivative bookkeeping of the final force pass is specified by the source (§6); the engine's
  GPU re-expression (gather, deterministic) is designed in ADR-006 and validated in M6 against the FP64 CPU path.
* Virial/stress: reference tallies per-interaction (`v_tally*`); equivalent `Σ r⊗F` formulation to be chosen at M6.
* QEq history carry-over across MD steps and across minimiser calls (LAMMPS `s_hist/t_hist` shifts) — M5/M7.
* Ghost-shell sufficiency (REF-GHOST) and whether `bond_mark` must be replicated bit-for-bit or may be proven irrelevant (M1/M4).
* Per-atom energy/virial semantics (`ev_tally` equivalents) for the adapter (M7); NPT virial identity check against stock `reaxff` (M6).
