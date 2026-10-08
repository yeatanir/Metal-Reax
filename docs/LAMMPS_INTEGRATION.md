# LAMMPS Integration Architecture (M0.5)

Owner directive (after M0): ReaxMetal is **usable directly inside LAMMPS**. We do **not** build a standalone MD engine that
duplicates LAMMPS integrators, thermostats, minimisers, atom management, trajectory handling or domain infrastructure.
Priority order: (1) backend-independent ReaxFF force/energy library, (2) native Metal backend, (3) thin LAMMPS pair-style
integration `reaxff/metal`, (4) CPU reference path for validation, (5) standalone evaluation/Python later.

**Evidence tags** used throughout — **[V]** verified by an executed run in this session (Linux x86-64, serial stock LAMMPS
`stable_30Sep2026` @ `8de817dd…`, g++ 13.3, probe plugin in `plugin/probe/`, harness `tests/lammps/run_probe.py`);
**[S]** established by reading the pinned source (file:line); **[U]** unverified — cannot be tested in this environment
(notably anything on macOS/Metal or with MPI > 1 rank) or not yet tested. Nothing tagged [U] may be reported as working.

## 1. Decisions in one table

| Question | Decision | Basis |
|---|---|---|
| How is `reaxff/metal` delivered? | **Loadable LAMMPS plugin** (DSO, `plugin load` or `LAMMPS_PLUGIN_PATH`), built against the pinned LAMMPS; an in-tree package patch is a possible later step, not required | plugin API [S][V] |
| Base class | **`Pair`**, *not* `PairReaxFF` | §3.3: no need for the PuReMD host machinery; charge fixes work via `extract()` [V]; avoids inheriting LAMMPS' memory heuristics and ffield parser |
| Style name | `reaxff/metal`; the regex `^reaxff` used by the charge fixes matches it [S][V] | `Force::pair_match` `force.cpp:252-274` |
| What LAMMPS keeps | integrators, thermostats, minimisers, domain/PBC, atom exchange, **ghost-atom creation and communication**, skin/rebuild decision, thermo/dump, `fix qeq/*` (initially) | §5 |
| What we own | ffield parsing, all ReaxFF terms, lists, EEM solve (later), device buffers/kernels | §5 |
| Periodicity | **Ghost-native**: the engine consumes LAMMPS' owned+ghost atom set; the standalone harness generates the same structure with an image expander | §4, ADR-013 (amends ADR-004) |
| Neighbor lists | **Not taken from LAMMPS in production** (device-built from positions); LAMMPS lists only for the CPU prototype | §6 [V: `neigh none` is allowed] |
| MPI | single rank; multi-rank must fail explicitly | §4 C8 |
| Charges, first prototype | stock `fix qeq/reaxff` or `fix qeq/shielded` (CPU) fed by `extract("chi"|"eta"|"gamma")` | §7 [V] |
| First validation oracle | **in-LAMMPS A/B**: same input, `pair_style reaxff` vs `reaxff/metal backend cpu64`, same ghosts/neighbor skin/QEq | §9 |

## 2. LAMMPS plugin mechanism (what we must conform to)

* A plugin is a DSO exporting `extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)`; it fills a
  `lammpsplugin_t {version, style, name, info, author, creator.v1, handle}` and calls the registration function
  (`doc/src/Developer_plugins.rst`, `src/lammpsplugin.h`) **[S]**. Our probe does exactly this and loads **[V]**.
* `plugin.version` should carry `LAMMPS_VERSION`; the loader does **not** enforce equality — it only logs "compiled for LAMMPS version X, loaded into Y" on a mismatch
  (`src/PLUGIN/plugin.cpp:179-182`) **[S]**. The hard constraint is ABI: the DSO must be compiled with the **same compiler family and settings** as LAMMPS
  (MPI on/off, `bigint/tagint` sizes, `LAMMPS_EXCEPTIONS`), otherwise symbols fail to resolve at load **[S]**; the adapter should therefore also verify the version string itself and refuse to run on a mismatch.
  Our CMake exposes `REAXMETAL_LAMMPS_MPI` and defines `LAMMPS_EXCEPTIONS`; a serial build uses `src/STUBS` for `mpi.h` **[V]**.
* Style factories live in a **process-global registry** since the 2 Sep 2026 change (a plugin survives `clear`; loading twice is a no-op;
  replacing needs `plugin unload`) **[S]**. A plugin may re-register an existing style name and override it **[S]** — not used.
* Class names must be unique in the whole executable (`PairReaxFFMetal…` prefix) **[S]**.
* Link model: on Linux the plugin leaves LAMMPS symbols undefined and `lmp`/`liblammps.so` provides them (`-rdynamic`); the stock build must be
  `BUILD_SHARED_LIBS=on` (or CMake ≥ 3.27 `COMPILE_ONLY`, per `examples/plugins/CMakeLists.txt`) **[S][V]**. On macOS the examples use
  `-Wl,-undefined,dynamic_lookup` and keep the `.so` suffix **[S]**; **not tested here [U]**.
* The adapter TU compiles as **C++20** against the pinned LAMMPS headers and loads **[V]** (the project is C++20; LAMMPS' own minimum is 17).
  Public headers shared between core and adapter may therefore use C++20, but `plugin/CMakeLists.txt` currently pins the adapter to 17 until
  the core is linked (decision at M2).
* Plugin interface headers (`lammpsplugin.h`, `library.h`) carry the **plain LAMMPS GPL notice**, not an LGPL one in this tree **[S]**
  (`third_party/lammps/LICENSE_AUDIT.tsv`): the plugin is a GPL-2.0 derivative work — consistent with the owner's GPL-2.0-only decision.

## 3. Pair-style contract

### 3.1 What LAMMPS calls, in order (single rank, `run`) **[S]** `verlet.cpp`, `pair.cpp`
`settings()` → `coeff()` → `Pair::init()`: `init_style()` (we make neighbor requests, checks) then `init_one(i,j)` for every type pair
(returns the cutoff; `cutforce = max`) → `setup()` → per step: `initial_integrate` → [`neighbor->decide()`: if rebuild: exchange,
**borders (new ghost set)**, neighbor build; else **forward comm of ghost positions**] → `force_clear()` (zeroes `f` for owned+ghost when newton)
→ **`pre_force` fixes (`fix qeq/*` update q and forward-communicate it)** → `pair->compute(eflag, vflag)` → **`reverse_comm()` folds ghost
forces onto owners** (newton on) → `post_force` / `final_integrate`. Minimisation uses the same pair call with `min_pre_force` for the QEq fix.

### 3.2 Flags the adapter sets (all verified to be accepted **[V]**)
`manybody_flag=1`, `one_coeff=1`, `restartinfo=0`, `single_enable=0`, `centroidstressflag=CENTROID_NOTAVAIL`,
`no_virial_fdotr_compute=0` (global virial from `Σ x·f` over **owned+ghost** atoms, `Pair::virial_fdotr_compute`, `pair.cpp:1832-1861`),
`nextra=14` + `pvector` (so `compute pair reaxff/metal` returns the 14 LAMMPS energy slots: read back marker values 1 and 14 **[V]**),
`ghostneigh` only if we ask LAMMPS for ghost neighbor lists (we normally do not).
`ev_init(eflag, vflag)` decodes: `eflag_global`, `eflag_atom`, `vflag_global`, `vflag_fdotr`, `vflag_atom` (`pair.cpp:989-1008`) — the engine skips energy
reductions on steps where `eflag_either == 0` (most steps).

### 3.3 Why `Pair`, not `PairReaxFF`
Tested **[V]** with a `Pair`-derived `reaxff/metal` and the stock `fix qeq/reaxff … reaxff`:

* The fix finds the parameters through `force->pair_match("^reaxff",0)->extract("chi"|"eta"|"gamma")` (`fix_qeq_reaxff.cpp:228-234`) — no cast needed.
* Its `dynamic_cast<PairReaxFF*>` result is `nullptr` and the fix then uses **its own neighbor list** (`fix_qeq_reaxff.cpp:537-547`, `:603-613`).
* Result: charges **bitwise identical** (max|Δq| = 0.0, N = 64) to stock `pair_style reaxff` + the same fix, for each of four neighbor-request modes of the pair.
* Consequence of not being a `PairReaxFF`: `fix reaxff/bonds`, `fix reaxff/species`, `compute reaxff/atom` refuse with an explicit LAMMPS error
  (`fix_reaxff_bonds.cpp:112`) **[V]**; `fix acks2/reaxff` needs extra `extract` keys and refuses **[V]**; `qtpie`/`qeq/rel` were not exercised **[U]** — all Deferred in FEATURE_MATRIX.

## 4. Single-rank atom / neighbor / ghost data contract

Measured on the LAMMPS regression geometry (64 atoms, 7.54 Å cubic periodic cell, `cutoff 10 Å`, default real-units skin 2 Å) **[V]**:

| Fact | Measured |
|---|---|
| ghost shell width | `cutforce + skin = 12 Å` (`Neighbor::cutneighmax`; `Comm::get_comm_cutoff` = `max(comm_modify cutoff, cutneighmax)`, `comm.cpp:659-704`) |
| atoms | `nlocal = 64`, `nghost = 4541`, `nall = 4605`; copies of one tag up to **125 = 5³** (images −2…+2 per dimension) |
| ghost = owner + integer lattice shift | held for **every** ghost on **every** one of 61 MD steps (positions refreshed each step) |
| ghost forces | adding +1 to `f_x` of **all** `nall` atoms inside `compute()` gives Σfx = **4605 = nall**, and `fx[1] = 64 = copies(tag 1)` — LAMMPS folds ghost forces onto owners exactly, in all six neighbor-request modes |
| neighbor list is optional | `pair_style … neigh none` (no `add_request`) is accepted; ghosts and cutoffs are still provided |
| list sizes (same system) | half/newton-on 34 737 entries (20 214 within 10 Å); half/newton-off 67 458 (38 417); full 69 474 (40 428); **full + ghost neighbors 3 106 696 (1 967 422 within 10 Å)** |
| rebuilds | owned order and `nghost` change **only** at steps with `neighbor->ago == 0` (0 violations over 61 steps; with `atom_modify sort 5 2.0` the owned-atom order changed at 12 rebuilds) |
| tags/types/charges | present for owned **and** ghost atoms; ghost charges updated by the fix each step (forward comm) before `compute` |
| `minimize` | drives the same `compute()` (2 calls; the zero-energy probe converges immediately — **not** a minimiser-contract test beyond "it is called") |

### Normative contract for the adapter (rules C1–C12)

* **C1 Inputs per call**: `nlocal`, `nghost`, `x[nall][3]` (double), `type[nall]` (mapped to ffield types via the `pair_coeff` map), `tag[nall]`, `q[nall]` (read **after** `pre_force`), box (only needed if the adapter builds its own lists).
* **C2 Topology epoch**: when `neighbor->ago == 0` *everything topological is stale* — `nlocal`, `nghost`, owned order, ghost set and ordering. The adapter re-uploads `type/tag`, rebuilds device lists, resizes buffers. When `ago > 0` only `x` and `q` change; indices are stable **[V]**. Never cache anything index-based across an `ago == 0` call.
* **C3 Ghost shell**: `init_one` returns `max(nonb_cut, hbond_cut, 2·bond_cut)` as the force cutoff (the reference warns when `cutmax < 2·bond_cut`, `pair_reaxff.cpp:372-375`; we **error** instead). The truly required shell is established by experiment REF-GHOST (VALIDATION) — a *requirement on the host*, checked in `init_style`.
* **C4 Forces**: the engine returns forces for **all** `nall` atoms; the adapter **adds** them to `atom->f` (LAMMPS cleared it; fixes may already have contributed) and never touches the reverse communication.
* **C5 Energies**: only if `eflag_global`: `eng_vdwl = Σ(non-Coulomb terms)`, `eng_coul = e_ele + e_pol` (the reference tallies `e_pol` as Coulomb, `reaxff_nonbonded.cpp:58`), `pvector[0..13]` from `to_lammps_pvector`.
* **C6 Virial**: `if (vflag_fdotr) virial_fdotr_compute()` after the forces were added (needs forces on ghosts). `vflag_global` without fdotr, and any `vflag_atom/cvflag_atom`, are **refused with an explicit error** until validated (M6/M7). Barostat runs are therefore refused until `lammps.virial_fdotr` is Implemented.
* **C7 Per-atom outputs**: `eflag_atom`/`vflag_atom` → explicit error until implemented (`out.per_atom_energy`, M7).
* **C8 Host requirements checked in `init_style`** **[V]** each with an explicit LAMMPS error: `atom->q_flag`, `tag_enable`, `newton_pair == on`, **`comm->nprocs == 1`** (the check exists in the probe; **a multi-rank run cannot be tested with the serial LAMMPS built here [U]**).
* **C9 Charges**: `q` comes from the host/fix; the adapter only reads it. With `checkqeq no` the user's static charges are used (parity with `pair_style reaxff checkqeq no`).
* **C10 Atom identity**: ReaxFF reference rules order interactions by `tag` (with coordinate tie-breaks for self-images, ENGINE_SPEC Q-07). The adapter passes `tag[]` for ghosts; `atom->map()` is **not** needed.
* **C11 Precision of coordinates**: positions are converted double→float *relative to the box lower corner* (wrapped coordinates, ghosts within ±(cutoff+skin) of the box); the engine never sees absolute unwrapped coordinates (NUMERICAL_POLICY 4.1).
* **C12 Errors**: engine failures (list overflow beyond growth limit, EEM non-convergence in strict mode, NaN) are raised with `error->all`/`error->one`; **nothing is returned silently wrong**.

## 5. Responsibility split

| Concern | LAMMPS host | Adapter (thin, CPU) | Core library (backend-independent) | Metal backend |
|---|---|---|---|---|
| time integration, thermostats, minimiser, restart, output | ● | | | |
| periodic images, ghost creation, forward/reverse comm | ● | consumes | (standalone: image expander) | |
| skin & rebuild decision | ● (`ago`) | reads | | |
| `pair_style` options, `pair_coeff`, element→type map | | ● parse | ffield/control parser, tables, compat flags | |
| `extract(chi/eta/gamma)`, pvector, tallies, virial call | | ● | | |
| host-rule checks (newton, q, nprocs, ghost shell) | | ● | | |
| double↔float conversion, upload/download | | ● | | buffers |
| cell/neighbor lists over owned+ghost | (CPU prototype only) | | CPU lists | ● device lists |
| bond orders, all energy terms, forces, coefficients | | | ● CPU-64/CPU-32 | ● kernels |
| EEM solve | ● stock fix (prototype) | verification | CPU solver (reference) | ● GPU-resident (M5+) |
| strict-convergence policy | | enforces | defines | reports status |

## 6. Neighbor-list strategy

* **Production**: the backend builds its own cell/neighbor lists on the device from `x[nall]` at `ago == 0` (and reuses them until the next
  rebuild — LAMMPS' skin already guarantees validity). Rationale **[V]**: asking LAMMPS for full lists with ghost neighbors costs 3.1 M entries
  for a 64-atom cell (1.6× more than needed within the 10 Å cutoff); converting/transferring that to the device per rebuild is wasteful and ties
  us to LAMMPS' list formats. `neigh none` is permitted **[V]**.
* **CPU prototype / A-B tests**: may request `REQ_FULL|REQ_GHOST` or half/newton-off lists to reproduce the reference traversal exactly (the
  reference uses a half list with ghosts, `pair_reaxff.cpp:370`).
* Bond, 3-body and H-bond lists are always engine-internal and rebuilt every evaluation (they depend on bond orders), as in the reference.

## 7. Charge-model (EEM) integration

### 7.1 Fix compatibility audit (stock tree, with a `Pair`-derived `reaxff/metal`)
| Fix / compute | Needs from the pair | With `reaxff/metal` | Evidence |
|---|---|---|---|
| `fix qeq/reaxff … reaxff` | `extract chi,eta,gamma`; pair name `^reaxff`; own neighbor list if pair is not `PairReaxFF` | **works**, charges bit-identical to stock | [V] |
| `fix qeq/shielded … reaxff` | same keys, `^reaxff`; own **full** list; H elements ×0.5 | **works**, agrees with qeq/reaxff to 3.3e-15 | [V] `fix_qeq_shielded.cpp:83-107,238` |
| `fix qeq/rel/reaxff` | same + `PairReaxFF` cast | not tested (takes extra arguments; my one-line probe command was rejected for argument count) | [U] |
| `fix acks2/reaxff` | extra keys `bcut_acks2`, `bond_softness`; PairReaxFF cast | refuses explicitly ("could not extract params from pair reaxff") | [V] |
| `fix qtpie/reaxff` | `extract chi,eta,gamma` + `PairReaxFF` cast (`fix_qtpie_reaxff.cpp:157,289-295`); needs extra arguments | not tested | [U] |
| `fix reaxff/bonds`, `fix reaxff/species`, `compute reaxff/atom` | `dynamic_cast<PairReaxFF*>` | refuse explicitly | [V] |
| `fix efield` with QEq | — | Deferred (`qeq.efield`) | — |

`extract()` convention (what LAMMPS expects from our style): arrays indexed **1..ntypes by LAMMPS type**, `dim = 1`, `chi` (eV) unchanged from the ffield,
**`eta` = 2 × ffield value**, `gamma` = ffield value (the fix forms `(γ_iγ_j)^(-3/2)` itself); a type mapped to `NULL` has all three equal to 0.

### 7.2 What is known about "EEM" vs LAMMPS "QEq" — and what is not
| Item | LAMMPS `fix qeq/reaxff` / `qeq/shielded` | AMS-ReaxFF EEM | Status |
|---|---|---|---|
| Energy functional | `E = Σ χq + ½ηq² + Σ_{i<j} H_ij q_i q_j`, `Σq = 0` | EEM: same family | LAMMPS **[S]**; AMS only via a web-search summary **[U]** |
| Kernel | `H_ij = 14.4·Tap(r)/(r³+(γ_iγ_j)^(-3/2))^(1/3)`, taper `Tap` 7th order on `[swa,swb]` from **fix arguments** | not retrievable (the SCM page fetch was blocked) | LAMMPS **[S]**; AMS **[U]** |
| `qeq/shielded` vs `qeq/reaxff` | **same kernel and constants** (14.4, shielding); different list (full, H ×0.5), different history (5 deep, initialised from `q`, vs 4 deep initialised to zero, `fix_qeq.cpp:95,134-135` vs `fix_qeq_reaxff.cpp:119,199-200`), taper lower bound fixed at `swa = 0` (`fix_qeq.cpp:84`) | — | **[S]**; numerically equal to 3.3e-15 on the test system **[V]** |
| Units | Å, eV, e hard-coded in the fix; `eta` file value ×2 | — | **[S]** |
| Total-charge constraint | `Σq = 0` for the fix group (warning if the initial charges are non-neutral); no per-region constraints | AMS documents regional net-charge constraints (search-summary only) | LAMMPS **[S]**; AMS **[U]** |
| Convergence criterion | relative **preconditioned** residual `√(rᵀM⁻¹r)/‖b‖ ≤ tol`, `maxiter 200`, warn-and-continue | AMS: tolerance on a sum of squared charge residuals, default 1e-6; solver options Direct/CG/MINRES-QLP/SparseCG; "Simple" predictor (search-summary only) | LAMMPS **[S]**; AMS **[U]** |
| Stability condition | not checked | AMS 2019.3 warning page (search-summary): `eta > 7.2·gamma` else "polarization catastrophe" risk | **[U]**; candidate input check, to be confirmed against the primary page |
| Literature | LAMMPS manual: Rappé–Goddard QEq as formulated by Nakano; ReaxFF's Coulomb eq. (13) | Mortier EEM = bare 1/r kernel; Rappé–Goddard QEq replaces it by a shielded kernel; a forum thread (secondhand) claims the ReaxFF paper treats the two as compensable via the shielding parameter | **[U]** (secondary sources) |

**Conclusion carried into the design (owner's instruction: do not create a physically different model):** the engine's charge model is named **EEM** and is
*defined to be* the LAMMPS-compatible model above (the one the bundled ffields are used with in LAMMPS). We claim **no** equivalence with AMS
output beyond what is verified; an AMS comparison needs AMS reference charges for a fixture (open question to the owner) — until then the
statement is "LAMMPS-compatible EEM". The M0.5 run also measured the small-cell caveat (§7.3).

### 7.3 `fix qeq/reaxff` in cells smaller than the cutoff — measured
LAMMPS documents that the fix "does not correctly handle interactions involving multiple periodic images of the same atom" for cells smaller than the non-bonded
cutoff (`doc/src/fix_qeq_reaxff.rst`, Restrictions) **[S]**. I compared the fix against an independent dense solve with **explicit periodic images**
(`tools/eem_dense_check.py`, ENGINE_SPEC §7 formulas, taper 8.0 Å) on the regression geometry **[V]**:

| cell | positions | max\|q_dense − q_LAMMPS\| | minimum-image-only model error |
|---|---|---|---|
| 2×2×2 (7.54 Å) | displaced 0.1 Å | 1.9e-14 | 7.76 e |
| 3×3×3 (11.31 Å) | displaced 0.1 Å | 2.0e-14 | 6.5e-2 e |
| 2×2×2 | perfect lattice (tie-break paths) | 2.7e-14 | 5.5e-1 e |
| 3×3×3 | perfect lattice | 2.2e-14 | 7.9e-2 e |

The documented restriction is **not reproduced** in these four cubic cases (agreement to round-off), so the stock fix is usable as an oracle there. It remains open for triclinic cells and
pathological tie-break geometries (VALIDATION `REF-QEQ-CELL`, M1). Minimum-image-only treatment is wrong by 0.06–7.8 e even at 11.3 Å: it must never be used.

### 7.4 Strict convergence with the stock fix (owner decision: strict by default)
The stock fix can only warn (`fix_qeq_reaxff.cpp:812-814`). Its `swa`, `swb`, `tolerance`, `H`, `s`, `t` are **protected** with no `extract()`
and no public convergence status (`fix_qeq_reaxff.h:30-80`) **[S]**, and `compute_scalar()` returns only `matvecs/2`. A pair style therefore cannot observe non-convergence. Options, in the order I recommend:
1. **Derived fix plugin** (`fix qeq/reaxff/metal`): subclass `FixQEqReaxFF` (the REAXFF package must be present anyway as the oracle), override `pre_force`/`min_pre_force`,
   call the base, then verify the true residual of `s` and `t` with the protected `H`, `b_s`, `b_t`, `Hdia_inv`, `tolerance` (one extra forward comm of `s,t`); strict mode
   raises `error->one`; `compat warn` mode reproduces the stock behaviour. Same argument syntax as `qeq/reaxff`. *Feasibility: members are protected → accessible to a subclass **[S]**; not yet implemented [U]*.
2. A fully own GPU-resident EEM fix (later; subject to numerical validation).
3. Adapter-side residual check with user-repeated parameters (fragile; not recommended).
Until (1) exists, **runs with the unmodified stock fix are by definition in "compat warn" mode** and the documentation must say so.

## 8. Host/device memory ownership and synchronisation (Metal) **[U — design; nothing here ran on a GPU]**

Ownership: LAMMPS owns `atom->x/f/q/type/tag` (host, double). The adapter owns **float staging buffers** allocated as `MTLBuffer` with
`MTLResourceStorageModeShared` (Apple-silicon unified memory: no copy, but explicit ordering). The backend owns private device buffers: topology, lists, per-atom intermediates, per-bond coefficients, reductions.

| # | Sync point (per `compute()`) | Action | Cost class |
|---|---|---|---|
| S0 | `ago == 0` only | upload `type`, `tag`; (re)size buffers; rebuild device cell/neighbor lists; check/grow list capacities | O(nall) + list build |
| S1 | every call | convert `x`, `q` double→float into the shared buffers (wrapped, box-relative) | O(nall), CPU |
| S2 | every call | encode one command buffer (all kernels), commit | — |
| S3 | every call | `waitUntilCompleted` — LAMMPS needs forces on return; CPU must not touch shared buffers while the GPU runs | blocking |
| S4 | every call | convert `f` float→double and **add** to `atom->f` (owned+ghost) | O(nall), CPU |
| S5 | `eflag_global`/`vflag` | read back energy partial array, reduce in a **fixed order in FP64**; fill `eng_*`, `pvector` | small |
| S6 | every call | read a 32-bit status word: list overflow, NaN, EEM non-convergence → grow-and-retry or raise | tiny |
Rules: no hidden CPU↔GPU copies; the *only* per-step transfers are `x,q` in and `f` out (plus S5/S6). A GPU-resident EEM removes the `q` input transfer and lets the solver overlap with bonded kernels.
`compute()` is synchronous because the LAMMPS time loop is; asynchronous overlap would need a LAMMPS-side change and is out of scope.

## 9. CPU-only plugin stub for Linux builds (design) and validation strategy

* **Targets** (CMake, `-DREAXMETAL_BUILD_LAMMPS_PLUGIN=ON -DREAXMETAL_LAMMPS_SOURCE_DIR=<lammps>/src`):
  `reaxmetal_core` (static, PIC) → `reaxmetal_lammps_adapter` (object lib, adapter TU) → `reaxmetalplugin.so` (single DSO registering `pair reaxff/metal`, later `fix qeq/reaxff/metal`).
* **Backends** selected at runtime: `pair_style reaxff/metal NULL backend cpu64|cpu32|metal`. On Linux only `cpu64`/`cpu32` exist; `backend metal` on a platform without it is an explicit error
  (`backend.metal_fp32` gating). The Metal sources are compiled only on Apple.
* **Increments** (each independently testable on Linux): **A0** probe (done, §10). **A1 (M2)**: real parse of ffield/control via our parser, real `extract` for all bundled ffields, element-flag derivation, host checks — compared to stock `extract` values. **A2 (M4–M6)**: CPU-64 backend behind the adapter, validated **in-LAMMPS A/B** against `pair_style reaxff` on the same input (same ghosts, skin, QEq fix): energies per pvector slot, total forces, virial. **A3**: CPU-32 twin. **A4 (M3/M6)**: Metal backend behind the same adapter — **runs only on the owner's Mac**. **A5 (M5+)**: derived/own EEM fix.
* **Why A/B in LAMMPS is the primary oracle:** it removes every host-side difference (ghosts, skin, neighbor ordering, QEq) from the comparison; M1 fixtures (hashed, instrumented) remain for localisation and for CI without a LAMMPS build.
* **Stock reference build** (reused by M1): `tools/fetch_lammps.sh --full <dir>` then `tools/build_lammps_reference.sh <dir> <build> <install>` (serial, shared lib, REAXFF+QEQ+PLUGIN; verifies pristine pinned commit). Executed here: configure+build+install ≈ 3 min on 4 cores (file timestamps) **[V]**.

## 10. Minimal end-to-end smoke test (implemented as the A0 probe) **[V]**

`python3 -I tests/lammps/run_probe.py --lmp <install>/bin/lmp --plugin <build>/reaxmetalprobeplugin.so --potentials <lammps>/potentials --work <dir>` (0.5 s) asserts:

1. plugin loads and `pair_style reaxff/metal` instantiates;
2. for six neighbor-request modes: ghost = owner + lattice shift; Σfx == nall and fx[1] == copies(tag 1) (reverse-comm fold-back);
3. 61-step MD, with and without a pair neighbor request and with `atom_modify sort 5`: no order/`nghost` change while `ago > 0`; ghost positions refreshed every step;
4. `fix qeq/reaxff` (four modes) and `fix qeq/shielded` give charges equal to stock `pair_style reaxff` + `fix qeq/reaxff` (0.0 / 3.3e-15);
5. `compute pair reaxff/metal` returns the 14-slot pvector; `minimize` calls the style;
6. explicit errors for `newton off`, missing charge attribute, `fix reaxff/bonds`, `fix acks2/reaxff`;
7. informational: dense explicit-image EEM equals the stock fix (≤ 2.7e-14) while minimum-image does not.
**Result of the recorded run: SUMMARY failures=0.** CTest registration of this test is gated by `REAXMETAL_LAMMPS_PREFIX` (not part of the default build).

## 11. Target machine facts (owner-reported) and consequences **[U — not reproducible here]**
Apple M5 Max (18 CPU / 40 GPU cores, 64 GB), macOS 26.3.1, SDK 26.2, Apple clang 17.0.0, CMake 4.2.3, Python 3.14.3, **Command Line Tools only (no Xcode, no `metal` shader compiler)**, GPU reports Metal 4.
* Offline `.metallib` generation (`xcrun metal`) is **unavailable** until Xcode is installed → shaders must be compiled **at runtime from source** via metal-cpp (`MTL::Device::newLibrary(source, options, &err)`). Whether the runtime compiler is usable without Xcode on that machine is the **first M3 check** [U].
* metal-cpp is header-only; building against the CLT SDK's Metal framework is plausible but **unverified** [U]. Metal 4 APIs will not be used until verified; the baseline is the MSL feature set the runtime compiler accepts.
* Plugin build on macOS: Apple clang, no OpenMP, serial (`BUILD_MPI=off`) LAMMPS built from the pinned tree; plugin linked with `-Wl,-undefined,dynamic_lookup` [S per examples, U in practice].
* CMake 4.x: project minimum is 3.24; LAMMPS' own minimum (3.20) and the example plugin files are compatible [S]. The Linux sandbox used 3.28.

## 12. Open items and risks
| ID | Item | Plan |
|---|---|---|
| I-1 | multi-rank refusal untested (serial build) | build an MPI LAMMPS on the Mac (or here if MPI is installable) and test the `nprocs` error |
| I-2 | strict EEM with stock fix impossible without a derived fix | §7.4 option 1 at M5 (earlier if you prefer) |
| I-3 | required ghost-shell width | REF-GHOST (M1): sweep `comm_modify cutoff` on real fixtures |
| I-4 | per-atom energy/virial, NPT | `lammps.virial_fdotr` (M6), `out.per_atom_energy` (M7) |
| I-5 | AMS equivalence | needs AMS reference charges; otherwise the claim stays "LAMMPS-compatible EEM" |
| I-6 | triclinic + QEq small cell | `REF-QEQ-CELL` (M1) |
| I-7 | Metal compile without Xcode, metal-cpp against CLT SDK | first task of M3 on the Mac |
| I-8 | `fix qeq/rel/reaxff` with `Pair`-derived style untested | low priority (Deferred) |

## M1 addenda (executed)
* **Single-rank contract (C8) verified with a real MPI build** (OpenMPI 4.1.6, probe plugin built with `REAXMETAL_LAMMPS_MPI=ON`): `mpirun -np 1` runs; `-np 2` and `-np 4` abort with `Pair style reaxff/metal supports a single MPI rank only`. (macOS/Apple MPI not tested.)
* **QEq taper vs ghost shell:** the adapter must refuse a QEq taper radius larger than `max(nonb_cut, hbond_cut, bond_cut) + skin` — the stock fix truncates silently (ENGINE_SPEC Q-35, capability `eem.taper_within_ghost_shell`).
* **Ghost shell:** results were identical for shells 7.5–14 Å on the tested systems; the 2·`bond_cut` warning of `pair_reaxff.cpp:373` is conservative for ordinary bond lengths (ENGINE_SPEC §3.1).
* **Image rules the host must preserve:** tags are the tie-break for equal-image pairs; a host that renumbers tags or merges images changes results (Q-32 shows the reference itself already misbehaves for self-images in H-bonds).

## M2 addenda (executed): adapter A1
* **A1** = `reaxff/metal` registered by `plugin/adapter` (`reaxmetaladapterplugin.so`): parses settings/control file/ffield with `reaxmetal_core`, maps `pair_coeff` elements, exposes `extract(chi|eta|gamma)` (index 1..ntypes, eta = 2× file value, as `pair reaxff`), runs the host checks of §3 in `init_style` (exactly one `qeq/reaxff`|`qeq/shielded`, newton on, `q` present, single rank, deferred fixes refused) and **refuses in `compute()`** with an explicit error. It never returns zero energy. Tested against the stock build through the LAMMPS C library (`tests/lammps/run_a1.py`, CTest `lammps_a1`).
* **Build-ABI hazard found in M2.** A plugin compiled as C++20 with clang fails to `dlopen` into the default (C++17) LAMMPS: LAMMPS' bundled `fmt/format.h` switches `fmt::format_args` to `std::format_args` when `<version>` reports `__cpp_lib_format` (clang, GCC ≥ 14), which changes the mangled `Error::_all/_one/_warning` symbols. `plugin load` only prints the dlopen failure, so the symptom is a later "Unrecognized pair style". The adapter therefore force-includes `plugin/adapter/lammps_fmt_abi.h` (withdraws the macro) unless `-DREAXMETAL_LAMMPS_STD_FORMAT=ON` is given for a C++20 LAMMPS; the test harness asserts that the style registered and names this cause otherwise. Not verified on macOS/Apple clang (the macro logic is the same, the result is unknown).
