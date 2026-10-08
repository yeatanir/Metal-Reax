# Prompt for Claude Code running on the Apple machine

Open Claude Code in a terminal **on the Mac**, in an empty working directory, and paste everything between the two lines below.
(The repository is public to your account; Claude Code needs `git`, `cmake` >= 3.24 and Apple clang, i.e. Command Line Tools. No Xcode.)

---------------------------------------------------------------------------------------------------------------------------------

You are helping validate the Metal part of milestone M3 of the ReaxMetal project on THIS Apple-silicon Mac. The code was written on a Linux
machine that has no Metal, so none of it has ever been built or run on Apple hardware. Your job is to run three scripted check steps,
report exactly what happened, and make only the smallest fixes needed to get past compile errors. Honesty matters more than a green result.

SETUP
1. `git clone <the ReaxMetal repository URL I give you> ReaxMetal && cd ReaxMetal && git checkout claude/friendly-ride-rwz7xu`
   (if I did not give you a URL, ask me for it). Then `git checkout -b mac/m3-bringup`. Do NOT push to `claude/friendly-ride-rwz7xu`.
2. Read `docs/MAC_VALIDATION.md` (what each step checks and what failures probably mean) and the header comments of
   `tools/metal_check.cpp`, `src/metal/metal_backend.mm` and `src/metal/shaders/reaxmetal_m3.metal`.

RUN, IN ORDER, STOPPING AT THE FIRST STEP THAT FAILS
- `tools/mac/step1_bringup.sh`, then `tools/mac/step2_neighbor.sh`, then `tools/mac/step3_reduce.sh`.
  Each writes `mac-reports/stepN-<time>.txt` (environment, build log, results). Never run a later step after an earlier one failed.

RULES
- Report facts only. Never say a step passed unless its report ends with `STEP n RESULT: PASS`. A pass of one check is not a pass of the step.
- If the CMake configure or the build fails: the Objective-C++ in `src/metal/metal_backend.mm` was only syntax-checked against stub headers I
  wrote from memory, so API mismatches and typos are expected. You MAY fix compile/link errors in `src/metal/metal_backend.mm`,
  `CMakeLists.txt`, `tools/mac/*.sh`, and `tools/metal_check.cpp` (build problems only). Keep each fix minimal, make one commit per fix on
  `mac/m3-bringup`, and quote the compiler error and the diff in your report. Then rerun the same step.
- If the runtime shader compilation fails, show the full compiler message. You MAY fix a plain MSL syntax error in
  `src/metal/shaders/reaxmetal_m3.metal` (same rules: minimal, own commit, quoted).
- You MUST NOT, without asking me first: change any tolerance, margin (`1e-3`), expected value, comparison or PASS condition in
  `tools/metal_check.cpp` or the tests; change the kernels' numerical behaviour (the order of additions in `rm_partial_sums` /
  `rm_sum_partials`, the cutoff test in `rm_far_rows`); switch compiler options such as fast-math; install Xcode or any other software;
  edit `docs/` or `tests/`. If a check FAILS at run time (e.g. `missing > 0`, a bitwise mismatch, NaN not detected), do not "fix" it:
  stop, and report the failing line, the numbers, and your best explanation.
- Do not run `tools/mac` steps beyond 3, do not try to build LAMMPS or the plugin, do not open a pull request, and do not push anything
  unless I tell you to (then only branch `mac/m3-bringup`).

REPORT (always, even when everything passes)
1. Per step: PASS / FAIL / NOT RUN, and the path of its report file. Paste the full `STEP n RESULT` line and every line containing FAIL.
2. For step 1 also paste: the device line (`MET-1a`), the compile time (`MET-1b`), the kernel table, `MET-1e`, and `MET-1f` (the
   fused-multiply-add probe). For step 2: the GPU time and `band_pairs` per geometry. For step 3: the best GPU time line.
3. A list of every file you changed with `git diff main..HEAD --stat` (or against the starting commit) and the reason for each change.
4. Anything surprising, even if it did not fail (compiler warnings from the Metal compiler, deprecation warnings, slow steps).
5. Finally, `cat` the content of each `mac-reports/step*.txt` file in full so I can paste it back to the person who wrote the code.

---------------------------------------------------------------------------------------------------------------------------------

Notes for you (not part of the prompt)
* Replace "<the ReaxMetal repository URL I give you>" by telling Claude Code the URL as your first message, or edit the line.
* If you would rather not let Claude Code change code at all, delete the "MAY fix" sentences; it will then just run and report.
* Whatever it reports, send me the three `mac-reports/step*.txt` files (or its final report); I will update `docs/VALIDATION.md`
  (MET-1, NBR-1, FORCE-2, MET-4) from them, and review any commits on `mac/m3-bringup` before merging anything.
