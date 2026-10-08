# src/metal — Metal backend (M3+, Apple platforms only)
Empty at M0 by design: `REAXMETAL_ENABLE_METAL=ON` is refused by CMake until M3. Design in
`docs/ARCHITECTURE_DECISIONS.md` ADR-006/ADR-008. Metal code written in this repository cannot be built or run
in the Linux development sandbox; every Metal result must be produced on an Apple-silicon machine and recorded in
`docs/VALIDATION.md` with the device and OS version.
