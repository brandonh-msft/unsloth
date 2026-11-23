---
name: unsloth-build-install
description: Build and install Unsloth Studio from the local Unsloth and Unsloth Zoo checkouts. Use when installing, rebuilding, or updating a local source-based Studio installation.
---

# Build and install Unsloth Studio from local source

Use the checked-out source, including uncommitted changes. This installs Studio
and its dependencies; it does not build the native desktop application.

## Select sources and install location

1. Identify the Unsloth checkout and inspect its Git status, branch, commit,
   and configured upstream. Do the same for Zoo after it is selected.
2. Ask the user which local Unsloth Zoo checkout to use, unless they already
   specified its path. Present discovered candidates when possible. Confirm
   the path contains the Zoo project, not just an installed package.
3. Check whether each checkout is behind its configured upstream. State that
   remote-tracking information may be stale unless fetched. If the user wants
   newer remote commits, ask before fetching or changing either checkout;
   never discard local changes, switch branches, or pull without permission.
   Otherwise install the current files and do not call them the latest remote
   commits.
4. Identify the intended Studio home and existing installation before making
   changes. Respect `UNSLOTH_STUDIO_HOME`, `STUDIO_HOME`, and `UNSLOTH_HOME`
   and the installer's precedence. The default is `~/.unsloth/studio`. Ask
   when the target is unclear. The installer may replace its managed virtual
   environment and download dependencies; preserve its normal rollback.
   Tell the user to close a running Studio instance before replacement.

## Install

1. Run the selected Unsloth checkout's installer from its root:
   - Windows PowerShell: `.\install.ps1 --local`
   - Linux, macOS, or WSL: `./install.sh --local`

   Set `UNSLOTH_SKIP_AUTOSTART=1` for the installer process. Use the selected
   Studio home consistently for installation and launch. Do not use
   `--no-rollback` unless the user explicitly requests it. Follow the
   installer's platform and accelerator choices; do not replace its PyTorch
   installation with a generic wheel.

2. Check the installer exit status and output. Stop on failure; report
   degraded components and warnings rather than calling a partial install
   complete. The installer builds the Studio frontend and installs Unsloth
   editable, but its local mode also installs Zoo from Git.
3. Resolve the Python interpreter in the actual Studio home:
   `unsloth_studio\Scripts\python.exe` on Windows or
   `unsloth_studio/bin/python` on Linux, macOS, or WSL. Use that interpreter,
   not a system Python or another virtual environment.
4. After the installer and its setup finish, install the chosen Zoo checkout
   editable into that environment without resolving dependencies again. Use
   `uv pip install --python <studio-python> --no-deps --reinstall-package
   unsloth-zoo -e <zoo-checkout>`, or
   `<studio-python> -m pip install --no-deps --force-reinstall -e
   <zoo-checkout>` if that environment has pip. Replace the angle-bracketed
   values with the resolved paths and pass paths as separate quoted arguments.
   Do not run Studio setup again afterward: it can replace the Zoo overlay.
   If the overlay fails, report that the installer may still contain the
   Git-sourced Zoo, not the requested checkout.

Do not rebuild or replace llama.cpp as part of this skill. Preserve an
explicitly configured local llama.cpp build if the user wants to keep it; use
the current installer's supported local-build option and verify its path
before installation. Otherwise let the installer manage its normal runtime.

## Verify and report

1. Use the Studio interpreter's distribution metadata, including
   `direct_url.json`, to verify that both `unsloth` and `unsloth-zoo` are
   editable installs from the selected checkout paths. Compare resolved paths,
   not just version strings. When the selected mode supports imports, import
   both packages and verify their module paths too. For a no-torch install,
   do not require training imports that the mode intentionally omits.
2. Check the installed dependency requirements against the selected Zoo
   version and run the environment's available package consistency check.
   Investigate and report conflicts, including platform-specific or
   optional-package gaps; do not silently upgrade or replace accelerator
   packages to make the check pass. A missing required dependency or an
   incompatible Zoo version means the requested install is not fully
   verified.
3. Verify the installed Studio launcher starts and, when practical, start a
   temporary instance on an unused loopback port and check `/api/health`.
   Confirm the response belongs to that instance, then stop only the process
   started for the check. Report if startup or health could not be verified.
4. Report the two checkout paths and commits, Studio home, launch command,
   and any remaining failures or degraded capabilities. Explain that future
   installer or Studio update runs can replace the local Zoo overlay, so this
   skill must be run again to restore it.
