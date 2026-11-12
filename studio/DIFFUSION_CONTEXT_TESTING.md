# Test the coordinated DiffusionGemma context fix

Use the `diffusion-context-settings` branch from **all three** forks:

| Component | Fork branch |
| --- | --- |
| Studio | <https://github.com/brandonh-msft/unsloth/tree/diffusion-context-settings> |
| Zoo shim | <https://github.com/brandonh-msft/unsloth-zoo/tree/diffusion-context-settings> |
| Native runner | <https://github.com/brandonh-msft/llama.cpp/tree/diffusion-context-settings> |

These are development branches for [unslothai/unsloth#17558](https://github.com/unslothai/unsloth/issues/17558),
not released packages. Installing only the Studio fork does not replace an
installed Zoo shim or native executable.

An explicit positive context must be used exactly, or loading must fail with an
actionable error. Zero retains native automatic sizing. DiffusionGemma remains
single-request: Studio displays one parallel slot instead of offering settings
that the native runner cannot honor.

## Prerequisites

- An installed, working Unsloth Studio Python environment and NVIDIA driver.
- Git, PowerShell, and `wslc` with working NVIDIA GPU container access for the
  Windows adapter below. This adapter targets Windows x64 or ARM64; it is not
  a general Docker or Linux launcher.
- The host's corporate npm and PyPI feed configuration. Builds receive these
  files as BuildKit secrets, not repository files or build arguments. Preserve
  certificate verification. If your feed needs a corporate CA, use an approved
  base image with that CA installed.
- Enough memory for the **real** model and selected context. Small synthetic
  tests validate propagation and execution, not full-model VRAM requirements.

Clone sibling directories into a new testing workspace:

```powershell
gh repo clone brandonh-msft/unsloth -- --branch diffusion-context-settings
gh repo clone brandonh-msft/unsloth-zoo -- --branch diffusion-context-settings
gh repo clone brandonh-msft/llama.cpp -- --branch diffusion-context-settings

$Studio = (Resolve-Path .\unsloth).Path
$Zoo = (Resolve-Path .\unsloth-zoo).Path
$Native = (Resolve-Path .\llama.cpp).Path
$Dev = Join-Path $Native 'examples\diffusion-gemma-server\dev'
$Frontend = Join-Path $Studio 'studio\frontend'
```

Keep feed files and the driver outside these checkouts. Set these to your
existing corporate configuration files; do not check them in:

```powershell
$PipConfig = (Resolve-Path 'C:\private\pip.conf').Path
$NpmConfig = (Resolve-Path 'C:\private\.npmrc').Path
$Private = Join-Path $env:TEMP ("diffusion-build-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $Private | Out-Null
```

For an environment-variable-only host configuration, create those private files
from the configured corporate `PIP_INDEX_URL` and `npm_config_registry` values
first. Do not silently fall back to public feeds. Include any required feed
authentication in the private configuration, never in the Dockerfiles.

## Native Windows runtime

Use a native Windows CUDA runner when testing on the Windows host. The
container runner and Windows adapter described below are an alternative
**container runtime**, not a cross-compiled Windows runner: the adapter starts
a WSL container for every model load. Building a Linux executable in WSL does
not make it runnable on Windows.

On Windows ARM64, open an ARM64 Visual Studio Developer PowerShell with CMake
and Ninja available. Set `CUDA_PATH` to a Windows CUDA toolkit containing
`bin\nvcc.exe` and the ARM64 target libraries in `lib\arm64`. CUDA 13.4.92
supports the tested RTX Spark N1X target and the Visual Studio 2026 compiler.
Keep this toolkit outside the checkouts.

```powershell
$Build = Join-Path $Native 'build-windows-arm64'
cmake -S $Native -B $Build -G Ninja `
  "-DCMAKE_TOOLCHAIN_FILE=$Native\cmake\arm64-windows-msvc-cuda.cmake" `
  "-DCUDAToolkit_ROOT=$env:CUDA_PATH" `
  "-DCMAKE_CROSSCOMPILING_EMULATOR=$env:ComSpec;/c" `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121a-real `
  -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DGGML_NATIVE=ON `
  -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF `
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_TESTS=ON `
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TOOLS=OFF `
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=ON
if ($LASTEXITCODE) { throw 'Native Windows configuration failed' }
cmake --build $Build --target llama-diffusion-gemma-visual-server `
  test-diffusion-context-sizing test-llama-archs --parallel 4
if ($LASTEXITCODE) { throw 'Native Windows runner build failed' }

# The target matches this host, so compiler probes can run directly through cmd.
# Use a CUDA architecture appropriate for the destination GPU on other machines.
Get-ChildItem "$env:CUDA_PATH\bin\arm64" -Filter '*.dll' |
  Copy-Item -Destination "$Build\bin"

$env:DG_BRIDGE_PYTHON = (Resolve-Path "$HOME\.unsloth\studio\unsloth_studio\Scripts\python.exe").Path
$env:DG_VISUAL_BIN = Join-Path $Build 'bin\llama-diffusion-gemma-visual-server.exe'
$env:DG_ZOO_SOURCE = $Zoo
$env:PYTHONPATH = "$Studio;$Zoo"
& $env:DG_VISUAL_BIN --capabilities
& "$Build\bin\test-diffusion-context-sizing.exe"
& "$Build\bin\test-llama-archs.exe" -a diffusion-gemma -b CUDA0
```

The capability response must include `exact_context_v1`. Skip the container
runner, Windows adapter, and container-only synthetic checks below, then build
the UI and launch Studio with these environment variables. The model runner
must appear as a Windows process; no model container should start.

Windows GPU memory budgets can include RAM outside the CPU-visible memory
pool. The native runner checks CPU and GPU budgets separately. Do not compare
the combined GPU context and host staging allocations against CPU RAM alone,
or replace the CUDA budget with WSL guest RAM.

## Container alternative: build the CUDA runner

The development image installs its compiler tools and CUDA toolkit **inside
the container**. No CMake, Ninja, Zig, or CUDA toolkit installation on Windows
is required. `BASE_IMAGE` can select an approved Python image containing Git,
GCC and G++; its default is `python:3.14-bookworm`.

The build needs a link-time CUDA driver library. Copy the library already
provided by the host's WSL GPU integration, rather than downloading a driver
or committing one:

```powershell
$DriverContainer = "diffusion-driver-" + [guid]::NewGuid().ToString('N')
wslc run --name $DriverContainer --gpus all --entrypoint sh python:3.14-bookworm `
  -c 'cp /usr/lib/wsl/lib/libcuda.so /tmp/libcuda.so'
if ($LASTEXITCODE) { throw 'WSL CUDA driver extraction failed' }
wslc cp "${DriverContainer}:/tmp/libcuda.so" "$Private\"
wslc remove $DriverContainer
$Driver = Join-Path $Private 'libcuda.so'
```

Choose `CUDA_ARCHITECTURES` for the **destination GPU**, not the original
machine. `121a-real` is the tested RTX Spark N1X target; it is not portable
to other GPU architectures. Use a target supported by the included CUDA
compiler and your GPU, such as `89-real` for Ada or `120-real` for applicable
Blackwell desktop GPUs.

```powershell
$CudaArchitecture = '121a-real' # Change for the destination GPU.
wslc build --file "$Dev\Dockerfile" `
  --secret "id=pip_config,src=$PipConfig" `
  --secret "id=cuda_driver,src=$Driver" `
  --build-arg "CUDA_ARCHITECTURES=$CudaArchitecture" --build-arg BUILD_JOBS=4 `
  --tag diffusion-context-runner:local $Native
if ($LASTEXITCODE) { throw 'CUDA runner build failed' }

wslc run --rm --gpus all --network none diffusion-context-runner:local --capabilities
wslc run --rm --gpus all --network none --entrypoint /build/bin/test-diffusion-context-sizing `
  diffusion-context-runner:local
wslc run --rm --gpus all --network none --entrypoint /build/bin/test-llama-archs `
  diffusion-context-runner:local -a diffusion-gemma -b CUDA0
```

The capability response must include `exact_context_v1`. Runtime driver access
comes from `--gpus all`; the build secret is not copied into the image. CUDA
compilation may take several minutes. The BuildKit cache preserves compiled
objects across development rebuilds.

## Container alternative: build the Windows adapter

The adapter exposes the native stdin/stdout protocol as a Windows executable.
It preserves Unicode paths, mounts only the model directory and a temporary
request directory read-only, disables container networking, and removes its
named model container on normal shutdown or launcher termination.

Build the executable for the Windows host architecture, independently of the
GPU architecture:

```powershell
$WindowsTarget = 'x86_64-windows-gnu' # Use aarch64-windows-gnu on Windows ARM64.
wslc run --rm --network none `
  --mount "type=bind,source=$Dev,target=/bridge" --entrypoint python3 `
  diffusion-context-runner:local -m ziglang cc -target $WindowsTarget `
  -municode -O2 /bridge/launcher.c -o /bridge/diffusion-context-runner.exe
if ($LASTEXITCODE) { throw 'Windows adapter build failed' }

$env:DG_BRIDGE_PYTHON = (Resolve-Path "$HOME\.unsloth\studio\unsloth_studio\Scripts\python.exe").Path
$env:DG_VISUAL_BIN = Join-Path $Dev 'diffusion-context-runner.exe'
$env:DG_ZOO_SOURCE = $Zoo
$env:PYTHONPATH = "$Studio;$Zoo"
$env:DG_CONTAINER_IMAGE = 'diffusion-context-runner:local'
& $env:DG_VISUAL_BIN --capabilities
```

Adjust `DG_BRIDGE_PYTHON` if Studio is installed elsewhere. The launcher
requires it explicitly; it never assumes another user's Python path.
`DG_WSLC` can override the `wslc` executable path if it is not discoverable.
Keep `bridge.py` beside the generated executable. No installed native binary
is overwritten.

## Optional synthetic GPU checks

These generate a tiny random-weight test model, **not DiffusionGemma weights**.
The fixture intentionally reuses this branch's internal architecture test
helpers. Generate it after building the matching native branch:

```powershell
New-Item -ItemType Directory -Force "$Dev\fixtures" | Out-Null
wslc run --rm --gpus all --network none `
  --mount "type=bind,source=$Dev,target=/bridge" --entrypoint sh `
  diffusion-context-runner:local -c `
  'c++ -std=c++17 -O2 -ffunction-sections -fdata-sections -I/src/common -I/src/include -I/src/ggml/include -I/src/src /bridge/make_fixture.cpp -L/build/bin -Wl,--gc-sections -Wl,-rpath-link,/opt/cuda/lib -Wl,-rpath-link,/usr/lib/wsl/lib -lllama-common -lllama -lggml -lggml-base -o /tmp/make-fixture && /tmp/make-fixture /bridge/fixtures/diffusion-gemma-synthetic.gguf'
if ($LASTEXITCODE) { throw 'Synthetic fixture generation failed' }

& $env:DG_BRIDGE_PYTHON "$Dev\verify_bridge.py"
if ($LASTEXITCODE) { throw 'GPU bridge verification failed' }
& $env:DG_BRIDGE_PYTHON "$Dev\verify_shim.py"
if ($LASTEXITCODE) { throw 'Shim verification failed' }
```

The bridge probe verifies READY and generation at 4,096, 32,768 and **65,535**
tokens, an approximately 49,726-token prompt rejected at 32,768 but accepted
at 65,535, subsequent short-prompt reuse, Unicode filenames, and normal/forced
container cleanup. The shim probe checks that `/health` publishes both
requested and resolved context as 65,535. Neither establishes that the real
model fits a particular context on your GPU.

## Build the Studio UI and launch source

The UI development image receives the host's configured corporate npm feed
as a secret. The output is copied out; Node tooling stays in the container.

```powershell
wslc build --file "$Frontend\dev\Dockerfile" `
  --secret "id=npm_config,src=$NpmConfig" `
  --tag diffusion-context-frontend:local $Frontend
if ($LASTEXITCODE) { throw 'Frontend build failed' }
$UiContainer = "diffusion-ui-" + [guid]::NewGuid().ToString('N')
wslc run --name $UiContainer --entrypoint sh diffusion-context-frontend:local -c true
wslc cp "${UiContainer}:/app/dist" "$Frontend\"
wslc remove $UiContainer

$env:UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK = '1'
& $env:DG_BRIDGE_PYTHON "$Studio\studio\backend\run.py" `
  --host 127.0.0.1 --port 8888 --no-cloudflare --frontend "$Frontend\dist"
```

Stop other Studio instances before launching. Open <http://127.0.0.1:8888>.
The `PYTHONPATH` and `DG_VISUAL_BIN` overrides must remain set in the process
launching Studio; otherwise the installed Zoo or runner may be selected.
Existing Studio network-access preferences still apply.

Load your real GGUF, explicitly select a context larger than the previous
32,768-token automatic limit, and inspect its native startup log. Expect
`requested=<selected context>` and the same resolved `MAXTOK` / READY context,
or an explicit memory/compatibility error. There must be no silent reduction
to 32,768, 8,192 or 4,096. Confirm that Studio displays fixed concurrency of one.
Then run the original long conversation, reload at a smaller context, and
confirm the overflow error reports that newly selected context accurately.

On Linux, build the native fork with your approved CUDA/CMake toolchain and
point `DG_VISUAL_BIN` directly at
`build/bin/llama-diffusion-gemma-visual-server`; do not use the Windows adapter.
Use `:` rather than `;` to separate source directories in `PYTHONPATH`.

Generated executables, PDBs, fixtures, logs and UI `dist` are not part of the
forks. Keep real model weights, corporate feed credentials, and driver
libraries outside Git. Full-size model testing on the larger-VRAM machine
remains necessary.
