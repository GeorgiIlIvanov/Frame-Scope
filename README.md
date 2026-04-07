The purpose of this repo is to provide feature geometry visualization and analytics.

The fundamental unit of analysis is the frame operators of each layer.

## Local Gemma Scope 1B setup

We are starting locally with the Gemma 3 `1b` family plus a selective subset of Gemma Scope 2 SAEs so we have enough headroom for:

- local token generation
- frame-operator extraction and visualization
- feature-vector projection analysis during generation

The initial local target is:

- base model: `google/gemma-3-1b-it`
- optional analysis base: `google/gemma-3-1b-pt`
- SAE repos: `google/gemma-scope-2-1b-it` and `google/gemma-scope-2-1b-pt`
- sites: `resid_post`, `attn_out`, `mlp_out`, `transcoder`
- layers: `7`, `13`, `17`, `22`
- width/L0: `65k`, `medium`

### Repo layout

- `manifests/gemma_scope_1b.json`: selective local download manifest
- `scripts/bootstrap_macos.sh`: creates a project virtualenv and installs dependencies
- `scripts/download_gemma_scope_1b.py`: downloads the selected Gemma 3 and Gemma Scope assets
- `scripts/materialize_gemma_scope_cfgs.py`: converts downloaded raw Gemma Scope folders into `sae-lens` local disk format by generating `cfg.json`

### Planned local artifact layout

- `artifacts/models/gemma-3-1b-it`
- `artifacts/models/gemma-3-1b-pt`
- `artifacts/saes/gemma-scope-2-1b-it`
- `artifacts/saes/gemma-scope-2-1b-pt`

### Notes

- The Gemma Scope repos are open and can be downloaded selectively.
- The base Gemma 3 repos are gated, so local download requires a Hugging Face account that has accepted the Gemma license.
- We are intentionally not mirroring the full Gemma Scope repos because they are far larger than what we need for Frame-Scope iteration.
