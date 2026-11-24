# Image to Three.js miner

Generates a Three.js module for a prompt image through the competition witness: drafts, selection, rewrites against
renders and a final pick, within the per-prompt budget. Every module is centered on the origin and scaled uniformly to
fit the unit cube.

Entry point: `main.py` -> `async generate(image, seed, witness) -> str | None`. The plan is `ACTIVE_PLAN` in
`miner404/versions.py`.

## Live run

```sh
uv venv --python 3.13 .venv-live
uv pip install --python .venv-live/bin/python -r requirements.txt
git clone --branch miner-runtime-v0.1.5 --depth 1 https://github.com/404-Repo/404-gen-subnet.git ../404-sdk
```

Create `witness.json` next to `main.py`, with `witness_url` set to `https://witness.subnet404.xyz` and
`proof_parameters` set to the absolute path of `404-sdk/witness-server/verifier/params`. Then:

```sh
.venv-live/bin/python main.py --prompts prompts.txt --seed PUBLISHED_SEED --wallet-name NAME --wallet-hotkey HOTKEY
```

## Third-party code

`miner404/patch.py` and the SEARCH/REPLACE edit format come from kaito0311/3d-gen-sub (MIT License, see `NOTICE.md`).
