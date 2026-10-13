# CS336 Spring 2026 Assignment 2: Systems

For a full description of the assignment, see the assignment handout at
[cs336_assignment2_systems.pdf](./cs336_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

The project contains two Python packages at the repository root:

- `cs336_basics/`: the Transformer model and utilities from Assignment 1.
- `cs336_systems/`: profiling and systems work for Assignment 2.
- `tests/`: the assignment tests.

Both packages are installed by the root `pyproject.toml`. The original Basics
README is preserved in `docs/basics.md`.

With uv, install dependencies and check that the model is importable:

```sh
uv run python -c "from cs336_basics.model import BasicsTransformerLM"
```

On zkrh, use the existing GPU environment:

```sh
source /data/private/niebinbin/envs/gpu/bin/activate
cd /data/private/niebinbin/test_and_study/cs336/assignment2-systems
python -m pip install -e . --no-deps
python -m cs336_basics.model
```

`--no-deps` uses the dependencies already installed in the GPU environment.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.
