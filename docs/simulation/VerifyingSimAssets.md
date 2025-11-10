# Verifying Simulation Assets

This document explains how to run tests for the simulation-ready assets in Infinigen via `pytest`.

## Running Tests

Tests are run using pytest:

```bash
# Run all tests for all assets
pytest tests/sim

# Run all tests for particular asset
pytest tests/sim --asset {asset name}

# Run specific test file for all assets
pytest tests/sim/test_blender_assets.py
```

## Options

| Option    | Description                                                        |
| --------- | ------------------------------------------------------------------ |
| `--asset` | Specify assets to test (e.g., `--asset door lamp`). Default: "all" |
| `--nr`    | Number of seeds to test per asset. Default: 10                     |

**Note**: If you are contributing a new simulation-ready asset, ensure that all tests are successful for your asset for 10 seeds (i.e. make sure to include `--nr 10` in your command).

## Additional Examples

```bash
# Run all tests for door
pytest tests/sim --asset door

# Run all tests for door using 10 different seeds per test
pytest tests/sim --asset door --nr 10

# Check vertex count for all assets. Will check seeds 0 through 4 for all assets.
pytest tests/sim/test_blender_assets.py::TestBlenderAssets::test_asset_vertex_count

# Check vertex count for door and drawer. Check for 10 different seeds (0 through 9).
pytest tests/sim/test_blender_assets.py::TestBlenderAssets::test_asset_vertex_count --asset door drawer --nr 10
```
