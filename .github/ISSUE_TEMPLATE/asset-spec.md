---
name: New Asset Specification
about: Create a proposal for a new asset to be developed
title: ""
labels: 'enhancement'
assignees: ''

---

### Goal

> Briefly describe the asset. What real-world object or phenomenon does it represent? What is its intended usage within the larger infinigen system?

### Reference Images

<!-- For each reference image, create a section like below. Describe the important
     visual features that the procedural generator should capture: shape, texture,
     color variation, surface detail, structural patterns, scale, etc. -->

#### Reference 1

> **Image:** (paste image here)
>
> **Key features:** Describe the important visual characteristics shown in this image that the generator should reproduce.

#### Reference 2

> **Image:** (paste image here)
>
> **Key features:** Describe the important visual characteristics shown in this image that the generator should reproduce.

<!-- Add more Reference sections as needed -->

### Interface

```python
def my_asset(
    ...ARGS...
) -> TYPE:
    pass

def my_asset_rand(
    rng: pf.RNG,
    ...ARGS...
) -> TYPE:
    pass
```

### Starter Code

> Is there any existing file or stub function that should be used as a starting point?

### Evaluation

> What commands will be used to evaluate if it works?

```bash
seq 8 | xargs -P 4 -n1 -I{} uv run infinigen !!!!!REPLACE!!!!! render_cycles --seed {} --output outputs/bricks/{} --quiet | ./scripts/image-grid.py 4x output.png -o
```

###### Baseline results

> If the existing branch can produce _something_ for this same command, paste the image here.
