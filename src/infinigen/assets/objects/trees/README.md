# Tree Generation

Overview of the generation pipeline for trees and other plants.

## Initial geometry

To start, we define a skeleton for the tree. That is, we procedurally place and connect vertices defining the core tree shape. There are two approaches to this that are complementary to each other. Note, compelling trees can be made exclusively by using just one method or the other.

### Recursive paths

### Space colonization

## Branch thickness

With the skeleton laid out, we need to fill out the tree by defining the thickness of stems and branches. Originally this was done with a skinning modifier, but this has been updated as defining the radius in Geometry Nodes (GN) is easier to tune and yields a better end result.

We define a nodegroup that takes as input the tree skeleton and produces a filled out mesh by leveraging the function `Curve to Mesh`. The radius calculation is made possible by providing the additional attribute `rev depth` as input which specifies the distance to a leaf node for each vertex in the tree.

We get instant feedback when adjusting parameters and can carefully tune a function that looks nice and behaves well. A handful of parameters are sufficient to govern a variety of looks and these can easily be recorded for future use for a particular class of tree/plant.

Odd artifacts can pop up when converting to a mesh, and there are a couple of details which control the final look of the tree which we adjust to get the best looking mesh without going overboard on the number of vertices.

 - we turn off `Shade smooth` which is turned on by default with `Curve to Mesh`
 - the resolution of the profile curve can be adjusted with the exposed variable: `profile res`
 - weird behavior is usually due to vertices that are too close to each other, we can add `Merge by distance` to remove duplicates. Unfortunately this collapses the thin twigs. One way around this is to use `Separate Geometry` and apply the merge to the main trunk and large branches. This is not too much trouble to set up, but currently not implemented

**Note on materials:** Currently there's one pain point in terms of materials which is that we cannot get a good UV unwrapping until the mesh is finalized. The current code assigns a material to the generated mesh after `set radius` but you'll just notice a color change mostly because it does not map properly across the tree geometry. We provide a function `finalize` that can be called that will separate out the tree from its instances, finalize the mesh, and call the appropriate UV unwrapping.


## Instancing child geometry

## Nodes Tips + Tricks

In some ways the node setup makes life easier, but in others it can be a bit frustrating to write code to implement new ideas in nodes. While tedious, there is some incredible flexibility that is introduced. Here are some features that are made possible that I think are worth paying attention to:

- **partitioning of geometry:** Most functions in nodes take `selection` as an additional input which means you can apply a function to a subset of a given mesh. With the attributes we have available for the tree geometry this makes it straightforward to apply filters such that different functions are applied in different ways to different parts of the tree. For example, a `level` is defined to distinguish recursive levels of the tree paths, so the trunk is `level=0` and the roots are `level=-1`. We make use of this to instance child geometry on leaves of the tree while excluding the "leaves" on the roots. Another thing we can do is have different levels of detail at different regions, e.g. using a `Profile res` of `4` for small twigs and `16` for the trunk and branches _(not currently implemented)_. This reduces vertices without sacrificing visual fidelity.
