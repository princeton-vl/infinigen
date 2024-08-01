# Static Assets: Import External Assets to Infinigen Indoors

In this guide, we will show you how to import external (static) assets into Infinigen. This is useful if you want to create scenes with categories of assets that are not covered by Infinigen (say sculptures, etc), or if you just want to add more variety to your scenes with custom assets. 


## Download the Objects

We will get started by downloading the objects we want to import. For this example, we will use a shelf, a leather couch, and a table from [Objaverse](https://objaverse.allenai.org/). You can use any objects you like, as long as it is one of the following formats:
- .dae
- .abc
- .usd
- .obj
- .ply
- .stl
- .fbx
- .glb
- .gltf
- .blend

I would recommend using .glb or .gltf files since they are the most extensively tested.

1. Download the following assets in .glb format from Sketchfab: [Iron Shelf](https://sketchfab.com/3d-models/iron-shelf-fd0cd420ffe04ac08174926f6b175d3f), [Office Couch](https://sketchfab.com/3d-models/office-couch-ca63db1db205476fa6f54e1603b7d15d), [Table](https://sketchfab.com/3d-models/de-table-63e4d8faac73435fa7e9e929baa2c175).
<p align="center">
  <img src="images/static_assets/shelf.jpg" width="236" />
  <img src="images/static_assets/couch.jpg" width="300" />
  <img src="images/static_assets/table.jpg" width="300" />
</p>

2. Create a folder for each category you want to import in `infinigen/assets/static_assets/source`. That is, create the folders `infinigen/assets/static_assets/Shelf`, `infinigen/assets/static_assets/Couch`, and `infinigen/assets/static_assets/Table`.

```bash
mkdir infinigen/assets/static_assets/source
mkdir infinigen/assets/static_assets/source/Shelf
mkdir infinigen/assets/static_assets/source/Couch
mkdir infinigen/assets/static_assets/source/Table
```

3. Place the downloaded .glb files in the corresponding folders. For this example, you should have three folders with one .glb file each.

Now, whenever we tell Infinigen to import a static Shelf asset in the scene, it will look for the .glb file in the Shelf folder we created. If you have multiple objects within the same category, Infinigen will randomly choose one of them each time that it wants to place in the scene. You can also put objects of different formats in the same folder: `shelf1.glb, shelf2.obj, shelf3.fbx`, etc.

NOTE: Objaverse supports downloads using python [API](https://colab.research.google.com/drive/15XpZMjrHXuky0IgBbXcsUtb_0g-XWYmN?usp=sharing) if you want to download a large number of objects. 

## Preprocess the Objects

We will need to preprocess the objects to make them compatible with Infinigen scenes. For each object you downloaded, do the following:

1. Launch Blender and remove the default cube, lighting, etc.

2. Click file &rarr; Import &rarr; glTF 2.0 (.glb/.gltf) and select the .glb file you downloaded.

3. Click on the object in the Outliner panel on the right hand side and press `s` to scale it to a reasonable size in meters. 

4. Choose ViewLayer &rarr; Transform &rarr; Rotation &rarr; XYZ Euler. Rotate the object so that the front is facing the positive x-axis. The bottom of the object should face the -z direction.

5. Click file &rarr; Export &rarr; glTF 2.0 (.glb/.gltf) and replace the original .glb file with the new one.

You might not need to do all these steps for all objects, but you should make sure that:

- The objects are a reasonable size. This is because Infinigen cannot infer what the size of the object should be, so it will use the size of the object in the .glb file. If the object is too big or too small, it will look out of place in the scene or it will not fit in the scene which means Infinigen will fail to place it. 
- The object is facing the right direction. This is because Infinigen assumes that the front is facing +x, the back is facing -x, the bottom is facing -z, and the top is facing +z. For instance, this information is used when placing a couch against the wall. 

Here's an example of adjusted dimensions and orientation for the shelf and the couch:

<p align="center">
  <img src="images/static_assets/shelf_edit.jpg" width="480" />
  <img src="images/static_assets/couch_edit.jpg" width="480" />
</p>

## Create Static Asset Category

Now that we have our objects ready, we just need to tell Infinigen to treat them as it would any other asset class. This means that we need to define an `AssetFactory`, which can call `fac.create_asset()` to create a new asset. In our case, we want `fac.create_asset()` to import the external assets we just downloaded. Luckily, the `static_category_factory()` does all the heavy lifting for us. We just need to define `Static{CategoryName}Factory = static_category_factory({CategoryName})`. Specifically, you would add the following lines to `infinigen/assets/static_assets/static_category.py`:

![alt text](images/static_assets/image.jpg)

These lines should already be in that file, but if you want to add more categories, you can add more lines like the ones above. Just replace `{CategoryName}` with the name of the category you want to import. Make sure that the name of the category matches the name of the folder you created in the `static_assets/source` folder. Also, don't forget to add `Static{CategoryName}Factory` to `infinigen/assets/static_assets/__init__.py` so that we can import it later. Again, this step is already done for our example: 

![alt text](images/static_assets/image2.jpg)

## Define Semantics

Infinigen allows the user to specify high-level semantics for the objects in the scene. These semantics are then used to define high-level constraints. For example, we want to say that our static shelf factory is a type of storage unit, which will be placed against the wall, and there will be a bunch of objects on top of it. In general, if you want your static object factory to be treated like an existing asset factory, you can just imitate the semantics of the existing asset factory. Let's demonstrate this idea by defining semantics for our static shelf. We go to `infinigen_examples/indoor_asset_semantics.py` and search for `LargeShelfFactory`. We see that it is used as `Semantics.Storage` and `Semantics.AssetPlaceholderForChildren`. We want our static shelf to be used as a storage unit as well, so we add a line for our new static factory:
![alt text](images/static_assets/image3.jpg)

Similarly, we add `StaticShelfFactory` to `Semantics.AssetPlaceholderForChildren`. This will replace the placeholder bounding box for the shelf before placing the small objects. 
![alt text](images/static_assets/image7.jpg)

The semantics for the couch and the table is very similar. We just define the same semantics as the existing asset factories that are similar to our static ones: 

![alt text](images/static_assets/image4.jpg)

![alt text](images/static_assets/image5.jpg)

![alt text](images/static_assets/image6.jpg)

If your category is not similar to any existing category, you would need to think about it a little bit more and define your own semantics. For example, I found that the sofa semantics work well for vending machine as they are both placed against the wall, etc. 

## Add Constraints

We are almost there! The last step is to add constraints for our static assets. We want to make sure that the shelf is placed against the wall, the couch is placed in the living room, and the table is placed in the dining room, etc. Luckily our new static assets are similar to existing assets, so we can just replace the existing constraints with our new static assets. We go to `infinigen_examples/indoor_constraint_examples.py` and search for `LargeShelfFactory`, `SofaFactory`, and `TableDiningFactory`. We just replace these constraints with our new static assets: 

![alt text](images/static_assets/image8.jpg)

![alt text](images/static_assets/image9.jpg)

![alt text](images/static_assets/image10.jpg)

If you have some new asset that does not behave like any of the existing assets, you would need to define new constraints.


## Generate Scene

And, that's it! You can now generate a scene with your new static assets. We just use the regular Infinigen commands:

```bash
python -m infinigen_examples.generate_indoors --seed 0 --task coarse --output_folder outputs/indoors/coarse0 -g fast_solve.gin singleroom.gin -p compose_indoors.terrain_enabled=False restrict_solving.restrict_parent_rooms=\[\"LivingRoom\"\]
```

![](images/static_assets/untitled8.jpg)

```bash
python -m infinigen_examples.generate_indoors --seed 1 --task coarse --output_folder outputs/indoors/coarse1 -g fast_solve.gin singleroom.gin -p compose_indoors.terrain_enabled=False restrict_solving.restrict_parent_rooms=\[\"LivingRoom\"\]
```
![](images/static_assets/untitled9.jpg)

```bash
python -m infinigen_examples.generate_indoors --seed 11 --task coarse --output_folder outputs/indoors/coarse0dining -g fast_solve.gin singleroom.gin -p compose_indoors.terrain_enabled=False restrict_solving.restrict_parent_rooms=\[\"DiningRoom\"\]
```
![](images/static_assets/untitled11.jpg)

You can see that our new static assets are placed in the scene. What's more is that they interact with the existing Infinigen assets. For example, we have a bowl of fruit and a glass on top of the table. 

## Modify Objects with Code (Optional)

If you want to post-process the imported static objects or customize the object creation, you can define your own static category class similar to `StaticCategoryFactory` in `infinigen/assets/static_assets/static_category.py`. For instance, you can add a `finalize_assets(self, assets)` function to your class to post-process the imported assets. 

## Support Surfaces (Optional)

A shelf should have small objects placed on it. But, how can Infinigen know where to place these objects? The default is that the top surface of the object is used as the surface for placing objects. For instance, in the above example, we saw that some objects were automatically placed on the table even though we did not specify where to place them. A shelf has multiple surfaces, so we need to tag these surfaces as so-called "support surfaces". This is very easy to do in the case where the support surfaces are the distinct planes along the z direction, as is the case with shelves. We just enable the option `StaticShelfFactory = static_category_factory("Shelf", tag_support=True)` in `infinigen/assets/static_assets/static_category.py`:

![alt text](images/static_assets/image11.jpg)

Now when we generate the scene, we see that the objects are placed on all surfaces of the shelf:

![alt text](images/static_assets/shelf_support.jpg)

## Summary for arbitrary objects

Let us summarize all the steps you would need to follow to import arbitrary objects into Infinigen:

1. Download the objects in one of the supported formats.

2. Create a folder for this object category in `infinigen/assets/static_assets/source`. E.g.

```bash
mkdir infinigen/assets/static_assets/source/MyCategory
```

3. Place the downloaded objects in the folder you created.

4. Open each object in Blender and adjust the size and orientation so that the size is reasonable and the front is facing the positive x-axis.

5. Export the objects in the same format and replace the old versions in the folder.

6. Add a line to `infinigen/assets/static_assets/static_category.py` to define the factory for this category. E.g.
    
```python
StaticMyCategoryFactory = static_category_factory("MyCategory")
```

7. Add a line to `infinigen/assets/static_assets/__init__.py` to import the factory from other files.

8. Define the semantics for the objects in `infinigen_examples/indoor_asset_semantics.py`. E.g.

```python
used_as[Semantics.Furniture] = {...
                                static_assets.StaticMyCategoryFactory}
```

9. Define the constraints for the objects in `infinigen_examples/indoor_constraint_examples.py`. E.g.

```python
my_cat_against_wall = wallfurn[static_assets.StaticMyCategoryFactory]
...
```

10. Generate the scene using the regular Infinigen commands.

## Some Other Examples

Now that you know how to import static assets, you can import all kinds of different objects. Here are some creative examples:

<p align="center">
  <img src="images/static_assets/untitled12.jpg" width="500" />
  <img src="images/static_assets/untitled13.jpg" width="500" />
  <img src="images/static_assets/untitled14.jpg" width="500" />
  <img src="images/static_assets/untitled15.jpg" width="500" />
  <img src="images/static_assets/untitled16.jpg" width="500" />
  <img src="images/static_assets/vending.jpg" width="500" />
</p>

