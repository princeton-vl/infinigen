# Generate a scene layout
python infinigen_examples/generate_nature.py -- --seed 0 --task coarse -g desert.gin simple.gin --output_folder outputs/helloworld/coarse

# Populate unique assets
python infinigen_examples/generate_nature.py -- --seed 0 --task populate fine_terrain -g desert.gin simple.gin --input_folder outputs/helloworld/coarse --output_folder outputs/helloworld/fine

# Render RGB images
python infinigen_examples/generate_nature.py -- --seed 0 --task render -g desert.gin simple.gin --input_folder outputs/helloworld/fine --output_folder outputs/helloworld/frames

# Render again for accurate ground-truth
python infinigen_examples/generate_nature.py -- --seed 0 --task render -g desert.gin simple.gin --input_folder outputs/helloworld/fine --output_folder outputs/helloworld/frames -p render.render_image_func=@flat/render_image 