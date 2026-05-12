# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def merge_images(
    image_path1, image_path2, text1="Program A", text2="Program B", strip_width=5
):
    # Open the images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Resize the larger image to match the width of the smaller one
    if image1.width > image2.width:
        # Calculate new height to maintain aspect ratio
        new_height = int((image2.width / image1.width) * image1.height)
        image1 = image1.resize((image2.width, new_height))
    elif image2.width > image1.width:
        # Calculate new height to maintain aspect ratio
        new_height = int((image1.width / image2.width) * image2.height)
        image2 = image2.resize((image1.width, new_height))

    # Determine the max height
    max_height = max(image1.height, image2.height)

    # Create a new image with the combined width plus the strip width and the max height
    combined_width = image1.width + image2.width + strip_width
    combined_image = Image.new("RGB", (combined_width, max_height), "black")

    # Paste the two images into the new image
    # Adjust the position if one image is shorter than the other
    image1_y = (max_height - image1.height) // 2
    image2_y = (max_height - image2.height) // 2

    combined_image.paste(image1, (0, image1_y))
    combined_image.paste(image2, (image1.width + strip_width, image2_y))

    # Add text
    font_size = 40
    draw = ImageDraw.Draw(combined_image)
    try:
        # Load a specific TrueType or OpenType font file
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Black.ttf", font_size
        )
    except IOError:
        # If the specific font file is not found, load the default font
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    text_color = (255, 0, 0)  # White color

    # Calculate text position
    text1_x = 10
    text1_y = 10
    text2_x = image1.width + strip_width + 10
    text2_y = 10

    draw.text((text1_x, text1_y), text1, fill=text_color, font=font)
    draw.text((text2_x, text2_y), text2, fill=text_color, font=font)

    # Save the combined image
    return combined_image


def merge_images2(
    image_path1, image_path2, text1="Program A", text2="Program B", strip_width=5
):
    # Open the images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Resize the larger image to match the height of the smaller one
    if image1.height > image2.height:
        # Calculate new width to maintain aspect ratio
        new_width = int((image2.height / image1.height) * image1.width)
        image1 = image1.resize((new_width, image2.height))
    elif image2.height > image1.height:
        # Calculate new width to maintain aspect ratio
        new_width = int((image1.height / image2.height) * image2.width)
        image2 = image2.resize((new_width, image1.height))

    # Determine the max width after resizing
    max_width = image1.width + image2.width + strip_width

    # Create a new image with the max width and the combined height
    combined_image = Image.new("RGB", (max_width, image1.height), "black")

    # Paste the two images into the new image
    image1_x = (max_width - image1.width - image2.width - strip_width) // 2
    image2_x = image1_x + image1.width + strip_width

    combined_image.paste(image1, (image1_x, 0))
    combined_image.paste(image2, (image2_x, 0))

    # Add text
    font_size = 40
    draw = ImageDraw.Draw(combined_image)
    try:
        # Load a specific TrueType or OpenType font file
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", font_size
        )
    except IOError:
        # If the specific font file is not found, load the default font
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    text_color = (255, 0, 0)  # Red color

    # Calculate text position
    text1_x = 10
    text1_y = 10
    text2_x = image1.width + strip_width + 10
    text2_y = 10

    draw.text((text1_x, text1_y), text1, fill=text_color, font=font)
    draw.text((text2_x, text2_y), text2, fill=text_color, font=font)

    # Save the combined image
    return combined_image


if __name__ == "__main__":
    # methods = ['eevee', 'fastsynth']
    # perspective = 'first_person'
    main_directory = sys.argv[1]
    methods = sys.argv[2]
    perspective = sys.argv[3]
    output_directory = sys.argv[4]

    random_seed = 1234  # You can choose any number as the seed
    random.seed(random_seed)

    k = 50
    # Set your main directory, methods, and perspective here

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Building paths for both methods
    path_method1 = os.path.join(main_directory, methods[0], perspective)
    path_method2 = os.path.join(main_directory, methods[1], perspective)

    # List of images in each method's perspective directory
    images_method1 = os.listdir(path_method1)
    images_method2 = os.listdir(path_method2)

    # Randomly select k images from each list (or all images if there are fewer than k)
    random_images_method1 = random.sample(images_method1, min(k, len(images_method1)))
    random_images_method2 = random.sample(images_method2, min(k, len(images_method2)))

    # Iterate over each randomly selected image in method1 and pair it with each randomly selected image in method2
    for img1 in tqdm(random_images_method1):
        for img2 in random_images_method2:
            image_path_1 = os.path.join(path_method1, img1)
            image_path_2 = os.path.join(path_method2, img2)
            # Extracting image identifiers
            img_0_id = img1.split(".")[0]
            img_1_id = img2.split(".")[0]

            # skip if not image
            if not (image_path_1.endswith(".png") or image_path_1.endswith(".jpg")):
                continue
            if not (image_path_2.endswith(".png") or image_path_2.endswith(".jpg")):
                continue

            # Creating a unique filename for the merged image
            merged_filename = (
                f"{perspective}-{methods[0]}-{img_0_id}-{methods[1]}-{img_1_id}.jpg"
            )
            merged_image_path = os.path.join(output_directory, merged_filename)

            # Merge and save images
            merged_img = merge_images2(image_path_1, image_path_2)
            merged_img.save(merged_image_path, "JPEG")
