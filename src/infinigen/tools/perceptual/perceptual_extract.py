# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import os
import shutil
import sys

if __name__ == "__main__":
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    # Supported image formats
    image_formats = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for folder_name in os.listdir(input_directory):
        folder_path = os.path.join(input_directory, folder_name)

        if os.path.isdir(folder_path):
            # Find the image file inside the folder
            for file_name in os.listdir(folder_path):
                if any(file_name.lower().endswith(ext) for ext in image_formats):
                    old_file_path = os.path.join(folder_path, file_name)
                    new_file_name = (
                        f"{folder_name}.png"  # Change the extension if needed
                    )
                    new_file_path = os.path.join(output_directory, new_file_name)

                    # Move and rename the image file
                    shutil.copy(old_file_path, new_file_path)
                    break  # Assuming only one image per folder
