# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import os
import sys

if __name__ == "__main__":
    # Set the directory containing your files
    directory = sys.argv[1]

    # List all files in the directory
    files = os.listdir(directory)

    # Sort files if necessary
    files.sort()  # This sorts in lexicographical order

    # Rename each file
    for i, filename in enumerate(files, start=1):
        old_path = os.path.join(directory, filename)
        _, file_extension = os.path.splitext(filename)
        new_path = os.path.join(directory, f"{i}{file_extension}")
        os.rename(old_path, new_path)
