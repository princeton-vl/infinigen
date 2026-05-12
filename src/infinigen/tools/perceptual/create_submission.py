# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import csv
import os
import random
import sys


def select_random_files_to_csv(folder_path, k, output_directory):
    # Get all files in the folder
    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    # Select k random files
    selected_files = random.sample(files, min(k, len(files)))

    # Create CSV file with the name of the folder
    folder_name = os.path.basename(folder_path)
    csv_file_path = os.path.join(output_directory, f"{folder_name}-{k}.csv")

    # Write the selected file names to the CSV
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_url"])  # Header
        for file in selected_files:
            writer.writerow([file])

    print(f"CSV file created at {csv_file_path}")


if __name__ == "__main__":
    k = 50
    output_directory = sys.argv[1]
    folder_path = sys.argv[2]
    # Example usage
    select_random_files_to_csv(folder_path, k, output_directory)
