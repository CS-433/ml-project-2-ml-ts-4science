#!/usr/bin/env python3
import os
import csv


def main():
    # Base BRACS directory
    bracs_dir = "/scratch/izar/dlopez/ml4science/data/BRACS"
    images_dir = os.path.join(bracs_dir, "images")
    output_file = os.path.join(bracs_dir, "labels.csv")

    # We'll look for all files in images_dir
    image_files = os.listdir(images_dir)

    # Open the CSV file for writing
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["file_name", "class"])

        for filename in image_files:
            # We expect something like BRACS_747_ADH_46.png
            # Let's split by underscore and extract the class
            # Filename format: BRACS_{n}_{CLASS}_{number}.png
            if filename.startswith("BRACS_") and filename.endswith(".png"):
                parts = filename.split("_")
                # Sanity check: we expect at least 4 parts: ["BRACS", "n", "CLASS", "number.png"]
                if len(parts) >= 4:
                    # class is in position 2
                    image_class = parts[2]

                    # Write line: file_name,class
                    writer.writerow([filename, image_class])


if __name__ == "__main__":
    main()
