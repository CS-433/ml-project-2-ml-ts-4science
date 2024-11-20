from glob import glob
import pandas as pd
import os
import yaml


if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    dataset = config["dataset"]

    images_dir = os.path.join(config["data_dir"], dataset, "images")
    list_of_images = sorted(glob(f"{images_dir}/*"))
    print(len(list_of_images))

    list_of_basenames = [os.path.basename(image_name) for image_name in list_of_images]
    # list_of_classes = [os.path.basename(image_name).split("-")[0] for image_name in list_of_images]

    data_df = pd.DataFrame({"Image Name": list_of_basenames, 
                            # "Tissue Class": list_of_classes
                            })
    print(data_df)

    save_dir = os.path.join(config["data_dir"], dataset, "images_metadata.csv")
    data_df.to_csv(save_dir)
