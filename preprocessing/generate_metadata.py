from glob import glob
import pandas as pd
import os



if __name__ == "__main__":
    dataset = "BRACS_RoI"

    if dataset == "BRACS_RoI":
        images_dir = f"/capstor/scratch/cscs/vsubrama/data/{dataset}/images/"
        list_of_images = sorted(glob(f"{images_dir}/*"))
        print(len(list_of_images))

        list_of_basenames = [os.path.basename(image_name) for image_name in list_of_images]
        # list_of_classes = [os.path.basename(image_name).split("-")[0] for image_name in list_of_images]

        data_df = pd.DataFrame({"Image Name": list_of_basenames, 
                                # "Tissue Class": list_of_classes
                                })
        print(data_df)

        data_df.to_csv(f"/capstor/scratch/cscs/vsubrama/data/{dataset}/images_metadata.csv")
