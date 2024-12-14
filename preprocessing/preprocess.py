import openslide
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
from typing import List, Tuple, Literal
from tqdm import tqdm
import yaml

Image.MAX_IMAGE_PIXELS = None

valid_datasets = ["BACH", "BRACS", "BreakHis"]

mpp_to_magnification = {
    0.25: "_40x",
    0.5: "_20x",
    1: "_10x",
    2: "_5x",
}

base_mpps_dict = {
    "BACH": 0.42,
    "BRACS": 0.25,  # in microns per pixel (MPP)
    "BreakHis": 0.25,
}

def make_sample_grid(
    image,
    patch_size=224,
    mpp=0.5,
    mult=1,
    base_mpp=None,
):
    """
    Script that given an openslide object return a list of tuples
    in the form of (x,y) coordinates for patch extraction of sample patches.
    mult is used to increase the resolution of the thumbnail to get finer  tissue extraction
    """

    rows = int(np.ceil(image.shape[0] * 1.0 / (patch_size * mpp / base_mpp))) * mult
    cols = int(np.ceil(image.shape[1] * 1.0 / (patch_size * mpp / base_mpp))) * mult

    if rows == 0 or cols == 0:
        raise ZeroDivisionError(
            f"Image size is too small for patch size {patch_size} and mpp {mpp}"
        )
    img = image[
        :: int(np.ceil(image.shape[0] / rows)), :: int(np.ceil(image.shape[1] / cols)), :
    ]
    img = np.ones((img.shape[0], img.shape[1]))

    w = np.where(img > 0)

    grid = list(
        zip(
            (w[1] * (patch_size * mpp / base_mpp)).astype(int),
            (w[0] * (patch_size * mpp / base_mpp)).astype(int),
        )
    )

    return grid

def add_subfolder(base_dir: str, subfolder_name: str) -> None:
    subdirs = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    for subdir in subdirs:
        tiles_metadata_path = os.path.join(subdir, subfolder_name)
        os.makedirs(tiles_metadata_path, exist_ok=True)

def save_metadata_to_file(
    data_dir: str,
    dataset: Literal[tuple(valid_datasets)],
    project_id: str,
    slide_id: str,
    tiles: List[Tuple[int]],
    patch_size: int,
    mpp: float,
    base_mpp: float,
    mult: float,
    magnification: str = "",
) -> None:
    if dataset in valid_datasets:
        json_file = f"{data_dir}/tiles_metadata_{patch_size}{magnification}/{slide_id}.json"
    else:
        raise ValueError(f"Unknown dataset {dataset}!")
    with open(json_file, "w") as f:
        json.dump(
            {
                "slide_id": slide_id,
                "project_id": project_id,
                "num_tiles": len(tiles),
                "patch_size": patch_size,
                "mpp": mpp,
                "base_mpp": base_mpp,
                "mult": mult,
                "tiles": tiles,
            },
            f,
            indent=4,
        )


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)


    ####### Configs ############################################
    dataset: Literal[tuple(valid_datasets)] = config["dataset"]

    image_base_mpp = base_mpps_dict.get(dataset)

    if not image_base_mpp:
        print(f"Unknown dataset {dataset}")
        return
    
    id_columns = {k: v["id_column"] for k, v in config["datasets"].items()}
    id_column = id_columns[dataset]

    data_dir = os.path.join(config["data_dir"], dataset)
    slide_metadata_path = f"{data_dir}/images_metadata.csv"

    patch_size = 224
    mpp = config["mpp"]
    magnification = mpp_to_magnification.get(mpp, "")
    mult = 1
    ############################################################

    excluded_or_missing = []
    df = pd.read_csv(slide_metadata_path, delimiter=None)

    df_slurm = df[[id_column]]
    df_slurm["slide_path"] = data_dir + "/images/" + df[id_column]
    df_slurm[f"metadata_path_{patch_size}"] = (
        data_dir + f"/tiles_metadata_{patch_size}{magnification}/" + df[id_column] + ".json"
    )

    print(df_slurm)

    num_tiles = []

    os.makedirs(f"{data_dir}/tiles_metadata_{patch_size}{magnification}", exist_ok=True)
    with tqdm(total=len(df)) as pbar:
        print(df_slurm)

        for idx, row in df_slurm.iterrows():
            slide_id = row[id_column]
            pbar.set_postfix_str(slide_id)
            pbar.update(1)
            project_id = slide_id

            try:
                image = np.array(Image.open(row["slide_path"]))
            except openslide.lowlevel.OpenSlideUnsupportedFormatError:
                print(f"{row['slide_path']}: Unsupported or missing image file")
                excluded_or_missing.append(slide_id)
                df[df[id_column].isin(excluded_or_missing)].to_csv(
                    f"{data_dir}/excluded_or_missing.csv", index=False
                )
                num_tiles.append(0)
                continue

            grid = make_sample_grid(
                image,
                patch_size=patch_size,
                mpp=mpp,
                mult=mult,
                base_mpp=image_base_mpp,
            )

            save_metadata_to_file(
                data_dir=data_dir,
                dataset=dataset,
                project_id=project_id,
                slide_id=slide_id,
                tiles=[tuple(map(int, tile)) for tile in grid],
                patch_size=patch_size,
                mpp=mpp,
                base_mpp=image_base_mpp,
                mult=mult,
                magnification=magnification,
            )
            num_tiles.append(len(grid))

        print(len(df_slurm))
        print(len(num_tiles))
        df_slurm["num_tiles"] = num_tiles
        df_slurm.to_csv(
            f"{data_dir}/images_metadata_slurm{magnification}.csv", index=False
        )

if __name__ == "__main__":
    main()
