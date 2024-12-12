from utils.SlideTileExtractor import extract_tissue
from utils.ImageTileExtractor import extract_tissue_from_images
import openslide
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
from typing import List, Tuple, Dict, Any, Literal
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import time
import yaml


valid_datasets = ["TCGA", "GTEx", "MHIST", "CRC100k", "PANDA", "BACH", "MIDOG", "BRACS"]

mpp_to_magnification = {
    0.25: "_40x",
    0.5: "_20x",
    1: "_10x",
    2: "_5x",
}


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
    min_cc_size: int,
    max_ratio_size: float,
    dilate: bool,
    erode: bool,
    prune: bool,
    overlap: float,
    maxn: int,
    bmp: float,
    oversample: bool,
    mult: float,
    remove_white_areas_bool: bool = False,
    allow_tile_overlap: bool = True,
    magnification: str = "",
) -> None:
    if dataset == "TCGA":
        json_file = (
            f"{data_dir}/{project_id}/tiles_metadata_{patch_size}/{slide_id}.json"
        )
    elif dataset in valid_datasets:
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
                "min_cc_size": min_cc_size,
                "max_ratio_size": max_ratio_size,
                "dilate": dilate,
                "erode": erode,
                "prune": prune,
                "overlap": overlap,
                "maxn": maxn,
                "bmp": bmp,
                "oversample": oversample,
                "mult": mult,
                "tiles": tiles,
                "remove_white_areas_bool": remove_white_areas_bool,
                "allow_tile_overlap": allow_tile_overlap,
            },
            f,
            indent=4,
        )


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    slide_or_image = {k: v["type"] for k, v in config["datasets"].items()}
    id_columns = {k: v["id_column"] for k, v in config["datasets"].items()}

    # samples_id = {"TCGA": "TCGA-22-1017-01Z-00-DX1.9562FE79-A261-42D3-B394-F3E0E2FF7DDA",
    #               "GTEx": "GTEX-1269W-0126",
    #               "MHIST": "MHIST_aaa.png",
    #               "CRC100k": "LYM-CLHGDLYK.tif",
    #               "PANDA": "8352937c62fad694f28fdeef9fc8c464.tiff",
    #               "BACH": "n020.tif",
    #               "BRACS": "BRACS_1499_UDH_2.png",
    #               "MIDOG": "001.tiff"}}

    ####### Configs ############################################
    dataset: Literal[tuple(valid_datasets)] = config["dataset"]
    input_prefix = slide_or_image[dataset]

    # data_dir = f"/store/swissai/a02/health_pathology/data/{dataset}" # for todi
    # data_dir = f"/capstor/scratch/cscs/vsubrama/data/{dataset}/" # for bristen
    data_dir = os.path.join(config["data_dir"], dataset)
    slide_metadata_path = f"{data_dir}/{input_prefix}_metadata."
    file_extension = "tsv" if dataset == "TCGA" else "csv"
    slide_metadata_path = f"{data_dir}/{input_prefix}_metadata.{file_extension}"
    mode: Literal["generate_metadata", "debug"] = "generate_metadata"
    patch_size = 224
    mpp = config["mpp"]
    magnification = mpp_to_magnification.get(mpp, "")
    min_cc_size = 10
    max_ratio_size = 10
    overlap = 1
    dilate = False
    erode = False
    prune = False
    oversample = False
    mult = 1
    maxn = None
    bmp = None
    id_column = id_columns[dataset]
    allow_tile_overlap = False
    ############################################################
    excluded_or_missing = []
    df = pd.read_csv(slide_metadata_path, delimiter="\t" if dataset == "TCGA" else None)
    if dataset == "TCGA":
        df_slurm = df[[id_column]]
        df_slurm["slide_path"] = (
            data_dir
            + "/"
            + df["Project ID"].str.split("TCGA-").str[-1]
            + "/slides/"
            + df[id_column]
        )

        df_slurm[f"metadata_path_{patch_size}"] = (
            data_dir
            + "/"
            + df["Project ID"].str.split("TCGA-").str[-1]
            + f"/tiles_metadata_{patch_size}/"
            + df[id_column].str.split(".svs").str[0]
            + ".json"
        )
    else:
        if input_prefix == "slides":
            df_slurm = df[[id_column]]
            df_slurm["slide_path"] = data_dir + "/slides/" + df[id_column] + ".svs"
            df_slurm[f"metadata_path_{patch_size}"] = (
                data_dir + f"/tiles_metadata_{patch_size}/" + df[id_column] + ".json"
            )
        else:  # input_prefix == images
            df_slurm = df[[id_column]]
            df_slurm["slide_path"] = data_dir + "/images/" + df[id_column]  # + ".tiff"
            df_slurm[f"metadata_path_{patch_size}"] = (
                data_dir + f"/tiles_metadata_{patch_size}{magnification}/" + df[id_column] + ".json"
            )

    df_slurm.to_csv(f"{data_dir}/{input_prefix}_metadata_slurm{magnification}.csv", index=False)
    print(df_slurm)

    if mode == "generate_metadata":
        num_tiles = []
        if input_prefix == "slides":
            if dataset == "TCGA":
                add_subfolder(data_dir, subfolder_name=f"tiles_metadata_{patch_size}")
            elif dataset in ["GTEx", "PANDA"]:
                os.makedirs(f"{data_dir}/tiles_metadata_{patch_size}", exist_ok=True)
            with tqdm(total=len(df)) as pbar:
                for idx, row in df_slurm.iterrows():
                    slide_id = (
                        row[id_column].split(".svs")[0]
                        if dataset == "TCGA"
                        else row[id_column]
                    )
                    pbar.set_postfix_str(slide_id)
                    pbar.update(1)
                    project_id = (
                        row["Project ID"].split("TCGA-")[-1]
                        if dataset == "TCGA"
                        else (
                            row["Subject ID"]
                            if dataset == "GTEx"
                            else row["Image_Name"]
                        )
                    )
                    if not os.path.exists(
                        row[f"metadata_path_{patch_size}"]
                    ) and os.path.exists(row["slide_path"]):
                        # start_time = time.time()
                        try:
                            slide = openslide.OpenSlide(row["slide_path"])
                        except openslide.lowlevel.OpenSlideUnsupportedFormatError:
                            print(
                                f"{row['slide_path']}: Unsupported or missing image file"
                            )
                            excluded_or_missing.append(slide_id)
                            df[df[id_column].isin(excluded_or_missing)].to_csv(
                                f"{data_dir}/excluded_or_missing.csv", index=False
                            )
                            num_tiles.append(0)
                            continue
                            # raise openslide.lowlevel.OpenSlideUnsupportedFormatError
                        slide_base_mpp = extract_tissue.slide_base_mpp(slide, slide_id)
                        if slide_base_mpp is not None:
                            try:
                                grid = extract_tissue.make_sample_grid(
                                    slide,
                                    patch_size,
                                    mpp=mpp,
                                    min_cc_size=min_cc_size,
                                    max_ratio_size=max_ratio_size,
                                    dilate=dilate,
                                    erode=erode,
                                    prune=prune,
                                    overlap=overlap,
                                    maxn=maxn,
                                    bmp=bmp,
                                    oversample=oversample,
                                    mult=mult,
                                    base_mpp=slide_base_mpp,
                                )
                            except NotImplementedError:
                                print("Skipping slide: ", slide)
                        else:
                            grid = []
                            df[df[id_column].isin(excluded_or_missing)].to_csv(
                                f"{data_dir}/excluded_or_missing.csv", index=False
                            )

                        save_metadata_to_file(
                            data_dir=data_dir,
                            dataset=dataset,
                            project_id=project_id,
                            slide_id=slide_id,
                            tiles=[tuple(map(int, tile)) for tile in grid],
                            patch_size=patch_size,
                            mpp=mpp,
                            min_cc_size=min_cc_size,
                            max_ratio_size=max_ratio_size,
                            dilate=dilate,
                            erode=erode,
                            prune=prune,
                            overlap=overlap,
                            maxn=maxn,
                            bmp=bmp,
                            oversample=oversample,
                            mult=mult,
                            base_mpp=slide_base_mpp,
                        )
                        num_tiles.append(len(grid))
                        # end_time = time.time()
                        # elapsed_time = end_time - start_time
                        # print("Elapsed time: ", elapsed_time)
        else:  # debugging!! ,
            remove_white_areas_bool = False  # This should be turned to TRUE if the images are actually WSIs but without header information
            os.makedirs(f"{data_dir}/tiles_metadata_{patch_size}{magnification}", exist_ok=True)
            with tqdm(total=len(df)) as pbar:
                print(df_slurm)
                for idx, row in df_slurm.iterrows():
                    slide_id = (
                        row[id_column].split(".")[0]
                        if dataset == "TCGA"
                        else row[id_column]
                    )
                    pbar.set_postfix_str(slide_id)
                    pbar.update(1)
                    project_id = (
                        row["Project ID"].split("TCGA-")[-1]
                        if dataset == "TCGA"
                        else row["Subject ID"] if dataset == "GTEx" else row[id_column]
                    )

                    # start_time = time.time()
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
                        # raise openslide.lowlevel.OpenSlideUnsupportedFormatError
                    image_base_mpp = extract_tissue_from_images.image_base_mpp(dataset)
                    if image_base_mpp is not None:
                        try:
                            grid = extract_tissue_from_images.make_sample_grid(
                                image,
                                patch_size=patch_size,
                                mpp=mpp,
                                min_cc_size=min_cc_size,
                                max_ratio_size=max_ratio_size,
                                dilate=dilate,
                                erode=erode,
                                prune=prune,
                                overlap=overlap,
                                maxn=maxn,
                                bmp=bmp,
                                oversample=oversample,
                                mult=mult,
                                base_mpp=image_base_mpp,
                                remove_white_areas_bool=remove_white_areas_bool,
                                allow_tile_overlap=allow_tile_overlap,
                            )
                        except (NotImplementedError, ZeroDivisionError):
                            print("Skipping image: ", row["Image Name"])
                            grid = []
                    else:
                        grid = []
                        df[df[id_column].isin(excluded_or_missing)].to_csv(
                            f"{data_dir}/excluded_or_missing.csv", index=False
                        )

                    save_metadata_to_file(
                        data_dir=data_dir,
                        dataset=dataset,
                        project_id=project_id,
                        slide_id=slide_id,
                        tiles=[tuple(map(int, tile)) for tile in grid],
                        patch_size=patch_size,
                        mpp=mpp,
                        min_cc_size=min_cc_size,
                        max_ratio_size=max_ratio_size,
                        dilate=dilate,
                        erode=erode,
                        prune=prune,
                        overlap=overlap,
                        maxn=maxn,
                        bmp=bmp,
                        oversample=oversample,
                        mult=mult,
                        base_mpp=image_base_mpp,
                        remove_white_areas_bool=remove_white_areas_bool,
                        allow_tile_overlap=allow_tile_overlap,
                        magnification=magnification,
                    )
                    num_tiles.append(len(grid))
                    # end_time = time.time()
                    # elapsed_time = end_time - start_time
                    # print("Elapsed time: ", elapsed_time)

            print(len(df_slurm))
            print(len(num_tiles))
            df_slurm["num_tiles"] = num_tiles
            df_slurm.to_csv(
                f"{data_dir}/{input_prefix}_metadata_slurm{magnification}.csv", index=False
            )

    # else:
    #     if input_prefix == "slides":
    #         #df.sample(n=10) #
    #         df_sample = df_slurm[df[id_column]==f"{sample_id}.svs"] if dataset == "TCGA" else df_slurm[df_slurm[id_column]==sample_id]
    #         for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    #             print(row["slide_path"], os.path.exists(row["slide_path"]))
    #             slide = openslide.open_slide(row["slide_path"]) #OpenSlide(row["slide_path"])
    #             extract_tissue.plot_extraction(
    #                 slide,
    #                 patch_size=patch_size,
    #                 mpp=mpp,
    #                 min_cc_size=min_cc_size,
    #                 max_ratio_size=max_ratio_size,
    #                 dilate=dilate,
    #                 erode=erode,
    #                 prune=prune,
    #                 overlap=overlap,
    #                 maxn=maxn,
    #                 bmp=bmp,
    #                 oversample=oversample,
    #                 mult=mult,
    #                 base_mpp=image_base_mpp,
    #                 save=f'shared/code/health-pathology/test/{sample_id}_tiling_overview.png',
    #             )
    #     else: #debug mode
    #         print("slurm", df_slurm)
    #         print(id_column, sample_id)

    #         df_sample = df_slurm[df_slurm[id_column]==sample_id] #.split(".")[0]]
    #         df_sample[id_column] = df_sample[id_column]
    #         print("sample", df_sample)
    #         for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):

    #             print(row["slide_path"], os.path.exists(row["slide_path"]))
    #             # if os.path.splitext(row["slide_path"])[-1] == "png":
    #             #     image = Image.open(row["slide_path"])

    #             print(dataset, row["slide_path"])
    #             image = np.array(Image.open(row["slide_path"])) #OpenSlide(row["slide_path"])
    #             # print(slide.properties)
    #             image_base_mpp = extract_tissue_from_images.image_base_mpp(dataset)
    #             extract_tissue_from_images.plot_extraction(
    #                 image,
    #                 patch_size=patch_size,
    #                 mpp=mpp,
    #                 min_cc_size=min_cc_size,
    #                 max_ratio_size=max_ratio_size,
    #                 dilate=dilate,
    #                 erode=erode,
    #                 prune=prune,
    #                 overlap=overlap,
    #                 maxn=maxn,
    #                 bmp=bmp,
    #                 oversample=oversample,
    #                 mult=mult,
    #                 base_mpp=image_base_mpp,
    #                 save=f'{sample_id}_tiling_overview.png',
    #             )


if __name__ == "__main__":
    main()
