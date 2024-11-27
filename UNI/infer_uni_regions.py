import os
import torch
import timm
from torchvision import transforms
import numpy as np
import json
from typing import Dict
from openslide import open_slide
from torch.utils.data import DataLoader, Dataset
import argparse
from PIL import Image


class TileDataset(Dataset):
    def __init__(self, slide_path: str, metadata: Dict, transform, debug_save_path="debug"):
        base_mpp = metadata["base_mpp"]
        target_mpp = metadata["mpp"]
        patch_size = metadata["patch_size"]
        coordinates = metadata["tiles"]
        downsample = target_mpp / base_mpp

        self.base_mpp = base_mpp
        self.target_mpp = target_mpp
        self.patch_size = patch_size
        self.downsample = downsample
        self.coordinates = coordinates
        self.transform = transform
        self.slide_path = slide_path
        self.debug_save_path = debug_save_path  # Path to save tiles for debugging

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with open_slide(self.slide_path) as slide:  # Must be initialized in each worker
            level = slide.get_best_level_for_downsample(self.downsample)
            lvl_f = slide.level_downsamples
            patch_size_src = round(
                self.patch_size * (self.target_mpp * lvl_f[level] / self.base_mpp)
            )

            x, y = self.coordinates[idx]
            tile = np.array(
                slide.read_region(
                    location=(x, y), size=(patch_size_src, patch_size_src), level=level
                ).convert("RGB")
            )

        # Save the tile for debugging purposes if debug_save_path is set
        if self.debug_save_path is not None:
            os.makedirs(self.debug_save_path, exist_ok=True)
            debug_tile_path = os.path.join(self.debug_save_path, f"tile_{idx}.png")
            Image.fromarray(tile).save(debug_tile_path)

        return self.transform(tile)


class SlideProcessor:
    def __init__(
        self,
        model_path: str,
        patch_size: int = 224,
        input_size: int = 224,
        batch_size: int = 32,
        device="cuda:0",
    ):
        self.model = timm.create_model(
            "vit_large_patch16_224",
            img_size=input_size,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device).eval()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(patch_size),
                # transforms.CenterCrop(input_size),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        self.batch_size = batch_size

    def process_tiles(self, slide_path: str, metadata: Dict) -> np.ndarray:
        dataset = TileDataset(slide_path, metadata, self.transform)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True
        )
        embeddings = []

        with torch.no_grad():
            for batch in loader:
                embeddings.append(self.model(batch.to(device)).cpu().numpy())
                break

        return np.vstack(embeddings), np.array(metadata["tiles"])


def infer_model_on_region(
    slide_path: str,
    metadata_path: str,
    model_path: str,
    device: str,
    patch_size: int,
    input_size: int,
) -> None:
    # save embeddings and coordinates as npz
    save_filepath = slide_path.replace("images", "embeddings/embeddings_uni").replace(
        ".tiff", ".npz"
    )

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    if len(metadata["tiles"]) > 0 and not os.path.exists(save_filepath):
        processor = SlideProcessor(
            model_path=model_path,
            patch_size=patch_size,
            input_size=input_size,
            device=device,
        )
        embeddings, coordinates = processor.process_tiles(slide_path, metadata)

        print("embeddings: ", embeddings.shape)
        print("coordinates: ", coordinates.shape)

        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
        np.savez_compressed(
            save_filepath, embeddings=embeddings, coordinates=coordinates
        )
        print("npz file saved: ")
        print(save_filepath)
    else:
        print(f"Skipping {save_filepath}")


if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()

    ####### Configs ############################################
    parser = argparse.ArgumentParser(description="Run inference on a wsi using UNI")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        default="/store/swissai/a02/health_pathology/data/GTEx",
        help="path where data is present",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="/scratch/izar/carlos/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin",
        help="path for saved chkpt",
    )

    parser.add_argument(
        "--metadata_path",
        type=str,
        required=False,
        default="/store/swissai/a02/health_pathology/data/TCGA/LUSC/tiles_metadata_256/TCGA-18-3410-01Z-00-DX1.DB186D75-4AEE-4E1B-83D5-5A1970F03581.json",
        # "/store/swissai/a02/health_pathology/data/TCGA/LUSC/tiles_metadata_256/TCGA-22-1017-01Z-00-DX1.9562FE79-A261-42D3-B394-F3E0E2FF7DDA.json",
        help="path where metadata of wsi resides",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        required=False,
        default=224,
        help="patch size to extract embeddings",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        required=False,
        default=224,
        help="size of tile inputted to the model",
    )

    parser.add_argument(
        "--gpu_node",
        type=int,
        required=False,
        default=0,
        help="which of the 0-3 gpu node to use for the wsi",
    )

    args = parser.parse_args()
    device = torch.device(
        "cuda:" + str(args.gpu_node) if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)

    data_dir = args.data_dir
    model_path = args.model_path
    patch_size = args.patch_size
    metadata_path = args.metadata_path
    input_size = args.input_size
    ############################################################
    print("metadata_path: ", metadata_path)
    sample = metadata_path.split("/")[-1].split(".json")[0]
    print("sample: ", sample)

    if data_dir.split("/")[-1] == "TCGA":
        slide_path = metadata_path.replace(
            f"tiles_metadata_{patch_size}", "images"
        ).replace(".json", ".tiff")

    else:
        slide_path = metadata_path.split("/")
        slide_path[-2] = "images" # replace tiles_metadata... with images
        slide_path = "/".join(slide_path).replace(".json", "")

    magnification = ""
    magnification = next((mag for mag in ["_5x", "_10x", "_20x", "_40x"] if mag in metadata_path), "")
    print("magnification: ", magnification)

    print("slide path: ", slide_path)

    infer_model_on_region(
        slide_path, metadata_path, model_path, device, patch_size, input_size
    )

    # pr.disable()
    # stats = Stats(pr)
    # stats.sort_stats('cumtime').print_stats(40)
