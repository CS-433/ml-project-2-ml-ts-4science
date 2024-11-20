import numpy as np
import openslide
from utils.SlideTileExtractor.extract_tissue import image2array
import h5py
from typing import List, Tuple, Dict, Any, Literal


def extract_and_rescale_tile(
    slide: openslide.OpenSlide,
    x: int,
    y: int,
    base_mpp: float,
    target_mpp: float,
    patch_size: int,
) -> np.ndarray:
    """Extracts and rescale a tile from a slide."""
    rescale_factor = base_mpp / target_mpp
    base_patch_size = int(patch_size * rescale_factor)
    tile = slide.read_region((x, y), 0, (base_patch_size, base_patch_size))

    # Convert to RGB format and remove alpha channel
    tile = image2array(tile)
    tile = cv2.resize(tile, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    return tile


def process_slide(
    slide_path: str,
    tile_coordinates: List[Tuple[int, int]],
    base_mpp: float,
    target_mpp: float,
    patch_size: int,
    output_file: str,
) -> None:
    """Processes a single slide and extracts tiles."""
    slide = openslide.OpenSlide(slide_path)
    slide_id = os.path.basename(slide_path).split(".")[0]
    tiles = []
    for x, y in tile_coordinates:
        tile = extract_and_rescale_tile(slide, x, y, base_mpp, target_mpp, patch_size)
        tiles.append((tile, (x, y)))

    with h5py.File(file_path, "a") as hdf5_file:
        group = hdf5_file.require_group(slide_id)
        for tile, (x, y) in zip(tiles, coordinates):
            dataset_name = f"tile_{x}_{y}"
            group.create_dataset(dataset_name, data=tile, compression="gzip")


def read_metadata(metadata_path: str) -> Tuple[int, List[Tuple[int, int]]]:
    """Reads tile coordinates and patch size from a JSON metadata file."""
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    patch_size = metadata["patch_size"]
    tiles = [tuple(coord) for coord in metadata["tiles"]]
    return patch_size, tiles


def process_slides_in_batches(
    slides: List[str],
    metadata_paths: List[str],
    base_mpp: float,
    target_mpp: float,
    slides_per_file: int,
    output_dir: str,
) -> None:
    """Processes slides in batches and stores each batch in a separate HDF5 file."""
    num_files = (len(slides) + slides_per_file - 1) // slides_per_file
    for file_index in range(num_files):
        output_file = os.path.join(output_dir, f"tiles_batch_{file_index + 1}.hdf5")
        start_index = file_index * slides_per_file
        end_index = min((file_index + 1) * slides_per_file, len(slides))
        for slide_path, metadata_path in zip(
            slides[start_index:end_index], metadata_paths[start_index:end_index]
        ):
            patch_size, tile_coordinates = read_metadata(metadata_path)
            process_slide(
                slide_path,
                tile_coordinates,
                base_mpp,
                target_mpp,
                patch_size,
                output_file,
            )
