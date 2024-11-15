# SlideTileExtractor
This module allows for the extraction of tiles from slides. It leverages openslide, so it is hopefully slide format agnostic.
The are a series of submodules that perform different tasks:
* `extract_tissue.py`: it is the main submodule. It can extract tiles from tissue and it has an option to extract tiles from the intersection of tissue and a BMP annotation. Note: it will extract from any annotation regardless of class.

All submodules handle tile extraction at different resolutions in microns per pixel (MPP).

## Requirements
* `opencv`
* `skimage`
* `openslide`
* `numpy`
* `PIL`

## `extract tissue`
The main function to use is `make_sample_grid`. It accepts the following parameters:
* `slide`: `openslide` object;
* `patch_size`: tile size;
* `mpp`: requested resolution in MPP;
* `base_mpp`: level 0 mpp for the slide;
* `power`: resolution in objective power. Will be converted to MPP automatically;
* `min_cc_size`: minimum size of connected components considered tissue;
* `max_ratio_size`: deprecated parameter, has no effect;
* `dilate`: after thresholding it will dilate the tissue mask;
* `erode`: after thresholding it will erode the tissue mask;
* `prune`: checks whether each found tile contains enough tissue. *Note: this is very slow*;
* `overlap`: how much overlap between consecutive tiles in a non overlapping grid;
* `maxn`: if not `None`, is the maximum number of sampled tiles *per class*;
* `bmp`: if not `None`, is the path to the BMP annotation file from which extract the tissue tiles;
* `oversample`: to extract the tiles in a grid from the highest resolution regardless of the requested resolution. Good for slides that have little tissue.
* `mult`: used to refine extraction.
* `centerpixel`: to return center of tile. By default return the top left corner.
It returns:
* list of coordinate tuples (x,y).

## `slide_base_mpp`
Returns the MPP of level 0.

## `find_level`
Another important function is `find_level`, which given a slide object, the requested resolution and tile size, calculates:
* from which level you need to extract tiles;
* what scale you need to apply to your extracted tiles to get the requested resolution.

*Note: if the appropriate tile size is less than 10% different than the working tile size (e.g. for a 224x224px tile, 22px tolerance around 224), no scaling will be applied to avoid interpolation artifacts.*

## `plot_extraction`
The function `plot_extraction` allows to show the result of the tile extraction.

Example usage:
```python
from SlideTileExtractor.extract_tissue import extract_tissue, find_level, plot_extraction
slide = openslide.OpenSlide('path')
base_mpp = slide_base_mpp(slide)
grid = extract_tissue(slide, patch_size=224, mpp=0.5, base_mpp=base_mpp, mult=4)
```
