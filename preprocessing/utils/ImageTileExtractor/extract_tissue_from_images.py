import cv2
import numpy as np
import openslide
from skimage.morphology import (
    binary_erosion,
    binary_dilation,
    label,
    dilation,
    square,
    skeletonize,
)
from skimage.filters import threshold_otsu
import PIL.Image as Image
import random
import pdb

# isyntax_mpp = 0.25
# isyntax_mpp = 0.5
MAX_PIXEL_DIFFERENCE = 0.2  # difference must be within 20% of image size

base_mpps_dict = {
    "BACH": 0.42,
    "BRACS": 0.25,  # in microns per pixel (MPP)
    "MIDOG": 0.25,  # TODO: check this value for the images from the scanners. We know which scanner was used for each image
}


def image_base_mpp(dataset):
    return base_mpps_dict[dataset]


# def find_level(image, mpp, patchsize=224, base_mpp=None):

#     downsample = mpp / base_mpp
#     # for i in range(slide.level_count)[::-1]:
#     #     if abs(downsample / slide.level_downsamples[i] * patchsize - patchsize) < MAX_PIXEL_DIFFERENCE * patchsize or downsample > slide.level_downsamples[i]:
#     #         level = i
#     #         mult = downsample / slide.level_downsamples[level]
#     #         break
#     # else:
#     #     raise Exception('Requested resolution ({} mpp) is too high'.format(mpp))
#     # #move mult to closest pixel
#     mult = downsample
#     mult = np.round(mult*patchsize)/patchsize
#     if abs(mult*patchsize - patchsize) < MAX_PIXEL_DIFFERENCE * patchsize:
#         mult = 1.
#     return 1, mult


def image2array(img):
    if img.__class__.__name__ == "Image":
        if img.mode == "RGB":
            img = np.array(img)
            r, g, b = np.rollaxis(img, axis=-1)
            img = np.stack([r, g, b], axis=-1)
        elif img.mode == "RGBA":
            img = np.array(img)
            r, g, b, a = np.rollaxis(img, axis=-1)
            img = np.stack([r, g, b], axis=-1)
        else:
            sys.exit("Error: image is not RGB Image")
    img = np.uint8(img)
    return img  # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def is_sample(
    img, threshold=0.9, ratioCenter=0.1, wholeAreaCutoff=0.5, centerAreaCutoff=0.9
):
    nrows, ncols = img.shape
    timg = cv2.threshold(img, 255 * threshold, 1, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    cimg = cv2.morphologyEx(timg[1], cv2.MORPH_CLOSE, kernel)
    crow = np.rint(nrows / 2).astype(int)
    ccol = np.rint(ncols / 2).astype(int)
    drow = np.rint(nrows * ratioCenter / 2).astype(int)
    dcol = np.rint(ncols * ratioCenter / 2).astype(int)
    centerw = cimg[crow - drow : crow + drow, ccol - dcol : ccol + dcol]
    if (np.count_nonzero(cimg) < nrows * ncols * wholeAreaCutoff) & (
        np.count_nonzero(centerw) < 4 * drow * dcol * centerAreaCutoff
    ):
        return False
    else:
        return True


def threshold(image, size, mpp, base_mpp, mult=1):
    # print(image.shape, type(image))
    # print(image[::2, ::2, :].shape)
    w = int(np.round(image.shape[0] * 1.0 / (size * mpp / base_mpp))) * mult
    h = int(np.round(image.shape[1] * 1.0 / (size * mpp / base_mpp))) * mult
    # print(w, h)
    # print(np.ceil(image.shape[0]/w), np.ceil(image.shape[1]/h))
    thumbnail = image[
        :: int(np.ceil(image.shape[0] / w)), :: int(np.ceil(image.shape[1] / h)), :
    ]  # .get_thumbnail((w,h))
    # thumbnail = thumbnail.resize((w,h))
    img_c = image2array(thumbnail)
    # print("img_c", img_c.shape)
    # calc std on color image
    std = np.std(img_c, axis=-1)
    # image to bw
    # img_g = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    img_g = cv2.cvtColor(img_c, cv2.COLOR_RGB2GRAY)

    # print("img_g", img_g.shape)
    # Detect markers
    marker = detect_marker(img_c, base_mpp / mpp * mult)
    # img_g[np.nonzero(marker)] = np.mean(img_g)

    # Otsu
    img_g = cv2.GaussianBlur(img_g, (5, 5), 0)
    # print("img_g", img_g.shape)
    # New Otsu with skimage and masked arrays
    # print("marker found?", marker is not None)
    if marker is not None:
        # masked = np.ma.masked_array(img_g, (marker>0)|(std<5))
        masked = np.ma.masked_array(img_g, (marker > 0) | (img_g == 255))
        # print(masked.compressed().shape)
        t = threshold_otsu(masked.compressed())
        img_g = cv2.threshold(img_g, t, 255, cv2.THRESH_BINARY)[1]

    else:
        # masked = np.ma.masked_array(img_g, std<5)
        masked = np.ma.masked_array(img_g, img_g == 255)
        t = threshold_otsu(masked.compressed())
        img_g = cv2.threshold(img_g, t, 255, cv2.THRESH_BINARY)[1]
        # t, img_g = cv2.threshold(img_g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Exclude marker
    if marker is not None:
        img_g = cv2.subtract(~img_g, marker)
        # img_g = cv2.erode(img_g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
        # img_g = cv2.dilate(img_g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
    else:
        img_g = 255 - img_g

    # Remove grays
    img_g[std < 5] = 0

    # Rescale
    if mult > 1:
        img_g = img_g.reshape(h // mult, mult, w // mult, mult).max(axis=(1, 3))

    return img_g, t


def remove_black_ink(img_g, th=50, delta=50):
    """
    image in gray scale
    returns mask where ink is positive
    th=50 and delta=50 was chosen based on some image
    """
    dist = np.clip(img_g - float(th), 0, None)
    mask = dist < delta
    if mask.sum() > 0:
        mask_s = skeletonize(mask)
        d = int(np.round(0.1 * mask.sum() / mask_s.sum()))
        mask = dilation(mask, square(2 * d + 1))
        return mask
    else:
        return None


def filter_regions(img, min_size):
    l, n = label(img, return_num=True)
    for i in range(1, n + 1):
        # filter small regions
        if l[l == i].size < min_size:
            l[l == i] = 0
    return l


def add(overlap):
    return np.linspace(0, 1, overlap + 1)[1:-1]


def add2offset(img, image, patch_size, mpp, maxmpp):
    size_x = img.shape[1]
    size_y = img.shape[0]
    offset_x = np.floor(
        (image.shape[0] * 1.0 / (patch_size * mpp / maxmpp) - size_x)
        * (patch_size * mpp / maxmpp)
    )
    offset_y = np.floor(
        (image.shape[1] * 1.0 / (patch_size * mpp / maxmpp) - size_y)
        * (patch_size * mpp / maxmpp)
    )
    add_x = np.linspace(0, offset_x, size_x).astype(int)
    add_y = np.linspace(0, offset_y, size_y).astype(int)
    return add_x, add_y


def addoverlap(w, grid, overlap, patch_size, mpp, maxmpp, img, offset=0):
    o = (add(overlap) * (patch_size * mpp / maxmpp)).astype(int)
    ox, oy = np.meshgrid(o, o)
    connx = np.zeros(img.shape).astype(bool)
    conny = np.zeros(img.shape).astype(bool)
    connd = np.zeros(img.shape).astype(bool)
    connu = np.zeros(img.shape).astype(bool)
    connx[:, :-1] = img[:, 1:]
    conny[:-1, :] = img[1:, :]
    connd[:-1, :-1] = img[1:, 1:]
    connu[1:, :-1] = img[:-1, 1:] & (~img[1:, 1:] | ~img[:-1, :-1])
    connx = connx[w]
    conny = conny[w]
    connd = connd[w]
    connu = connu[w]
    extra = []
    for i, (x, y) in enumerate(grid):
        if connx[i]:
            extra.extend(zip(o + x - offset, np.repeat(y, overlap - 1) - offset))
        if conny[i]:
            extra.extend(zip(np.repeat(x, overlap - 1) - offset, o + y - offset))
        if connd[i]:
            extra.extend(zip(ox.flatten() + x - offset, oy.flatten() + y - offset))
        if connu[i]:
            extra.extend(zip(x + ox.flatten() - offset, y - oy.flatten() - offset))
    return extra


def make_sample_grid(
    image,
    patch_size=224,
    mpp=0.5,
    min_cc_size=10,
    max_ratio_size=10,
    dilate=False,
    erode=False,
    prune=False,
    overlap=1,
    maxn=None,
    bmp=None,
    oversample=False,
    mult=1,
    centerpixel=False,
    base_mpp=None,
    remove_white_areas_bool=False,
):
    """
    Script that given an openslide object return a list of tuples
    in the form of (x,y) coordinates for patch extraction of sample patches.
    It has an erode option to make sure to get patches that are full of tissue.
    It has a prune option to check if patches are sample. It is slow.
    If bmp is given, it samples from within areas of the bmp that are nonzero.
    If oversample is True, it will downsample for full resolution regardless of what resolution is requested.
    mult is used to increase the resolution of the thumbnail to get finer  tissue extraction
    """
    if base_mpp is None:
        base_mpp = isyntax_mpp

    if remove_white_areas_bool:
        if oversample:
            img, th = threshold(image, patch_size, base_mpp, base_mpp, mult)
        else:
            img, th = threshold(image, patch_size, mpp, base_mpp, mult)

        if bmp:
            bmplab = Image.open(bmp)
            thumbx, thumby = img.shape
            bmplab = bmplab.resize((thumby, thumbx), Image.ANTIALIAS)
            bmplab = np.array(bmplab)
            bmplab[bmplab > 0] = 1
            img = np.logical_and(img, bmplab)

        img = filter_regions(img, min_cc_size)
        img[img > 0] = 1
        if erode:
            img = binary_erosion(img)
        if dilate:
            img = binary_dilation(img)

    else:
        w = int(np.round(image.shape[0] * 1.0 / (patch_size * mpp / base_mpp))) * mult
        h = int(np.round(image.shape[1] * 1.0 / (patch_size * mpp / base_mpp))) * mult

        if w == 0 or h == 0:
            raise ZeroDivisionError(
                f"Image size is too small for patch size {patch_size} and mpp {mpp}"
            )
        img = image[
            :: int(np.ceil(image.shape[0] / w)), :: int(np.ceil(image.shape[1] / h)), :
        ]  # .get_thumbnail((w,h))
        img = np.ones((img.shape[0], img.shape[1]))

    if oversample:
        add_x, add_y = add2offset(img, image, patch_size, base_mpp, base_mpp)
    else:
        add_x, add_y = add2offset(img, image, patch_size, mpp, base_mpp)

    w = np.where(img > 0)

    # grid=zip(w[1]*patch_size,w[0]*patch_size)
    if oversample:
        offset = int(0.5 * patch_size * ((mpp / base_mpp) - 1))
        grid = list(
            zip(
                (w[1] * (patch_size) + add_x[w[1]] - offset).astype(int),
                (w[0] * (patch_size) + add_y[w[0]] - offset).astype(int),
            )
        )
    else:
        grid = list(
            zip(
                (w[1] * (patch_size * mpp / base_mpp) + add_x[w[1]]).astype(int),
                (w[0] * (patch_size * mpp / base_mpp) + add_y[w[0]]).astype(int),
            )
        )

    # connectivity
    if overlap > 1:
        if oversample:
            extra = addoverlap(
                w, grid, overlap, patch_size, base_mpp, base_mpp, img, offset=offset
            )
            grid.extend(extra)
        else:
            extra = addoverlap(w, grid, overlap, patch_size, mpp, base_mpp, img)
            grid.extend(extra)

    # center pixel offset
    if centerpixel:
        offset = int(mpp / base_mpp * patch_size // 2)
        grid = [(x[0] + offset, x[1] + offset) for x in grid]

    # prune squares
    # if prune:
    #     level, mult = find_level(slide,mpp,base_mpp)
    #     psize = int(patch_size*mult)
    #     truegrid = []
    #     for tup in grid:
    #         reg = slide.read_region(tup,level,(psize,psize))
    #         if mult != 1:
    #             reg = reg.resize((224,224),Image.BILINEAR)
    #         reg = image2array(reg)
    #         if is_sample(reg,th/255,0.2,0.4,0.5):
    #             truegrid.append(tup)
    # else:

    truegrid = grid

    # sample if maxn
    if maxn:
        truegrid = random.sample(truegrid, min(maxn, len(truegrid)))

    return truegrid


def make_hires_map(image, pred, grid, patch_size, mpp, maxmpp, overlap):
    """
    Given the list of predictions and the known overlap it gives the hires probability map
    """
    W = image.shape[0]
    H = image.shape[1]
    w = int(np.round(W * 1.0 / (patch_size * mpp / maxmpp)))
    h = int(np.round(H * 1.0 / (patch_size * mpp / maxmpp)))

    newimg = np.zeros((h * overlap, w * overlap)) - 1
    offset_x = np.floor(
        (W * 1.0 / (patch_size * mpp / maxmpp) - w) * (patch_size * mpp / maxmpp)
    )
    offset_y = np.floor(
        (H * 1.0 / (patch_size * mpp / maxmpp) - h) * (patch_size * mpp / maxmpp)
    )
    add_x = np.linspace(0, offset_x, w).astype(int)
    add_y = np.linspace(0, offset_y, h).astype(int)
    for i, (xgrid, ygrid) in enumerate(grid):
        yindx = int(ygrid / (patch_size * mpp / maxmpp))
        xindx = int(xgrid / (patch_size * mpp / maxmpp))
        y = np.round(
            (ygrid - add_y[yindx]) * overlap / (patch_size * mpp / maxmpp)
        ).astype(int)
        x = np.round(
            (xgrid - add_x[xindx]) * overlap / (patch_size * mpp / maxmpp)
        ).astype(int)
        newimg[y, x] = pred[i]
    return newimg


def make_hires_map_stride(image, pred, grid, stride):
    """
    Given the list of predictions and the stride it gives the hires probability map
    Grid ndarray specify the center pixel of a tile
    """
    W = image.shape[0]
    H = image.shape[1]
    w = int(round(W * 1.0 / stride))
    h = int(round(H * 1.0 / stride))

    # Scale grid to pixels
    ngrid = np.floor(grid.astype(float) / stride).astype(int)

    # Make image
    newimg = np.zeros((h, w)) - 2

    # Add tissue
    tissue = threshold_stride(image, stride)
    newimg[tissue > 0] = -1

    # paint predictions
    for i in range(len(ngrid)):
        x, y = ngrid[i]
        newimg[y, x] = pred[i]

    return newimg


def threshold_stride(image, stride):
    W = image.shape[0]
    H = image.shape[1]

    w = int(np.round(image.shape[0] * 1.0 / (size * mpp / base_mpp))) * mult
    h = int(np.round(image.shape[1] * 1.0 / (size * mpp / base_mpp))) * mult
    thumbnail = image[
        :: int(np.ceil(image.shape[0] / w)), :: int(np.ceil(image.shape[1] / h)), :
    ]  # .get_thumbnail((w,h))

    # w = int(np.ceil(W*1./stride))
    # h = int(np.ceil(H*1./stride))
    # thumbnail = slide.get_thumbnail((w,h))
    # thumbnail = thumbnail.resize((w,h))
    img = image2array(thumbnail)
    # calc std on color image
    std = np.std(img, axis=-1)
    # image to bw
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## remove black dots ##
    _, tmp = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    tmp = cv2.dilate(tmp, kernel, iterations=1)
    img[tmp == 255] = 255
    img = cv2.GaussianBlur(img, (5, 5), 0)
    t, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = 255 - img
    img[std < 5] = 0
    return img


def plot_extraction(
    image,
    patch_size=224,
    mpp=0.5,
    min_cc_size=10,
    max_ratio_size=10,
    dilate=False,
    erode=False,
    prune=False,
    overlap=1,
    maxn=None,
    bmp=None,
    oversample=False,
    mult=1,
    base_mpp=None,
    save="",
):
    """Script that shows the result of applying the detector in case you get weird results"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    downsample = 5.0
    dpi = 100
    if base_mpp is None:
        base_mpp = isyntax_mpp

    if save:
        plt.switch_backend("agg")

    grid = make_sample_grid(
        image,
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
        base_mpp=base_mpp,
    )
    print(image.shape, grid)

    thumb = image[:: int(downsample), :: int(downsample), :]
    width, height, _ = np.round(thumb.shape)

    ps = []
    for tup in grid:
        ps.append(
            patches.Rectangle(
                (tup[0] / downsample, tup[1] / downsample),
                patch_size / downsample * (mpp / base_mpp),
                patch_size / downsample * (mpp / base_mpp),
                fill=False,
                edgecolor="red",
            )
        )

    fig = plt.figure(figsize=(int(width / dpi), int(height / dpi)), dpi=dpi)
    ax = fig.add_subplot(111, aspect="equal")
    ax.imshow(thumb)
    for p in ps:
        ax.add_patch(p)
    if save:
        plt.savefig(save)
    else:
        plt.show()


def detect_marker(thumb, mult):
    ksize = int(max(1, mult))
    # ksize = 1
    img = cv2.GaussianBlur(thumb, (5, 5), 0)

    hsv_origimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Extract marker
    black_marker = cv2.inRange(
        hsv_origimg, np.array([0, 0, 0]), np.array([180, 255, 125])
    )  # black marker
    blue_marker = cv2.inRange(
        hsv_origimg, np.array([90, 30, 30]), np.array([130, 255, 255])
    )  # blue marker
    green_marker = cv2.inRange(
        hsv_origimg, np.array([40, 30, 30]), np.array([90, 255, 255])
    )  # green marker
    # red_marker_lower = cv2.inRange(hsv_origimg, np.array([0, 30, 30]), np.array([5, 255, 255])) # red marker
    # red_marker = cv2.inRange(hsv_origimg, np.array([170, 30, 30]), np.array([180, 255, 255])) # red marker
    # red_marker = cv2.bitwise_or(red_marker_lower, red_marker_upper)
    # print("red marker?", np.count_nonzero(red_marker))
    mask_hsv = cv2.bitwise_or(cv2.bitwise_or(black_marker, blue_marker), green_marker)
    mask_hsv = cv2.erode(
        mask_hsv, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    )
    mask_hsv = cv2.dilate(
        mask_hsv, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize * 3, ksize * 3))
    )
    if np.count_nonzero(mask_hsv) > 0:
        return mask_hsv
    else:
        return None


# kernel=np.ones((5,5),np.uint8)
# cimg=cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)

# out=[]
# for tup in grid:
#    out.append(sum([1 if x==tup else 0 for x in grid]))
# np.unique(np.array(out))
# TESTS
# from SlideTileExtractor import extract_tissue
# import openslide
# slide = openslide.OpenSlide('/lila/data/fuchs/projects/lung/impacted/1142892.svs')
# extract_tissue.plot_extraction(slide, patch_size=224, mpp=0.5, power=None, min_cc_size=10, max_ratio_size=10, dilate=False, erode=False, prune=False, overlap=1, maxn=None, bmp=None, oversample=False, mult=1, save='')