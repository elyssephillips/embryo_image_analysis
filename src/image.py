from __future__ import annotations
import numpy as np
import scipy.ndimage as nd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage.measure import regionprops


def get_user_rotation(dapi_mip, identifier):
    """Displays the DAPI MIP and asks the user for a rotation angle."""
    plt.imshow(dapi_mip, cmap='gray')
    plt.title(f"Image: {identifier}\nEnter rotation angle in terminal (0-360):")
    plt.show(block=False)
    angle = float(input(f"Enter rotation angle for {identifier} (or 0 to skip): "))
    plt.close()
    return angle


def rotate_full_stack(img, angle):
    """Rotates the full 3D/4D stack on the Y-X plane."""
    if angle == 0:
        return img
    # Microscopy axes: (C, Z, Y, X) -> rotate Y and X (axes 2 and 3)
    # If image is (Z, Y, X) -> rotate axes 1 and 2
    axes = (2, 3) if img.ndim == 4 else (1, 2)
    return nd.rotate(img, angle, axes=axes, reshape=True, order=1)


def make_anisotropic_selem(erode_xy_px, erode_z_px):
    """
    Creates a 3D ellipsoid (structuring element) for erosion.
    If Z erosion is 0, it creates a 2D disk inside a 3D volume.
    """
    zz, rr = int(max(0, erode_z_px)), int(max(0, erode_xy_px))
    z, y, x = np.arange(-zz, zz+1), np.arange(-rr, rr+1), np.arange(-rr, rr+1)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    denom_z = zz if zz > 0 else 1
    denom_r = rr if rr > 0 else 1
    return (Z / denom_z)**2 + (Y / denom_r)**2 + (X / denom_r)**2 <= 1.0


def erode_labels_optimized(labels_zyx, erode_xy_px, erode_z_px):
    """
    Erodes each nucleus ID individually using bounding boxes for speed.
    Prevents nuclei from bleeding into each other during erosion.
    """
    out = np.zeros_like(labels_zyx)
    selem = make_anisotropic_selem(erode_xy_px, erode_z_px)
    props = regionprops(labels_zyx)
    for prop in props:
        label_id = prop.label
        z0, y0, x0, z1, y1, x1 = prop.bbox
        z_start, z_end = max(0, z0-1), min(labels_zyx.shape[0], z1+1)
        y_start, y_end = max(0, y0-1), min(labels_zyx.shape[1], y1+1)
        x_start, x_end = max(0, x0-1), min(labels_zyx.shape[2], x1+1)
        crop = labels_zyx[z_start:z_end, y_start:y_end, x_start:x_end]
        mask = (crop == label_id)
        eroded_mask = ndi.binary_erosion(mask, structure=selem)
        out[z_start:z_end, y_start:y_end, x_start:x_end][eroded_mask] = label_id
    return out
