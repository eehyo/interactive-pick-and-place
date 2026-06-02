import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
try:
    from termcolor import colored
except Exception:
    def colored(text, color=None):
        return text


def get_colors(n_color=10, cmap_name="gist_rainbow", alpha=1.0):
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(np.linspace(0, 1, int(n_color)))
    rgba[:, 3] = alpha
    return [tuple(r) for r in rgba]

def imread(
        impath,
        verbose    = True,
        return_np  = True,
        return_pil = False,
        show_image = True,
        fig_w      = 4,
        fig_h      = None,
        uint8      = True,
        out_w      = None,
        out_h      = None,
    ):
    """
    Read an image from a file and convert it to an RGB numpy array.
    It returns a PIL image if return_pil is True, and a numpy array if return_np is True.

    Parameters:
        impath (str): Path to the image file.
        verbose (bool): Whether to print debug information.
        show_image (bool): Whether to display the image.
        return_np (bool): Whether to return the image as a numpy array.
        return_pil (bool): Whether to return the image as a PIL Image.
        fig_w (float): Figure width for displaying the image.
        fig_h (float or None): Figure height for displaying the image. If None, it is calculated to preserve aspect ratio.
        uint8 (bool): Whether to convert the image to uint8 format.
        out_w (int or None): Optional output image width. If only out_w is given,
            out_h is computed automatically to preserve aspect ratio.
        out_h (int or None): Optional output image height. If only out_h is given,
            out_w is computed automatically to preserve aspect ratio.

    Returns:
        np.array or PIL.Image: RGB image as a numpy array or PIL image.
    """
    if (not bool(return_np)) and (not bool(return_pil)):
        raise ValueError("At least one of return_np or return_pil must be True.")

    # Load the image using PIL and convert to RGB format
    img_pil = Image.open(impath).convert("RGB") # pil image in RGB format

    # Optionally resize the image
    if (out_w is not None) or (out_h is not None):
        src_w,src_h = img_pil.size

        if out_w is not None:
            out_w = int(out_w)
            if out_w <= 0:
                raise ValueError("out_w must be a positive integer, got %s"%(str(out_w)))

        if out_h is not None:
            out_h = int(out_h)
            if out_h <= 0:
                raise ValueError("out_h must be a positive integer, got %s"%(str(out_h)))

        # Preserve aspect ratio if only one side is specified
        if (out_w is not None) and (out_h is None):
            out_h = max(1,int(round(float(src_h)*float(out_w)/float(src_w))))
        elif (out_w is None) and (out_h is not None):
            out_w = max(1,int(round(float(src_w)*float(out_h)/float(src_h))))

        # PIL compatibility across versions
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR

        img_pil = img_pil.resize((out_w,out_h),resample=resample)

    img_np = np.array(img_pil) # numpy array in RGB format

    # Convert dtype/range if requested
    if bool(uint8):
        if img_np.dtype != np.uint8:
            x  = img_np.astype(np.float32)
            mx = float(np.max(x)) if x.size > 0 else 0.0
            if mx <= 1.0+1e-6:
                x = x*255.0
            img_np = np.clip(x,0.0,255.0).astype(np.uint8)
    else:
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32)/255.0
        else:
            x  = img_np.astype(np.float32)
            mx = float(np.max(x)) if x.size > 0 else 0.0
            if mx > 1.0+1e-6:
                x = x/255.0
            img_np = np.clip(x,0.0,1.0).astype(np.float32)

    # Print debug information
    if verbose:
        print("[imread] Loaded image from [%s] with shape %s dtype=%s"%(str(impath),str(img_np.shape),str(img_np.dtype)))

    # Show the image if requested
    if bool(show_image):
        try:
            imshow(img_np,fig_w=fig_w,fig_h=fig_h)
        except Exception as e:
            if verbose:
                print("[imread] warning: show_image failed (%s)"%str(e))

    # Return the image in the requested formats
    if bool(return_np) and bool(return_pil):
        return img_np,img_pil
    elif bool(return_np):
        return img_np
    else:
        return img_pil

def imshow(
        img,
        fig_w=4,
        fig_h=None,
        title=None,
        title_fs=8,
        axis_off=True,
        cmap="viridis",
        colorbar=False,
        cbar_fs=6,
        caxis=None,
        dpi=120,
        pad_px=0,
        title_overlay=True,
        cbar_inset=True,
    ):
    if img is None:
        print("[imshow] 'img' is None. Nothing to display.")
        return

    h, w = img.shape[:2]
    fig_h = float(fig_w) * float(h) / float(w) if fig_h is None else float(fig_h)
    fig = plt.figure(figsize=(float(fig_w), float(fig_h)), dpi=int(dpi))
    pad_px = int(max(0, pad_px))
    if pad_px > 0:
        fw_px = float(fig.get_figwidth() * fig.dpi)
        fh_px = float(fig.get_figheight() * fig.dpi)
        l, b = pad_px / fw_px, pad_px / fh_px
        r, t = 1.0 - pad_px / fw_px, 1.0 - pad_px / fh_px
        ax = fig.add_axes([l, b, max(0.0, r - l), max(0.0, t - b)])
    else:
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    if img.ndim == 2:
        im = ax.imshow(img, cmap=cmap, vmin=None if caxis is None else caxis[0], vmax=None if caxis is None else caxis[1])
    else:
        im = ax.imshow(img)
    if bool(axis_off):
        ax.set_axis_off()
    if title is not None:
        if bool(title_overlay):
            ax.text(
                0.01, 0.99, str(title),
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=int(title_fs),
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="none"),
            )
        else:
            fig.text(0.01, 0.99, str(title), ha="left", va="top", fontsize=int(title_fs))
    if bool(colorbar) and img.ndim == 2:
        if bool(cbar_inset):
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            cax = inset_axes(ax, width="3.5%", height="45%", loc="lower right", borderpad=0.8)
            cbar = fig.colorbar(im, cax=cax)
        else:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=int(cbar_fs))
    plt.show()


def print_red(text):
    print(colored(text, "red"))


def print_yellow(text):
    print(colored(text, "yellow"))


def print_green(text):
    print(colored(text, "green"))


def meters2xyz(depth_img, cam_matrix):
    fx = cam_matrix[0][0]
    cx = cam_matrix[0][2]
    fy = cam_matrix[1][1]
    cy = cam_matrix[1][2]
    height, width = depth_img.shape[:2]
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - cx) * z_e / fx
    y_e = (indices[..., 0] - cy) * z_e / fy
    return np.stack([z_e, -x_e, -y_e], axis=-1)
