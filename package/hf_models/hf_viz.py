"""
hf_viz.py
"""
import numpy as np
import matplotlib.pyplot as plt

def to_int_xy(p):
    """
    Convert a point to integer coordinates.

    Parameters:
        p (iterable): an iterable of two numbers (x,y).

    Returns:
        list: A list containing the integer coordinates [x, y], or None if conversion fails.
    """
    if p is None:
        return None
    try:
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        return [x,y]
    except Exception:
        return None

def norm_points_xy_list(xy_list):
    """
    Normalize a list of points to integer coordinates.
    
    Parameters:
        xy_list (iterable): an iterable of points, where each point is an iterable of two numbers (x,y).

    Returns:
        list: A list of points with integer coordinates.
    """
    if xy_list is None:
        return []
    out = []
    for p in xy_list:
        q = to_int_xy(p)
        if q is not None:
            out.append(q)
    return out

def norm_xy(p,default=(0,0)):
    """
    Normalize a single point to integer coordinates, with a default fallback.

    Parameters:
        p (iterable): an iterable of two numbers (x,y).
        default (iterable): an iterable of two numbers (x,y) to use if conversion fails.

    Returns:
        list: A list containing the integer coordinates [x, y], or the default coordinates if conversion fails.
    """
    q = to_int_xy(p)
    if q is None:
        return [int(default[0]),int(default[1])]
    return q


def rgb_img_np(img_np):
    """
    Convert an image to 3-channel RGB numpy array.

    Parameters:
        img_np (array-like): The input image array.

    Returns:
        np.ndarray: A 3-channel RGB image array.
    """
    a = np.asarray(img_np)
    if a.ndim==2:
        a = np.stack([a,a,a],axis=-1)
    return a


def viz_make_fig_ax(fig_w,fig_h,dpi,pad_px,w,h):
    """
    Create a matplotlib figure and axis for visualization.

    Parameters:
        fig_w (float): Figure width in inches.
        fig_h (float): Figure height in inches.
        dpi (int): Dots per inch for the figure.
        pad_px (int): Padding in pixels.
        w (int): Width of the axis in data coordinates.
        h (int): Height of the axis in data coordinates.

    Returns:
        tuple: A tuple containing the figure and axis objects (fig, ax).
    """
    fig = plt.figure(figsize=(float(fig_w),float(fig_h)),dpi=int(dpi),constrained_layout=False)
    fig.set_constrained_layout(False)

    p = int(max(0,pad_px))
    if p>0:
        fw_px = float(fig.get_figwidth()*fig.dpi)
        fh_px = float(fig.get_figheight()*fig.dpi)
        l     = float(p)/fw_px
        b     = float(p)/fh_px
        r     = 1.0-float(p)/fw_px
        t     = 1.0-float(p)/fh_px
        ax    = fig.add_axes([l,b,max(0.0,r-l),max(0.0,t-b)])
    else:
        ax    = fig.add_axes([0.0,0.0,1.0,1.0])

    ax.set_xlim(-0.5,float(w)-0.5)
    ax.set_ylim(float(h)-0.5,-0.5)
    ax.set_autoscale_on(False)
    ax.set_axis_off()
    return fig,ax

def viz_title(fig,ax,title,title_fs,title_overlay):
    """
    Add a title to a matplotlib figure or axis.

    Parameters:
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes.Axes): The axis object.
        title (str): The title text.
        title_fs (int): Font size of the title.
        title_overlay (bool): Whether to overlay the title on the axis.

    Returns:
        None
    """
    if title is None:
        return
    if bool(title_overlay):
        ax.text(
            0.01,0.99,str(title),
            transform=ax.transAxes,
            ha="left",va="top",
            fontsize=int(title_fs),
            color="white",
            bbox=dict(boxstyle="round,pad=0.2",facecolor="black",alpha=0.35,edgecolor="none"),
            clip_on=False,
        )
    else:
        fig.text(0.01,0.99,str(title),ha="left",va="top",fontsize=int(title_fs))

def viz_text_bbox(facecolor="white",alpha=0.7,pad=0.05,edgecolor="none"):
    """
    Create a text bounding box dict for ax.text.

    Parameters:
        facecolor (str): The face color of the bounding box.
        alpha (float): The transparency of the bounding box.
        pad (float): The padding inside the bounding box.
        edgecolor (str): The edge color of the bounding box.

    Returns:
        dict: A dictionary of bounding box properties.
    """
    return dict(
        boxstyle  = "square,pad=%0.2f"%float(pad),
        facecolor = str(facecolor),
        alpha     = float(alpha),
        edgecolor = str(edgecolor),
    )

def viz_label(ax,x,y,txt,dx,dy,fs,bbox,clip_on=True):
    """
    Add a label to a matplotlib axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis object.
        x (float): The x-coordinate of the label.
        y (float): The y-coordinate of the label.
        txt (str): The text of the label.
        dx (float): The x-offset of the label.
        dy (float): The y-offset of the label.
        fs (int): The font size of the label.
        bbox (dict): The bounding box properties.
        clip_on (bool): Whether to clip the label to the axis.

    Returns:
        None
    """
    ax.text(
        float(x)+float(dx),
        float(y)+float(dy),
        str(txt),
        fontsize=int(fs),
        bbox=bbox,
        clip_on=bool(clip_on),
    )

def viz_close(fig):
    """
    Close a matplotlib figure.

    Parameters:
        fig (matplotlib.figure.Figure): The figure object to close.

    Returns:
        None
    """
    try:
        plt.close(fig)
    except Exception:
        pass

# -------------------------------------------------------------------------    