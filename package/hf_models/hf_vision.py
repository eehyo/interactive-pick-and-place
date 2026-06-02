import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import binary_fill_holes, label
from transformers import (
    Sam2Model, 
    Sam2Processor, 
    Owlv2Processor, 
    Owlv2ForObjectDetection,
    )

from package.hf_models.hf_common import ensure_local_ckpt, get_device
from package.hf_models.hf_viz import (
    norm_points_xy_list,
    rgb_img_np,
    viz_make_fig_ax,
    viz_title,
    viz_text_bbox,
    viz_label,
    viz_close,
)


class PointSam2(object):
    """Point-prompted segmentation using SAM2."""

    def __init__(
            self,
            ckpt="facebook/sam2.1-hiera-base-plus",
            device=None,
            verbose=True,
            local_dir="./models/sam2",
            revision=None,
        ):
        self.ckpt = ensure_local_ckpt(
            ckpt,
            local_dir=local_dir,
            revision=revision,
            verbose=verbose,
        )
        self.processor = Sam2Processor.from_pretrained(self.ckpt)
        self.verbose = bool(verbose)
        self.device = get_device(device=device, verbose=self.verbose)
        self.model = Sam2Model.from_pretrained(self.ckpt).to(self.device)
        self.model.eval()
        if self.verbose:
            print("[PointSam2] ckpt=[%s] ready." % self.ckpt)

    def _keep_largest_cc(self, mask):
        mask = np.asarray(mask).astype(bool)
        structure = np.ones((3, 3), dtype=np.int8)
        lab, n = label(mask, structure=structure)
        if int(n) == 0:
            return mask
        sizes = np.bincount(lab.ravel())
        if sizes.size > 0:
            sizes[0] = 0
        k = int(np.argmax(sizes)) if sizes.size > 0 else 0
        return lab == k

    def _select_candidate_mask(self, cand_masks, pos, neg, prefer_pos_mask=False):
        cand_masks = np.asarray(cand_masks).astype(bool)
        if cand_masks.ndim != 3:
            raise ValueError("cand_masks must have shape (K,H,W).")

        if not bool(prefer_pos_mask):
            cand_i = int(np.argmax(cand_masks.reshape(cand_masks.shape[0], -1).sum(axis=1)))
            return cand_masks[cand_i]

        h, w = cand_masks.shape[1:]
        pos = norm_points_xy_list(pos)
        neg = norm_points_xy_list(neg)
        best_i = 0
        best_key = None

        for i, mask in enumerate(cand_masks):
            pos_hit = 0
            neg_hit = 0
            for x, y in pos:
                if (0 <= x < w) and (0 <= y < h) and bool(mask[y, x]):
                    pos_hit += 1
            for x, y in neg:
                if (0 <= x < w) and (0 <= y < h) and bool(mask[y, x]):
                    neg_hit += 1
            area = int(mask.sum())
            key = (pos_hit, -neg_hit, -area)
            if (best_key is None) or (key > best_key):
                best_key = key
                best_i = i
        return cand_masks[best_i]

    def predict_mask(
            self,
            img_pil,
            pos=None,
            neg=None,
            fill_holes=True,
            keep_largest=True,
            prefer_pos_mask=False,
        ):
        if pos is None:
            pos = []
        if neg is None:
            neg = []
        if (len(pos) == 0) and (len(neg) == 0):
            raise ValueError("At least one of pos or neg points must be provided.")

        pos = norm_points_xy_list(pos)
        neg = norm_points_xy_list(neg)
        pts = pos + neg
        labels = [1] * len(pos) + [0] * len(neg)
        input_points = [[pts]]
        input_labels = [[labels]]

        inputs = self.processor(
            images=img_pil,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )
        for k in inputs:
            if hasattr(inputs[k], "to"):
                inputs[k] = inputs[k].to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )[0]
        cand_masks = masks[0].numpy().astype(bool)
        mask = self._select_candidate_mask(
            cand_masks=cand_masks,
            pos=pos,
            neg=neg,
            prefer_pos_mask=prefer_pos_mask,
        )

        if bool(fill_holes):
            mask = binary_fill_holes(mask)
        if bool(keep_largest):
            mask = self._keep_largest_cc(mask)
        return np.asarray(mask).astype(bool)

    def run(
            self,
            img_pil,
            pos=None,
            neg=None,
            fill_holes=True,
            keep_largest=True,
            prefer_pos_mask=False,
        ):
        pos_n = norm_points_xy_list(pos)
        neg_n = norm_points_xy_list(neg)
        mask = self.predict_mask(
            img_pil,
            pos_n,
            neg_n,
            fill_holes=fill_holes,
            keep_largest=keep_largest,
            prefer_pos_mask=prefer_pos_mask,
        )
        if self.verbose:
            print(
                "[PointSam2] Predicted segmentation mask of size [%d,%d] with [%d] positive and [%d] negative points."
                % (mask.shape[0], mask.shape[1], len(pos_n), len(neg_n))
            )
        return mask

    @staticmethod
    def plot_segmask(
            img_np,
            mask=None,
            pos=None,
            neg=None,
            alpha=0.5,
            bright=0.5,
            mask_color=(255, 0, 0),
            fig_w=4,
            dpi=120,
            pad_px=0,
            marker_size=60,
            text_offset=(5, 5),
            text_fs=8,
            title_fs=8,
            title_overlay=True,
            cmap_mask="gray",
        ):
        img_np = np.asarray(img_np)
        h, w = img_np.shape[:2]
        fig_h = float(fig_w) * float(h) / max(1.0, float(w))

        pos = norm_points_xy_list(pos)
        neg = norm_points_xy_list(neg)

        overlay = None
        if mask is not None:
            m = np.asarray(mask).astype(bool)
            base = rgb_img_np(img_np).astype(np.float32).copy()
            a = max(0.0, min(1.0, float(alpha)))
            b = max(0.0, min(1.0, float(bright)))
            base = base * (1.0 - b) + 255.0 * b
            mc = np.array(mask_color, dtype=np.float32).reshape((1, 1, 3))
            base[m] = a * base[m] + (1.0 - a) * mc
            overlay = np.clip(base, 0, 255).astype(np.uint8)

        bbox_xy = viz_text_bbox(facecolor="white", alpha=0.7, pad=0.05, edgecolor="none")
        dx, dy = float(text_offset[0]), float(text_offset[1])

        fig, ax = viz_make_fig_ax(fig_w, fig_h, dpi, pad_px, w, h)
        ax.imshow(img_np, cmap="viridis" if img_np.ndim == 2 else None)
        for p in pos:
            x, y = int(p[0]), int(p[1])
            ax.scatter([x], [y], s=float(marker_size), marker="o", facecolors="none", edgecolors="r")
            viz_label(ax, x + 10, y, "(%d,%d)" % (x, y), dx, dy, text_fs, bbox_xy, clip_on=True)
        for p in neg:
            x, y = int(p[0]), int(p[1])
            ax.scatter([x], [y], s=float(marker_size), marker="x", color="b")
            viz_label(ax, x + 10, y, "(%d,%d)" % (x, y), dx, dy, text_fs, bbox_xy, clip_on=True)
        viz_title(fig, ax, "original + pos/neg", title_fs, title_overlay)
        plt.show()
        viz_close(fig)

        fig, ax = viz_make_fig_ax(fig_w, fig_h, dpi, pad_px, w, h)
        if mask is None:
            ax.imshow(np.zeros((h, w), dtype=np.uint8), cmap=str(cmap_mask), vmin=0, vmax=255)
            viz_title(fig, ax, "mask (None)", title_fs, title_overlay)
        else:
            ax.imshow(np.asarray(mask).astype(np.uint8) * 255, cmap=str(cmap_mask), vmin=0, vmax=255)
            viz_title(fig, ax, "mask", title_fs, title_overlay)
        plt.show()
        viz_close(fig)

        fig, ax = viz_make_fig_ax(fig_w, fig_h, dpi, pad_px, w, h)
        if overlay is None:
            ax.imshow(img_np, cmap="viridis" if img_np.ndim == 2 else None)
            viz_title(fig, ax, "overlay (None)", title_fs, title_overlay)
        else:
            ax.imshow(overlay)
            viz_title(fig, ax, "overlay", title_fs, title_overlay)
        plt.show()
        viz_close(fig)





class PromptOwlv2(object):
    """
    Text-prompted object detection using OWLv2.
    """
    def __init__(
            self,
            ckpt      = "google/owlv2-base-patch16",
            device    = None,
            verbose   = True,
            local_dir = "./models/owlv2",
            revision  = None,
        ):
        """
        Initialize the PromptOwlv2 object.

        Parameters:
            ckpt (str): The checkpoint name or path.
            device (str or torch.device): The device to run the model on.
            verbose (bool): Whether to print verbose messages.
            local_dir (str): The local directory to store the model.
            revision (str): The model revision to use.
        """
        self.ckpt      = ensure_local_ckpt(ckpt,local_dir=local_dir,revision=revision,verbose=verbose)
        self.processor = Owlv2Processor.from_pretrained(self.ckpt)
        self.verbose   = bool(verbose)
        self.device    = get_device(device=device,verbose=self.verbose)
        self.model     = Owlv2ForObjectDetection.from_pretrained(self.ckpt).to(self.device)
        self.model.eval()
        if self.verbose:
            print("[PromptOwlv2] ckpt=[%s] ready."%self.ckpt)

    def predict_boxes(
            self,
            img_pil,
            text_queries,
            threshold=0.1,
        ):
        """
        Predict bounding boxes for the given image and text queries.

        Parameters:
            img_pil (PIL.Image.Image): The input image.
            text_queries (str or list of str): The text queries for object detection.
            threshold (float): The confidence threshold for filtering detections.
        """
        if text_queries is None:
            raise ValueError("text_queries must be provided.")
        if isinstance(text_queries,str):
            text_queries=[text_queries]
        if len(text_queries)==0:
            raise ValueError("text_queries must be non-empty.")

        inputs = self.processor(
            images         = img_pil,
            text           = text_queries,
            return_tensors = "pt",
        )
        for k in inputs:
            if hasattr(inputs[k],"to"):
                inputs[k]=inputs[k].to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        h,w          = img_pil.size[1],img_pil.size[0]
        target_sizes = torch.tensor([[h,w]],dtype=torch.int64)

        if hasattr(self.processor,"post_process_object_detection"):
            results = self.processor.post_process_object_detection(
                outputs,
                threshold    = float(threshold),
                target_sizes = target_sizes,
            )[0]
        elif hasattr(self.processor,"image_processor") and hasattr(self.processor.image_processor,"post_process_object_detection"):
            results = self.processor.image_processor.post_process_object_detection(
                outputs,
                threshold    = float(threshold),
                target_sizes = target_sizes,
            )[0]
        else:
            raise AttributeError("No post_process_object_detection found in Owlv2Processor/image_processor. Update transformers.")

        # convert to numpy arrays on CPU for easier downstream processing (and to detach from computation graph just in case)
        boxes  = results["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = results["scores"].detach().cpu().numpy().astype(np.float32)
        labels = results["labels"].detach().cpu().numpy().astype(np.int64)

        # Return boxes, scores, and labels as numpy arrays.
        return boxes,scores,labels

    def plot_dets(
            self,
            img_np,
            boxes,
            scores,
            labels,
            text_queries  = None,
            fig_w         = 4,
            show_scores   = True,
            score_fmt     = "%.2f",
            linewidth     = 2,
            edgecolor     = "r",
            text_offset   = (2,2),
            dpi           = 120,
            pad_px        = 0,
            title         = None,
            title_fs      = 8,
            title_overlay = True,
            cmap          = "viridis",
            clip_text     = False,
            bbox_fc       = "white",
            bbox_alpha    = 0.7,
        ):
        """
        Visualize the detected bounding boxes on the image.

        Parameters:
            img_np (numpy.ndarray): The original image as a numpy array.
            boxes (numpy.ndarray): The detected bounding boxes.
            scores (numpy.ndarray): The confidence scores for each detected box.
            labels (numpy.ndarray): The class labels for each detected box.
            text_queries (list of str): The text queries corresponding to the class labels.
            fig_w (float): Figure width in inches for visualization.
            show_scores (bool): Whether to display confidence scores in the labels.
            score_fmt (str): The format string for displaying scores.
            linewidth (float): The line width for the bounding box edges.
            edgecolor (str): The color for the bounding box edges.
            text_offset (tuple): Offset for the text labels relative to the bounding box corners.
            dpi (int): Dots per inch for the figure.
            pad_px (int): Padding in pixels for the figure.
            title (str): The title for the visualization.
            title_fs (int): Font size for the title.
            title_overlay (bool): Whether to overlay the title on the image.
            cmap (str): Colormap to use for the image if it's grayscale.
            clip_text (bool): Whether to clip text labels to the image boundaries.
            bbox_fc (str): Face color for the bounding box text background.
            bbox_alpha (float): Alpha transparency for the bounding box text background.
        """
        img_np = np.asarray(img_np)
        h,w    = img_np.shape[:2]
        fig_h  = float(fig_w)*float(h)/max(1.0,float(w))

        fig,ax = viz_make_fig_ax(fig_w,fig_h,dpi,pad_px,w,h)

        if img_np.ndim==2:
            ax.imshow(img_np,cmap=str(cmap))
        else:
            ax.imshow(img_np)

        n = int(boxes.shape[0]) if boxes is not None else 0

        bbox_txt = viz_text_bbox(facecolor=str(bbox_fc),alpha=float(bbox_alpha),pad=0.05,edgecolor="none")
        dx,dy    = float(text_offset[0]),float(text_offset[1])

        for i in range(n): # iterate over detected boxes
            x1,y1,x2,y2 = boxes[i].tolist()
            x1=float(x1); y1=float(y1); x2=float(x2); y2=float(y2)

            # Draw the bounding box as a rectangle patch on the axis.
            rect = plt.Rectangle(
                (x1,y1),
                x2-x1,
                y2-y1,
                fill      = False,
                linewidth = float(linewidth),
                edgecolor = str(edgecolor),
            )
            ax.add_patch(rect)

            # Determine the label name for the detected box based on the class label and the provided text queries, and format the label text to include the confidence score if show_scores is True.
            lab = int(labels[i]) if labels is not None else -1
            if (text_queries is not None) and (0<=lab<len(text_queries)):
                name = text_queries[lab]
            else:
                name = "q%d"%lab

            # Format the label text to include the confidence score if show_scores is True and scores are available, otherwise just use the name.
            if bool(show_scores) and (scores is not None):
                s   = float(scores[i])
                txt = "%s %s"%(name,(score_fmt%s))
            else:
                txt = "%s"%name

            # Add the label text near the top-left corner of the bounding box, with an optional background for better visibility, and optionally clip the text to the image boundaries if clip_text is True.
            fs = 8 # default font size for labels, can be adjusted as needed
            viz_label(ax,x1,y1,txt,dx,dy,fs,bbox_txt,clip_on=bool(clip_text))

        # Add the title to the visualization using the viz_title helper function, and display the plot.
        viz_title(fig,ax,title,title_fs,title_overlay)
        plt.show()
        viz_close(fig)

    def run(
            self,
            img_pil,
            img_np,
            text_queries,
            threshold = 0.1,
            plot      = False,
            fig_w     = 4,
        ):
        """
        Run the object detection and optionally plot the results.

        Parameters:
            img_pil (PIL.Image.Image): The input image as a PIL image.
            img_np (numpy.ndarray): The input image as a numpy array (for plotting).
            text_queries (str or list of str): The text queries for object detection.
            threshold (float): The confidence threshold for filtering detections.
            plot (bool): Whether to plot the detected bounding boxes on the image.
            fig_w (float): Figure width in inches for visualization if plotting is enabled.
        """
        # Run the box prediction using the predict_boxes method.
        boxes,scores,labels = self.predict_boxes(
            img_pil,
            text_queries,
            threshold=threshold,
        )
        if self.verbose:
            qn = len(text_queries) if not isinstance(text_queries,str) else 1
            print("[PromptOwlv2] det:%d, threshold:%.3f, queries:%d"%(boxes.shape[0],float(threshold),qn))

        if plot:
            tq = text_queries if not isinstance(text_queries,str) else [text_queries]
            self.plot_dets(img_np,boxes,scores,labels,text_queries=tq,fig_w=fig_w)

        return boxes,scores,labels

class PromptOwlv2Sam2(object):
    """
    Text-based segmentation using:
      - OWLv2 for text-prompted detection (boxes)
      - SAM2 for point-prompted segmentation (mask)
    """
    def __init__(self,owl,sam2,verbose=True):
        """
        Initialize the PromptOwlv2Sam2 object.

        Parameters:
            owl: The OWLv2 model for text-prompted detection.
            sam2: The SAM2 model for point-prompted segmentation.
            verbose (bool): Whether to print verbose output.
        """
        self.owl     = owl
        self.sam2    = sam2
        self.verbose = bool(verbose)

    @staticmethod
    def _clip_xy(x,y,w,h):
        x = int(np.clip(int(x),0,w-1))
        y = int(np.clip(int(y),0,h-1))
        return [x,y]  # canonical

    def _box_to_points(self,box,w,h,neg_margin=10):
        """
        Convert a bounding box to positive and negative points for segmentation.

        Parameters:
            box (list or tuple): The bounding box coordinates [x1, y1, x2, y2].
            w (int): The width of the image.
            h (int): The height of the image.
            neg_margin (int): The margin for negative points around the bounding box.

        Returns:
            pos (list): List of positive points inside the bounding box.
            neg (list): List of negative points outside the bounding box.
        """
        x1,y1,x2,y2 = [float(v) for v in box]
        cx          = int(round(0.5*(x1+x2)))
        cy          = int(round(0.5*(y1+y2)))
        pos         = [self._clip_xy(cx,cy,w,h)]

        nm  = int(neg_margin)
        neg = [
            self._clip_xy(int(round(x1-nm)),int(round(y1-nm)),w,h),
            self._clip_xy(int(round(x2+nm)),int(round(y1-nm)),w,h),
            self._clip_xy(int(round(x1-nm)),int(round(y2+nm)),w,h),
            self._clip_xy(int(round(x2+nm)),int(round(y2+nm)),w,h),
        ]
        return pos,neg

    def _overlay_mask(self,img_np,mask,alpha=0.5,bright=0.35,mask_color=(255,0,0)):
        """
        Overlay a segmentation mask on the image.

        Parameters:
            img_np (numpy.ndarray): The input image as a numpy array.
            mask (numpy.ndarray): The segmentation mask.
            alpha (float): The alpha blending factor for the mask.
            bright (float): The brightness factor for the background.
            mask_color (tuple): The color of the mask.

        Returns:
            numpy.ndarray: The image with the mask overlay.
        """
        if mask is None:
            return img_np

        m = np.asarray(mask).astype(bool)

        if np.asarray(img_np).ndim==2:
            out = np.stack([img_np,img_np,img_np],axis=-1).astype(np.float32)
        else:
            out = np.asarray(img_np).astype(np.float32).copy()

        alpha  = float(alpha);  alpha=max(0.0,min(1.0,alpha))
        bright = float(bright); bright=max(0.0,min(1.0,bright))

        out[~m] = out[~m]*(1.0-bright)+255.0*bright
        mc      = np.array(mask_color,dtype=np.float32).reshape((1,1,3))
        out[m]  = alpha*out[m]+(1.0-alpha)*mc

        return np.clip(out,0,255).astype(np.uint8)

    def plot_det_seg(
            self,
            img_np,
            results,
            fig_w          = 4,
            show_scores    = True,
            score_fmt      = "%.2f",
            linewidth      = 2,
            edgecolor      = "r",
            text_offset    = (2,2),
            overlay_alpha  = 0.5,
            overlay_bright = 0.35,
            mask_color     = (255,0,0),
            dpi            = 120,
            pad_px         = 0,
            cmap           = "viridis",
            clip_text      = False,
        ):
        """
        Visualize the detected bounding boxes and segmentation masks on the image.

        Parameters:
            img_np (numpy.ndarray): The original image as a numpy array.
            results (list of dict): The list of detection and segmentation results, where each dict contains keys like 'box', 'score', 'label', 'mask', etc.
            fig_w (float): Figure width in inches for visualization.
            show_scores (bool): Whether to display confidence scores in the labels.
            score_fmt (str): The format string for displaying scores.
            linewidth (float): The line width for the bounding box edges.
            edgecolor (str): The color for the bounding box edges.
            text_offset (tuple): Offset for the text labels relative to the bounding box corners.
            overlay_alpha (float): The alpha blending factor for the mask overlay.
            overlay_bright (float): The brightness factor for the background when overlaying the mask.
            mask_color (tuple): The color of the mask overlay.
            dpi (int): Dots per inch for the figure.
            pad_px (int): Padding in pixels for the figure.
            cmap (str): Colormap to use for the image if it's grayscale.
            clip_text (bool): Whether to clip text labels to the image boundaries.
        """
        img_np = np.asarray(img_np)
        h,w    = img_np.shape[:2]
        fig_h  = float(fig_w)*float(h)/max(1.0,float(w))

        n = int(len(results)) if results is not None else 0

        bbox_txt = viz_text_bbox(facecolor="white",alpha=0.7,pad=0.05,edgecolor="none")
        dx,dy    = float(text_offset[0]),float(text_offset[1])

        if n==0:
            fig,ax = viz_make_fig_ax(fig_w,fig_h,dpi,pad_px,w,h)
            if img_np.ndim==2:
                ax.imshow(img_np,cmap=str(cmap))
            else:
                ax.imshow(img_np)
            plt.show()
            viz_close(fig)
            return

        for i in range(n):
            r       = results[i]
            fig,ax  = viz_make_fig_ax(fig_w,fig_h,dpi,pad_px,w,h)

            alpha   = float(r.get("alpha",overlay_alpha))
            bright  = float(r.get("bright",overlay_bright))
            mcolor  = r.get("mask_color",mask_color)

            if r.get("mask",None) is None:
                vis = img_np
            else:
                vis = self._overlay_mask(img_np,r["mask"],alpha=alpha,bright=bright,mask_color=mcolor)

            if np.asarray(vis).ndim==2:
                ax.imshow(vis,cmap=str(cmap))
            else:
                ax.imshow(vis)

            x1,y1,x2,y2 = [float(v) for v in r["box"]]
            rect = plt.Rectangle(
                (x1,y1),
                x2-x1,
                y2-y1,
                fill      = False,
                linewidth = float(linewidth),
                edgecolor = str(edgecolor),
            )
            ax.add_patch(rect)

            name = str(r.get("query","q%d"%int(r.get("label",-1))))
            if bool(show_scores):
                s   = float(r.get("score",0.0))
                txt = "%s %s"%(name,(score_fmt%s))
            else:
                txt = "%s"%name

            viz_label(ax,x1,y1,txt,dx,dy,9,bbox_txt,clip_on=bool(clip_text))

            plt.show()
            viz_close(fig)

    def run(
            self,
            img_pil,
            text_queries,
            det_threshold  = 0.1,
            topk_per_query = 1,
            neg_margin     = 10,
            use_neg_points = True,
            fill_holes     = True,
            keep_largest   = True,
            prefer_pos_mask = True,
            overlay_alpha  = 0.5,
        ):
        """
        Run the combined OWLv2 detection and SAM2 segmentation for the given image and text queries.

        Parameters:
            img_pil (PIL.Image.Image): The input image as a PIL image.
            text_queries (str or list of str): The text queries for object detection and segmentation.
            det_threshold (float): The confidence threshold for filtering detections from OWLv2.
            topk_per_query (int): The maximum number of detections to consider per text query for segmentation.
            neg_margin (int): The margin in pixels for generating negative points around the detected bounding box.
            use_neg_points (bool): Whether to use negative points around the
                detected bounding box. Set False to match point-only SAM2
                prompting used by notebook/10_interactive_pick_and_place.ipynb.
            fill_holes (bool): Whether to fill holes in the predicted segmentation masks.
            keep_largest (bool): Whether to keep only the largest connected component in the predicted segmentation masks.
            prefer_pos_mask (bool): Whether to prefer SAM2 candidate masks that contain the bbox center positive point.
            overlay_alpha (float): The alpha blending factor for the mask overlay when visualizing results.
        """
        if isinstance(text_queries,str):
            text_queries=[text_queries]
        if (text_queries is None) or (len(text_queries)==0):
            raise ValueError("text_queries must be non-empty.")

        W,H = img_pil.size

        boxes,scores,labels = self.owl.predict_boxes(
            img_pil,
            text_queries,
            threshold=det_threshold,
        )

        results = []
        for qi in range(len(text_queries)):
            idxs = np.where(labels==qi)[0]
            if idxs.size==0:
                continue

            idxs = idxs[np.argsort(scores[idxs])[::-1]]
            idxs = idxs[:int(topk_per_query)]

            for j in idxs.tolist():
                box   = boxes[j].astype(np.float32)
                score = float(scores[j])

                pos,neg = self._box_to_points(box,W,H,neg_margin=neg_margin)
                if not bool(use_neg_points):
                    neg = []

                # unify inter-class point format: list of [x,y] ints
                pos = norm_points_xy_list(pos)
                neg = norm_points_xy_list(neg)

                mask = self.sam2.predict_mask(
                    img_pil,
                    pos             = pos,
                    neg             = neg,
                    fill_holes      = fill_holes,
                    keep_largest    = keep_largest,
                    prefer_pos_mask = prefer_pos_mask,
                )

                results.append(dict(
                    query = text_queries[qi],
                    box   = box,
                    score = score,
                    label = int(qi),
                    pos   = pos,
                    neg   = neg,
                    mask  = np.asarray(mask).astype(bool),
                    alpha = float(overlay_alpha),
                ))

        if self.verbose:
            print("[PromptOwlv2Sam2] queries:%d, det_threshold:%.3f, results:%d"%
                  (len(text_queries),float(det_threshold),len(results)))

        return results
