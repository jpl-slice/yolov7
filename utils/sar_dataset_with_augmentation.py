import contextlib
import glob
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2  # For augmentations like random_perspective and augment_hsv
import numpy as np
import rasterio
import torch
from pycocotools.coco import COCO
from rasterio import windows
from torch import Tensor
from torch.utils.data import Dataset

# Assuming torchvision.transforms.Compose might still be used if self.transform is set
from torchvision.transforms import Compose
from tqdm.auto import tqdm

# --- Import YOLOv7 augmentation utilities ---
# These paths might need adjustment based on your project structure.
# CRITICAL: random_perspective in your yolov7/utils/datasets.py MUST be modified
# to accept and use a 'border_value' argument for nodata handling.
from utils.datasets import augment_hsv, pastein, random_perspective

MAX_NODATA_FRACTION = 0.4


def get_nodata_from_src(src: rasterio.DatasetReader) -> float:
    # return self.src.nodata if self.src.nodata is not None else -9999.0
    if src.nodata is not None:
        return src.nodata
    else:
        data = src.read(
            1,
            window=windows.Window(0, 0, min(128, src.width), min(128, src.height)),
            boundless=True,
            fill_value=0,
        )  # Read a sample
        if np.any(data < -9000):  # Heuristic from user
            # If nodata is not set, assume a common nodata value for SAR data
            return -9999.0
        else:
            return 0.0  # Default to 0 if not otherwise determined.


class SARTileDetectionDataset(Dataset):
    """
    A sliding-window detection dataset over one or many SAR GeoTIFF(s),
    returning COCO-style targets per-tile. Incorporates YOLOv7-style augmentations.

    Image and Annotation File Handling:
    -----------------------------------
    The dataset can be initialized in a few ways depending on how image paths
    are specified and where the image files are located:

    1. `image_input_path` is `None` (Default):
       - Image paths are sourced directly from the `file_name` field in the
         `ann_file` (COCO JSON).
       - If `file_name` is an absolute path, it's used as is.
       - If `file_name` is a relative path or just a basename (e.g., "image1.tif"),
         the dataset first attempts to locate the image relative to the current
         working directory.
       - If not found, it then tries to resolve the path relative to the
         directory containing the `ann_file` itself.
         (e.g., if `ann_file` is "/path/to/annotations/coco.json" and `file_name`
         is "image1.tif", it will look for "/path/to/annotations/image1.tif").

    2. `image_input_path` is a directory path (e.g., "/path/to/images/"):
       - The dataset will look for image files within this specified directory.
       - It expects the `file_name` field in the `ann_file` to contain
         basenames (e.g., "image1.tif", "image2.tif").
       - Only images whose basenames are listed in the `ann_file` and are
         found in `image_input_path` will be processed.

    3. `image_input_path` is a path to a single image file (e.g., "/path/to/images/image1.tif"):
       - The dataset will process only this single image file.
       - It expects the basename of this provided image file (e.g., "image1.tif")
         to be present in one of the `file_name` fields within the `ann_file`.

    Attributes:
        win_size (int): Size of the sliding window (tile height and width).
        stride (int): Stride of the sliding window.
        transform (Optional[Compose]): Torchvision transforms to apply to each tile.
        hyp (Optional[Dict[str, Any]]): Hyperparameters dictionary, typically from YOLOv7.
                                        Used for augmentation parameters.
        augment (bool): If True, apply YOLOv7-style augmentations.
        img_size (int): Target image size after potential resizing (used for mosaic border,
                        though mosaic is currently disabled).
        coco (COCO): COCO API instance for the annotation file.
        ann_file (str): Path to the COCO annotation JSON file.
        _windows (List[windows.Window]): List of rasterio window objects for valid tiles.
        labels (List[np.ndarray]): List of labels for each tile, in
                                   [class_idx, cx_norm, cy_norm, w_norm, h_norm] format.
        shapes (List[Tuple[int, int]]): List of (height, width) for each original tile.
        _window_paths (List[str]): List of source GeoTIFF paths for each window.
        img_files (List[str]): Generated unique names for each tile/window.
        n (int): Total number of valid tiles.
        indices (List[int]): List of indices for data shuffling/access.
        names (List[str]): List of category names from COCO annotations.
    """

    def __init__(
        self,
        ann_file: str,
        image_input_path: Optional[str] = None,
        window_size: int = 448,
        stride: int = 224,
        transform: Optional[Compose] = None,
        hyp: Optional[Dict[str, Any]] = None,
        augment: bool = False,
        img_size: int = 448,
    ):
        self.win_size = window_size
        self.stride = stride
        self.transform = transform

        self.hyp = hyp if hyp is not None else {}
        self.augment = augment
        self.img_size = img_size
        self.mosaic = False
        self.mosaic_border = [-img_size // 2, -img_size // 2]

        self.coco = COCO(ann_file)
        self.ann_file = ann_file

        self._windows: List[windows.Window] = []
        self.labels: List[np.ndarray] = []
        self.shapes: List[Tuple[int, int]] = []
        self._window_paths: List[str] = []

        if image_input_path is None:
            coco_images_info = self.coco.dataset.get("images", [])
            if not coco_images_info:
                raise ValueError(
                    f"No images found in the annotation file: {self.ann_file}"
                )
            processed_count = 0
            for img_info in coco_images_info:
                image_path_str = img_info.get("file_name")
                if not image_path_str:
                    print(
                        f"Warning: Image info in {self.ann_file} missing 'file_name': {img_info}. Skipping."
                    )
                    continue
                image_path = Path(image_path_str)
                if not image_path.exists():
                    # Try to resolve relative to annotation file's directory
                    # alt_image_path = Path(self.ann_file).parent / Path(image_path_str).name
                    alt_image_path = Path(
                        os.path.join(
                            os.path.dirname(self.ann_file),
                            os.path.basename(image_path_str),
                        )
                    )
                    if not alt_image_path.exists():
                        print(
                            f"Warning: Image file not found: {image_path_str} (also tried {alt_image_path}). Skipping this image."
                        )
                        continue
                    image_path = alt_image_path  # Use resolved path
                try:
                    if self._process_single_image(image_path):
                        processed_count += 1
                except ValueError as e:
                    print(
                        f"Warning: Could not process image {image_path_str} from {self.ann_file} due to: {e}. Skipping."
                    )
            if coco_images_info and processed_count == 0:
                print(
                    f"Warning: Images were listed in {self.ann_file}, but none could be processed successfully."
                )
        else:
            p = Path(image_input_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"Provided image_input_path does not exist: {image_input_path}"
                )

            if p.is_dir():
                tif_paths = sorted(glob.glob(str(p / "*.tif"))) + sorted(
                    glob.glob(str(p / "*.tiff"))
                )
                tif_basenames = {Path(tp).name for tp in tif_paths}

                coco_img_filenames = {
                    Path(img["file_name"]).name
                    for img in self.coco.dataset.get("images", [])
                }

                relevant_tif_paths = [
                    tp for tp in tif_paths if Path(tp).name in coco_img_filenames
                ]

                missing_in_folder = [
                    name for name in coco_img_filenames if name not in tif_basenames
                ]
                if missing_in_folder:
                    print(
                        f"Warning: The following images listed in {ann_file} are missing in {image_input_path}: {missing_in_folder}"
                    )

                if not relevant_tif_paths:
                    raise ValueError(
                        f"No relevant .tif/.tiff files found in directory {image_input_path} that are also listed in {ann_file}."
                    )

                for tif_path_str in tqdm(
                    relevant_tif_paths, desc="Processing images from directory"
                ):
                    self._process_single_image(Path(tif_path_str))
            else:  # Single file mode
                # Check if this single file is listed in COCO annotations
                is_in_coco = any(
                    Path(img_info["file_name"]).name == p.name
                    for img_info in self.coco.dataset.get("images", [])
                )
                if not is_in_coco:
                    raise ValueError(
                        f"The single image file {p.name} provided via image_input_path is not listed in the annotation file {ann_file}."
                    )
                self._process_single_image(p)

        if not self._windows:
            error_msg = f"No valid windows were generated. "
            if image_input_path is None:
                error_msg += f"Images sourced from {self.ann_file}. "
            else:
                error_msg += f"Images sourced from {image_input_path} and filtered by {self.ann_file}. "
            error_msg += (
                "Ensure image files exist, are accessible, have corresponding annotations, "
                "and meet windowing criteria (e.g., nodata fraction)."
            )
            raise ValueError(error_msg)

        self.img_files = [
            f"{Path(fp).stem}_win{i}" for i, fp in enumerate(self._window_paths)
        ]
        self.shapes = np.array(self.shapes, dtype=np.int64)
        self.n = len(self._windows)
        self.indices = list(range(self.n))

        cat_ids = self.coco.getCatIds()
        self.names = [cat["name"] for cat in self.coco.loadCats(cat_ids)]

    def __len__(self) -> int:
        return len(self._windows)

    def _process_single_image(self, tif_path: Path) -> bool:
        """Process one GeoTIFF and extend class-level lists in-place. Returns True if successful."""
        try:
            self._initialize_raster_properties(tif_path)
            initial_windows = self._compute_initial_windows()

            basename = self._load_coco_annotations_and_image_info(tif_path)
            if basename is None:
                return False

            kept_wins, kept_lbls, kept_shps = self._build_filtered_labels_and_windows(
                initial_windows, self.win_size
            )

            if kept_wins:
                self._windows.extend(kept_wins)
                self.labels.extend(kept_lbls)
                self.shapes.extend(kept_shps)
                self._window_paths.extend([str(tif_path)] * len(kept_wins))
            return True
        except Exception as e:
            print(f"Error processing single image {tif_path}: {e}")
            return False
        finally:
            if hasattr(self, "src") and self.src is not None:
                with contextlib.suppress(Exception):
                    self.src.close()
                self.src = None

    def _initialize_raster_properties(self, geotiff_path: Path):
        self.src = rasterio.open(str(geotiff_path))
        self.nodata_val = get_nodata_from_src(self.src)

    def _compute_initial_windows(self) -> List[windows.Window]:
        h, w = self.src.height, self.src.width
        initial_windows_list: List[windows.Window] = []
        if h < self.win_size or w < self.win_size:
            return []

        for y_offset in range(0, h - self.win_size + 1, self.stride):
            for x_offset in range(0, w - self.win_size + 1, self.stride):
                win = windows.Window(x_offset, y_offset, self.win_size, self.win_size)
                data = self.src.read(
                    1, window=win, boundless=True, fill_value=self.nodata_val
                )
                if (data == self.nodata_val).mean() < MAX_NODATA_FRACTION:
                    initial_windows_list.append(win)
                    if np.nanmean(data) == 0:
                        import pdb

                        pdb.set_trace()
        return initial_windows_list

    def _load_coco_annotations_and_image_info(
        self, geotiff_path: Path
    ) -> Optional[str]:
        basename = geotiff_path.name
        img_ids = [
            img["id"]
            for img in self.coco.dataset.get("images", [])
            if Path(img["file_name"]).name == basename
        ]

        if not img_ids:
            print(
                f"Warning: No image entry for {basename} in {self.ann_file} based on filename. Skipping this image for annotations."
            )
            return None

        self.img_id = img_ids[0]
        self.anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[self.img_id]))
        return basename

    def _build_filtered_labels_and_windows(
        self,
        initial_windows: List[windows.Window],
        window_size: int,
    ) -> Tuple[List[windows.Window], List[np.ndarray], List[Tuple[int, int]]]:
        NEG_KEEP_PROB = self.hyp.get("keep_neg_prob", 0.0)

        kept_windows, kept_labels, kept_shapes = [], [], []
        for win in initial_windows:
            x0_win, y0_win = win.col_off, win.row_off
            boxes_in_window = []
            for ann in self.anns:
                x1_abs, y1_abs, w_abs, h_abs = ann["bbox"]
                x2_abs, y2_abs = x1_abs + w_abs, y1_abs + h_abs

                xi1 = max(x1_abs, x0_win)
                yi1 = max(y1_abs, y0_win)
                xi2 = min(x2_abs, x0_win + window_size)
                yi2 = min(y2_abs, y0_win + window_size)

                if xi2 > xi1 and yi2 > yi1:
                    bx_rel = xi1 - x0_win
                    by_rel = yi1 - y0_win
                    bw_rel = xi2 - xi1
                    bh_rel = yi2 - yi1

                    cx = (bx_rel + bw_rel / 2) / window_size
                    cy = (by_rel + bh_rel / 2) / window_size
                    bw = bw_rel / window_size
                    bh = bh_rel / window_size

                    cat_id = ann.get("category_id")
                    if cat_id is None:
                        continue

                    # Assuming COCO category IDs are 1-indexed. Map to 0-indexed.
                    # Verify this mapping against your specific COCO file.
                    class_idx = cat_id - 1
                    # You might need a more robust mapping if IDs are not contiguous or not 1-based.
                    # e.g. self.coco_cat_id_to_label = {coco_id: i for i, coco_id in enumerate(self.coco.getCatIds())}
                    # class_idx = self.coco_cat_id_to_label.get(cat_id)
                    # if class_idx is None: continue

                    boxes_in_window.append([class_idx, cx, cy, bw, bh])

            if not boxes_in_window:
                if random.random() > NEG_KEEP_PROB:
                    continue
                current_labels = np.zeros((0, 5), dtype=np.float32)
            else:
                current_labels = np.array(boxes_in_window, dtype=np.float32)

            kept_windows.append(win)
            kept_labels.append(current_labels)
            kept_shapes.append((window_size, window_size))

        return kept_windows, kept_labels, kept_shapes

    def _get_individual_object_samples_from_tile(
        self, tile_idx: int
    ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        img_hwc = self._load_and_preprocess_tile(tile_idx)
        labels_normalized_5_col = self.labels[tile_idx].copy()

        s_labels, s_images, s_masks = [], [], []
        if len(labels_normalized_5_col) == 0:
            return s_labels, s_images, s_masks

        h_tile, w_tile = img_hwc.shape[:2]

        for obj_label_norm in labels_normalized_5_col:
            cls_idx = obj_label_norm[0]
            cx_norm, cy_norm, bw_norm, bh_norm = obj_label_norm[1:]

            x1 = int((cx_norm - bw_norm / 2) * w_tile)
            y1 = int((cy_norm - bh_norm / 2) * h_tile)
            x2 = int((cx_norm + bw_norm / 2) * w_tile)
            y2 = int((cy_norm + bh_norm / 2) * h_tile)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_tile, x2), min(h_tile, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            obj_img_crop = img_hwc[y1:y2, x1:x2]
            obj_mask_crop = np.full(obj_img_crop.shape[:2], 255, dtype=np.uint8)

            if obj_img_crop.size == 0 or obj_mask_crop.size == 0:
                continue

            s_labels.append(float(cls_idx))
            s_images.append(obj_img_crop)
            s_masks.append(obj_mask_crop)

        return s_labels, s_images, s_masks

    def __getitem__(self, index: int) -> Tuple[
        Tensor,
        Tensor,
        str,
        Tuple[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]],
    ]:
        actual_index = self.indices[index]
        img_hwc_orig = self._load_and_preprocess_tile(actual_index)

        labels_np_5col = (
            self.labels[actual_index].copy()
            if len(self.labels[actual_index]) > 0
            else np.zeros((0, 5), dtype=np.float32)
        )

        h0, w0 = img_hwc_orig.shape[:2]  # Current tile dimensions

        img_augmented = img_hwc_orig.copy()
        # Labels for augmentation need to be in [cls, x1, y1, x2, y2] pixel format
        # Convert normalized [cls, cx, cy, w, h] to pixel [cls, x1, y1, x2, y2]
        labels_pixel_xyxy = np.zeros_like(labels_np_5col)
        if len(labels_np_5col) > 0:
            labels_pixel_xyxy[:, 0] = labels_np_5col[:, 0]  # class
            labels_pixel_xyxy[:, 1] = (
                labels_np_5col[:, 1] - labels_np_5col[:, 3] / 2
            ) * w0  # x1
            labels_pixel_xyxy[:, 2] = (
                labels_np_5col[:, 2] - labels_np_5col[:, 4] / 2
            ) * h0  # y1
            labels_pixel_xyxy[:, 3] = (
                labels_np_5col[:, 1] + labels_np_5col[:, 3] / 2
            ) * w0  # x2
            labels_pixel_xyxy[:, 4] = (
                labels_np_5col[:, 2] + labels_np_5col[:, 4] / 2
            ) * h0  # y2
            # Clip to image boundaries
            labels_pixel_xyxy[:, [1, 3]] = np.clip(labels_pixel_xyxy[:, [1, 3]], 0, w0)
            labels_pixel_xyxy[:, [2, 4]] = np.clip(labels_pixel_xyxy[:, [2, 4]], 0, h0)

        labels_augmented_xyxy = labels_pixel_xyxy.copy()

        if self.augment:
            # Store original for fallback
            img_backup = img_augmented.copy()
            labels_backup = labels_augmented_xyxy.copy()

            if not self.mosaic:  # This class doesn't implement mosaic
                img_augmented, labels_augmented_xyxy = random_perspective(
                    img_augmented,
                    targets=labels_augmented_xyxy,
                    segments=(),  # No segments for SAR tiles in this context
                    degrees=self.hyp.get("degrees", 0.0),
                    translate=self.hyp.get("translate", 0.0),
                    scale=self.hyp.get("scale", 0.0),
                    shear=self.hyp.get("shear", 0.0),
                    perspective=self.hyp.get("perspective", 0.0),
                    # border=self.mosaic_border,
                    border_value=(
                        0,
                        0,
                        0,
                    ),  # CRITICAL: ensure your random_perspective uses this
                )

                # Check if augmentation resulted in too much nodata
                valid_pixels = img_augmented != 0  # Assuming 0 is nodata
                nodata_fraction = 1.0 - np.mean(valid_pixels)

                if nodata_fraction > MAX_NODATA_FRACTION:
                    # Revert to original if augmentation created too much nodata
                    img_augmented = img_backup
                    labels_augmented_xyxy = labels_backup

            img_augmented_hsv = img_augmented.copy()
            augment_hsv(
                img_augmented_hsv,
                hgain=self.hyp.get("hsv_h", 0.0),
                sgain=self.hyp.get("hsv_s", 0.0),
                vgain=self.hyp.get("hsv_v", 0.0),
            )

            hsv_nodata_fraction = np.mean(img_augmented_hsv == 0)
            if hsv_nodata_fraction > MAX_NODATA_FRACTION:
                print(
                    f"Warning: Too much nodata after augment_hsv ({hsv_nodata_fraction:.2%}). Reverting to original tile."
                )
            else:
                img_augmented = img_augmented_hsv

            if random.random() < self.hyp.get("paste_in", 0.0):
                (
                    collected_sample_labels,
                    collected_sample_images,
                    collected_sample_masks,
                ) = ([], [], [])
                num_objects_target = self.hyp.get("paste_in_objects_total", 10)
                attempts = 0
                max_load_attempts = self.hyp.get(
                    "paste_in_load_attempts",
                    len(self._windows) * 2 if self._windows else 20,
                )

                while (
                    len(collected_sample_labels) < num_objects_target
                    and attempts < max_load_attempts
                ):
                    if not self._windows:
                        break
                    sample_tile_idx = random.randint(0, len(self._windows) - 1)
                    try:
                        ind_labels, ind_images, ind_masks = (
                            self._get_individual_object_samples_from_tile(
                                sample_tile_idx
                            )
                        )
                        if ind_labels:
                            collected_sample_labels.extend(ind_labels)
                            collected_sample_images.extend(ind_images)
                            collected_sample_masks.extend(ind_masks)
                    except Exception:
                        pass
                    attempts += 1

                if collected_sample_labels:
                    labels_augmented_xyxy = pastein(
                        image=img_augmented,
                        labels=labels_augmented_xyxy,
                        sample_labels=collected_sample_labels,
                        sample_images=collected_sample_images,
                        sample_masks=collected_sample_masks,
                    )

            if random.random() < self.hyp.get("flipud", 0.0):
                img_augmented = np.flipud(img_augmented)
                if len(labels_augmented_xyxy):
                    labels_augmented_xyxy[:, [2, 4]] = (
                        img_augmented.shape[0] - labels_augmented_xyxy[:, [4, 2]]
                    )  # y1, y2 swap and subtract

            if random.random() < self.hyp.get("fliplr", 0.0):
                img_augmented = np.fliplr(img_augmented)
                if len(labels_augmented_xyxy):
                    labels_augmented_xyxy[:, [1, 3]] = (
                        img_augmented.shape[1] - labels_augmented_xyxy[:, [3, 1]]
                    )  # x1, x2 swap and subtract

        # Convert labels from [cls, x1,y1,x2,y2] (pixels) back to [cls, cx,cy,w,h] (normalized)
        # This is the format expected by the collate_fn for YOLO training
        final_labels_normalized_xywh = np.zeros(
            (len(labels_augmented_xyxy), 5), dtype=np.float32
        )
        if len(labels_augmented_xyxy) > 0:
            final_labels_normalized_xywh[:, 0] = labels_augmented_xyxy[:, 0]  # class
            img_h_aug, img_w_aug = img_augmented.shape[:2]

            x1 = labels_augmented_xyxy[:, 1]
            y1 = labels_augmented_xyxy[:, 2]
            x2 = labels_augmented_xyxy[:, 3]
            y2 = labels_augmented_xyxy[:, 4]

            final_labels_normalized_xywh[:, 1] = ((x1 + x2) / 2) / img_w_aug  # cx
            final_labels_normalized_xywh[:, 2] = ((y1 + y2) / 2) / img_h_aug  # cy
            final_labels_normalized_xywh[:, 3] = (x2 - x1) / img_w_aug  # w
            final_labels_normalized_xywh[:, 4] = (y2 - y1) / img_h_aug  # h

            # Clip normalized values to [0, 1]
            final_labels_normalized_xywh[:, 1:] = np.clip(
                final_labels_normalized_xywh[:, 1:], 0, 1
            )

            # Remove invalid boxes (zero or negative width/height)
            valid_boxes_mask = (final_labels_normalized_xywh[:, 3] > 0) & (
                final_labels_normalized_xywh[:, 4] > 0
            )
            final_labels_normalized_xywh = final_labels_normalized_xywh[
                valid_boxes_mask
            ]

        # Prepare final labels tensor (N, 6) [batch_idx, cls, cx, cy, w, h]
        output_labels_torch = torch.zeros(
            (len(final_labels_normalized_xywh), 6), dtype=torch.float32
        )
        if len(final_labels_normalized_xywh) > 0:
            output_labels_torch[:, 1:] = torch.from_numpy(final_labels_normalized_xywh)
        elif len(final_labels_normalized_xywh) == 0 and len(labels_augmented_xyxy) > 0:
            raise ValueError(
                f"No valid labels after augmentation for tile index {actual_index}. "
                "Check if the augmentations are too aggressive or if the input data is too sparse."
            )

        # Prepare final labels tensor (N, 6) [batch_idx, cls, cx, cy, w, h]
        output_labels_torch = torch.zeros(
            (len(final_labels_normalized_xywh), 6), dtype=torch.float32
        )
        if len(final_labels_normalized_xywh) > 0:
            output_labels_torch[:, 1:] = torch.from_numpy(final_labels_normalized_xywh)

        # Convert image to PyTorch tensor (CHW from HWC) and normalize to [0,1]
        # UPDATE: Do NOT normalize to [0,1] here, leave it as uint8 for consistency with YOLOv7 expectations
        img_tensor = torch.from_numpy(
            np.ascontiguousarray(img_augmented.transpose(2, 0, 1))
        )  # .float() / 255.0

        path_out = (
            self.img_files[actual_index]
            if actual_index < len(self.img_files)
            else f"sar_tile_{actual_index}"
        )

        # shapes_for_collate: ((original_H, original_W), ((final_H_scale, final_W_scale), (pad_W, pad_H)))
        # For SAR tiles, original H, W is win_size. Scale is 1 if no resize. Pad is 0 if no letterbox.
        # This might need adjustment if letterbox is used after augmentations.
        current_h_aug, current_w_aug = img_augmented.shape[:2]
        scale_ratio_h = current_h_aug / h0
        scale_ratio_w = current_w_aug / w0
        # Padding is assumed to be zero unless a letterbox step is explicitly added after augmentations
        # and before this return statement.
        padding_values = (0.0, 0.0)
        shapes_for_collate = (
            (h0, w0),
            ((scale_ratio_h, scale_ratio_w), padding_values),
        )

        return img_tensor, output_labels_torch, path_out, shapes_for_collate

    def _load_and_preprocess_tile(self, idx: int) -> np.ndarray:
        win = self._windows[idx]
        tif_path = self._window_paths[idx]
        tif_path_str = str(tif_path)

        with rasterio.open(tif_path_str) as src:
            arr = src.read(
                1, window=win, boundless=True, fill_value=get_nodata_from_src(src)
            ).astype(np.float32)
            nodata = get_nodata_from_src(src)

        arr[arr == nodata] = np.nan
        valid_mask = ~np.isnan(arr)

        if not np.any(valid_mask):
            processed_arr = np.zeros((self.win_size, self.win_size), dtype=np.uint8)
        else:
            arr_valid_pixels = arr[valid_mask]
            # Use try-except for nanpercentile in case all valid pixels are identical (rare)
            try:
                lo, hi = np.nanpercentile(arr_valid_pixels, (1, 99))
            except (
                ValueError
            ):  # Handles cases like all valid pixels being the same value
                lo, hi = arr_valid_pixels[0], arr_valid_pixels[0]

            if hi - lo < 1e-6:
                # All valid pixels are almost the same. Scale them to 1 (min valid value).
                # Or map to a specific gray level e.g. 128 if you prefer.
                # Scaling to 1 ensures they are not 0 (nodata).
                scaled_valid_pixels = np.ones_like(arr_valid_pixels, dtype=np.uint8)
            else:
                normalized_valid_pixels = (arr_valid_pixels - lo) / (hi - lo + 1e-6)
                normalized_valid_pixels = np.clip(normalized_valid_pixels, 0, 1)
                scaled_valid_pixels = (
                    normalized_valid_pixels * 254.0
                ) + 1.0  # Scale to [1, 255]

            processed_arr = np.zeros_like(
                arr, dtype=np.uint8
            )  # Initialize with nodata value (0)
            processed_arr[valid_mask] = scaled_valid_pixels.astype(np.uint8)

        img_hwc = np.stack([processed_arr] * 3, axis=-1)
        if np.all(img_hwc == 0):
            raise ValueError(
                f"Processed image from {tif_path_str} is all zeros after scaling. Check nodata handling or input data."
            )
        return img_hwc

    @staticmethod
    def collate_fn(batch):
        imgs, labels, paths, shapes = zip(*batch)
        for i, l_tensor in enumerate(labels):
            if l_tensor.shape[0] > 0:
                l_tensor[:, 0] = i
        return torch.stack(imgs, 0), torch.cat(labels, 0), paths, shapes
