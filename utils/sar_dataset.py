import contextlib
import glob  # Added import
import random  # Added import
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
import torch
from pycocotools.coco import COCO
from rasterio import windows
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose

MAX_NODATA_FRACTION = 0.4


class SARTileDetectionDataset(Dataset):
    """
    A sliding-window detection dataset over one **or many** SAR GeoTIFF(s),
    returning COCO-style targets per-tile.

    Can be initialized in three ways:
    1. ann_file only: If *image_input_path* is None, images are sourced
       directly from the 'file_name' entries in the *ann_file*.
       'file_name' in COCO JSON should be a full or resolvable relative path.
    2. Single .tif file: If *image_input_path* is a path to a .tif file,
       that specific file is processed if listed in *ann_file*.
    3. Directory of .tif files: If *image_input_path* is a path to a directory,
       all ``*.tif`` files within that directory that are also listed in
       *ann_file* are processed. The annotation file must list images present in this directory.
    """

    def __init__(
        self,
        ann_file: str,
        image_input_path: Optional[
            str
        ] = None,  # Not optional, and may be a file or a directory now
        window_size: int = 448,
        stride: int = 224,
        transform: Optional[Compose] = None,
    ):
        # ── 0) basic attrs ───────────────────────────────────────────────
        self.win_size = window_size
        self.stride = stride
        self.transform = transform

        # ── 1) load COCO dataset once ───────────────────────────────────
        self.coco = COCO(ann_file)
        self.ann_file = ann_file

        # ── 2) containers that will be progressively filled ─────────────
        self._windows: List[windows.Window] = []
        self.labels: List[np.ndarray] = []
        self.shapes: List[Tuple[int, int]] = []
        self._window_paths: List[str] = []  # full path for each window

        if image_input_path is None:
            # ── Mode 1: ann_file only ───────────────────────────────────────
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
                    print(
                        f"Warning: Image file not found: {image_path_str} (listed in {self.ann_file}). Skipping this image."
                    )
                    continue

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
                # ── Mode 3: directory ──────────────────────────────────────
                tif_paths = sorted(glob.glob(str(p / "*.tif")))
                tif_basenames = {Path(tp).name for tp in tif_paths}  # O(1) look‑ups
                # Ensure all images referenced in COCO exist in the folder
                missing = [
                    img["file_name"]
                    for img in self.coco.dataset["images"]
                    if Path(img["file_name"]).name not in tif_basenames
                ]
                if missing:
                    raise FileNotFoundError(
                        f"The following images listed in {ann_file} are missing in {image_input_path}: {missing}"
                    )

                for tif_path in tif_paths:
                    self._process_single_image(Path(tif_path))
            else:
                # ── Mode 2: single-file mode (original behavior) ───────────────────
                _processed = self._process_single_image(p)
                if _processed is None:
                    raise ValueError(
                        f"No image entry for {p.name} in {ann_file}. "
                        "Ensure the GeoTIFF file is listed in the COCO annotations."
                    )

        # ── 3) finalise common attrs ────────────────────────────────────
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
        self.n = len(self.img_files)
        self.indices = list(range(self.n))
        self.mosaic_border = [-window_size // 2, -window_size // 2]
        self.names = self.coco.loadCats(self.coco.getCatIds())

    def __len__(self) -> int:
        return len(self._windows)

    # ─────────────────────────────────────────────────────────────────────
    # internal helpers
    # ─────────────────────────────────────────────────────────────────────
    def _process_single_image(self, tif_path: Path):
        """Process one GeoTIFF and extend class-level lists in-place."""
        self._initialize_raster_properties(tif_path)
        initial_windows = self._compute_initial_windows()

        basename = self._load_coco_annotations_and_image_info(tif_path)
        if basename is None:
            return  # skip unannotated images

        kept_wins, kept_lbls, kept_shps = self._build_filtered_labels_and_windows(
            initial_windows, self.win_size
        )
        self._windows.extend(kept_wins)
        self.labels.extend(kept_lbls)
        self.shapes.extend(kept_shps)
        self._window_paths.extend([str(tif_path)] * len(kept_wins))

        # Close raster to avoid exhausting file handles
        with contextlib.suppress(Exception):
            self.src.close()
        return basename

    def _initialize_raster_properties(self, geotiff_path: Path):
        # ── open the raster ────────────────────────────────────────────
        self.src = rasterio.open(str(geotiff_path))
        self.nodata = self.src.nodata if self.src.nodata is not None else -9999.0
        print(f"{geotiff_path.name} {self.nodata=}")

    def _compute_initial_windows(self) -> List[windows.Window]:
        # ── precompute all valid windows ───────────────────────────────
        h, w = self.src.height, self.src.width
        initial_windows_list: List[windows.Window] = []
        for y in range(0, h - self.win_size + 1, self.stride):
            for x in range(0, w - self.win_size + 1, self.stride):
                win = windows.Window(x, y, self.win_size, self.win_size)
                data = self.src.read(1, window=win)
                if (data == self.nodata).mean() < MAX_NODATA_FRACTION:
                    initial_windows_list.append(win)
        return initial_windows_list

    def _load_coco_annotations_and_image_info(self, geotiff_path: Path) -> str:
        # ── load COCO annotations for this image ───────────────────────
        basename = geotiff_path.name
        matches = [
            img
            for img in self.coco.dataset["images"]
            if Path(img["file_name"]).name == basename
        ]
        if not matches:
            # raise ValueError(f"No image entry for {basename} in {self.ann_file}")
            print(f"No image entry for {basename} in {self.ann_file}")
            return None
        self.img_id = matches[0]["id"]
        self.anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[self.img_id]))
        return basename

    def _build_filtered_labels_and_windows(
        self,
        initial_windows: List[windows.Window],
        window_size: int,
    ) -> Tuple[List[windows.Window], List[np.ndarray], List[Tuple[int, int]]]:
        NEG_KEEP_PROB = 0.0  # fraction of empty windows to keep

        kept_windows, kept_labels, kept_shapes = [], [], []
        for win in initial_windows:
            x0, y0 = win.col_off, win.row_off
            boxes = []
            for ann in self.anns:
                x1, y1, w1, h1 = ann["bbox"]
                x2, y2 = x1 + w1, y1 + h1
                xi1, yi1 = max(x1, x0), max(y1, y0)
                xi2, yi2 = min(x2, x0 + window_size), min(y2, y0 + window_size)
                if xi2 > xi1 and yi2 > yi1:
                    bx = xi1 - x0
                    by = yi1 - y0
                    bw = xi2 - xi1
                    bh = yi2 - yi1
                    cx = (bx + bw / 2) / window_size
                    cy = (by + bh / 2) / window_size
                    bw /= window_size
                    bh /= window_size
                    boxes.append([ann["category_id"] - 1, cx, cy, bw, bh])

            if len(boxes) == 0:
                boxes = np.zeros((0, 5), dtype=np.float32)
                if random.random() > NEG_KEEP_PROB:
                    continue

            kept_windows.append(win)
            kept_labels.append(np.array(boxes, dtype=np.float32))
            kept_shapes.append((window_size, window_size))
        return kept_windows, kept_labels, kept_shapes

    def __getitem__(self, idx):
        img = self._load_and_preprocess_tile(idx)
        labels_out = self._prepare_tile_labels(idx)
        shapes = (
            (self.win_size, self.win_size),
            ((1.0, 1.0), (0.0, 0.0)),
        )
        return img, labels_out, self.img_files[idx], shapes

    def _load_and_preprocess_tile(self, idx: int) -> Tensor:
        win = self._windows[idx]
        tif_path = self._window_paths[idx]
        with rasterio.open(tif_path) as src:
            arr = src.read(1, window=win).astype(np.float32)
            nodata = src.nodata if src.nodata is not None else -9999.0

        arr[arr == nodata] = np.nan
        lo, hi = np.nanpercentile(arr, (1, 99))
        arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1) * 255
        arr = np.nan_to_num(arr, nan=0.0)
        img = np.stack([arr] * 3, axis=2).astype(np.uint8)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def _prepare_tile_labels(self, idx: int) -> Tensor:
        labels = self.labels[idx]
        nL = labels.shape[0]
        labels_out = torch.zeros((nL, 6), dtype=torch.float32)
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
        return labels_out

    @staticmethod
    def collate_fn(batch):
        imgs, labels, paths, shapes = zip(*batch)
        for i, l in enumerate(labels):
            l[:, 0] = i
        return torch.stack(imgs, 0), torch.cat(labels, 0), paths, shapes
