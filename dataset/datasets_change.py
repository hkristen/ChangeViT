from typing import Any, Sequence, Callable
import torch
from torch import Tensor
from torchgeo.datasets.utils import BoundingBox
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import percentile_normalization
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class BinaryChangeDetectionDataset(RasterDataset):
    """Dataset for detecting binary changes between pairs of GeoTIFF images.

    A dataset that loads pairs of GeoTIFF images and their corresponding change mask.
    The images are returned as separate 3-channel tensors.

    Attributes:
        is_image: Flag indicating if this is an image dataset
        image1_path: Path to the first (before) image
        image2_path: Path to the second (after) image
        mask_path: Path to the change mask
    """

    is_image: bool = False

    def __init__(
        self,
        image1_path: str,
        image2_path: str,
        mask_path: str,
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            image1_path: Path to the first (before) image
            image2_path: Path to the second (after) image
            mask_path: Path to the change mask
            bands: Names of bands to load
            transforms: Callable to transform the data samples
            cache: Whether to cache the index
        """
        self.image1_path = image1_path
        self.image2_path = image2_path
        self.mask_path = mask_path

        super().__init__(
            paths=[image1_path, image2_path, mask_path],
            bands=bands,
            transforms=transforms,
            cache=cache,
        )

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Get a data sample for a given bounding box.

        Args:
            query: Bounding box defining the region to load

        Returns:
            Dictionary containing:
                - image1: 'Before' image tensor [3, H, W]
                - image2: 'After' image tensor [3, H, W]
                - mask: Binary change mask tensor [H, W]

        Raises:
            IndexError: If query bounds are not found in the dataset
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # Load both images and mask
        image1_tensor = self._merge_files(
            [self.image1_path], query, self.band_indexes
        )  # [3, H, W]
        image2_tensor = self._merge_files(
            [self.image2_path], query, self.band_indexes
        )  # [3, H, W]
        mask_tensor = self._merge_files([self.mask_path], query)  # [1, H, W]

        # Normalize images to [0, 1]
        image1_tensor = image1_tensor.float() / 255.0
        image2_tensor = image2_tensor.float() / 255.0

        # Remove channel dimension from mask, keep original integer class labels
        mask_tensor = mask_tensor.squeeze(0)  # [H, W]

        sample = {
            "image1": image1_tensor,  # [3, H, W]
            "image2": image2_tensor,  # [3, H, W]
            "mask": mask_tensor,  # [H, W]
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Creates a figure with three subplots showing the before image, after image,
        and the after image with a change mask overlay.

        Args:
            sample: A sample returned by __getitem__ containing:
                   - image1: 'Before' image tensor [3, H, W]
                   - image2: 'After' image tensor [3, H, W]
                   - mask: Binary change mask tensor [H, W]
            show_titles: Flag indicating whether to show titles above each panel
            suptitle: Optional suptitle to use for figure

        Returns:
            A matplotlib Figure with the rendered sample
        """
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

        # Normalize and prepare images ensuring RGB order
        def prepare_image(img_tensor: torch.Tensor) -> np.ndarray:
            """Prepare image for visualization with correct RGB ordering.

            Args:
                img_tensor: Input image tensor in [C, H, W] format, values in [0,1]

            Returns:
                Normalized numpy array in [H, W, C] format with values in [0,1] for display
            """
            # img_tensor is in (C,H,W) format with R=0, G=1, B=2
            img = img_tensor.permute(1, 2, 0).numpy()  # to (H,W,C)
            # Values are already expected to be [0,1], clip to be safe for display
            # Percentile normalization might not be needed if input is already 0-1
            # If further normalization for visualization is desired, it can be added here.
            return np.clip(img, 0, 1)

        image1_vis = prepare_image(sample["image1"])
        image2_vis = prepare_image(sample["image2"])

        # Create mask overlay for the third subplot
        mask = sample["mask"].numpy()
        mask_rgba_overlay = np.zeros((*mask.shape, 4))
        mask_rgba_overlay[mask == 1] = [
            0,
            1,
            0,
            0.6,
        ]  # Green with 60% opacity where mask == 1

        # Plot before image
        axs[0].imshow(image1_vis)
        axs[0].axis("off")

        # Plot after image
        axs[1].imshow(image2_vis)
        axs[1].axis("off")

        # Plot after image with change overlay
        axs[2].imshow(image2_vis)  # Show image2
        axs[2].imshow(mask_rgba_overlay)  # Overlay the mask
        axs[2].axis("off")

        if show_titles:
            axs[0].set_title("Before (t1)")
            axs[1].set_title("After (t2)")
            axs[2].set_title("After (t2) with Changes")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class MultiClassChangeDetectionDataset(BinaryChangeDetectionDataset):

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
        num_classes: int = 9,  # Default, can be overridden by datamodule if needed
    ) -> Figure:
        """Plot a sample from the MultiClass dataset.

        Creates a figure with three subplots: before image, after image, and the change mask.
        The change mask is colored based on class.

        Args:
            sample: A sample returned by __getitem__ containing:
                   - image1: 'Before' image tensor [3, H, W]
                   - image2: 'After' image tensor [3, H, W]
                   - mask: Multi-class change mask tensor [H, W]
            show_titles: Flag indicating whether to show titles above each panel
            suptitle: Optional suptitle to use for figure
            num_classes: Number of classes for colormap (relevant if not passed from datamodule).

        Returns:
            A matplotlib Figure with the rendered sample
        """
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

        # Normalize and prepare images ensuring RGB order
        def prepare_image(img_tensor: torch.Tensor) -> np.ndarray:
            """Prepare image for visualization with correct RGB ordering.

            Args:
                img_tensor: Input image tensor in [C, H, W] format, values in [0,1]

            Returns:
                Normalized numpy array in [H, W, C] format with values in [0,1] for display
            """
            img = img_tensor.permute(1, 2, 0).numpy()  # to (H,W,C)
            return np.clip(img, 0, 1)

        image1_vis = prepare_image(sample["image1"])
        image2_vis = prepare_image(sample["image2"])

        mask_np = sample["mask"].numpy()

        # Determine actual number of unique classes present in the mask for colormap
        # or use the provided num_classes if it's more reliable (e.g. from datamodule config)
        # For visualization, it's often better to use the dataset's configured num_classes
        # to ensure consistent coloring across samples, even if a particular sample
        # doesn't contain all classes.
        # unique_classes = np.unique(mask_np)
        # current_num_classes = len(unique_classes) if len(unique_classes) > 1 else 2 # Ensure at least 2 for cmap

        cmap_num_classes = num_classes
        if (
            hasattr(self, "num_classes") and self.num_classes is not None
        ):  # If dataset has it
            cmap_num_classes = self.num_classes

        cmap = plt.cm.get_cmap("viridis", cmap_num_classes)
        vmin = 0
        vmax = cmap_num_classes - 1

        # Plot before image
        axs[0].imshow(image1_vis)
        axs[0].axis("off")

        # Plot after image
        axs[1].imshow(image2_vis)
        axs[1].axis("off")

        # Plot multi-class mask directly
        # Mask values should correspond to class indices
        im = axs[2].imshow(mask_np, cmap=cmap, vmin=vmin, vmax=vmax)
        # fig.colorbar(im, ax=axs[2], ticks=np.arange(vmin, vmax + 1)) # Optional: add a colorbar

        if show_titles:
            axs[0].set_title("Before (t1)")
            axs[1].set_title("After (t2)")
            axs[2].set_title("Change Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()  # Adjust layout to prevent overlap
        return fig
