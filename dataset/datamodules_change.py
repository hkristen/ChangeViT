import rasterio
from typing import Optional, Tuple
from torch import Tensor
import torch
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.samplers import RandomGeoSampler
from .single_samplers import RandomGeoSamplerIntersectingPolygons
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from .datasets_change import (
    BinaryChangeDetectionDataset,
    MultiClassChangeDetectionDataset,
)
import kornia.augmentation as K
import geopandas as gpd
from torchgeo.datasets.utils import BoundingBox
import matplotlib.pyplot as plt
import warnings


class BinaryChangeDetectionDataModule(NonGeoDataModule):
    """NonGeoDataModule for binary change detection between pairs of satellite images.

    This module handles loading and preprocessing pairs of satellite images and their corresponding
    change masks for training change detection models. It supports:
    - Loading train/val/test splits based on ROI (region of interest) files
    - Patch-based sampling from the input images
    - Data augmentation using Kornia transforms
    - Normalization based on per-band statistics for each image separately.
    """

    def __init__(
        self,
        image1_path: str,
        image2_path: str,
        mask_path: str,
        train_roi_path: str,
        val_roi_path: str,
        test_roi_path: str,
        label_poly_path: str,
        num_classes: int = 2,  # Remains for compatibility, primarily used by MultiClass version
        patch_size: tuple[int, int] = (256, 256),
        batch_size: int = 32,
        num_workers: int = 4,
        samples_per_epoch: int | None = None,
        dataset_class=BinaryChangeDetectionDataset,  # Allow subclass to specify dataset
    ) -> None:
        """Initialize the data module.

        Args:
            image1_path: Path to the first (before) image
            image2_path: Path to the second (after) image
            mask_path: Path to the change mask
            train_roi_path: Path to training region of interest geopackage
            val_roi_path: Path to validation region of interest geopackage
            test_roi_path: Path to test region of interest geopackage
            label_poly_path: Path to the geopackage/shapefile containing label polygons for sampling.
            num_classes: Number of classes (used by MultiClassChangeDetectionDataModule).
            patch_size: Size of image patches to sample (height, width)
            batch_size: Number of samples per batch
            num_workers: Number of parallel workers for data loading
            samples_per_epoch: Number of patches to sample per epoch (optional)
            dataset_class: The dataset class to use (e.g., BinaryChangeDetectionDataset).
        """
        super().__init__(dataset_class, batch_size=batch_size, num_workers=num_workers)

        self.image1_path = image1_path
        self.image2_path = image2_path
        self.mask_path = mask_path
        self.train_roi_path = train_roi_path
        self.val_roi_path = val_roi_path
        self.test_roi_path = test_roi_path
        self.label_poly_path = label_poly_path
        # self.num_classes is primarily for MultiClass. Binary is implicitly 2.
        # It's set here if MultiClassChangeDetectionDataModule calls super.
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.samples_per_epoch = samples_per_epoch

        def get_image_stats_uint8_normalized(image_path: str) -> Tuple[Tensor, Tensor]:
            """Calculate per-band mean and standard deviation statistics for a multi-band image and normalize it to [0,1] range.

            Args:
                image_path: Path to the raster input image file in uint8 format.

            Returns:
                Tuple containing:
                    - Tensor of normalized mean values for each band (3 channels expected)
                    - Tensor of normalized standard deviation values for each band (3 channels expected)

            The statistics are normalized by dividing by 255.0 to get values in [0,1] range.
            """
            with rasterio.open(image_path) as src:
                if src.count != 3:  # Assuming 3-channel images for ChangeViT
                    warnings.warn(
                        f"Image {image_path} has {src.count} channels, expected 3. Statistics might be misaligned if not RGB."
                    )
                band_means = []
                band_stds = []
                for band_idx in range(
                    1, src.count + 1
                ):  # Iterate through available bands
                    stats = src.statistics(band_idx, approx=False)
                    if src.profile["dtype"] == "uint8":
                        band_means.append(stats.mean / 255.0)
                        band_stds.append(stats.std / 255.0)
                    else:
                        # Your dataset divides by 255.0, so this implies uint8 input.
                        # If other dtypes are used, normalization in dataset and here needs adjustment.
                        raise ValueError(
                            f"Only uint8 images are supported by current normalization logic. Found {src.profile['dtype']}."
                        )

            return (torch.tensor(band_means), torch.tensor(band_stds))

        # Calculate and store means and stds for each image separately
        self.means1, self.stds1 = get_image_stats_uint8_normalized(self.image1_path)
        self.means2, self.stds2 = get_image_stats_uint8_normalized(self.image2_path)

        # Augmentations will now operate on 'image1', 'image2', 'mask'
        # Normalization is handled separately in on_after_batch_transfer
        self.train_aug = K.AugmentationSequential(
            # K.Normalize is removed here, will be applied per image in on_after_batch_transfer
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomBrightness(brightness=(0.5, 1.5), p=0.5),
            K.RandomRotation(degrees=(-15, 15), p=0.5),
            K.RandomResizedCrop(
                size=self.patch_size, scale=(0.8, 1.0), ratio=(1, 1), p=1.0
            ),
            # Ensure geometric augmentations are applied consistently
            # Use 'input' for generic image-like tensors that should receive the same augmentations.
            # Use 'mask' for the mask tensor.
            data_keys=["input", "input", "mask"],
        )

        self.val_aug = K.AugmentationSequential(
            # K.Normalize removed
            data_keys=["input", "input", "mask"],
            same_on_batch=True,
        )
        self.test_aug = K.AugmentationSequential(
            # K.Normalize removed
            data_keys=["input", "input", "mask"],
            same_on_batch=True,
        )

        # Initialize label_polygons attribute, will be populated in setup
        self.label_polygons = None

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply augmentations after batch is transferred to device.

        Args:
            batch: Dictionary containing image1, image2 and mask data
            dataloader_idx: Index of the dataloader

        Returns:
            Batch with augmentations applied
        """

        image1 = batch["image1"]  # [B, 3, H, W]
        image2 = batch["image2"]  # [B, 3, H, W]
        mask = batch["mask"].float().unsqueeze(1)  # Add channel dim [B, 1, H, W]

        # Apply normalization per image
        # Ensure means/stds are on the same device as images
        device = image1.device
        means1_dev = self.means1.to(device)
        stds1_dev = self.stds1.to(device)
        means2_dev = self.means2.to(device)
        stds2_dev = self.stds2.to(device)

        # Create Normalize instances on the fly or pre-initialize them
        normalize1 = K.Normalize(mean=means1_dev, std=stds1_dev)
        normalize2 = K.Normalize(mean=means2_dev, std=stds2_dev)

        image1_norm = normalize1(image1)
        image2_norm = normalize2(image2)

        if self.trainer.training:
            # Apply training augmentations to both images and masks
            # Pass as a list: [image1, image2, mask]
            transformed = self.train_aug(image1_norm, image2_norm, mask)
            batch["image1"] = transformed[0]
            batch["image2"] = transformed[1]
            batch["mask"] = transformed[2].squeeze(1).long()
        else:
            # For validation/test, only normalization is applied here.
            # If val_aug/test_aug had other transforms, they would be applied similarly.
            # transformed = self.val_aug(image1_norm, image2_norm, mask) # if val_aug is not Identity
            # batch["image1"] = transformed[0]
            # batch["image2"] = transformed[1]
            # batch["mask"] = transformed[2].squeeze(1).long()
            # For now, just use the normalized images if val_aug is effectively Identity
            batch["image1"] = image1_norm
            batch["image2"] = image2_norm
            batch["mask"] = mask.squeeze(1).long()

        return batch

    def mask_to_long(self, sample: dict) -> dict:
        """Convert mask to PyTorch long tensor for loss functions.

        Args:
            sample: Dictionary containing image and mask data

        Returns:
            Sample with mask converted to long tensor
        """
        return {**sample, "mask": sample["mask"].long()}

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation and testing.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """

        # self.dataset_class is set by __init__
        dataset_class = self.dataset_class

        # The transforms passed to the dataset are minimal, as most are done in on_after_batch_transfer
        # mask_to_long might still be useful if masks aren't long type from dataset.
        # Your dataset already returns mask as [H,W] integer tensor.
        # Let's assume dataset returns mask appropriately for loss function after .long()
        dataset_transforms = Compose([self.mask_to_long])

        print(f"Setting up datasets using {dataset_class.__name__}...")
        if stage == "fit" or stage is None:
            self.train_dataset = dataset_class(
                image1_path=self.image1_path,
                image2_path=self.image2_path,
                mask_path=self.mask_path,
                transforms=dataset_transforms,  # Pass minimal dataset specific transforms
            )

            self.val_dataset = dataset_class(
                image1_path=self.image1_path,
                image2_path=self.image2_path,
                mask_path=self.mask_path,
                transforms=dataset_transforms,
            )

        # Test dataset setup (conditional on stage or always if needed)
        # Assuming test_dataset is always needed if setup is called broadly.
        # Or, could be conditional: if stage == 'test' or stage is None:
        self.test_dataset = dataset_class(
            image1_path=self.image1_path,
            image2_path=self.image2_path,
            mask_path=self.mask_path,
            transforms=dataset_transforms,
        )

        # Load ROIs from geopackage files and convert to BoundingBox objects

        def bounds_to_bbox(bounds):
            return BoundingBox(
                minx=bounds[0],
                maxx=bounds[2],
                miny=bounds[1],
                maxy=bounds[3],
                mint=0,
                maxt=1,
            )

        rois = {
            "train": self.train_roi_path,
            "val": self.val_roi_path,
            "test": self.test_roi_path,
        }

        for split, path in rois.items():
            bounds = gpd.read_file(path).total_bounds
            setattr(self, f"{split}_roi", bounds_to_bbox(bounds))

        # Load label polygons for sampling
        try:
            label_gdf = gpd.read_file(self.label_poly_path)

            # Use the CRS from one of the initialized datasets
            dataset_crs = self.train_dataset.crs
            if label_gdf.crs != dataset_crs:
                print(
                    f"Reprojecting label polygons from {label_gdf.crs} to {dataset_crs}..."
                )
                label_gdf = label_gdf.to_crs(dataset_crs)

            self.label_polygons = [
                geom
                for geom in label_gdf.geometry.tolist()
                if geom.is_valid and not geom.is_empty
            ]
            print(f"Loaded {len(self.label_polygons)} valid label polygons.")

            if not self.label_polygons:
                warnings.warn(
                    "No valid label polygons were loaded. Sampler might not yield any samples."
                )

        except Exception as e:
            print(f"Error loading or processing label polygons: {e}")
            self.label_polygons = []  # Set to empty list to avoid downstream errors
            warnings.warn(
                "Failed to load label polygons. Proceeding without polygon intersection sampling."
            )

        print("Datasets created successfully")

    def train_dataloader(self):
        """Create the training data loader.

        Returns:
            DataLoader for training data
        """
        if self.label_polygons is None:
            raise RuntimeError(
                "Label polygons have not been loaded. Ensure setup() was called."
            )

        if not self.label_polygons:
            warnings.warn(
                "Training sampler created with no label polygons. May yield no samples."
            )
            # Fall back to regular RandomGeoSampler if no polygons are available
            sampler = RandomGeoSampler(
                dataset=self.train_dataset,
                size=self.patch_size,
                roi=self.train_roi,
                length=self.samples_per_epoch,
            )
        else:
            sampler = RandomGeoSamplerIntersectingPolygons(
                dataset=self.train_dataset,
                label_polygons=self.label_polygons,
                size=self.patch_size,
                roi=self.train_roi,
                length=self.samples_per_epoch,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=False,
        )

    def val_dataloader(self):
        """Create the validation data loader.

        Returns:
            DataLoader for validation data
        """
        val_length = (
            self.samples_per_epoch // 5 if self.samples_per_epoch is not None else 1000
        )
        if val_length == 0 and self.samples_per_epoch is not None:
            val_length = 1  # Ensure at least 1 sample if samples_per_epoch is small

        if self.label_polygons is None:
            raise RuntimeError(
                "Label polygons have not been loaded. Ensure setup() was called."
            )

        if not self.label_polygons:
            warnings.warn(
                "Validation sampler created with no label polygons. May yield no samples."
            )
            # Fall back to regular RandomGeoSampler if no polygons are available
            sampler = RandomGeoSampler(
                dataset=self.val_dataset,
                size=self.patch_size,
                roi=self.val_roi,
                length=val_length,
            )
        else:
            sampler = RandomGeoSamplerIntersectingPolygons(
                dataset=self.val_dataset,
                label_polygons=self.label_polygons,
                size=self.patch_size,
                roi=self.val_roi,
                length=val_length,
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self):
        """Create the test data loader.

        Returns:
            DataLoader for test data
        """
        print("Creating test dataloader")

        test_length = (
            self.samples_per_epoch // 5 if self.samples_per_epoch is not None else 1000
        )
        if test_length == 0 and self.samples_per_epoch is not None:
            test_length = 1  # Ensure at least 1 sample if samples_per_epoch is small

        if self.label_polygons is None:
            raise RuntimeError(
                "Label polygons have not been loaded. Ensure setup() was called."
            )

        if not self.label_polygons:
            warnings.warn(
                "Test sampler created with no label polygons. May yield no samples."
            )
            # Fall back to regular RandomGeoSampler if no polygons are available
            sampler = RandomGeoSampler(
                dataset=self.test_dataset,
                size=self.patch_size,
                roi=self.test_roi,
                length=test_length,
            )
        else:
            sampler = RandomGeoSamplerIntersectingPolygons(
                dataset=self.test_dataset,
                label_polygons=self.label_polygons,
                size=self.patch_size,
                roi=self.test_roi,
                length=test_length,
            )

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=False,
        )

    def plot_sample(self, sample, title=None):
        """Plot a single sample."""
        import matplotlib.pyplot as plt
        from torchgeo.datasets.utils import percentile_normalization
        import numpy as np

        # Create figure with three subplots, ensuring equal sizes
        fig = plt.figure(figsize=(15, 5))
        gs = plt.GridSpec(1, 3, figure=fig)
        gs.update(wspace=0.05)  # Reduce spacing between subplots

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        # Prepare images for visualization
        def prepare_image(img_tensor):
            # Convert to dense tensor if sparse
            if img_tensor.is_sparse:
                img_tensor = img_tensor.to_dense()
            img = img_tensor.permute(1, 2, 0).numpy()
            img = percentile_normalization(img)
            return np.clip(img, 0, 1)

        # Plot images
        image1 = prepare_image(sample["image1"])
        image2 = prepare_image(sample["image2"])

        # Create mask visualization
        mask = sample["mask"].numpy()
        mask_rgb = np.zeros((*mask.shape, 3))
        mask_rgb[mask == 1] = [1, 0, 0]  # Red for changes

        # Plot with equal aspect ratios
        ax1.imshow(image1, aspect="equal")
        ax1.axis("off")
        ax1.set_title("Before (t1)")

        ax2.imshow(image2, aspect="equal")
        ax2.axis("off")
        ax2.set_title("After (t2)")

        ax3.imshow(mask_rgb, aspect="equal")
        ax3.axis("off")
        ax3.set_title("Changes")

        if title:
            plt.suptitle(title)

        return fig

    def visualize_samples(self, num_samples=4, split="train"):
        """Visualize samples from the datamodule.

        Args:
            datamodule: The ChangeDetectionDataModule instance (self)
            num_samples: Number of samples to visualize
            split: Either 'train' or 'val'
        """

        # Make sure the datamodule is set up
        if split == "train" and not hasattr(self, "train_dataset"):
            self.setup("fit")
        elif split == "val" and not hasattr(self, "val_dataset"):
            self.setup("validate")  # or 'fit' if val is part of fit stage
        elif not hasattr(self, "train_dataset") and not hasattr(
            self, "val_dataset"
        ):  # Fallback
            self.setup()

        # Get the appropriate dataloader
        if split == "train":
            dataloader = self.train_dataloader()
            # roi = self.train_roi # roi not used in this plotting logic
            title_prefix = "Training"
        elif split == "val":  # Added 'val' condition
            dataloader = self.val_dataloader()
            # roi = self.val_roi
            title_prefix = "Validation"
        else:
            raise ValueError(
                f"Unsupported split for visualization: {split}. Choose 'train' or 'val'."
            )

        # Get a batch of data
        # Note: on_after_batch_transfer would have already run if this is from a PL trainer hook.
        # If calling manually, batch is direct from DataLoader (before on_after_batch_transfer).
        # For visualization, we typically want to see the augmented data.
        # However, this method as written pulls a raw batch and then calls self.plot_sample.
        # self.plot_sample expects normalized image data (e.g. 0-1).
        # The dataset now provides image1, image2 as 0-1 float tensors.

        batch = next(iter(dataloader))  # Raw batch from dataloader

        # Create a figure. Subplots are handled by self.plot_sample or dataset.plot
        # fig = plt.figure(figsize=(15, 7 * num_samples)) # Figure size might need adjustment

        # Plot each sample
        # The batch keys should be 'image1', 'image2', 'mask' from your updated dataset
        for i in range(
            min(num_samples, batch["image1"].shape[0])
        ):  # Use shape of image1 for count
            sample_to_plot = {
                "image1": batch["image1"][i],  # Already [3, H, W], float 0-1
                "image2": batch["image2"][i],  # Already [3, H, W], float 0-1
                "mask": batch["mask"][i],  # Already [H, W], long or int
            }

            # Determine if we should use dataset.plot or self.plot_sample
            # If dataset has a plot method, it's usually preferred.
            # Your dataset classes (BinaryChangeDetectionDataset) now have a .plot method.

            # Get the dataset instance
            current_dataset = dataloader.dataset
            if hasattr(current_dataset, "plot") and callable(
                getattr(current_dataset, "plot")
            ):
                # The dataset's plot method should handle figure creation and display.
                # It might take num_classes for MultiClass.
                if isinstance(current_dataset, MultiClassChangeDetectionDataset) or (
                    hasattr(self, "dataset_class")
                    and self.dataset_class == MultiClassChangeDetectionDataset
                ):
                    fig_sample = current_dataset.plot(
                        sample_to_plot,
                        suptitle=f"{title_prefix} Sample {i+1}",
                        num_classes=self.num_classes,
                    )
                else:  # BinaryChangeDetectionDataset
                    fig_sample = current_dataset.plot(
                        sample_to_plot, suptitle=f"{title_prefix} Sample {i+1}"
                    )
                # fig_sample.show() # Or handle display outside if preferred
            else:
                # Fallback to datamodule's plot_sample if dataset.plot is not available
                # This datamodule's plot_sample is designed for binary.
                self.plot_sample(sample_to_plot, title=f"{title_prefix} Sample {i+1}")

        plt.show()  # Show all generated figures if not shown by plot methods themselves
        # plt.tight_layout() # May or may not be needed depending on how dataset.plot behaves


class MultiClassChangeDetectionDataModule(BinaryChangeDetectionDataModule):
    """NonGeoDataModule for multi-class change detection between pairs of satellite images.
    Uses MultiClassChangeDetectionDataset.
    """

    def __init__(
        self,
        image1_path: str,
        image2_path: str,
        mask_path: str,
        train_roi_path: str,
        val_roi_path: str,
        test_roi_path: str,
        label_poly_path: str,
        num_classes: int,  # Specific to multi-class
        patch_size: tuple[int, int] = (256, 256),
        batch_size: int = 32,
        num_workers: int = 4,
        samples_per_epoch: int | None = None,
    ) -> None:
        """Initialize the data module.
        Args:
            num_classes: Total number of classes in the dataset (including background).
        """
        # Explicitly pass MultiClassChangeDetectionDataset to the superclass constructor
        super().__init__(
            image1_path=image1_path,
            image2_path=image2_path,
            mask_path=mask_path,
            train_roi_path=train_roi_path,
            val_roi_path=val_roi_path,
            test_roi_path=test_roi_path,
            label_poly_path=label_poly_path,
            num_classes=num_classes,  # Pass num_classes to super
            patch_size=patch_size,
            batch_size=batch_size,
            num_workers=num_workers,
            samples_per_epoch=samples_per_epoch,
            dataset_class=MultiClassChangeDetectionDataset,  # Specify the correct dataset
        )
        # num_classes is set by super().__init__ now, but can be asserted or re-set if needed
        # self.num_classes = num_classes # Already set by super's __init__ if it takes num_classes

    # plot_sample is inherited from BinaryChangeDetectionDataModule.
    # If a different plotting is needed specifically for MultiClass at datamodule level,
    # it can be overridden here. The current self.plot_sample is for binary.
    # However, visualize_samples now calls dataset.plot(), which is correctly implemented
    # for MultiClassChangeDetectionDataset. So this plot_sample might not be directly used by visualize_samples.

    # The MultiClassChangeDetectionDataModule's own plot_sample (if it were to override)
    # would be similar to the one you had, but it's often cleaner to use dataset.plot().
    # The existing plot_sample in BinaryChangeDetectionDataModule is for binary masks.
    # If MultiClassChangeDetectionDataModule needs its own distinct plot_sample for some reason,
    # it should be implemented here. Otherwise, visualization relies on the dataset's plot method.
    # For now, we rely on the dataset's plot via visualize_samples.
    # The inherited plot_sample from Binary... will plot a binary-style mask if called directly.

    # If a distinct plot_sample method is desired for MultiClassDataModule itself:
    def plot_sample(self, sample, title=None):
        """Plot a single multi-class sample. (Overrides Binary's version)"""
        import matplotlib.pyplot as plt

        # from torchgeo.datasets.utils import percentile_normalization # Not needed if images are 0-1
        import numpy as np

        fig = plt.figure(figsize=(15, 5))  # Adjusted for 3 subplots
        gs = plt.GridSpec(1, 3, figure=fig)
        gs.update(wspace=0.05)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        def prepare_image(img_tensor):  # Assumes img_tensor is already 0-1 float
            img = img_tensor.permute(1, 2, 0).numpy()
            return np.clip(img, 0, 1)  # Clip just in case

        image1 = prepare_image(sample["image1"])
        image2 = prepare_image(sample["image2"])

        mask = sample["mask"].numpy()
        # print(f"Plotting multi-class mask. Unique values: {np.unique(mask)}. Num classes for cmap: {self.num_classes}")

        cmap = plt.cm.get_cmap("viridis", self.num_classes)
        vmin = 0
        vmax = self.num_classes - 1

        ax1.imshow(
            image1, aspect="auto"
        )  # Changed from 'equal' to 'auto' for general use
        ax1.axis("off")
        ax1.set_title("Before (t1)")

        ax2.imshow(image2, aspect="auto")
        ax2.axis("off")
        ax2.set_title("After (t2)")

        im_mask = ax3.imshow(mask, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax3.axis("off")
        ax3.set_title(f"Changes ({self.num_classes} classes)")
        # Optional: Add colorbar
        # cbar = fig.colorbar(im_mask, ax=ax3, ticks=np.arange(vmin, vmax + 1), orientation='vertical', fraction=0.046, pad=0.04)
        # cbar.set_label('Class ID')

        if title:
            plt.suptitle(title)

        plt.tight_layout(rect=[0, 0, 1, 0.96] if title else None)  # Adjust for suptitle
        return fig

    # setup method is inherited and should work correctly due to dataset_class mechanism.
    # train_dataloader, val_dataloader, test_dataloader are inherited and should also work.
    # visualize_samples is inherited and now calls dataset.plot(), which handles multi-class.
