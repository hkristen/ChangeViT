import sys

from model.trainer import Trainer
from pathlib import Path

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler

import dataset.dataset as myDataLoader
import dataset.Transforms as myTransforms
from model.metric_tool import ConfuseMatrixMeter
from model.utils import BCEDiceLoss

from PIL import Image

import os, time
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import rasterio


@torch.no_grad()
def val(args, val_loader, model):
    model.eval()

    salEvalVal = ConfuseMatrixMeter(n_class=2)

    epoch_loss = []

    total_batches = len(val_loader)
    print(len(val_loader))
    for iter, batched_inputs in enumerate(val_loader):

        img, target = batched_inputs
        img_name = val_loader.sampler.data_source.file_list[iter]
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        start_time = time.time()

        if args.onGPU == True:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the mdoel
        output = model(pre_img_var, post_img_var)
        loss = BCEDiceLoss(output, target_var)

        pred = torch.where(
            output > 0.5, torch.ones_like(output), torch.zeros_like(output)
        ).long()

        # torch.cuda.synchronize()
        time_taken = time.time() - start_time

        epoch_loss.append(loss.data.item())

        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)

        f1 = salEvalVal.update_cm(pred.cpu().numpy(), target_var.cpu().numpy())

        if iter % 5 == 0:
            print(
                "\r[%d/%d] F1: %3f loss: %.3f time: %.3f"
                % (iter, total_batches, f1, loss.data.item(), time_taken),
                end="",
            )

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_val, scores


def ValidateSegmentation(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.backends.cudnn.benchmark = True

    model = Trainer(args.model_type).float()

    args.savedir = (
        args.savedir
        + "_"
        + args.file_root
        + "_iter_"
        + str(args.max_steps)
        + "_lr_"
        + str(args.lr)
        + "/"
    )

    if args.file_root == "LEVIR":
        args.file_root = "./levir_cd_256"
    elif args.file_root == "WHU":
        args.file_root = "./whu_cd_256"
    elif args.file_root == "CLCD":
        args.file_root = "./clcd_256"
    elif args.file_root == "OSCD":
        args.file_root = "oscd_256"
    else:
        raise TypeError("%s has not defined" % args.file_root)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if args.onGPU:
        model = model.cuda()

    mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # compose the data with transforms
    valDataset = myTransforms.Compose(
        [
            myTransforms.Normalize(mean=mean, std=std),
            myTransforms.Scale(args.inWidth, args.inHeight),
            myTransforms.ToTensor(),
        ]
    )

    test_data = myDataLoader.Dataset(
        file_root=args.file_root, mode="test", transform=valDataset
    )
    testLoader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    if args.onGPU:
        cudnn.benchmark = True

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, "a")
    else:
        logger = open(logFileLoc, "w")
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s\t%s"
            % ("Epoch", "Kappa", "IoU", "F1", "R", "P", "OA")
        )
    logger.flush()

    # load the model
    model_file_name = args.savedir + "best_model.pth"
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    loss_test, score_test = val(args, testLoader, model)
    print(
        "\nTest :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f OA (te) = %.4f"
        % (
            score_test["Kappa"],
            score_test["IoU"],
            score_test["F1"],
            score_test["recall"],
            score_test["precision"],
            score_test["OA"],
        )
    )
    logger.write(
        "\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f"
        % (
            "Test",
            score_test["Kappa"],
            score_test["IoU"],
            score_test["F1"],
            score_test["recall"],
            score_test["precision"],
            score_test["OA"],
        )
    )
    logger.flush()
    logger.close()


def predict_large_image_changevit(
    model,
    image1_path,
    image2_path,
    output_path,
    num_classes=2,
    tile_size=512,  # Should be a multiple of 16
    overlap=0,
    save_probability_map=True,
):
    """
    Predict on large images using a ChangeViT model by processing them in tiles.
    Note: tile_size must be a multiple of 16 (the model's patch size).
    """
    # Ensure tile_size is a multiple of 16
    if tile_size % 16 != 0:
        tile_size = ((tile_size + 15) // 16) * 16
        print(f"Adjusted tile_size to {tile_size} to be a multiple of 16")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    output_path = Path(output_path)

    # Open the images
    with rasterio.open(image1_path) as src1, rasterio.open(image2_path) as src2:
        # Check that the images have the same dimensions and CRS
        if src1.shape != src2.shape or src1.crs != src2.crs:
            print(f"Shape mismatch: {src1.shape} vs {src2.shape}")
            print(f"CRS mismatch: {src1.crs} vs {src2.crs}")
            raise ValueError("Input images must have the same dimensions and CRS")

        # Get metadata for prediction output (class indices)
        prediction_meta = src1.meta.copy()
        prediction_meta.update(
            {
                "count": 1,
                "dtype": "uint8",  # Assuming number of classes < 256
                "driver": "COG",
                "compress": "LZW",
                "predictor": 1,  # Predictor 1 for uint8 data
            }
        )

        # Get metadata for probability map output (confidence of predicted class)
        prob_meta = src1.meta.copy()
        prob_meta.update(
            {
                "count": 1,
                "dtype": "float32",  # Use float32 for probabilities
                "driver": "COG",
                "compress": "ZSTD",
                "predictor": 3,  # Predictor 3 for floating point
            }
        )

        # Create output arrays
        height, width = src1.shape
        prediction = np.zeros((height, width), dtype=np.uint8)
        probability_map = np.full((height, width), np.nan, dtype=np.float32)

        # Calculate effective tile size with overlap
        if overlap >= tile_size // 2:
            raise ValueError(
                f"Overlap ({overlap}) must be less than half the tile size ({tile_size // 2})"
            )
        effective_size = tile_size - overlap
        stride = tile_size - overlap
        n_tiles_h = int(np.ceil(height / stride))
        n_tiles_w = int(np.ceil(width / stride))

        print(
            f"Image size: {height}x{width}, Tile size: {tile_size}x{tile_size}, Overlap: {overlap}, Stride: {stride}"
        )
        print(
            f"Processing {n_tiles_h} x {n_tiles_w} = {n_tiles_h * n_tiles_w} tiles..."
        )

        tile_count = 0
        # Process each tile
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                tile_count += 1
                # Calculate tile coordinates
                y_start = i * stride
                x_start = j * stride
                read_y_end = min(y_start + tile_size, height)
                read_x_end = min(x_start + tile_size, width)
                read_y_start = max(0, read_y_end - tile_size)
                read_x_start = max(0, read_x_end - tile_size)
                window_h = read_y_end - read_y_start
                window_w = read_x_end - read_x_start

                # Read tile data
                img1_tile = src1.read(
                    window=((read_y_start, read_y_end), (read_x_start, read_x_end))
                )
                img2_tile = src2.read(
                    window=((read_y_start, read_y_end), (read_x_start, read_x_end))
                )

                # Convert to PyTorch tensors and normalize
                norm_factor = 255.0 if src1.profile["dtype"] == "uint8" else 65535.0
                img1_tensor = torch.from_numpy(img1_tile).float() / norm_factor
                img2_tensor = torch.from_numpy(img2_tile).float() / norm_factor

                # Move to device
                img1_tensor = img1_tensor.to(device)
                img2_tensor = img2_tensor.to(device)

                # Ensure batch dimension
                if img1_tensor.ndim == 3:
                    img1_tensor = img1_tensor.unsqueeze(0)
                if img2_tensor.ndim == 3:
                    img2_tensor = img2_tensor.unsqueeze(0)

                # After dividing by 255, concatenate the two images along channel dim
                img_cat = torch.cat([img1_tensor, img2_tensor], dim=1)  # (1, 6, H, W)
                # Normalize with mean=0.5, std=0.5 for all 6 channels
                mean = torch.tensor([0.5] * 6, device=device).view(1, 6, 1, 1)
                std = torch.tensor([0.5] * 6, device=device).view(1, 6, 1, 1)
                img_cat = (img_cat - mean) / std
                # Then split back if your model expects two images, or pass as is if it expects concatenated input
                pre_img = img_cat[:, :3, :, :]
                post_img = img_cat[:, 3:, :, :]

                # Pad if the tile is smaller than tile_size
                pad_h = tile_size - pre_img.shape[2]
                pad_w = tile_size - pre_img.shape[3]
                if pad_h > 0 or pad_w > 0:
                    padding = (0, pad_w, 0, pad_h)
                    pre_img = torch.nn.functional.pad(
                        pre_img, padding, mode="constant", value=0
                    )
                    post_img = torch.nn.functional.pad(
                        post_img, padding, mode="constant", value=0
                    )

                # Make prediction
                with torch.no_grad():
                    output = model(pre_img, post_img)

                    if num_classes == 2:
                        # Binary case: output is already logits, apply sigmoid
                        probabilities = torch.sigmoid(output)
                        pred = (probabilities > 0.5).long()
                        max_prob = probabilities
                    else:
                        # Multiclass case: output is logits, apply softmax
                        probabilities = torch.softmax(output, dim=1)
                        max_prob, pred = torch.max(probabilities, dim=1)

                    # Remove batch and channel dimensions
                    pred = pred.squeeze(0).squeeze(0).to(torch.uint8)
                    max_prob = max_prob.squeeze(0).squeeze(0)

                # Crop the prediction and probability map if padding was added
                if pad_h > 0 or pad_w > 0:
                    pred = pred[:window_h, :window_w]
                    max_prob = max_prob[:window_h, :window_w]

                # Handle overlap
                write_y_start = read_y_start
                write_x_start = read_x_start
                write_y_end = read_y_end
                write_x_end = read_x_end

                tile_read_y_start = 0
                tile_read_x_start = 0
                tile_read_y_end = window_h
                tile_read_x_end = window_w

                if overlap > 0:
                    if i > 0:
                        write_y_start += overlap // 2
                        tile_read_y_start += overlap // 2
                    if j > 0:
                        write_x_start += overlap // 2
                        tile_read_x_start += overlap // 2
                    if i < n_tiles_h - 1:
                        write_y_end -= (overlap + 1) // 2
                        tile_read_y_end -= (overlap + 1) // 2
                    if j < n_tiles_w - 1:
                        write_x_end -= (overlap + 1) // 2
                        tile_read_x_end -= (overlap + 1) // 2

                # Ensure indices are valid
                write_y_start = max(0, write_y_start)
                write_x_start = max(0, write_x_start)
                write_y_end = min(height, write_y_end)
                write_x_end = min(width, write_x_end)

                tile_read_y_start = max(0, tile_read_y_start)
                tile_read_x_start = max(0, tile_read_x_start)
                tile_read_y_end = min(window_h, tile_read_y_end)
                tile_read_x_end = min(window_w, tile_read_x_end)

                # Extract the relevant part of the tile prediction
                pred_tile_section = pred[
                    tile_read_y_start:tile_read_y_end, tile_read_x_start:tile_read_x_end
                ]
                prob_tile_section = max_prob[
                    tile_read_y_start:tile_read_y_end, tile_read_x_start:tile_read_x_end
                ]

                # Move data to CPU just before assignment
                pred_to_write = pred_tile_section.cpu().numpy()
                prob_to_write = prob_tile_section.cpu().numpy()

                # Update the prediction and probability map
                target_shape = (
                    write_y_end - write_y_start,
                    write_x_end - write_x_start,
                )
                if pred_to_write.shape == target_shape:
                    prediction[write_y_start:write_y_end, write_x_start:write_x_end] = (
                        pred_to_write
                    )
                    if save_probability_map:
                        probability_map[
                            write_y_start:write_y_end, write_x_start:write_x_end
                        ] = prob_to_write
                else:
                    print(
                        f"Warning: Shape mismatch during write. Target: {target_shape}, Source: {pred_to_write.shape}. Skipping write for this section of tile {i},{j}."
                    )
                    print(
                        f"Write coords: y={write_y_start}:{write_y_end}, x={write_x_start}:{write_x_end}"
                    )
                    print(
                        f"Tile read coords: y={tile_read_y_start}:{tile_read_y_end}, x={tile_read_x_start}:{tile_read_x_end}"
                    )

                if tile_count % 10 == 0:
                    print(f"Processed tile {tile_count}/{n_tiles_h * n_tiles_w}...")

        # Write the prediction (class indices) to disk
        print(f"Writing prediction to {output_path}...")
        with rasterio.open(output_path, "w", **prediction_meta) as dst:
            dst.write(prediction, 1)

        print(f"Prediction saved to {output_path}")

        # Write the probability map (confidence) to disk if requested
        if save_probability_map:
            prob_output_path = output_path.with_name(
                f"{output_path.stem}_probability{output_path.suffix}"
            )
            print(f"Writing probability map to {prob_output_path}...")
            with rasterio.open(prob_output_path, "w", **prob_meta) as dst:
                dst.write(probability_map, 1)

            print(f"Probability map saved to {prob_output_path}")

        return prediction, probability_map if save_probability_map else None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--file_root",
        default="LEVIR",
        help="Data directory | LEVIR | WHU | CLCD | OSCD ",
    )
    parser.add_argument("--inWidth", type=int, default=256, help="Width of RGB image")
    parser.add_argument("--inHeight", type=int, default=256, help="Height of RGB image")
    parser.add_argument(
        "--max_steps", type=int, default=80000, help="Max. number of iterations"
    )
    parser.add_argument(
        "--num_workers", type=int, default=3, help="No. of parallel threads"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--model_type", type=str, default="small", help="select vit model type"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument(
        "--savedir", default="./results", help="Directory to save the results"
    )
    parser.add_argument(
        "--logFile",
        default="testLog.txt",
        help="File that stores the training and validation logs",
    )
    parser.add_argument(
        "--onGPU",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="Run on CPU or GPU. If TRUE, then GPU.",
    )
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU id number")

    args = parser.parse_args()
    print("Called with args:")
    print(args)

    ValidateSegmentation(args)


def load_changevit_model(checkpoint_path, model_type="small", num_classes=2):
    """
    Load a trained ChangeViT model from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file (.pth.tar or .pth)
        model_type: 'tiny' or 'small' (default: 'small')
        num_classes: Number of classes (2 for binary, >2 for multiclass)

    Returns:
        Loaded model in eval mode
    """
    # Initialize the model
    model = Trainer(model_type=model_type)

    # Load the checkpoint
    checkpoint = torch.load(
        checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
    )

    # If the checkpoint is a .pth.tar file (full checkpoint)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    # If the checkpoint is a .pth file (just the model state)
    else:
        model.load_state_dict(checkpoint)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    return model
