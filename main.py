import sys

from model.trainer import Trainer

sys.path.insert(0, ".")

import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler

import wandb
from dataset.datamodules_change import BinaryChangeDetectionDataModule
from model.metric_tool import ConfuseMatrixMeter
from model.utils import BCEDiceLoss, adjust_learning_rate, init_seed, weight_init


@torch.no_grad()
def val(args, val_loader, model):
    model.eval()

    salEvalVal = ConfuseMatrixMeter(n_class=args.num_classes)

    epoch_loss = []

    total_batches = len(val_loader)
    print(f"Validation loader length: {len(val_loader)}")
    for iter, batch in enumerate(val_loader):

        image1 = batch["image1"]
        image2 = batch["image2"]
        target = batch["mask"]

        start_time = time.time()

        if args.onGPU == True:
            image1 = image1.cuda()
            image2 = image2.cuda()
            target = target.cuda()

        output = model(image1, image2)

        # Create class weights tensor
        class_weights = torch.tensor([args.background_weight, args.change_weight])
        if args.onGPU:
            class_weights = class_weights.cuda()

        loss = BCEDiceLoss(
            output, target.float().unsqueeze(1), class_weights=class_weights
        )

        pred = torch.where(
            output > 0.5, torch.ones_like(output), torch.zeros_like(output)
        ).long()

        time_taken = time.time() - start_time

        epoch_loss.append(loss.item())

        if args.onGPU and torch.cuda.device_count() > 1:
            pass

        f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())
        if iter % 5 == 0 and total_batches > 0:
            print(
                f"\rValidation: [{iter}/{total_batches}] F1: {f1:.3f} loss: {loss.item():.3f} time: {time_taken:.3f}s",
                end="",
            )

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0
    scores = salEvalVal.get_scores()
    if total_batches > 0:
        print()

    return average_epoch_loss_val, scores


def train(
    args,
    train_loader,
    model,
    optimizer,
    epoch,
    max_batches_epoch,
    global_step_offset,
    total_max_steps,
    lr_factor=1.0,
):
    model.train()

    salEvalVal = ConfuseMatrixMeter(n_class=args.num_classes)
    epoch_loss_list = []

    # Create class weights tensor
    class_weights = torch.tensor([args.background_weight, args.change_weight])
    if args.onGPU:
        class_weights = class_weights.cuda()

    for iter_in_epoch, batch in enumerate(train_loader):
        current_global_step = global_step_offset + iter_in_epoch
        if current_global_step >= total_max_steps:
            break

        image1 = batch["image1"]
        image2 = batch["image2"]
        target = batch["mask"]

        start_time = time.time()

        lr = adjust_learning_rate(
            args,
            optimizer,
            epoch,
            current_global_step,
            total_max_steps,
            lr_factor=lr_factor,
        )

        if args.onGPU == True:
            image1 = image1.cuda()
            image2 = image2.cuda()
            target = target.cuda()

        output = model(image1, image2)
        loss = BCEDiceLoss(
            output, target.float().unsqueeze(1), class_weights=class_weights
        )

        pred = torch.where(
            output > 0.5, torch.ones_like(output), torch.zeros_like(output)
        ).long()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss_list.append(loss.item())
        time_taken = time.time() - start_time

        remaining_steps = total_max_steps - current_global_step
        res_time_hours = (remaining_steps * time_taken) / 3600 if time_taken > 0 else 0

        if args.onGPU and torch.cuda.device_count() > 1:
            pass

        with torch.no_grad():
            f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())

        if iter_in_epoch % args.print_interval == 0:
            print(
                f"\rEpoch: {epoch} Iter: [{current_global_step}/{total_max_steps}] F1: {f1:.3f} LR: {lr:.7f} Loss: {loss.item():.3f} ETA: {res_time_hours:.2f}h",
                end="",
            )

        if args.use_wandb and (current_global_step % args.wandb_log_interval == 0):
            wandb.log(
                {
                    "train/loss_step": loss.item(),
                    "train/lr_step": lr,
                    "train/f1_step": f1,
                    "global_step": current_global_step,
                    "epoch": epoch,
                }
            )

    average_epoch_loss_train = (
        sum(epoch_loss_list) / len(epoch_loss_list) if epoch_loss_list else 0
    )
    scores = salEvalVal.get_scores()
    print()

    return average_epoch_loss_train, scores, lr, current_global_step + 1


def trainValidateSegmentation(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.backends.cudnn.benchmark = True
    init_seed(args.seed)

    # Define class weights from arguments
    class_weights = torch.tensor([args.background_weight, args.change_weight])
    if args.onGPU:
        class_weights = class_weights.cuda()

    if args.use_wandb:
        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
            print("Attempting to log in to W&B with provided API key.")
            try:
                wandb.login(key=args.wandb_api_key)
            except Exception as e:
                print(
                    f"Failed to login to W&B with provided key: {e}. Ensure WANDB_API_KEY is set or use 'wandb login'."
                )

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name if args.wandb_run_name else None,
            config=vars(args),
            resume="allow",
            id=args.wandb_run_id if args.wandb_run_id else None,
        )
        if wandb_run:
            print(f"W&B Run Initialized: {wandb_run.name} (ID: {wandb_run.id})")
            wandb.save(os.path.abspath(__file__))
        else:
            print("W&B initialization failed. Continuing without W&B logging.")
            args.use_wandb = False

    run_name_suffix = f"{args.model_type}_steps_{args.max_steps}_lr_{args.lr}_bs_{args.batch_size}_seed_{args.seed}"
    if args.wandb_run_name:
        run_name = args.wandb_run_name
    elif args.use_wandb and wandb_run:
        run_name = wandb_run.name
    else:
        run_name = run_name_suffix

    args.savedir = os.path.join(args.experiment_dir, run_name)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
        print(f"Created save directory: {args.savedir}")

    model = Trainer(args.model_type).float()
    if args.onGPU:
        model = model.cuda()

    weight_init(model)

    if args.use_wandb and wandb_run:
        wandb.watch(model, log="all", log_freq=args.wandb_watch_interval)

    if args.num_classes > 2:
        from dataset.datamodules_change import MultiClassChangeDetectionDataModule

        datamodule_class = MultiClassChangeDetectionDataModule
        print(
            f"Using MultiClassChangeDetectionDataModule with {args.num_classes} classes."
        )
    else:
        datamodule_class = BinaryChangeDetectionDataModule
        args.num_classes = 2
        print("Using BinaryChangeDetectionDataModule.")

    datamodule = datamodule_class(
        image1_path=args.image1_path,
        image2_path=args.image2_path,
        mask_path=args.mask_path,
        train_roi_path=args.train_roi_path,
        val_roi_path=args.val_roi_path,
        test_roi_path=args.test_roi_path,
        label_poly_path=args.label_poly_path,
        num_classes=args.num_classes,
        patch_size=(args.inHeight, args.inWidth),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples_per_epoch=args.samples_per_epoch,
    )

    datamodule.setup("fit")
    trainLoader = datamodule.train_dataloader()
    try:
        valLoader = datamodule.val_dataloader()
        if len(valLoader) == 0 and args.val_roi_path:
            print(
                "Val Dataloader is empty despite val_roi_path being set. Check val_roi. Falling back to Test Dataloader."
            )
            datamodule.setup("test")
            valLoader = datamodule.test_dataloader()
        elif not args.val_roi_path:
            print("val_roi_path not provided. Using Test Dataloader for validation.")
            datamodule.setup("test")
            valLoader = datamodule.test_dataloader()

    except (NotImplementedError, RuntimeError):
        print(
            "Failed to get val_dataloader, falling back to Test Dataloader for validation."
        )
        datamodule.setup("test")
        valLoader = datamodule.test_dataloader()

    max_batches_epoch = len(trainLoader)
    if max_batches_epoch == 0:
        print(
            "ERROR: Training loader has 0 batches. Check data paths, ROIs, and sampler logic."
        )
        if args.use_wandb and wandb_run:
            wandb.finish(exit_code=1)
        return

    print(f"Train loader: {max_batches_epoch} batches per epoch.")
    print(f"Validation loader: {len(valLoader)} batches.")

    if args.onGPU:
        cudnn.benchmark = True

    args.max_epochs = (
        int(np.ceil(args.max_steps / max_batches_epoch)) if max_batches_epoch > 0 else 1
    )
    start_epoch = 0
    current_global_step = 0
    max_F1_val = 0

    if args.use_wandb and wandb_run and wandb_run.resumed:
        print(f"W&B Run '{wandb_run.name}' (ID: {wandb_run.id}) is resuming.")

    if args.resume is not None:
        resume_path = args.resume
        if not os.path.isabs(resume_path) and not os.path.exists(resume_path):
            resume_path = os.path.join(args.savedir, "checkpoint.pth.tar")

        if os.path.isfile(resume_path):
            print(f"=> loading checkpoint '{resume_path}'")
            checkpoint = torch.load(
                resume_path, map_location="cuda" if args.onGPU else "cpu"
            )
            start_epoch = checkpoint.get("epoch", start_epoch)
            current_global_step = checkpoint.get(
                "current_global_step", start_epoch * max_batches_epoch
            )
            model.load_state_dict(checkpoint["state_dict"])
            max_F1_val = checkpoint.get("F_val", max_F1_val)
            if "optimizer" in checkpoint and args.resume_optimizer:
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("Optimizer state loaded from checkpoint.")

            print(
                f"=> loaded checkpoint '{resume_path}' (epoch {start_epoch}, global_step {current_global_step})"
            )
        else:
            print(f"=> no checkpoint found at '{resume_path}'")

    logFileLoc = os.path.join(args.savedir, args.logFile)
    log_mode = (
        "a"
        if os.path.isfile(logFileLoc)
        and (args.resume or (args.use_wandb and wandb_run and wandb_run.resumed))
        else "w"
    )
    logger = open(logFileLoc, log_mode)
    if log_mode == "w":
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s\t%s"
            % (
                "Epoch",
                "Kappa (val)",
                "IoU (val)",
                "F1 (val)",
                "R (val)",
                "P (val)",
                "OA (val)",
            )
        )
    logger.flush()

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        betas=(0.9, 0.99),
        eps=1e-08,
        weight_decay=args.weight_decay,
    )
    if (
        args.resume
        and os.path.isfile(resume_path)
        and "optimizer" in checkpoint
        and args.resume_optimizer
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Add early stopping variables
    best_iou = 0
    no_improvement_count = 0
    last_validation_step = 0

    for epoch in range(start_epoch, args.max_epochs):
        if current_global_step >= args.max_steps:
            print(
                f"Reached max_steps {args.max_steps} at epoch {epoch}. Stopping training."
            )
            break

        lossTr, score_tr, lr, next_global_step_offset = train(
            args,
            trainLoader,
            model,
            optimizer,
            epoch,
            max_batches_epoch,
            current_global_step,
            args.max_steps,
        )

        if args.use_wandb and wandb_run:
            wandb.log(
                {
                    "train/epoch_loss": lossTr,
                    "train/epoch_F1": score_tr["F1"],
                    "train/epoch_Kappa": score_tr["Kappa"],
                    "train/epoch_IoU": score_tr["IoU"],
                    "train/epoch_OA": score_tr["OA"],
                    "train/epoch_precision": score_tr["precision"],
                    "train/epoch_recall": score_tr["recall"],
                    "epoch": epoch,
                    "global_step": current_global_step,
                }
            )
        current_global_step = next_global_step_offset

        torch.cuda.empty_cache()

        # Update validation logic to be step-based
        perform_validation = False
        if len(valLoader) > 0:
            if (
                epoch == start_epoch
                and args.resume is None
                and not args.validate_first_epoch
            ):
                print(
                    f"Skipping validation for initial epoch {epoch} as per validate_first_epoch=False."
                )
            elif (current_global_step - last_validation_step) >= args.val_interval:
                perform_validation = True
                last_validation_step = current_global_step

        if perform_validation:
            lossVal, score_val = val(args, valLoader, model)
            torch.cuda.empty_cache()

            # Early stopping check
            current_iou = score_val["IoU"]
            if current_iou > best_iou + args.early_stopping_min_delta:
                best_iou = current_iou
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= args.early_stopping_patience:
                print(
                    f"\nEarly stopping triggered after {current_global_step} steps. No improvement in IoU for {args.early_stopping_patience} validations."
                )
                break

            print(
                f"\nEpoch {epoch}: Train Loss={lossTr:.4f}, Val Loss={lossVal:.4f}, F1(tr)={score_tr['F1']:.4f}, F1(val)={score_val['F1']:.4f}"
            )
            logger.write(
                f"\n{epoch}\t\t{score_val['Kappa']:.4f}\t\t{score_val['IoU']:.4f}\t\t{score_val['F1']:.4f}\t\t{score_val['recall']:.4f}\t\t{score_val['precision']:.4f}\t\t{score_val['OA']:.4f}"
            )
            logger.flush()

            if args.use_wandb and wandb_run:
                wandb.log(
                    {
                        "val/loss": lossVal,
                        "val/F1": score_val["F1"],
                        "val/Kappa": score_val["Kappa"],
                        "val/IoU": score_val["IoU"],
                        "val/OA": score_val["OA"],
                        "val/precision": score_val["precision"],
                        "val/recall": score_val["recall"],
                        "epoch": epoch,
                        "global_step": current_global_step,
                    }
                )

                # Log class-based metrics
                if "class_metrics" in score_val:
                    for class_id, metrics in score_val["class_metrics"].items():
                        for metric_name, value in metrics.items():
                            wandb.log(
                                {
                                    f"val/{class_id}/{metric_name}": value,
                                    "epoch": epoch,
                                    "global_step": current_global_step,
                                }
                            )

            is_best = score_val["F1"] > max_F1_val
            if is_best:
                max_F1_val = score_val["F1"]
                best_model_file_name = os.path.join(args.savedir, "best_model.pth")
                torch.save(model.state_dict(), best_model_file_name)
                print(
                    f"Saved new best model with F1: {max_F1_val:.4f} to {best_model_file_name}"
                )
                if args.use_wandb and wandb_run:
                    wandb.save(best_model_file_name)
                    wandb.summary["best_F1_val"] = max_F1_val
                    wandb.summary["best_val_epoch"] = epoch

        else:
            lossVal = -1
            if (
                epoch == start_epoch
                and args.resume is None
                and not args.validate_first_epoch
            ):
                pass
            elif len(valLoader) == 0:
                if epoch == start_epoch:
                    print("(No validation loader configured)")
            else:
                if epoch % args.print_interval == 0:
                    print(
                        f"(Validation skipped this epoch, interval: {args.val_interval})"
                    )

            print(
                f"\nEpoch {epoch}: Train Loss={lossTr:.4f}, F1(tr)={score_tr['F1']:.4f}"
            )
            logger.write(f"\n{epoch}\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\tN/A")
            logger.flush()
            is_best = False

        if (epoch + 1) % args.save_interval == 0 or (
            current_global_step >= args.max_steps
        ):
            checkpoint_path = os.path.join(args.savedir, "checkpoint.pth.tar")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "current_global_step": current_global_step,
                    "arch": args.model_type,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lossTr": lossTr,
                    "lossVal": (
                        lossVal if perform_validation and len(valLoader) > 0 else -1
                    ),
                    "F_Tr": score_tr["F1"],
                    "F_val": (
                        score_val["F1"]
                        if perform_validation
                        and len(valLoader) > 0
                        and "F1" in score_val
                        else -1
                    ),
                    "max_F1_val": max_F1_val,
                    "lr": lr,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")
            if args.use_wandb and wandb_run and args.wandb_save_checkpoints:
                wandb.save(checkpoint_path)

    print("\nTraining finished. Evaluating on test set...")
    datamodule.setup("test")
    testLoader = datamodule.test_dataloader()
    final_test_metrics = {}
    if len(testLoader) > 0:
        best_model_path = os.path.join(args.savedir, "best_model.pth")
        if os.path.exists(best_model_path):
            print(f"Loading best model for final test: {best_model_path}")
            state_dict = torch.load(
                best_model_path, map_location="cuda" if args.onGPU else "cpu"
            )
            model.load_state_dict(state_dict)
        else:
            print("Best model not found, using last model for testing.")

        loss_test, score_test = val(args, testLoader, model)
        print(
            f"\nTest Results: Kappa={score_test['Kappa']:.4f}, IoU={score_test['IoU']:.4f}, F1={score_test['F1']:.4f}, Recall={score_test['recall']:.4f}, Precision={score_test['precision']:.4f}, OA={score_test['OA']:.4f}"
        )

        # Print class-based metrics
        if "class_metrics" in score_test:
            print("\nPer-class metrics:")
            for class_id, metrics in score_test["class_metrics"].items():
                print(f"\n{class_id}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")

        logger.write(
            f"\nTest\t\t{score_test['Kappa']:.4f}\t\t{score_test['IoU']:.4f}\t\t{score_test['F1']:.4f}\t\t{score_test['recall']:.4f}\t\t{score_test['precision']:.4f}\t\t{score_test['OA']:.4f}"
        )
        final_test_metrics = {
            "test/loss": loss_test,
            "test/F1": score_test["F1"],
            "test/Kappa": score_test["Kappa"],
            "test/IoU": score_test["IoU"],
            "test/OA": score_test["OA"],
            "test/precision": score_test["precision"],
            "test/recall": score_test["recall"],
        }

        # Add class-based metrics to final test metrics
        if "class_metrics" in score_test:
            for class_id, metrics in score_test["class_metrics"].items():
                for metric_name, value in metrics.items():
                    final_test_metrics[f"test/{class_id}/{metric_name}"] = value

    logger.flush()
    logger.close()
    print(f"Log file saved to: {logFileLoc}")

    if args.use_wandb and wandb_run:
        if final_test_metrics:
            wandb.log(final_test_metrics)
        wandb.finish()
    print("Training is done!")
    if args.auto_shutdown:
        print("Shutting down system...")
        os.system("shutdown now")
    else:
        print("Training completed. System will not shut down automatically.")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--image1_path",
        type=str,
        default="/home/hkristen/habitalp2/data/processed/orthos_rgb_2003_2013/flug_2003_rgb.tif",
        help="Path to the first (before) image",
    )
    parser.add_argument(
        "--image2_path",
        type=str,
        default="/home/hkristen/habitalp2/data/processed/orthos_rgb_2003_2013/flug_2013_rgb.tif",
        help="Path to the second (after) image",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="/home/hkristen/habitalp2/data/processed/orthos_rgb_2003_2013/habitalp_change_2003_2013_B_C_prio3_rasterized_cog.tif",
        help="Path to the change mask",
    )
    parser.add_argument(
        "--label_poly_path",
        type=str,
        default="/home/hkristen/habitalp2/data/processed/orthos_rgb_2003_2013/habitalp_change_2003_2013_B_C_prio3_transitions_aggregated.gpkg",
        help="Path to label polygons for sampling",
    )
    parser.add_argument(
        "--train_roi_path",
        type=str,
        default="/home/hkristen/habitalp2/data/processed/orthos_rgb_2003_2013/split_train.gpkg",
        help="Path to training ROI",
    )
    parser.add_argument(
        "--val_roi_path",
        type=str,
        default="/home/hkristen/habitalp2/data/processed/orthos_rgb_2003_2013/split_val.gpkg",
        help="Path to validation ROI. If not provided, test set is used for validation.",
    )
    parser.add_argument(
        "--test_roi_path",
        type=str,
        default="/home/hkristen/habitalp2/data/processed/orthos_rgb_2003_2013/split_test.gpkg",
        help="Path to test ROI",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes (2 for binary change, >2 for multi-class)",
    )
    parser.add_argument(
        "--inWidth", type=int, default=256, help="Patch width (Size of image patches)"
    )
    parser.add_argument(
        "--inHeight", type=int, default=256, help="Patch height (Size of image patches)"
    )
    parser.add_argument(
        "--samples_per_epoch",
        type=int,
        default=25000,
        help="Number of samples per epoch for datamodule (can be None)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="No. of parallel threads for dataloading",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    parser.add_argument(
        "--model_type",
        type=str,
        default="small",
        choices=["tiny", "small"],
        help="Select vit model type for ChangeViT",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20000,
        help="Max. number of training steps (iterations)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.00008, help="Initial learning rate"
    )
    parser.add_argument(
        "--lr_mode",
        default="poly",
        choices=["step", "poly"],
        help="Learning rate policy",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Optimizer weight decay"
    )
    parser.add_argument(
        "--seed", type=int, default=15, help="Initialization seed number"
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="/home/hkristen/habitalp2/src/models/experiments",
        help="Base directory to save experiment results",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Path to checkpoint to resume training",
        # /home/hkristen/ChangeViT/checkpoint/deit_tiny_patch16_224-a1311bcf.pth
        # /home/hkristen/ChangeViT/checkpoint/dinov2_vits14_pretrain.pth
    )
    parser.add_argument(
        "--logFile",
        default="trainValLog.txt",
        help="File that stores the training and validation logs",
    )
    parser.add_argument(
        "--save_interval", type=int, default=25, help="Save checkpoint every N epochs"
    )

    parser.add_argument(
        "--onGPU",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="Run on CPU or GPU",
    )
    parser.add_argument(
        "--gpu_id", default=0, type=int, help="GPU id number if onGPU is true"
    )

    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="change-vit", help="W&B project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (username or team name)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="TEST_DEV",
        help="W&B run name (experiment name)",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default="0f05335491f6648e2beb8972f303e87b9bcab99b",
        help="W&B API key (prefer using env var WANDB_API_KEY or wandb login)",
    )
    parser.add_argument(
        "--wandb_log_interval",
        type=int,
        default=10,
        help="Log training step metrics to W&B every N global steps",
    )
    parser.add_argument(
        "--wandb_watch_interval",
        type=int,
        default=100,
        help="Log model gradients/parameters to W&B every N batches (log_freq for wandb.watch)",
    )
    parser.add_argument(
        "--wandb_save_checkpoints",
        action="store_true",
        help="Save model checkpoints as W&B artifacts",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="W&B run ID to resume a specific run.",
    )

    parser.add_argument(
        "--print_interval",
        type=int,
        default=10,
        help="Print training status every N iterations",
    )
    parser.add_argument(
        "--val_interval", type=int, default=10, help="Run validation every N steps"
    )

    parser.add_argument(
        "--resume_optimizer",
        action="store_true",
        help="Resume optimizer state from checkpoint if available",
    )
    parser.add_argument(
        "--validate_first_epoch",
        action="store_true",
        help="Run validation after the very first epoch, even if not resuming.",
    )

    # Add early stopping parameters
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=200,
        help="Number of validation steps to wait before early stopping",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.001,
        help="Minimum change in IoU to be considered as improvement",
    )

    # Add class weights parameters
    parser.add_argument(
        "--background_weight",
        type=float,
        default=0.55016005,
        help="Weight for the background class in loss calculation",
    )
    parser.add_argument(
        "--change_weight",
        type=float,
        default=1.44984,
        help="Weight for the change class in loss calculation",
    )

    # Add shutdown parameter
    parser.add_argument(
        "--auto_shutdown",
        action="store_true",
        help="Automatically shutdown the system after training completes",
    )

    args = parser.parse_args()

    if args.samples_per_epoch is not None and args.samples_per_epoch <= 0:
        args.samples_per_epoch = None

    if args.wandb_run_name == "TEST_DEV" and args.use_wandb:
        pass

    print("Called with args:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")

    trainValidateSegmentation(args)
