import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

np.random.seed(0)
torch.manual_seed(0)


def test(model, test_loader, criterion, device):
    """
    Function that can takes a model and a testing data loader
    and will get the test accuray/loss of the model
    """
    logger.info("Testing Model on Whole Testing Dataset")

    model.eval()
    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    logger.info(
        f"Test set accuracy: {100*total_acc}, Test set average loss: {total_loss}"
    )


def train(model, dataset_loader, criterion, optimizer, device, args):
    """ """
    best_loss = 1e6
    loss_counter = 0
    logger.info("Training")

    for epoch in range(args.epochs):
        for phase in ["train", "valid"]:
            logger.info(f"Epoch {epoch}, Phase {phase}")
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            for step, (inputs, labels) in enumerate(dataset_loader[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples += len(inputs)
                if running_samples % (args.batch_size * 20) == 0:
                    accuracy = running_corrects / running_samples
                    logger.info(
                        "Images [{}/{} ({:.0f}%)] Loss: {:.4f} Accuracy: {}/{} ({:.4f}%)".format(
                            running_samples,
                            len(dataset_loader[phase].dataset),
                            100.0
                            * (running_samples / len(dataset_loader[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0 * accuracy,
                        )
                    )

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples

            if phase == "train":
                logger.info(
                    f"Epoch training loss: {round(epoch_loss, 4)}; Epoch training accuracy: {round(epoch_acc, 4)}"
                )
            if phase == "valid":
                logger.info(
                    f"Epoch test loss: {round(epoch_loss, 4)}; Epoch test accuracy: {round(epoch_acc, 4)}"
                )

    return model


def get_pretrained_model(args, num_classes):
    """
    Function that initializes pre-trained model
    """

    if args.model_type == "resnet18":
        model = models.resnet18(pretrained=True)

    elif args.model_type == "resnet50":
        model = models.resnet50(pretrained=True)

    else:
        logger.info(f"{args.model_type} not found. Will use defaul option: ResNet18.")
        model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features

    model.fc = nn.Sequential(nn.Linear(num_features, num_classes))

    return model


def download_data():
    """ """
    dataset = torchvision.datasets.Caltech256(root="./data", download=True)
    data_path = dataset.root + "/256_ObjectCategories"
    return data_path


def create_data_loaders(data, args):
    """
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    """

    val_split = args.val_split
    targets = data.targets

    train_idx, valid_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_split,
        random_state=42,
        shuffle=True,
        stratify=targets,
    )

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        data, batch_size=args.test_batch_size, sampler=val_sampler
    )

    dataset_loader = {
        "train": train_loader,
        "valid": val_loader,
    }

    return dataset_loader


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    logger.info(f"Running on Device {device}")

    logger.info(f"Loading and Transforming Dataset")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_path = os.path.join(args.data_dir, args.data_path)

    try:
        dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    except Exception as err:
        logger.info(
            f"Couldn't load dataset from {data_path} path. Will try downloading."
        )
        data_path = args.data_dir + download_data()
        dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

    num_classes = len(set(dataset.targets))
    logger.info(f"--- num_classes: {num_classes} ---")

    logger.info("--- Getting training and testing data loaders... ---")
    dataset_loader = create_data_loaders(dataset, args)
    logger.info(f"Initializing pre-trained model")

    model = get_pretrained_model(args, num_classes)
    model = model.to(device)

    logger.info("--- Defining Criteria and Optimizer ---")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum)

    model = train(model, dataset_loader, criterion, optimizer, device, args)

    logger.info("Testing model.")
    test(model, dataset_loader["valid"], criterion, device)

    logger.info(f"Saving model to {args.model_dir}")
    os.makedirs(args.model_dir, exist_ok=True)
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 10)",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for testing (default: 10)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 5)",
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        metavar="VS",
        help="fraction of images to be used for test purposes (default: 0.1)",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="resnet18",
        help="Pytorch model to use for fine-tuning (default: resnet18)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/caltech256/256_ObjectCategories",
        help="Path with data (default: data/caltech256/256_ObjectCategories)",
    )

    # Container environment
    parser.add_argument(
        "--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"])
    )
    parser.add_argument(
        "--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()

    main(args)
