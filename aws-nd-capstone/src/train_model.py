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
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

np.random.seed(0)
torch.manual_seed(0)

def test(model, test_loader, criterion, device):

    hook = get_hook(create_if_not_exists=True)
    logger.info("--- Testing model on the whole test dataset ---")

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

    hook = get_hook(create_if_not_exists=True)
    best_loss = 1e6
    loss_counter = 0
    logger.info("--- Starting training process ---")

    if hook:
        hook.register_loss(criterion)

    for epoch in range(args.epochs):
        for phase in ["train", "valid"]:
            logger.info(f"Epoch {epoch}, Phase {phase}")
            if phase == "train":
                model.train()
                if hook:
                    hook.set_mode(modes.TRAIN)
            else:
                model.eval()
                if hook:
                    hook.set_mode(modes.EVAL)
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
                    f"Epoch Training Loss: {round(epoch_loss, 4)}; Epoch Training Accuracy: {round(epoch_acc*100, 4)}"
                )
            if phase == "valid":
                logger.info(
                    f"Epoch Test Loss: {round(epoch_loss, 4)}; Epoch Test Accuracy: {round(epoch_acc*100, 4)}"
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


def get_mean_and_std(loader):
    """
    Function that calculates mean and std of image dataset
    """
    mean = 0.
    std = 0.
    total_images_count = 0

    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch

    mean /= total_images_count
    std /= total_images_count

    return mean, std

def create_data_loaders(args):
    
    train_data_path = os.path.join(args.data_dir, 'train/')
    test_data_path = os.path.join(args.data_dir, 'test/')
    
    image_size = args.image_size
    batch_size= args.batch_size
    
    train_transform = transforms.Compose([    
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor()   
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
    
    train_mean, train_std = get_mean_and_std(train_loader)
    
    train_transform = transforms.Compose([    
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),   
            transforms.ToTensor(),    
            transforms.Normalize(torch.Tensor(train_mean), torch.Tensor(train_std))
        ])

    test_transform = transforms.Compose([    
            transforms.Resize((image_size,image_size)),     
            transforms.ToTensor(),    
            transforms.Normalize(torch.Tensor(train_mean), torch.Tensor(train_std))
        ])

    train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)

    num_classes = len(set(train_dataset.targets))
    logger.info(f"--- num_classes: {num_classes} ---")
    
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle = True
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle = False
    )

    dataset_loader = {
        "train": train_loader,
        "valid": val_loader,
    }

    return dataset_loader, num_classes


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps") # Apple Silicon Users like me
        else:
            device = torch.device("cpu")

    logger.info(f"--- Running on Device {device} ---")
    
    logger.info("--- Loading and Transforming Dataset ---")
    
    dataset_loader, num_classes = create_data_loaders(args)

    logger.info("--- Initializing pre-trained model ---")

    model = get_pretrained_model(args, num_classes)
    model = model.to(device)

    logger.info("--- Defining Criteria and Optimizer ---")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum)

    model = train(model, dataset_loader, criterion, optimizer, device, args)

    logger.info("--- Testing model ---")
    test(model, dataset_loader["valid"], criterion, device)

    logger.info(f"--- Saving model to {args.model_dir} ---")
    os.makedirs(args.model_dir, exist_ok=True)
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model, path)

#     model_scripted = torch.jit.script(model) # Export to TorchScript
#     path = os.path.join(args.model_dir, "model.pt")
#     model_scripted.save(path)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        metavar="N",
        help="input image size for training (default: 224)",
    )
    
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
        default=10,
        metavar="N",
        help="number of epochs to train (default: 5)",
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        metavar="VS",
        help="fraction of images to be used for test purposes (default: 0.1)",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="resnet50",
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
        "--local",
        type=bool,
        default=True,
        help="Flag to run traning locally or via AWS (default: True)",
    )
    
    args = parser.parse_args()
    
    # if args.local:
    #     logger.info('Running locally!')
    #     parser.add_argument("--model-dir", type=str, default='')
    #     parser.add_argument("--data-dir", type=str, default='')
        
    # else:
        
        # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument(
        "--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"])
    )
    parser.add_argument(
        "--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]
    )
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--gpu", type=str2bool, default=True)

    args = parser.parse_args()

    main(args)
