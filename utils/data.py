import torch
import random
import pandas as pd
from enum import Enum
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import Tuple, Optional


class CaseTypes(Enum):
    def __init__(self, id, description):
        self.id = id
        self.description = description

    akiec = 0, "Bowen's disease "
    bcc = 1, "basal cell carcinoma "
    bkl = 2, "benign keratosis-like lesions"
    df = 3, "dermatofibroma"
    mel = 4, "melanoma"
    nv = 5, "melanocytic nevi"
    vasc = 6, "vascular lesions"

    @classmethod
    def has_member(cls, value: str) -> bool:
        return value in cls.__members__


class SkinCancerDataset(torch.utils.data.Dataset):
    """Dataset class to handle inputs and targets for the CNN model"""

    def __init__(
        self,
        image_dir: Path,
        metadata_csv: Path,
        resize_image_shape: Optional[Tuple[int, int]] = None,
        normalize_images: bool = False,
        normalize_mean: Optional[Tuple[float, float, float]] = None,
        normalize_std: Optional[Tuple[float, float, float]] = None,
        target_key: str = "dx",
        target_types=CaseTypes,
    ):

        dataframe = pd.read_csv(metadata_csv)
        image_names = dataframe["image_id"].values.tolist()
        image_paths = [image_dir / Path(image_name + ".jpg") for image_name in image_names]

        assert all(
            [image_path.is_file() for image_path in image_paths]
        ), f"[HERA ERROR] Dataset is inconsistent. Could not find all images, listed in {metadata_csv}"
        assert len(image_paths) == len(
            [image_path for image_path in image_dir.iterdir() if image_path.suffix == ".jpg"]
        ), f"[HERA ERROR]: Dataset is inconsistent. There are more images than entries in {metadata_csv}"

        target_entries = dataframe[target_key].values.tolist()
        targets = [target_types[target_entry].id for target_entry in target_entries if target_types.has_member(target_entry)]

        assert len(image_paths) == len(
            targets
        ), f"[HERA ERROR] Dataset is inconsistent. Found {len(image_paths)} images, and {len(targets)} labels. Should be the same."

        self.data_points = [(image_path, target) for (image_path, target) in zip(image_paths, targets)]
        random.shuffle(self.data_points)

        print(f"[HERA INFO]: Created skin cancer dataset from image paths and labels for {len(self.data_points)} data points.")
        print(f"[HERA INFO]: Target key is {target_key} for targets of type {target_types}.")

        pil_image = Image.open(str(image_paths[0].absolute()))
        self.image_shape = (3,) + pil_image.size
        self.resize_shape = resize_image_shape

        transform_list = []

        transform_list.append(transforms.ToTensor())

        if resize_image_shape is not None:
            transform_list.append(transforms.Resize(resize_image_shape))

        if normalize_images:
            assert (
                normalize_mean is not None and normalize_std is not None
            ), "[HERA ERROR]: To normalize the images, please specify mean and standard deviation for all three color channels."

            transform_list.append(
                transforms.Normalize(
                    mean=normalize_mean,
                    std=normalize_std,
                )
            )

            print(f"[HERA INFO]: Images will be normalized with:\n mean: {normalize_mean} and\n std: {normalize_std}.")

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):

        image_path, target = self.data_points[index]

        pil_image = Image.open(str(image_path.absolute()))

        tensor_image = self.transform(pil_image)

        return tensor_image, target

    def __len__(self):
        return len(self.data_points)


def load_dataset(
    image_dir: Path,
    metadata_csv: Path,
    batch_size: int,
    resize_image_shape: Optional[Tuple[int, int]] = None,
    normalize_images: bool = False,
    normalize_mean: Optional[Tuple[float, float, float]] = None,
    normalize_std: Optional[Tuple[float, float, float]] = None,
    target_key: str = "dx",
    target_types=CaseTypes,
    do_shuffle: bool = True,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    single_loader: bool = False,
    num_workers: Optional[int] = None,
):
    """Returns DataLoader for a DataSet."""

    dataset = SkinCancerDataset(
        image_dir=image_dir,
        metadata_csv=metadata_csv,
        target_key=target_key,
        target_types=target_types,
        resize_image_shape=resize_image_shape,
        normalize_images=normalize_images,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    if single_loader:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=do_shuffle,
            num_workers=num_workers,
        )
        print(f"[HERA INFO]: Created a single DataLoader from DataSet with batch size {batch_size}.")
    else:
        assert (
            sum(split_ratios) == 1.0
        ), f"[HERA ERROR]: The sum of split_ratios should equal 100%, but is at {sum(split_ratios)*100}%."

        train_len = int(split_ratios[0] * len(dataset))
        val_len = int(split_ratios[1] * len(dataset))
        test_len = len(dataset) - val_len - train_len

        sets = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])

        loader = tuple(
            torch.utils.data.DataLoader(split, batch_size=batch_size, shuffle=do_shuffle, num_workers=num_workers)
            for split in sets
        )

        print(f"[HERA INFO]: Created three DataLoaders from DataSet with batch size {batch_size}.")
        print(
            f"[HERA INFO]: There are {len(sets[0])}, {len(sets[1])}, and {len(sets[2])} data points for training, validating, and testing, respectively."
        )

    print(f"[HERA INFO]: The image shape is {dataset.resize_shape if dataset.resize_shape is not None else dataset.image_shape}")
    print(f"[HERA INFO]: Datapoints will {'not ' if not do_shuffle else ''}be shuffled after a full pass.")

    return loader
