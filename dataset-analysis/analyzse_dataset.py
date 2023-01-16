from typing_extensions import dataclass_transform
import tensorflow as tf
from tensorflow_datasets.core import DatasetInfo
import numpy as np
import collections
from typing import List


def calculate_imbalance(class_sizes: List[int]) -> float:
    """
        Idea of using shannon entropy to calculate balance from here: https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
        For data of n instances, if we have k classes of size c_i, we can calculate the entropy

        class_sizes : List(int) - contains the number of elements in each class, the length of this list is the number of classes we have
        The number of instances can also be retrieved from this list.

        Returns the dataset's imbalance as a float.
    """

    # calculate shannon entropy
    n: int = sum(class_sizes)
    k: int = len(class_sizes)
    print(f"Calculate class imbalance for dataset with {k} classes and {n} items")

    H: float = 0.0
    for c in class_sizes:
        H += (c / n) * np.log((c / n))

    H *= -1

    # calculate balance value
    B: float = H / np.log(k)

    return B


def analyze_dataset(data: np.ndarray, ds_info: DatasetInfo):
    if ds_info is not None:
        print(f"----------------------------\nDataset Info: \n{ds_info}\n----------------------------")

    print(type(data))

    class_counter = collections.defaultdict(int)
    class_sizes: List[int] = []
    datapoint_entropy: List[float] = []

    #all_data_len = len(data)
    counter = 0

    print("Starting to analyse dataset...")
    for i in data:
        #print(f"{counter}/{all_data_len}", end='\r')
        print(f"{counter}", end='\r')
        counter += 1
        if type(i) == dict:
            data_tensor = i["images"]
            class_tensor = i["labels"][0]
        else:
            data_tensor, class_tensor = i

        class_counter[class_tensor] += 1

        # calculate entropy of each data_tensor
        value, counts = np.unique(data_tensor, return_counts=True)
        norm_counts = counts / counts.sum()
        entropy = -(norm_counts * np.log(norm_counts)).sum()
        entropy = entropy / np.log(len(value))
        datapoint_entropy.append(entropy)

    # fill list with classes and number of datapoints per class
    for _, v in class_counter.items():
        class_sizes.append(v)

    balance = calculate_imbalance(class_sizes)
    smallest_class = min(class_sizes)
    biggest_class = max(class_sizes)
    print(f"Class Balance: {balance:.4f}, Number of items in smallest class: {smallest_class}, biggest class: {biggest_class}")

    # analyze entropy list -> find min, max and average values
    max_entropy = max(datapoint_entropy)
    min_entropy = min(datapoint_entropy)
    avg_entropy = sum(datapoint_entropy) / len(datapoint_entropy)

    print(f"Average Data Entropy: {avg_entropy:.4f}, Min Entropy: {min_entropy:.4f}, Max Entropy: {max_entropy:.4f}")


# This function is used for the PurchaseX and TexasX numpy datasets
def analyze_numpy_dataset(features: np.ndarray, labels: np.ndarray):
    class_counter = collections.defaultdict(int)
    class_sizes: List[int] = []
    datapoint_entropy: List[float] = []

    all_data_len = len(features)
    counter = 0

    print("Starting to analyse dataset...")
    for i in zip(features, labels):
        print(f"{counter}/{all_data_len}", end='\r')
        counter += 1

        # reverse keras' to_categorical (binary label array to decimal integer)
        label = int(i[1].nonzero()[0][0]) + 1
        class_counter[label] += 1

        feature = i[0]

        # calculate entropy of each data_tensor
        value, counts = np.unique(feature, return_counts=True)
        norm_counts = counts / counts.sum()
        entropy = -(norm_counts * np.log(norm_counts)).sum()
        entropy = entropy / np.log(len(value))
        datapoint_entropy.append(entropy)

    # fill list with classes and number of datapoints per class
    for _, v in class_counter.items():
        class_sizes.append(v)

    balance = calculate_imbalance(class_sizes)
    smallest_class = min(class_sizes)
    biggest_class = max(class_sizes)
    print(f"Class Balance: {balance:.4f}, Number of items in smallest class: {smallest_class}, biggest class: {biggest_class}")

    # analyze entropy list -> find min, max and average values
    max_entropy = max(datapoint_entropy)
    min_entropy = min(datapoint_entropy)
    avg_entropy = sum(datapoint_entropy) / len(datapoint_entropy)

    print(f"Average Data Entropy: {avg_entropy:.4f}, Min Entropy: {min_entropy:.4f}, Max Entropy: {max_entropy:.4f}")
