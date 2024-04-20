""" Data utility functions. """

import gzip
import pickle as pkl

from core.utils.fctp import FCTP
from core.utils.fctp import CapacitatedFCTP
from core.utils.fctp import FixedStepFCTP


def dict_to_instance(instance_dict):
    """Convert instance dict into proper FCTP class object.

    Parameters
    ----------
    instance_dict: dict
        Dictionary containing instance attributes.

    Returns
    -------
    FCTP
        Some FCTP instance object.

    """
    if instance_dict["instance_type"] == "fctp":
        del instance_dict["instance_type"]
        return FCTP(**instance_dict)
    elif instance_dict["instance_type"] == "c-fctp":
        del instance_dict["instance_type"]
        return CapacitatedFCTP(**instance_dict)
    elif instance_dict["instance_type"] == "fs-fctp":
        del instance_dict["instance_type"]
        return FixedStepFCTP(**instance_dict)
    else:
        raise ValueError


def load_instance(instance_path):
    """Load instance from path.

    Parameters
    ----------
    instance_path: str
        Path to pickle file with instance data.

    Returns
    -------
    instance: FCTP
        FCTP instance.

    """
    with gzip.open(instance_path, "rb") as file:
        instance_dict = pkl.load(file)
    return dict_to_instance(instance_dict)


def load_sample(sample_path):
    """Load sample from path.

    Parameters
    ----------
    sample_path: str
        Path to pickle file with sample data.

    Returns
    -------
    dict
        Sample dictionary.

    """
    with gzip.open(sample_path, "rb") as file:
        sample = pkl.load(file)
    sample["instance"] = dict_to_instance(sample["instance"])
    return sample
