"""Utilities for downloading and initializing model weights."""
import filelock
import glob
import fnmatch
import json
import os
from collections import defaultdict
from typing import Any, Iterator, List, Optional, Tuple

from huggingface_hub import snapshot_download, HfFileSystem
import numpy as np
from safetensors.torch import load_file, save_file, safe_open
import torch
from tqdm.auto import tqdm

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (get_quantization_config,
                                                     QuantizationConfig)

logger = init_logger(__name__)


class Disabledtqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def get_lock(model_name_or_path: str, cache_dir: Optional[str] = None):
    lock_dir = cache_dir if cache_dir is not None else "/tmp"
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))
    return lock


def _shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for _, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def convert_bin_to_safetensor_file(
    pt_filename: str,
    sf_filename: str,
) -> None:
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared = _shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})

    # check file size
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size
    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """)

    # check if the tensors are the same
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


# TODO(woosuk): Move this to other place.
def get_quant_config(model_config: ModelConfig) -> QuantizationConfig:
    quant_cls = get_quantization_config(model_config.quantization)
    # Read the quantization config from the HF model config, if available.
    hf_quant_config = getattr(model_config.hf_config, "quantization_config",
                              None)
    if hf_quant_config is not None:
        return quant_cls.from_config(hf_quant_config)
    model_name_or_path = model_config.model
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        # Download the config files.
        with get_lock(model_name_or_path, model_config.download_dir):
            hf_folder = snapshot_download(model_name_or_path,
                                          revision=model_config.revision,
                                          allow_patterns="*.json",
                                          cache_dir=model_config.download_dir,
                                          tqdm_class=Disabledtqdm)
    else:
        hf_folder = model_name_or_path
    config_files = glob.glob(os.path.join(hf_folder, "*.json"))

    quant_config_files = [
        f for f in config_files if any(
            f.endswith(x) for x in quant_cls.get_config_filenames())
    ]
    if len(quant_config_files) == 0:
        raise ValueError(
            f"Cannot find the config file for {model_config.quantization}")
    if len(quant_config_files) > 1:
        raise ValueError(
            f"Found multiple config files for {model_config.quantization}: "
            f"{quant_config_files}")

    quant_config_file = quant_config_files[0]
    with open(quant_config_file, "r") as f:
        config = json.load(f)
    return quant_cls.from_config(config)


def prepare_hf_model_weights(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    fall_back_to_pt: bool = True,
    revision: Optional[str] = None,
) -> Tuple[str, List[str], bool]:
    # Download model weights from huggingface.
    is_local = os.path.isdir(model_name_or_path)
    use_safetensors = False
    # Some quantized models use .pt files for storing the weights.
    if load_format == "auto":
        allow_patterns = ["*.safetensors", "*.bin"]
    elif load_format == "safetensors":
        use_safetensors = True
        allow_patterns = ["*.safetensors"]
    elif load_format == "pt":
        allow_patterns = ["*.pt"]
    elif load_format == "npcache":
        allow_patterns = ["*.bin"]
    else:
        raise ValueError(f"Unknown load_format: {load_format}")

    if fall_back_to_pt:
        allow_patterns += ["*.pt"]

    if not is_local:
        # Before we download we look at that is available:
        fs = HfFileSystem()
        file_list = fs.ls(model_name_or_path, detail=False, revision=revision)

        # depending on what is available we download different things
        for pattern in allow_patterns:
            matching = fnmatch.filter(file_list, pattern)
            if len(matching) > 0:
                allow_patterns = [pattern]
                break

        logger.info(f"Using model weights format {allow_patterns}")
        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(model_name_or_path, cache_dir):
            hf_folder = snapshot_download(model_name_or_path,
                                          allow_patterns=allow_patterns,
                                          cache_dir=cache_dir,
                                          tqdm_class=Disabledtqdm,
                                          revision=revision)
    else:
        hf_folder = model_name_or_path
    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
        if len(hf_weights_files) > 0:
            if pattern == "*.safetensors":
                use_safetensors = True
            break
    if not use_safetensors:
        # Exclude files that are not needed for inference.
        # https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
        blacklist = [
            "training_args.bin",
            "optimizer.bin",
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
        ]
        hf_weights_files = [
            f for f in hf_weights_files
            if not any(f.endswith(x) for x in blacklist)
        ]

    if len(hf_weights_files) == 0:
        raise RuntimeError(
            f"Cannot find any model weights with `{model_name_or_path}`")

    return hf_folder, hf_weights_files, use_safetensors


def hf_model_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    revision: Optional[str] = None,
    fall_back_to_pt: Optional[bool] = True,
) -> Iterator[Tuple[str, torch.Tensor]]:
    hf_folder, hf_weights_files, use_safetensors = prepare_hf_model_weights(
        model_name_or_path,
        cache_dir=cache_dir,
        load_format=load_format,
        fall_back_to_pt=fall_back_to_pt,
        revision=revision)

    if load_format == "npcache":
        # Currently np_cache only support *.bin checkpoints
        assert use_safetensors is False

        # Convert the model weights from torch tensors to numpy arrays for
        # faster loading.
        np_folder = os.path.join(hf_folder, "np")
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, "weight_names.json")
        # Use file lock to prevent multiple processes from
        # dumping the same model weights to numpy at the same time.
        with get_lock(model_name_or_path, cache_dir):
            if not os.path.exists(weight_names_file):
                weight_names = []
                for bin_file in hf_weights_files:
                    state = torch.load(bin_file, map_location="cpu")
                    for name, param in state.items():
                        param_path = os.path.join(np_folder, name)
                        with open(param_path, "wb") as f:
                            np.save(f, param.cpu().detach().numpy())
                        weight_names.append(name)
                with open(weight_names_file, "w") as f:
                    json.dump(weight_names, f)

        with open(weight_names_file, "r") as f:
            weight_names = json.load(f)

        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, "rb") as f:
                param = np.load(f)
            yield name, torch.from_numpy(param)
    elif use_safetensors:
        for st_file in hf_weights_files:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    param = f.get_tensor(name)
                    yield name, param
    else:
        for bin_file in hf_weights_files:
            state = torch.load(bin_file, map_location="cpu")
            for name, param in state.items():
                yield name, param
            del state
            torch.cuda.empty_cache()


def convert_pyslice_to_tensor(x: Any) -> torch.Tensor:
    """convert PySafeSlice object from safetensors to torch.Tensor

    PySafeSlice object supports indexing, which is done before loading the
    actual tensor and can reduce the amount of memory being read into the
    memory. However, it does not support more advanced functionalities
    like `.view()` or `.t()`. Therefore, if we need to modify the loaded
    tensor with these more complicated operators, we need to convert to
    tensor first.
    """
    if not isinstance(x, torch.Tensor):
        x = x[:]
    return x


def default_weight_loader(param: torch.Tensor,
                          loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    assert param.size() == loaded_weight.size()
    param.data.copy_(loaded_weight)


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.
    """
    for param in model.state_dict().values():
        if torch.is_floating_point(param):
            param.data.uniform_(low, high)


import time
import boto3
from google.cloud import storage
from huggingface_hub import hf_hub_download

HF_PREFIX = "hf://"
MODEL_DIR = "/tmp/vllm_model"


def prepare_hf_model_weights_on_the_fly(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    use_safetensors: bool = False,
    fall_back_to_pt: bool = True,
    revision: Optional[str] = None,
) -> Tuple[List[str], bool]:
    logger.info("Loading weights on the fly.")
    lock = get_lock(model_name_or_path, cache_dir)

    hf_weights_files = []
    if use_safetensors:
        logger.info("Looking for .safetensors files")
        index_filename = "model.safetensors.index.json"
        allow_patterns = "*.safetensors"
    else:
        logger.info("Looking for .bin files")
        index_filename = "pytorch_model.bin.index.json"
        allow_patterns = "*.bin"
    if not os.path.isdir(model_name_or_path):
        try:
            with lock:
                index_file = hf_hub_download(repo_id=model_name_or_path,
                                             filename=index_filename,
                                             cache_dir=cache_dir)
        except:
            logger.info(
                "The model is in HF hub with 1 file, download it directly.")
            with lock:
                hf_folder = snapshot_download(repo_id=model_name_or_path,
                                              allow_patterns=allow_patterns,
                                              cache_dir=cache_dir,
                                              tqdm_class=Disabledtqdm)
            hf_weights_files = [
                x for x in glob.glob(os.path.join(hf_folder, allow_patterns))
            ]
        else:
            logger.info(
                "The model is in HF hub with multiple files, do not download it now."
            )
            with open(index_file, "r") as f:
                index = json.loads(f.read())
            weight_filenames = set(index["weight_map"].values())
            hf_weights_files = [
                f"{HF_PREFIX}{model_name_or_path}/{weight_filename}"
                for weight_filename in weight_filenames
            ]
    else:
        logger.info("The model is possibly in local disk.")
        hf_weights_files = [
            x for x in glob.glob(
                os.path.join(model_name_or_path, allow_patterns))
        ]

    if not use_safetensors:
        # Exclude files that are not needed for inference.
        # https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
        blacklist = [
            "training_args.bin",
            "optimizer.bin",
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
        ]
        hf_weights_files = [
            f for f in hf_weights_files
            if not any(f.endswith(x) for x in blacklist)
        ]
    hf_weights_files.sort()

    if not hf_weights_files and use_safetensors:
        return prepare_hf_model_weights_on_the_fly(model_name_or_path,
                                                   cache_dir=cache_dir,
                                                   use_safetensors=False,
                                                   fall_back_to_pt=False,
                                                   revision=revision)
    if not hf_weights_files:
        raise RuntimeError(f"No weight files found in {model_name_or_path}")
    logger.info(f"Fetched weight files: {hf_weights_files}")
    return hf_weights_files, use_safetensors


def hf_model_weights_iterator_download_on_the_fly(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    revision: Optional[str] = None,
    fall_back_to_pt: Optional[bool] = True,
) -> Iterator[Tuple[str, torch.Tensor]]:
    lock = get_lock(model_name_or_path, cache_dir)
    hf_weights_files, use_safetensors = prepare_hf_model_weights_on_the_fly(
        model_name_or_path=model_name_or_path,
        cache_dir=cache_dir,
        use_safetensors=True,
        fall_back_to_pt=fall_back_to_pt,
        revision=revision)
    os.makedirs(MODEL_DIR, exist_ok=True)
    for hf_weight_file in hf_weights_files:
        delete_download = False

        if os.path.exists(hf_weight_file):
            prefix = open(hf_weight_file, "rb").read(2)
            # Download from GCS.
            if prefix == b"gs":
                gcs_path = open(hf_weight_file).read()
                hf_weight_filename = gcs_path.split("/")[-1]
                local_file = os.path.join(MODEL_DIR, hf_weight_filename)
                with lock:
                    if not os.path.exists(local_file):
                        client = storage.Client()
                        with open(local_file, 'wb') as f:
                            logger.info(
                                f"Download {gcs_path} to {hf_weight_file}")
                            client.download_blob_to_file(gcs_path, f)
                hf_weight_file = local_file
                delete_download = True
            # Download from S3.
            elif prefix == b"s3":
                s3_path = open(hf_weight_file).read()
                hf_weight_filename = s3_path.split("/")[-1]
                local_file = os.path.join(MODEL_DIR, hf_weight_filename)

                bucket_name = s3_path.split('/')[2]
                obj_key = s3_path.split(bucket_name)[1][1:]
                with lock:
                    if not os.path.exists(local_file):
                        access_key_id = os.environ['AWS_ACCESS_KEY_ID']
                        secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
                        client = boto3.client(
                            's3',
                            aws_access_key_id=access_key_id,
                            aws_secret_access_key=secret_key,
                        )
                        with open(local_file, 'wb') as f:
                            logger.info(
                                f"Download {s3_path} to {hf_weight_file}")
                            client.download_fileobj(bucket_name, obj_key, f)
                hf_weight_file = local_file
                delete_download = True

        else:
            # Download from HF.
            assert hf_weight_file.startswith(HF_PREFIX)
            hf_weight_filename = os.path.basename(hf_weight_file)
            local_file = os.path.join(MODEL_DIR, hf_weight_filename)
            with lock:
                if not os.path.exists(local_file):
                    logger.info(
                        f"Download {model_name_or_path}/{hf_weight_filename} to {local_file}"
                    )
                    hf_hub_download(repo_id=model_name_or_path,
                                    filename=hf_weight_filename,
                                    local_dir=MODEL_DIR,
                                    local_dir_use_symlinks=False,
                                    force_download=True)
            hf_weight_file = local_file
            delete_download = True

        if use_safetensors:
            with safe_open(hf_weight_file, framework="pt") as f:
                for name in f.keys():
                    param = f.get_tensor(name)
                    yield name, param
                torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            logger.info(f"Load {hf_weight_file} to memory.")
            state = torch.load(hf_weight_file, map_location="cpu")
            for name, param in state.items():
                yield name, param
            del state
            torch.cuda.empty_cache()
            torch.distributed.barrier()

        if delete_download:
            with lock:
                if os.path.exists(hf_weight_file):
                    logger.info(f"Delete {hf_weight_file}")
                    os.remove(hf_weight_file)


hf_model_weights_iterator = hf_model_weights_iterator_download_on_the_fly
