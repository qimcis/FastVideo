# SPDX-License-Identifier: Apache-2.0

import dataclasses
import glob
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Generator, Iterable, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoImageProcessor, AutoTokenizer, PretrainedConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.hf_transformer_utils import (get_diffusers_config,
                                                      get_hf_config)
from fastvideo.v1.models.loader.fsdp_load import load_fsdp_model
from fastvideo.v1.models.loader.utils import set_default_torch_dtype
from fastvideo.v1.models.loader.weight_utils import (
    filter_duplicate_safetensors_files, filter_files_not_needed_for_inference,
    pt_weights_iterator, safetensors_weights_iterator)
from fastvideo.v1.models.registry import ModelRegistry
from fastvideo.v1.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class ComponentLoader(ABC):
    """Base class for loading a specific type of model component."""

    def __init__(self, device=None) -> None:
        self.device = device

    @abstractmethod
    def load(self, model_path: str, architecture: str,
             inference_args: InferenceArgs):
        """
        Load the component based on the model path, architecture, and inference args.
        
        Args:
            model_path: Path to the component model
            architecture: Architecture of the component model
            inference_args: Inference arguments
            
        Returns:
            The loaded component
        """
        raise NotImplementedError

    @classmethod
    def for_module_type(cls, module_type: str,
                        transformers_or_diffusers: str) -> 'ComponentLoader':
        """
        Factory method to create a component loader for a specific module type.
        
        Args:
            module_type: Type of module (e.g., "vae", "text_encoder", "transformer", "scheduler")
            transformers_or_diffusers: Whether the module is from transformers or diffusers
            
        Returns:
            A component loader for the specified module type
        """
        # Map of module types to their loader classes and expected library
        module_loaders = {
            "scheduler": (SchedulerLoader, "diffusers"),
            "transformer": (TransformerLoader, "diffusers"),
            "vae": (VAELoader, "diffusers"),
            "text_encoder": (TextEncoderLoader, "transformers"),
            "text_encoder_2": (TextEncoderLoader, "transformers"),
            "tokenizer": (TokenizerLoader, "transformers"),
            "tokenizer_2": (TokenizerLoader, "transformers"),
            "image_processor": (ImageProcessorLoader, "transformers"),
            "image_encoder": (ImageEncoderLoader, "transformers"),
        }

        if module_type in module_loaders:
            loader_cls, expected_library = module_loaders[module_type]
            # Assert that the library matches what's expected for this module type
            assert transformers_or_diffusers == expected_library, f"{module_type} must be loaded from {expected_library}, got {transformers_or_diffusers}"
            return loader_cls()

        # For unknown module types, use a generic loader
        logger.warning(
            "No specific loader found for module type: %s. Using generic loader.",
            module_type)
        return GenericComponentLoader(transformers_or_diffusers)


class TextEncoderLoader(ComponentLoader):
    """Loader for text encoders."""

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: Optional[list[str]] = None
        """If defined, weights will load exclusively using these patterns."""

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def _prepare_weights(
        self,
        model_name_or_path: str,
        fall_back_to_pt: bool,
        allow_patterns_overrides: Optional[list[str]],
    ) -> Tuple[str, List[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        # model_name_or_path = (self._maybe_download_from_modelscope(
        #     model_name_or_path, revision) or model_name_or_path)

        is_local = os.path.isdir(model_name_or_path)
        assert is_local, "Model path must be a local directory"

        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        allow_patterns = ["*.safetensors", "*.bin"]

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        hf_folder = model_name_or_path

        hf_weights_files: List[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file)
        else:
            hf_weights_files = filter_files_not_needed_for_inference(
                hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`")

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
            self, source: "Source"
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path, source.fall_back_to_pt,
            source.allow_patterns_overrides)
        if use_safetensors:
            weights_iterator = safetensors_weights_iterator(hf_weights_files)
        else:
            weights_iterator = pt_weights_iterator(hf_weights_files)

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor)
                for (name, tensor) in weights_iterator)

    def _get_all_weights(
        self,
        model_config: Any,
        model: nn.Module,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        primary_weights = TextEncoderLoader.Source(
            model_config.model,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides",
                                             None),
        )
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[TextEncoderLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def load(self, model_path: str, architecture: str,
             inference_args: InferenceArgs):
        """Load the text encoders based on the model path, architecture, and inference args."""
        model_config: PretrainedConfig = get_hf_config(
            model=model_path,
            trust_remote_code=inference_args.trust_remote_code,
            revision=inference_args.revision,
            model_override_args=None,
            inference_args=inference_args,
        )
        logger.info("HF Model config: %s", model_config)

        target_device = torch.device(inference_args.device_str)
        # TODO(will): add support for other dtypes
        return self.load_model(model_path, model_config, target_device,
                               inference_args.text_encoder_precision)

    def load_model(self,
                   model_path: str,
                   model_config,
                   target_device: torch.device,
                   dtype: str = "fp16"):
        with set_default_torch_dtype(PRECISION_TO_TYPE[dtype]):
            with target_device:
                architectures = getattr(model_config, "architectures", [])
                model_cls, _ = ModelRegistry.resolve_model_cls(architectures)
                model = model_cls(model_config)

            weights_to_load = {name for name, _ in model.named_parameters()}
            model_config.model = model_path
            loaded_weights = model.load_weights(
                self._get_all_weights(model_config, model))
            self.counter_after_loading_weights = time.perf_counter()
            logger.info(
                "Loading weights took %.2f seconds",
                self.counter_after_loading_weights -
                self.counter_before_loading_weights)
            # We only enable strict check for non-quantized models
            # that have loaded weights tracking currently.
            # if loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError("Following weights were not initialized from "
                                 f"checkpoint: {weights_not_loaded}")

        # TODO(will): add support for training/finetune
        return model.eval()


class ImageEncoderLoader(TextEncoderLoader):

    def load(self, model_path: str, architecture: str,
             inference_args: InferenceArgs):
        """Load the text encoders based on the model path, architecture, and inference args."""
        model_config: PretrainedConfig = get_hf_config(
            model=model_path,
            trust_remote_code=inference_args.trust_remote_code,
            revision=inference_args.revision,
            model_override_args=None,
            inference_args=inference_args,
        )
        logger.info("HF Model config: %s", model_config)

        target_device = torch.device(inference_args.device_str)
        # TODO(will): add support for other dtypes
        return self.load_model(model_path, model_config, target_device,
                               inference_args.image_encoder_precision)


class ImageProcessorLoader(ComponentLoader):
    """Loader for image processor."""

    def load(self, model_path: str, architecture: str,
             inference_args: InferenceArgs):
        """Load the image processor based on the model path, architecture, and inference args."""
        logger.info("Loading image processor from %s", model_path)

        image_processor = AutoImageProcessor.from_pretrained(model_path, )
        logger.info("Loaded image processor: %s",
                    image_processor.__class__.__name__)
        return image_processor


class TokenizerLoader(ComponentLoader):
    """Loader for tokenizers."""

    def load(self, model_path: str, architecture: str,
             inference_args: InferenceArgs):
        """Load the tokenizer based on the model path, architecture, and inference args."""
        logger.info("Loading tokenizer from %s", model_path)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            # TODO(will): pass these tokenizer kwargs from inference args? Maybe
            # other method of config?
            padding_size='right',
        )
        logger.info("Loaded tokenizer: %s", tokenizer.__class__.__name__)
        return tokenizer


class VAELoader(ComponentLoader):
    """Loader for VAE."""

    def load(self, model_path: str, architecture: str,
             inference_args: InferenceArgs):
        """Load the VAE based on the model path, architecture, and inference args."""
        # TODO(will): move this to a constants file
        config = get_diffusers_config(model=model_path)

        class_name = config.pop("_class_name")
        assert class_name is not None, "Model config does not contain a _class_name attribute. Only diffusers format is supported."
        config.pop("_diffusers_version")

        vae_cls, _ = ModelRegistry.resolve_model_cls(class_name)

        vae = vae_cls(**config).to(inference_args.device)

        # Find all safetensors files
        safetensors_list = glob.glob(
            os.path.join(str(model_path), "*.safetensors"))
        # TODO(PY)
        assert len(
            safetensors_list
        ) == 1, f"Found {len(safetensors_list)} safetensors files in {model_path}"
        loaded = safetensors_load_file(safetensors_list[0])
        vae.load_state_dict(loaded)
        dtype = PRECISION_TO_TYPE[inference_args.vae_precision]
        vae = vae.eval().to(dtype)

        return vae


class TransformerLoader(ComponentLoader):
    """Loader for transformer."""

    def load(self, model_path: str, architecture: str,
             inference_args: InferenceArgs):
        """Load the transformer based on the model path, architecture, and inference args."""
        model_config = get_diffusers_config(model=model_path)
        cls_name = model_config.pop("_class_name")
        if cls_name is None:
            raise ValueError(
                "Model config does not contain a _class_name attribute. "
                "Only diffusers format is supported.")
        model_config.pop("_diffusers_version")

        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        # Find all safetensors files
        safetensors_list = glob.glob(
            os.path.join(str(model_path), "*.safetensors"))
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {model_path}")

        logger.info("Loading model from %s safetensors files in %s",
                    len(safetensors_list), model_path)

        # initialize_sequence_parallel_group(inference_args.sp_size)
        default_dtype = PRECISION_TO_TYPE[inference_args.precision]

        # Load the model using FSDP loader
        logger.info("Loading model from %s", cls_name)
        model = load_fsdp_model(model_cls=model_cls,
                                init_params=model_config,
                                weight_dir_list=safetensors_list,
                                device=inference_args.device,
                                cpu_offload=inference_args.use_cpu_offload,
                                default_dtype=default_dtype)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded model with %.2fB parameters", total_params / 1e9)

        dtypes = set(param.dtype for param in model.parameters())
        if len(dtypes) > 1:
            model = model.to(default_dtype)
        model = model.eval()
        return model


class SchedulerLoader(ComponentLoader):
    """Loader for scheduler."""

    def load(self, model_path: str, architecture: str,
             inference_args: InferenceArgs):
        """Load the scheduler based on the model path, architecture, and inference args."""
        config = get_diffusers_config(model=model_path)

        class_name = config.pop("_class_name")
        assert class_name is not None, "Model config does not contain a _class_name attribute. Only diffusers format is supported."
        config.pop("_diffusers_version")

        scheduler_cls, _ = ModelRegistry.resolve_model_cls(class_name)

        scheduler = scheduler_cls(**config)
        if inference_args.flow_shift is not None:
            scheduler.set_shift(inference_args.flow_shift)

        return scheduler


class GenericComponentLoader(ComponentLoader):
    """Generic loader for components that don't have a specific loader."""

    def __init__(self, library="transformers") -> None:
        super().__init__()
        self.library = library

    def load(self, model_path: str, architecture: str,
             inference_args: InferenceArgs):
        """Load a generic component based on the model path, architecture, and inference args."""
        logger.warning("Using generic loader for %s with library %s",
                       model_path, self.library)

        if self.library == "transformers":
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=inference_args.trust_remote_code,
                revision=inference_args.revision,
            )
            logger.info("Loaded generic transformers model: %s",
                        model.__class__.__name__)
            return model
        elif self.library == "diffusers":
            logger.warning(
                "Generic loading for diffusers components is not fully implemented"
            )

            model_config = get_diffusers_config(model=model_path)
            logger.info("Diffusers Model config: %s", model_config)
            # This is a placeholder - in a real implementation, you'd need to handle this properly
            return None
        else:
            raise ValueError(f"Unsupported library: {self.library}")


class PipelineComponentLoader:
    """
    Utility class for loading pipeline components.
    This replaces the chain of if-else statements in load_pipeline_module.
    """

    @staticmethod
    def load_module(module_name: str, component_model_path: str,
                    transformers_or_diffusers: str, architecture: str,
                    inference_args: InferenceArgs):
        """
        Load a pipeline module.
        
        Args:
            module_name: Name of the module (e.g., "vae", "text_encoder", "transformer", "scheduler")
            component_model_path: Path to the component model
            transformers_or_diffusers: Whether the module is from transformers or diffusers
            architecture: Architecture of the component model
            inference_args: Inference arguments
            
        Returns:
            The loaded module
        """
        logger.info(
            "Loading %s using %s from %s",
            module_name,
            transformers_or_diffusers,
            component_model_path,
        )

        # Get the appropriate loader for this module type
        loader = ComponentLoader.for_module_type(module_name,
                                                 transformers_or_diffusers)

        # Load the module
        return loader.load(component_model_path, architecture, inference_args)
