import os
import types

import pytest
import torch

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.pipelines.stages.denoising import DenoisingStage

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")


@pytest.fixture(autouse=True)
def _patch_attention_backend(monkeypatch):
    class _DummyBackend:
        @staticmethod
        def get_builder_cls():
            return None

    monkeypatch.setattr("fastvideo.pipelines.stages.denoising.get_attn_backend",
                        lambda *args, **kwargs: _DummyBackend)
    monkeypatch.setattr(
        "fastvideo.pipelines.stages.denoising.get_local_torch_device",
        lambda: torch.device("cpu"))


class DummyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 64
        self.num_attention_heads = 8
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, *args, **kwargs):  # pragma: no cover - not used in tests
        return torch.zeros(1, 1)


class DummyScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([1.0, 0.5, 0.0])
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])
        self.num_train_timesteps = 1000
        self.order = 1

    def scale_model_input(self, latents, t):  # pragma: no cover - not used
        return latents

    def step(self, *args, **kwargs):  # pragma: no cover - not used
        return (torch.zeros_like(args[2]),)


def make_stage() -> DenoisingStage:
    transformer = DummyTransformer()
    transformer_2 = DummyTransformer()
    scheduler = DummyScheduler()
    return DenoisingStage(transformer=transformer,
                          transformer_2=transformer_2,
                          scheduler=scheduler)


def make_args(config: PipelineConfig,
              dit_cpu_offload: bool = False) -> types.SimpleNamespace:
    return types.SimpleNamespace(pipeline_config=config,
                                 dit_cpu_offload=dit_cpu_offload)


def test_sr_stays_on_sketch_before_min_step():
    config = PipelineConfig(sr_enabled=True,
                            sr_min_switch_step=5,
                            sr_max_switch_step=10)
    stage = make_stage()
    fastvideo_args = make_args(config)

    model, _ = stage._sr_select_model(
        i=0,
        current_model=stage.transformer,
        guidance_scale=1.0,
        fastvideo_args=fastvideo_args,
    )

    assert model is stage.transformer
    assert stage.sr_switched is False


def test_sr_switches_on_convergence():
    config = PipelineConfig(sr_enabled=True,
                            sr_diff_threshold=0.02,
                            sr_min_switch_step=1,
                            sr_max_switch_step=10,
                            sr_offload_unused=False)
    stage = make_stage()
    fastvideo_args = make_args(config)

    stage.sr_prev_prev_model_output = torch.ones(1, 1, 1)
    stage.sr_prev_model_output = torch.ones(1, 1, 1) * 0.99
    stage.sr_prev_diff = 0.015

    model, _ = stage._sr_select_model(
        i=6,
        current_model=stage.transformer,
        guidance_scale=1.0,
        fastvideo_args=fastvideo_args,
    )

    assert model is stage.transformer_2
    assert stage.sr_switched is True


def test_sr_forces_switch_at_max_step():
    config = PipelineConfig(sr_enabled=True,
                            sr_min_switch_step=1,
                            sr_max_switch_step=2,
                            sr_offload_unused=False)
    stage = make_stage()
    fastvideo_args = make_args(config)

    stage.sr_prev_prev_model_output = torch.zeros(1, 1, 1)
    stage.sr_prev_model_output = torch.ones(1, 1, 1) * 0.5
    stage.sr_prev_diff = 0.1

    model, _ = stage._sr_select_model(
        i=2,
        current_model=stage.transformer,
        guidance_scale=1.0,
        fastvideo_args=fastvideo_args,
    )

    assert model is stage.transformer_2
    assert stage.sr_switched is True


def test_sr_state_resets_between_batches():
    stage = make_stage()
    stage.sr_switched = True
    stage.sr_prev_model_output = torch.ones(1, 1, 1)
    stage.sr_prev_prev_model_output = torch.zeros(1, 1, 1)
    stage.sr_prev_diff = 0.5

    stage._sr_reset_state()

    assert stage.sr_switched is False
    assert stage.sr_prev_model_output is None
    assert stage.sr_prev_prev_model_output is None
    assert stage.sr_prev_diff == 0.0


def test_sr_and_wan22_mutually_exclusive():
    config = PipelineConfig(sr_enabled=True, boundary_ratio=0.5)

    with pytest.raises(ValueError):
        config.validate_sr_config()


def test_sr_records_denoised_prediction():
    config = PipelineConfig(sr_enabled=True,
                            sr_min_switch_step=1,
                            sr_max_switch_step=4)
    stage = make_stage()
    fastvideo_args = make_args(config)

    latents = torch.ones(1, 1, 1, 1, 1)
    noise_pred = torch.ones_like(latents) * 0.25
    timestep = torch.tensor([stage.scheduler.timesteps[0]])

    denoised = stage._sr_compute_denoised(latents=latents,
                                          noise_pred=noise_pred,
                                          timestep=timestep)
    assert denoised is not None
    expected = latents - stage.scheduler.sigmas[0] * noise_pred
    assert torch.allclose(denoised, expected)

    stage._sr_record_model_output(denoised)
    assert stage.sr_prev_model_output is not None
    assert torch.allclose(stage.sr_prev_model_output, expected)
