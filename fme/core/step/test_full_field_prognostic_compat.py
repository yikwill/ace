import pytest
from torch import nn

from fme.core.normalizer import NetworkAndLossNormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.step.single_module import ResidualPredictionConfig, SingleModuleStepConfig
from fme.core.step.spectral_lowpass import ResidualLowpassConfig
from fme.core.testing import trivial_normalization


def _state(**extra) -> dict:
    config = SingleModuleStepConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": nn.Identity()}),
        in_names=["a", "c"],
        out_names=["a", "b", "c"],
        normalization=NetworkAndLossNormalizationConfig(
            network=trivial_normalization(["a", "b", "c"]),
            residual=trivial_normalization(["a", "b", "c"]),
        ),
        residual_prediction=ResidualPredictionConfig(),
    )
    state = config.get_state()
    state["residual_prediction"] = True
    state.update(extra)
    return state


def test_full_field_names_become_the_residual_complement():
    config = SingleModuleStepConfig.from_state(
        _state(full_field_prognostic_names=["a"])
    )
    assert config.residual_prediction is not None
    assert config.residual_names == frozenset({"c"})


def test_full_field_name_cannot_be_lowpassed():
    state = _state(
        full_field_prognostic_names=["a"],
        residual_lowpass=[{"names": ["a"], "lmax": 4, "target": "residual"}],
    )
    with pytest.raises(ValueError, match="not stepped as a residual"):
        SingleModuleStepConfig.from_state(state)


def test_full_field_names_must_be_prognostic():
    with pytest.raises(ValueError, match="must be prognostic"):
        SingleModuleStepConfig.from_state(_state(full_field_prognostic_names=["b"]))


def test_listed_residual_can_still_be_lowpassed():
    config = SingleModuleStepConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": nn.Identity()}),
        in_names=["a", "c"],
        out_names=["a", "b", "c"],
        normalization=NetworkAndLossNormalizationConfig(
            network=trivial_normalization(["a", "b", "c"]),
        ),
        residual_prediction=ResidualPredictionConfig(names=["c"]),
        residual_lowpass=[ResidualLowpassConfig(["c"], 4, "skip")],
    )
    assert config.residual_names == frozenset({"c"})
