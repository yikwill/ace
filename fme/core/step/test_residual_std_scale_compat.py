import pytest
from torch import nn

from fme.core.normalizer import NetworkAndLossNormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.step.single_module import ResidualPredictionConfig, SingleModuleStepConfig
from fme.core.testing import trivial_normalization


def _state(**extra) -> dict:
    config = SingleModuleStepConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": nn.Identity()}),
        in_names=["a"],
        out_names=["a"],
        normalization=NetworkAndLossNormalizationConfig(
            network=trivial_normalization(["a"]),
            residual=trivial_normalization(["a"]),
        ),
        residual_prediction=ResidualPredictionConfig(),
    )
    state = config.get_state()
    # YAML written before residual_prediction grew options used a bool.
    state["residual_prediction"] = True
    state.update(extra)
    return state


def test_scale_residual_by_residual_std_sets_normalized():
    config = SingleModuleStepConfig.from_state(
        _state(scale_residual_by_residual_std=True)
    )
    assert config.residual_prediction is not None
    assert config.residual_prediction.normalized is True


def test_scale_residual_by_residual_std_requires_residual_prediction():
    state = _state(scale_residual_by_residual_std=True)
    state["residual_prediction"] = False
    with pytest.raises(ValueError, match="requires residual_prediction"):
        SingleModuleStepConfig.from_state(state)


def test_false_scale_flag_is_ignored():
    config = SingleModuleStepConfig.from_state(
        _state(scale_residual_by_residual_std=False)
    )
    assert config.residual_prediction is not None
    assert config.residual_prediction.normalized is False
