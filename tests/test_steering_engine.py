from __future__ import annotations

from dataclasses import dataclass

import pytest

from mlx_steer.steering.injector import SteeringEngine


@dataclass
class DummyLayer:
    def __call__(self, x):
        return x


@dataclass
class DummyInnerModel:
    layers: list


@dataclass
class DummyModel:
    model: DummyInnerModel


def test_steering_engine_enable_disable_restores_layer():
    base_layer = DummyLayer()
    m = DummyModel(model=DummyInnerModel(layers=[base_layer]))

    eng = SteeringEngine(m, layer_idx=0)

    with pytest.raises(ValueError):
        eng.enable()

    eng.set_vector(3)
    eng.enable()

    # Layer is replaced by wrapper; calling it adds the vector.
    assert m.model.layers[0](10) == 13

    eng.disable()
    assert m.model.layers[0] is base_layer
    assert m.model.layers[0](10) == 10

