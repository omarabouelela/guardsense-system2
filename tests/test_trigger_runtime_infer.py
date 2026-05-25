from __future__ import annotations

import numpy as np

from src.data.pose_preprocess import NormalizationConfig, SequenceAssemblyConfig
from src.trigger.infer import aggregate_window_probabilities, prepare_trigger_windows


def test_prepare_trigger_windows_matches_training_temporal_config() -> None:
    sequence = np.zeros((48, 17, 3), dtype=np.float32)
    sequence[..., 0] = 960.0
    sequence[..., 1] = 540.0
    sequence[..., 2] = 1.0

    windows = prepare_trigger_windows(
        sequence,
        normalization=NormalizationConfig(mode="frame", frame_width=1920, frame_height=1080),
        temporal=SequenceAssemblyConfig(window_size=32, overlap=0.5, short_policy="pad"),
    )

    assert windows.shape == (2, 32, 17, 3)
    np.testing.assert_allclose(windows[..., 0], 0.5)
    np.testing.assert_allclose(windows[..., 1], 0.5)
    np.testing.assert_allclose(windows[..., 2], 1.0)


def test_aggregate_window_probabilities_uses_max_fight_for_binary_event() -> None:
    window_probs = np.asarray(
        [
            [0.90, 0.10],
            [0.20, 0.80],
            [0.40, 0.60],
            [0.30, 0.70],
        ],
        dtype=np.float32,
    )

    output = aggregate_window_probabilities(
        window_probs,
        num_classes=2,
        fight_class_id=1,
        top_k=2,
        include_window_debug=True,
    )

    assert output["number_of_windows"] == 4
    assert output["predicted_label"] == 1
    assert output["trigger_label"] == 1
    np.testing.assert_allclose(output["class_probabilities"]["1"], 0.8)
    np.testing.assert_allclose(output["class_probabilities"]["0"], 0.2)
    np.testing.assert_allclose(output["max_fight_probability"], 0.8)
    np.testing.assert_allclose(output["topk_mean_fight_probability"], 0.75)
    assert [row["window_index"] for row in output["window_debug"]] == [0, 1, 2, 3]
    np.testing.assert_allclose([row["prob_1"] for row in output["window_debug"]], [0.1, 0.8, 0.6, 0.7])
