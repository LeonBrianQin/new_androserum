from __future__ import annotations

import torch

from androserum.train.dataset import (
    ContrastiveMethodDataset,
    MethodTextSample,
    normalize_susi_label,
)
from androserum.train.losses import (
    abe_contrastive_loss,
    abce_contrastive_loss,
    ab_contrastive_loss,
    build_ab_positive_mask,
    build_abe_positive_mask,
    build_abce_positive_mask,
    count_b_positive_pairs,
    count_c_positive_pairs,
    count_e_positive_pairs,
)
from androserum.train.samplers import PositiveGroupBatchSampler, SusiGroupBatchSampler


def test_normalize_susi_label_filters_missing_and_no_category():
    assert normalize_susi_label(None) is None
    assert normalize_susi_label("") is None
    assert normalize_susi_label("   ") is None
    assert normalize_susi_label("NO_CATEGORY") is None
    assert normalize_susi_label("network") == "network"


def test_dataset_stats_and_usable_groups():
    ds = ContrastiveMethodDataset(
        samples=[
            MethodTextSample("A" * 64, "id0", "x", "NETWORK", ["Ljava/lang/Runnable;->run()V"], ["EXACT_FULL_ID::same"]),
            MethodTextSample("A" * 64, "id1", "y", "NETWORK", ["Ljava/lang/Runnable;->run()V"], ["EXACT_FULL_ID::same"]),
            MethodTextSample("B" * 64, "id2", "z", "FILE", [], []),
            MethodTextSample("B" * 64, "id3", "w", None, [], []),
        ],
        source_shas=["A" * 64, "B" * 64],
        skipped_filtered=3,
        skipped_unlabeled=5,
    )
    stats = ds.stats()
    assert stats["samples_total"] == 4
    assert stats["samples_labeled"] == 3
    assert stats["samples_unlabeled"] == 1
    assert stats["susi_labels_total"] == 2
    assert stats["susi_labels_usable"] == 1
    assert stats["override_keys_total"] == 1
    assert stats["override_keys_usable"] == 1
    assert stats["library_keys_total"] == 1
    assert stats["library_keys_usable"] == 1
    assert ds.usable_label_to_indices == {"NETWORK": [0, 1]}
    assert ds.usable_override_to_indices == {"Ljava/lang/Runnable;->run()V": [0, 1]}
    assert ds.usable_library_to_indices == {"EXACT_FULL_ID::same": [0, 1]}


def test_build_ab_positive_mask_contains_a_and_b_links():
    labels = ["NETWORK", "NETWORK", None]
    mask = build_ab_positive_mask(labels)
    assert mask.shape == (6, 6)

    # A positives: view-1 <-> view-2 for the same sample.
    assert mask[0, 3]
    assert mask[3, 0]
    assert mask[1, 4]
    assert mask[4, 1]
    assert mask[2, 5]
    assert mask[5, 2]

    # B positives: same SuSi label across different methods.
    assert mask[0, 1]
    assert mask[0, 4]
    assert mask[3, 1]
    assert mask[3, 4]

    # Different labels should not be linked by B.
    assert not mask[0, 2]
    assert not mask[0, 5]


def test_count_b_positive_pairs():
    assert count_b_positive_pairs([None, None, None]) == 0
    assert count_b_positive_pairs(["A", "A", None]) == 1
    assert count_b_positive_pairs(["A", "A", "A"]) == 3
    assert count_b_positive_pairs(["A", "A", "B", "B"]) == 2


def test_build_abe_positive_mask_contains_e_links():
    labels = [None, None, None]
    override_keys = [["K1"], ["K1"], []]
    mask = build_abe_positive_mask(labels, override_keys)
    # A positives still exist.
    assert mask[0, 3]
    assert mask[1, 4]
    # E positives: shared override target.
    assert mask[0, 1]
    assert mask[0, 4]
    assert mask[3, 1]
    assert mask[3, 4]
    # No E link to the third sample.
    assert not mask[0, 2]
    assert not mask[0, 5]


def test_count_e_positive_pairs():
    assert count_e_positive_pairs([[], [], []]) == 0
    assert count_e_positive_pairs([["K1"], ["K1"], []]) == 1
    assert count_e_positive_pairs([["K1"], ["K1"], ["K1"]]) == 3
    assert count_e_positive_pairs([["K1", "K2"], ["K2"], ["K1"]]) == 2


def test_build_abce_positive_mask_contains_c_links():
    labels = [None, None, None]
    override_keys = [[], [], []]
    library_keys = [["L1"], ["L1"], []]
    mask = build_abce_positive_mask(labels, override_keys, library_keys)
    assert mask[0, 3]
    assert mask[1, 4]
    assert mask[0, 1]
    assert mask[0, 4]
    assert not mask[0, 2]


def test_count_c_positive_pairs():
    assert count_c_positive_pairs([[], [], []]) == 0
    assert count_c_positive_pairs([["L1"], ["L1"], []]) == 1
    assert count_c_positive_pairs([["L1"], ["L1"], ["L1"]]) == 3


def test_ab_contrastive_loss_is_finite():
    torch.manual_seed(0)
    view1 = torch.randn(4, 8)
    view2 = torch.randn(4, 8)
    labels = ["NETWORK", "NETWORK", None, "FILE"]
    loss = ab_contrastive_loss(view1, view2, labels)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_abe_contrastive_loss_is_finite():
    torch.manual_seed(0)
    view1 = torch.randn(4, 8)
    view2 = torch.randn(4, 8)
    labels = ["NETWORK", None, None, "FILE"]
    override_keys = [[], ["K1"], ["K1"], []]
    loss = abe_contrastive_loss(view1, view2, labels, override_keys)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_abce_contrastive_loss_is_finite():
    torch.manual_seed(0)
    view1 = torch.randn(4, 8)
    view2 = torch.randn(4, 8)
    labels = ["NETWORK", None, None, "FILE"]
    override_keys = [[], ["K1"], ["K1"], []]
    library_keys = [[], ["L1"], ["L1"], []]
    loss = abce_contrastive_loss(view1, view2, labels, override_keys, library_keys)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_group_sampler_emits_at_least_one_label_pair_per_batch():
    sampler = SusiGroupBatchSampler(
        all_indices=list(range(10)),
        label_to_indices={"NETWORK": [0, 1, 2], "FILE": [3, 4, 5]},
        batch_size=6,
        label_group_size=2,
        label_fraction=0.5,
        steps_per_epoch=5,
        seed=7,
    )
    for batch in sampler:
        assert len(batch) == 6
        network = sum(i in {0, 1, 2} for i in batch)
        file_ = sum(i in {3, 4, 5} for i in batch)
        assert network >= 2 or file_ >= 2


def test_positive_group_sampler_can_draw_from_override_groups():
    sampler = PositiveGroupBatchSampler(
        all_indices=list(range(10)),
        group_maps=[
            {"NETWORK": [0, 1, 2]},
            {"Ljava/lang/Runnable;->run()V": [6, 7, 8]},
        ],
        batch_size=6,
        group_size=2,
        grouped_fraction=0.5,
        steps_per_epoch=5,
        seed=7,
    )
    for batch in sampler:
        assert len(batch) == 6
        network = sum(i in {0, 1, 2} for i in batch)
        runnable = sum(i in {6, 7, 8} for i in batch)
        assert network >= 2 or runnable >= 2


def test_positive_group_sampler_can_draw_from_library_groups():
    sampler = PositiveGroupBatchSampler(
        all_indices=list(range(10)),
        group_maps=[
            {"NETWORK": [0, 1, 2]},
            {"EXACT_FULL_ID::same": [6, 7, 8]},
        ],
        batch_size=6,
        group_size=2,
        grouped_fraction=0.5,
        steps_per_epoch=5,
        seed=11,
    )
    for batch in sampler:
        assert len(batch) == 6
        network = sum(i in {0, 1, 2} for i in batch)
        libsame = sum(i in {6, 7, 8} for i in batch)
        assert network >= 2 or libsame >= 2
