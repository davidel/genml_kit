"""Tests for the COCO-style mAP meter, including hand-computed AP cases."""

import torch

from genml_kit.training.det_metrics import MAPMeter, _average_precision


class TestAveragePrecision:

  def test_perfect_ranking(self):
    # One GT, one detection at the top: precision 1, recall 1 -> AP 1.
    assert _average_precision([1.0], [1.0]) == 1.0

  def test_half_recall(self):
    # Two GT, one correct detection: AP = 0.5.
    ap = _average_precision([0.5], [1.0])
    assert abs(ap - 0.5) < 1e-9

  def test_false_positive_first(self):
    # FP then TP against one GT: precision at recall 1 is 0.5 -> AP 0.5.
    ap = _average_precision([0.5, 1.0], [0.0, 0.5])
    assert abs(ap - 0.5) < 1e-9

  def test_monotonic_precision_envelope(self):
    # Envelope: precision at recall 0.5 is max(1.0, 0.5)=1.0; at recall 1.0
    # it is 0.5. Area = 0.5*1.0 + 0.5*0.5 = 0.75.
    ap = _average_precision([0.5, 1.0], [1.0, 0.5])
    assert abs(ap - 0.75) < 1e-9


class TestMAPMeter:

  def _perfect_case(self, num_classes=1):
    meter = MAPMeter(num_classes=num_classes)
    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    gt_labels = torch.tensor([0])
    # A detection with a perfect IoU and a high score.
    meter.update(torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([0.9]),
                 torch.tensor([0]), gt_boxes, gt_labels)
    return meter

  def test_perfect_detection_gives_map_one(self):
    metrics = self._perfect_case().compute()
    assert abs(metrics["map"] - 1.0) < 1e-6
    assert abs(metrics["map_50"] - 1.0) < 1e-6

  def test_no_detections_gives_map_zero(self):
    meter = MAPMeter(num_classes=1)
    meter.update(torch.zeros((0, 4)), torch.zeros((0,)),
                 torch.zeros((0,), dtype=torch.long),
                 torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([0]))
    metrics = meter.compute()
    assert metrics["map"] == 0.0

  def test_no_ground_truth_gives_map_zero(self):
    # A class absent from the ground truth contributes no AP; with no class
    # present at all, the mean is defined as 0.
    meter = MAPMeter(num_classes=1)
    meter.update(torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([0.9]),
                 torch.tensor([0]), torch.zeros((0, 4)),
                 torch.zeros((0,), dtype=torch.long))
    assert meter.compute()["map"] == 0.0

  def test_low_iou_fails_high_thresholds(self):
    # IoU ~0.36: passes 0.5? No. Only thresholds <= 0.35 would pass, and the
    # smallest tracked threshold is 0.5, so every AP is 0.
    meter = MAPMeter(num_classes=1)
    meter.update(torch.tensor([[5.0, 5.0, 15.0, 15.0]]), torch.tensor([0.9]),
                 torch.tensor([0]), torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                 torch.tensor([0]))
    metrics = meter.compute()
    assert metrics["map"] == 0.0
    assert metrics["map_50"] == 0.0

  def test_duplicate_detection_counts_as_false_positive(self):
    # Two GTs, but both detections pile onto the first box (the second is a
    # duplicate -> FP) and the second GT is missed. recall tops out at 0.5, so
    # AP = 0.5. (A single GT would *not* be penalised: its recall hits 1.0 on
    # the first match, where the all-point envelope still sees precision 1.0.)
    meter = MAPMeter(num_classes=1)
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([0, 0])
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    meter.update(boxes, scores, labels, gt, torch.tensor([0, 0]))
    assert abs(meter.compute()["map"] - 0.5) < 1e-6

  def test_two_classes_averaged(self):
    meter = MAPMeter(num_classes=2)
    # Class 0 detected perfectly, class 1 missed entirely -> mean AP 0.5.
    meter.update(torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([0.9]),
                 torch.tensor([0]),
                 torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
                 torch.tensor([0, 1]))
    metrics = meter.compute()
    assert abs(metrics["map"] - 0.5) < 1e-6

  def test_reset_clears_state(self):
    meter = self._perfect_case()
    meter.reset()
    # After reset there is no data; every class is absent -> map 0.
    assert meter.compute()["map"] == 0.0

  def test_map_75_present_when_tracked(self):
    metrics = self._perfect_case().compute()
    assert "map_75" in metrics
    assert abs(metrics["map_75"] - 1.0) < 1e-6

  def test_custom_thresholds(self):
    meter = MAPMeter(num_classes=1, iou_thresholds=[0.5])
    meter.update(torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([0.9]),
                 torch.tensor([0]), torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                 torch.tensor([0]))
    assert abs(meter.compute()["map"] - 1.0) < 1e-6
