import paddle
import paddle.nn
import math


def test_metric_stats(device):
    paddle.device.set_device(device)
    from speechbrain.utils.metric_stats import MetricStats
    from speechbrain.nnet.losses import l1_loss

    l1_stats = MetricStats(metric=l1_loss)
    l1_stats.append(
        ids=["utterance1", "utterance2"],
        predictions=paddle.to_tensor([[0.1, 0.2], [0.1, 0.2]], ),
        targets=paddle.to_tensor([[0.1, 0.3], [0.2, 0.3]], ),
        length=paddle.ones([2], ),
        reduction="batch",
    )
    summary = l1_stats.summarize()
    assert math.isclose(summary["average"], 0.075, rel_tol=1e-5)
    assert math.isclose(summary["min_score"], 0.05, rel_tol=1e-5)
    assert summary["min_id"] == "utterance1"
    assert math.isclose(summary["max_score"], 0.1, rel_tol=1e-5)
    assert summary["max_id"] == "utterance2"


def test_error_rate_stats(device):
    paddle.device.set_device(device)
    from speechbrain.utils.metric_stats import ErrorRateStats

    wer_stats = ErrorRateStats()
    i2l = {1: "hello", 2: "world", 3: "the"}

    def mapper(batch):
        return [[i2l[int(x)] for x in seq] for seq in batch]

    wer_stats.append(
        ids=["utterance1", "utterance2"],
        predict=[[3, 2, 1], [2, 3]],
        target=paddle.to_tensor([[3, 2, 0], [2, 1, 0]], ),
        target_len=paddle.to_tensor([0.67, 0.67], ),
        ind2lab=mapper,
    )
    summary = wer_stats.summarize()
    assert summary["WER"] == 50.0
    assert summary["insertions"] == 1
    assert summary["substitutions"] == 1
    assert summary["deletions"] == 0
    assert wer_stats.scores[0]["ref_tokens"] == ["the", "world"]
    assert wer_stats.scores[0]["hyp_tokens"] == ["the", "world", "hello"]


def test_binary_metrics(device):
    paddle.device.set_device(device)
    from speechbrain.utils.metric_stats import BinaryMetricStats

    binary_stats = BinaryMetricStats()
    binary_stats.append(
        ids=["utt1", "utt2", "utt3", "utt4", "utt5", "utt6"],
        scores=paddle.to_tensor([0.1, 0.4, 0.8, 0.2, 0.3, 0.6], ),
        labels=paddle.to_tensor([1, 0, 1, 0, 1, 0], ),
    )
    summary = binary_stats.summarize(threshold=0.5)
    assert summary["TP"] == 1
    assert summary["TN"] == 2
    assert summary["FP"] == 1
    assert summary["FN"] == 2

    summary = binary_stats.summarize(threshold=None)
    assert summary["threshold"] >= 0.3 and summary["threshold"] < 0.4

    summary = binary_stats.summarize(threshold=None, max_samples=1)
    assert summary["threshold"] >= 0.1 and summary["threshold"] < 0.2


def test_EER(device):
    paddle.device.set_device(device)
    from speechbrain.utils.metric_stats import EER

    positive_scores = paddle.to_tensor([0.1, 0.2, 0.3], )
    negative_scores = paddle.to_tensor([0.4, 0.5, 0.6], )
    eer, threshold = EER(positive_scores, negative_scores)
    assert eer == 1.0
    assert threshold > 0.3 and threshold < 0.4

    positive_scores = paddle.to_tensor([0.4, 0.5, 0.6], )
    negative_scores = paddle.to_tensor([0.3, 0.2, 0.1], )
    eer, threshold = EER(positive_scores, negative_scores)
    assert eer == 0
    assert threshold > 0.3 and threshold < 0.4

    cos = paddle.nn.CosineSimilarity(axis=1, eps=1e-6)
    input1 = paddle.randn([1000, 64], )
    input2 = paddle.randn([1000, 64], )
    positive_scores = cos(input1, input2)

    input1 = paddle.randn([1000, 64], )
    input2 = paddle.randn([1000, 64], )
    negative_scores = cos(input1, input2)

    eer, threshold = EER(positive_scores, negative_scores)

    correct = (positive_scores > threshold).nonzero(as_tuple=False).shape[0] + (
        negative_scores < threshold
    ).nonzero(as_tuple=False).shape[0]

    assert correct > 900 and correct < 1100


def test_minDCF(device):
    paddle.device.set_device(device)
    from speechbrain.utils.metric_stats import minDCF

    positive_scores = paddle.to_tensor([0.1, 0.2, 0.3], )
    negative_scores = paddle.to_tensor([0.4, 0.5, 0.6], )
    min_dcf, threshold = minDCF(positive_scores, negative_scores)
    assert (0.01 - min_dcf) < 1e-4
    assert threshold >= 0.6

    positive_scores = paddle.to_tensor([0.4, 0.5, 0.6], )
    negative_scores = paddle.to_tensor([0.1, 0.2, 0.3], )
    min_dcf, threshold = minDCF(positive_scores, negative_scores)
    assert min_dcf == 0
    assert threshold > 0.3 and threshold < 0.4
