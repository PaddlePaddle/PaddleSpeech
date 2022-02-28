import pytest


def test_hpopt_generic():
    from io import StringIO
    from speechbrain.utils import hpopt as hp
    import json

    output = StringIO()

    reporter = hp.GenericHyperparameterOptimizationReporter(
        objective_key="per", output=output
    )
    result = {"train_loss": 0.9, "valid_loss": 1.2, "per": 0.10}
    reporter.report_objective(result)
    output.seek(0)
    output_result = json.load(output)
    assert output_result["train_loss"] == pytest.approx(0.9)
    assert output_result["valid_loss"] == pytest.approx(1.2)
    assert output_result["per"] == pytest.approx(0.10)
    assert output_result["objective"] == pytest.approx(0.10)


def test_hpopt_orion():
    from speechbrain.utils import hpopt as hp

    results = {}

    class MockOrion:
        def report_objective(self, value):
            results["value"] = value

    mock_orion = MockOrion()

    reporter = hp.OrionHyperparameterOptimizationReporter(
        objective_key="valid_loss"
    )
    reporter.orion_client = mock_orion

    result = {"train_loss": 0.9, "valid_loss": 1.2, "per": 0.10}
    reporter.report_objective(result)
    assert results["value"] == pytest.approx(1.2)


def test_hpopt_context():
    import json
    from speechbrain.utils import hpopt as hp
    from io import StringIO

    output = StringIO()
    reporter = hp.GenericHyperparameterOptimizationReporter(
        objective_key="per", output=output
    )

    with hp.hyperparameter_optimization() as hp_ctx:
        hp_ctx.reporter = reporter
        result = {"per": 10, "loss": 1.2}
        hp.report_result(result)

        result = {"per": 3, "loss": 1.3}
        hp.report_result(result)

    output.seek(0)
    output_result = json.load(output)
    assert output_result["per"] == 3
