from io import StringIO

import pytest

from pyGATH.reporting import (
    ProgressTracker,
    ReportingConfig,
    SimulationReporter,
    emit,
    get_reporter,
    reported_stage,
)


def test_reporter_formats_relative_time_and_filters_progress():
    stream = StringIO()
    reporter = SimulationReporter(
        ReportingConfig(verbosity=1, console=True), stream=stream
    )

    with reporter:
        emit(1, "build grid", "complete")
        emit(2, "deposit", "1/10 operations")

    output = stream.getvalue()
    assert "[+00:00:00." in output
    assert "simulation: simulation started" in output
    assert "build grid: complete" in output
    assert "1/10 operations" not in output
    assert "simulation: simulation complete" in output
    assert get_reporter() is None


def test_reporter_writes_progress_and_eta_to_file(tmp_path):
    path = tmp_path / "logs" / "simulation.log"
    reporter = SimulationReporter(
        ReportingConfig(
            verbosity=2,
            console=False,
            file=path,
            progress_interval_s=1.0e-9,
        )
    )

    with reporter:
        tracker = ProgressTracker("deposition", 4, "operations")
        tracker.update(2, force=True)
        reporter.callback(
            {
                "stage": "overlap",
                "status": "running",
                "message": "candidate pairs 3/8",
                "estimated_remaining_s": 2.5,
            }
        )

    output = path.read_text(encoding="utf-8")
    assert "deposition: 2/4 operations; ETA" in output
    assert "overlap: candidate pairs 3/8; ETA 00:00:02.500" in output


def test_reported_stage_reports_failure_and_reraises():
    stream = StringIO()

    @reported_stage("failing stage", synchronize_result=False)
    def fail():
        raise RuntimeError("broken")

    with (
        pytest.raises(RuntimeError, match="broken"),
        SimulationReporter(ReportingConfig(verbosity=1, console=True), stream=stream),
    ):
        fail()

    output = stream.getvalue()
    assert "failing stage: started" in output
    assert "failing stage: failed after" in output
    assert "simulation: simulation failed: RuntimeError: broken" in output
