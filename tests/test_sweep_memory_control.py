"""
Tests for Phase 6 sweep memory control features:
  1. _cleanup_petsc_objects — PETSc garbage + Python GC
  2. _get_process_memory_mb — RSS reporting for current process
  3. _get_pid_rss_mb — RSS reporting for arbitrary PID
  4. _run_in_subprocess — subprocess execution with memory watchdog
  5. _sweep_worker_main — subprocess worker entry point
  6. run_phase_6 — skip/resume, summary output, subprocess dispatch
  7. parse_args — --mem-limit-gb CLI argument
"""

import argparse
import gc
import json
import os
import platform
import signal
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path
from unittest import mock

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "experiments" / "shinnecock_study"))

from experiments.shinnecock_study.run_comparison import (
    _cleanup_petsc_objects,
    _get_pid_rss_mb,
    _get_process_memory_mb,
    _run_in_subprocess,
    _sweep_worker_main,
    MEM_WATCHDOG_INTERVAL_S,
    SWEEP_BATCH_SIZE,
    SWEEP_BASELINE,
    SWEEP_DIMS,
)


# ============================================================================
# Tests for _cleanup_petsc_objects
# ============================================================================


class TestCleanupPetscObjects:
    """Tests for the _cleanup_petsc_objects helper."""

    def test_calls_gc_collect_twice(self):
        """GC should be collected at least twice (before and after PETSc cleanup)."""
        call_count = 0
        original_collect = gc.collect

        def counting_collect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_collect(*args, **kwargs)

        with mock.patch("gc.collect", side_effect=counting_collect):
            _cleanup_petsc_objects()

        assert call_count >= 2, (
            f"Expected gc.collect called ≥2 times, got {call_count}"
        )

    def test_calls_petsc_garbage_cleanup(self):
        """Should call PETSc.garbage_cleanup() when petsc4py is available."""
        try:
            from petsc4py import PETSc
        except ImportError:
            pytest.skip("petsc4py not available")

        with mock.patch.object(PETSc, "garbage_cleanup", wraps=PETSc.garbage_cleanup) as spy:
            _cleanup_petsc_objects()

        spy.assert_called()

    def test_survives_missing_petsc(self):
        """Should not raise if petsc4py is not installed."""
        with mock.patch.dict("sys.modules", {"petsc4py": None}):
            _cleanup_petsc_objects()

    def test_survives_petsc_exception(self):
        """Should not raise if PETSc.garbage_cleanup raises."""
        mock_petsc = mock.MagicMock()
        mock_petsc.garbage_cleanup.side_effect = RuntimeError("PETSc error")
        with mock.patch.dict("sys.modules", {"petsc4py": mock.MagicMock(),
                                              "petsc4py.PETSc": mock_petsc}):
            _cleanup_petsc_objects()  # Should not raise

    def test_is_idempotent(self):
        """Calling multiple times should not raise or change behavior."""
        for _ in range(5):
            _cleanup_petsc_objects()


# ============================================================================
# Tests for _get_process_memory_mb
# ============================================================================


class TestGetProcessMemoryMb:
    """Tests for the _get_process_memory_mb helper."""

    def test_returns_positive_float(self):
        """Should return a positive number on any system with resource module."""
        result = _get_process_memory_mb()
        if platform.system() in ("Darwin", "Linux"):
            assert result is not None
            assert isinstance(result, float)
            assert result > 0, "RSS should be positive for a running process"

    def test_returns_reasonable_value(self):
        """RSS should be at least a few MB (Python itself uses ~20+ MB)."""
        result = _get_process_memory_mb()
        if result is not None:
            assert result > 1.0, f"RSS {result} MB seems unreasonably low"
            assert result < 10_000, f"RSS {result} MB seems unreasonably high"

    def test_darwin_uses_bytes_divisor(self):
        """On macOS, maxrss is in bytes so we divide by 1024*1024."""
        mock_usage = mock.MagicMock()
        mock_usage.ru_maxrss = 100 * 1024 * 1024  # 100 MB in bytes

        with mock.patch("platform.system", return_value="Darwin"):
            with mock.patch("resource.getrusage", return_value=mock_usage):
                result = _get_process_memory_mb()

        assert result == pytest.approx(100.0)

    def test_linux_uses_kb_divisor(self):
        """On Linux, maxrss is in KB so we divide by 1024."""
        mock_usage = mock.MagicMock()
        mock_usage.ru_maxrss = 100 * 1024  # 100 MB in KB

        with mock.patch("platform.system", return_value="Linux"):
            with mock.patch("resource.getrusage", return_value=mock_usage):
                result = _get_process_memory_mb()

        assert result == pytest.approx(100.0)

    def test_returns_none_on_exception(self):
        """Should return None gracefully if resource module fails."""
        with mock.patch("resource.getrusage", side_effect=OSError("no resource")):
            result = _get_process_memory_mb()
        assert result is None


# ============================================================================
# Tests for _get_pid_rss_mb
# ============================================================================


class TestGetPidRssMb:
    """Tests for the _get_pid_rss_mb helper (external PID RSS monitoring)."""

    def test_returns_positive_for_own_pid(self):
        """Should return a positive value for the current process."""
        result = _get_pid_rss_mb(os.getpid())
        assert result is not None
        assert result > 0

    def test_returns_reasonable_value_for_own_pid(self):
        """Own process RSS should be at least a few MB."""
        result = _get_pid_rss_mb(os.getpid())
        if result is not None:
            assert result > 1.0
            assert result < 10_000

    def test_returns_none_for_nonexistent_pid(self):
        """Should return None for a PID that doesn't exist."""
        # PID 99999999 is extremely unlikely to exist
        result = _get_pid_rss_mb(99999999)
        assert result is None

    def test_returns_value_for_child_process(self):
        """Should be able to monitor a child subprocess."""
        # Start a child that sleeps briefly
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            time.sleep(0.5)  # Let it start
            result = _get_pid_rss_mb(child.pid)
            assert result is not None
            assert result > 0
        finally:
            child.kill()
            child.wait()

    def test_falls_back_to_ps_when_psutil_unavailable(self):
        """Should use ps fallback when psutil import fails."""
        with mock.patch.dict("sys.modules", {"psutil": None}):
            # Force psutil import to fail inside _get_pid_rss_mb
            result = _get_pid_rss_mb(os.getpid())
            # ps fallback should still work on macOS/Linux
            if platform.system() in ("Darwin", "Linux"):
                assert result is not None
                assert result > 0


# ============================================================================
# Tests for _run_in_subprocess
# ============================================================================


class TestRunInSubprocess:
    """Tests for subprocess execution with memory watchdog."""

    def test_successful_child_returns_result(self, tmp_path):
        """A child that writes a valid result file should return it."""
        result_file = tmp_path / "result.json"
        expected = {"status": "success", "method": "4dvar",
                    "results": {"error_reduction": 8.5}}

        # Create a tiny script that writes the result and exits
        worker_script = tmp_path / "worker.py"
        worker_script.write_text(textwrap.dedent(f"""\
            import json, sys
            result_file = "{result_file}"
            result = {json.dumps(expected)}
            with open(result_file, "w") as f:
                json.dump(result, f)
            sys.exit(0)
        """))

        run_config = {
            "result_file": str(result_file),
            "phase_prefix": "6_test_",
            "sub_label": "a",
            "method": "4dvar",
        }

        # Use the worker script as the "script_path" but override the command
        # by mocking Popen to run our custom worker
        with mock.patch("subprocess.Popen") as mock_popen:
            mock_proc = mock.MagicMock()
            mock_proc.pid = os.getpid()  # Use own PID so watchdog can read RSS
            mock_proc.stdout = iter([])  # No output
            mock_proc.poll = mock.MagicMock(return_value=0)  # Already done
            mock_proc.wait = mock.MagicMock(return_value=0)
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc

            # Pre-write the result file (simulating child writing it)
            result_file.write_text(json.dumps(expected))

            result = _run_in_subprocess(run_config, mem_limit_mb=12000,
                                        script_path=str(worker_script))

        assert result["status"] == "success"
        assert result["results"]["error_reduction"] == 8.5

    def test_child_crash_returns_failure(self, tmp_path):
        """A child that exits with non-zero code should return a failure dict."""
        result_file = tmp_path / "result.json"

        run_config = {
            "result_file": str(result_file),
            "phase_prefix": "6_test_",
            "sub_label": "a",
            "method": "4dvar",
            "dim_name": "noise",
            "param_name": "obs_noise_level",
            "val": 0.01,
        }

        with mock.patch("subprocess.Popen") as mock_popen:
            mock_proc = mock.MagicMock()
            mock_proc.pid = os.getpid()
            mock_proc.stdout = iter(["Traceback...\n", "RuntimeError: boom\n"])
            mock_proc.poll = mock.MagicMock(return_value=1)
            mock_proc.wait = mock.MagicMock(return_value=1)
            mock_proc.returncode = 1
            mock_popen.return_value = mock_proc

            result = _run_in_subprocess(run_config, mem_limit_mb=12000,
                                        script_path="dummy.py")

        assert result["status"] == "failed"
        assert "code 1" in result["error"]
        # Should have written failure JSON
        assert result_file.exists()

    def test_watchdog_kills_memory_hog(self, tmp_path):
        """Watchdog should SIGKILL a child whose RSS exceeds the limit."""
        result_file = tmp_path / "result.json"

        run_config = {
            "result_file": str(result_file),
            "phase_prefix": "6_test_",
            "sub_label": "a",
            "method": "4dvar",
            "dim_name": "test",
            "param_name": "test_param",
            "val": 1.0,
        }

        # Launch a real child that just sleeps (simulating a long run)
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

        try:
            # Very low memory limit (1 MB) — any Python process exceeds this
            # Patch _get_pid_rss_mb to report high RSS immediately
            with mock.patch(
                "experiments.shinnecock_study.run_comparison._get_pid_rss_mb",
                return_value=5000.0,  # 5 GB
            ):
                # Patch Popen to return our real child
                with mock.patch("subprocess.Popen", return_value=child):
                    result = _run_in_subprocess(run_config, mem_limit_mb=1.0,
                                                script_path="dummy.py")

            assert result["status"] == "failed"
            assert "watchdog" in result["error"].lower()
            assert result_file.exists()
        finally:
            try:
                child.kill()
            except OSError:
                pass
            child.wait()

    def test_watchdog_interval_is_reasonable(self):
        """Watchdog check interval should be frequent enough to catch runaways."""
        assert MEM_WATCHDOG_INTERVAL_S <= 10
        assert MEM_WATCHDOG_INTERVAL_S >= 1

    def test_missing_result_file_returns_failure(self, tmp_path):
        """If child exits 0 but writes no result file, return failure."""
        result_file = tmp_path / "nonexistent_result.json"

        run_config = {
            "result_file": str(result_file),
            "phase_prefix": "6_test_",
            "sub_label": "a",
            "method": "4dvar",
        }

        with mock.patch("subprocess.Popen") as mock_popen:
            mock_proc = mock.MagicMock()
            mock_proc.pid = os.getpid()
            mock_proc.stdout = iter([])
            mock_proc.poll = mock.MagicMock(return_value=0)
            mock_proc.wait = mock.MagicMock(return_value=0)
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc

            result = _run_in_subprocess(run_config, mem_limit_mb=12000,
                                        script_path="dummy.py")

        assert result["status"] == "failed"
        assert "not found" in result["error"].lower()


# ============================================================================
# Tests for _sweep_worker_main
# ============================================================================


class TestSweepWorkerMain:
    """Tests for the subprocess worker entry point."""

    def test_calls_run_sub_experiment(self, tmp_path):
        """Worker should call _run_sub_experiment with correct args."""
        from experiments.shinnecock_study import run_comparison as rc

        result_file = tmp_path / "data" / "phase6_test_0p01_a_results.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "output_dir": str(tmp_path),
            "adios_file": "data/shinnecock_inlet",
            "sub_label": "a",
            "method": "4dvar",
            "nt_da": 12,
            "nt_ramp": 144,
            "phase_prefix": "6_test_0p01_",
            "sweep_params": {"obs_fraction": 0.1},
            "result_file": str(result_file),
            "dim_name": "test",
            "param_name": "obs_noise_level",
            "val": 0.01,
            "mem_limit_gb": 12.0,
        }

        fake_result = {"status": "success", "method": "4dvar",
                       "results": {"error_reduction": 5.0}}

        with mock.patch.object(rc, "_run_sub_experiment", return_value=fake_result):
            with pytest.raises(SystemExit) as exc_info:
                _sweep_worker_main(json.dumps(config))

        assert exc_info.value.code == 0

    def test_worker_saves_failure_on_exception(self, tmp_path):
        """Worker should save failure JSON and exit(1) when _run_sub_experiment raises."""
        from experiments.shinnecock_study import run_comparison as rc

        result_file = tmp_path / "data" / "phase6_test_a_results.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "output_dir": str(tmp_path),
            "adios_file": "data/shinnecock_inlet",
            "sub_label": "a",
            "method": "4dvar",
            "nt_da": 12,
            "nt_ramp": 144,
            "phase_prefix": "6_test_",
            "sweep_params": {},
            "result_file": str(result_file),
            "dim_name": "test",
            "param_name": "test",
            "val": 1.0,
            "mem_limit_gb": 12.0,
        }

        with mock.patch.object(rc, "_run_sub_experiment",
                               side_effect=RuntimeError("solver diverged")):
            with pytest.raises(SystemExit) as exc_info:
                _sweep_worker_main(json.dumps(config))

        assert exc_info.value.code == 1
        assert result_file.exists()
        with open(result_file) as f:
            data = json.load(f)
        assert data["status"] == "failed"
        assert "solver diverged" in data["error"]


# ============================================================================
# Tests for run_phase_6 orchestration
# ============================================================================


def _make_args(tmp_path, mem_limit_gb=12.0, sweep_dim="noise"):
    """Create a mock args namespace for run_phase_6."""
    return argparse.Namespace(
        phase="6",
        sweep_dim=sweep_dim,
        output_dir=str(tmp_path / "outputs"),
        adios_file="data/shinnecock_inlet",
        sub=None,
        verbose=True,
        mem_limit_gb=mem_limit_gb,
    )


def _make_import_patcher(mock_mpi):
    """Create a __import__ side_effect that intercepts mpi4py imports."""
    _real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def patched_import(name, *args, **kwargs):
        if name == "mpi4py" or name == "mpi4py.MPI":
            mod = types.ModuleType(name)
            mod.MPI = mock_mpi
            return mod
        return _real_import(name, *args, **kwargs)
    return patched_import


class TestRunPhase6SkipResume:
    """Tests that existing result files are skipped on re-run."""

    def test_skips_existing_result_files(self, tmp_path):
        """Runs that already have a result file should be skipped entirely."""
        from experiments.shinnecock_study import run_comparison as rc

        args = _make_args(tmp_path, sweep_dim="model_error")
        data_dir = Path(args.output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        values = SWEEP_DIMS["model_error"]["values"]
        for val in values:
            val_str = str(val).replace(".", "p")
            for sub in ["a", "b"]:
                fname = f"phase6_model_error_{val_str}_{sub}_results.json"
                result = {"status": "success", "method": "4dvar" if sub == "a" else "dcwme",
                          "results": {"error_reduction": 5.0}}
                (data_dir / fname).write_text(json.dumps(result))

        mock_comm = mock.MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_mpi = mock.MagicMock()
        mock_mpi.COMM_WORLD = mock_comm

        with mock.patch.object(rc, "_run_in_subprocess") as mock_run:
            with mock.patch("builtins.__import__", side_effect=_make_import_patcher(mock_mpi)):
                result = rc.run_phase_6(args)

        mock_run.assert_not_called()
        assert "model_error" in result
        assert len(result["model_error"]) == len(values)

    def test_resumes_after_partial_completion(self, tmp_path):
        """If only some result files exist, only the missing ones should run."""
        from experiments.shinnecock_study import run_comparison as rc

        args = _make_args(tmp_path, sweep_dim="cov_inflation")
        data_dir = Path(args.output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        values = SWEEP_DIMS["cov_inflation"]["values"]  # [1.0, 5.0, 10.0, 20.0]

        for val in values[:2]:
            val_str = str(val).replace(".", "p")
            for sub in ["a", "b"]:
                fname = f"phase6_cov_inflation_{val_str}_{sub}_results.json"
                result = {"status": "success", "method": "4dvar" if sub == "a" else "dcwme",
                          "results": {"error_reduction": 5.0}}
                (data_dir / fname).write_text(json.dumps(result))

        mock_comm = mock.MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_mpi = mock.MagicMock()
        mock_mpi.COMM_WORLD = mock_comm

        call_log = []

        def fake_subprocess(run_config, mem_limit_mb, script_path):
            call_log.append(run_config["phase_prefix"])
            result = {"status": "success", "method": run_config["method"],
                      "results": {"error_reduction": 3.0}}
            Path(run_config["result_file"]).write_text(json.dumps(result))
            return result

        with mock.patch.object(rc, "_run_in_subprocess", side_effect=fake_subprocess):
            with mock.patch("builtins.__import__", side_effect=_make_import_patcher(mock_mpi)):
                rc.run_phase_6(args)

        # 2 remaining values × 2 methods = 4 calls
        assert len(call_log) == 4
        prefixes = set(call_log)
        assert "6_cov_inflation_1p0_" not in prefixes
        assert "6_cov_inflation_5p0_" not in prefixes
        assert "6_cov_inflation_10p0_" in prefixes
        assert "6_cov_inflation_20p0_" in prefixes

    def test_failed_runs_skipped_on_rerun(self, tmp_path):
        """Failed result files should be skipped on re-run (user must delete to retry)."""
        from experiments.shinnecock_study import run_comparison as rc

        args = _make_args(tmp_path, sweep_dim="bg_error")
        data_dir = Path(args.output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        values = SWEEP_DIMS["bg_error"]["values"]
        for val in values:
            val_str = str(val).replace(".", "p")
            for sub in ["a", "b"]:
                fname = f"phase6_bg_error_{val_str}_{sub}_results.json"
                result = {"status": "failed", "error": "previous failure",
                          "method": "4dvar" if sub == "a" else "dcwme"}
                (data_dir / fname).write_text(json.dumps(result))

        mock_comm = mock.MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_mpi = mock.MagicMock()
        mock_mpi.COMM_WORLD = mock_comm

        with mock.patch.object(rc, "_run_in_subprocess") as mock_run:
            with mock.patch("builtins.__import__", side_effect=_make_import_patcher(mock_mpi)):
                rc.run_phase_6(args)

        mock_run.assert_not_called()


class TestRunPhase6SubprocessDispatch:
    """Tests that run_phase_6 dispatches to _run_in_subprocess correctly."""

    def test_passes_correct_config_to_subprocess(self, tmp_path):
        """run_config passed to _run_in_subprocess should have all required fields."""
        from experiments.shinnecock_study import run_comparison as rc

        args = _make_args(tmp_path, sweep_dim="obs_frequency", mem_limit_gb=10.0)
        data_dir = Path(args.output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        mock_comm = mock.MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_mpi = mock.MagicMock()
        mock_mpi.COMM_WORLD = mock_comm

        captured_configs = []

        def capture_subprocess(run_config, mem_limit_mb, script_path):
            captured_configs.append(run_config)
            result = {"status": "success", "method": run_config["method"],
                      "results": {"error_reduction": 3.0}}
            Path(run_config["result_file"]).write_text(json.dumps(result))
            return result

        with mock.patch.object(rc, "_run_in_subprocess", side_effect=capture_subprocess):
            with mock.patch("builtins.__import__", side_effect=_make_import_patcher(mock_mpi)):
                rc.run_phase_6(args)

        assert len(captured_configs) > 0

        # Check that each config has the required keys
        required_keys = {"output_dir", "adios_file", "sub_label", "method",
                         "nt_da", "nt_ramp", "phase_prefix", "sweep_params",
                         "result_file", "dim_name", "param_name", "val", "mem_limit_gb"}
        for config in captured_configs:
            assert required_keys.issubset(config.keys()), (
                f"Missing keys: {required_keys - config.keys()}"
            )

    def test_passes_correct_mem_limit(self, tmp_path):
        """Memory limit passed to _run_in_subprocess should match --mem-limit-gb."""
        from experiments.shinnecock_study import run_comparison as rc

        args = _make_args(tmp_path, sweep_dim="obs_frequency", mem_limit_gb=8.0)
        data_dir = Path(args.output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        mock_comm = mock.MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_mpi = mock.MagicMock()
        mock_mpi.COMM_WORLD = mock_comm

        captured_limits = []

        def capture_subprocess(run_config, mem_limit_mb, script_path):
            captured_limits.append(mem_limit_mb)
            result = {"status": "success", "method": run_config["method"],
                      "results": {"error_reduction": 3.0}}
            Path(run_config["result_file"]).write_text(json.dumps(result))
            return result

        with mock.patch.object(rc, "_run_in_subprocess", side_effect=capture_subprocess):
            with mock.patch("builtins.__import__", side_effect=_make_import_patcher(mock_mpi)):
                rc.run_phase_6(args)

        # 8 GB = 8192 MB
        assert all(lim == 8192.0 for lim in captured_limits)


class TestRunPhase6SummaryOutput:
    """Tests for summary file output."""

    def test_final_summary_written(self, tmp_path):
        """phase6_summary.json should be written at the end."""
        from experiments.shinnecock_study import run_comparison as rc

        args = _make_args(tmp_path, sweep_dim="obs_frequency")
        data_dir = Path(args.output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        mock_comm = mock.MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_mpi = mock.MagicMock()
        mock_mpi.COMM_WORLD = mock_comm

        def fake_subprocess(run_config, mem_limit_mb, script_path):
            result = {"status": "success", "method": run_config["method"],
                      "results": {"error_reduction": 5.0}}
            Path(run_config["result_file"]).write_text(json.dumps(result))
            return result

        with mock.patch.object(rc, "_run_in_subprocess", side_effect=fake_subprocess):
            with mock.patch("builtins.__import__", side_effect=_make_import_patcher(mock_mpi)):
                rc.run_phase_6(args)

        summary = data_dir / "phase6_summary.json"
        assert summary.exists()
        with open(summary) as f:
            data = json.load(f)
        assert "obs_frequency" in data
        assert len(data["obs_frequency"]) == len(SWEEP_DIMS["obs_frequency"]["values"])

    def test_failed_child_still_produces_summary(self, tmp_path):
        """Even if all children fail, a summary should be written."""
        from experiments.shinnecock_study import run_comparison as rc

        args = _make_args(tmp_path, sweep_dim="bg_error")
        data_dir = Path(args.output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        mock_comm = mock.MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_mpi = mock.MagicMock()
        mock_mpi.COMM_WORLD = mock_comm

        def failing_subprocess(run_config, mem_limit_mb, script_path):
            result = {
                "status": "failed",
                "error": "child crashed",
                "method": run_config["method"],
                "phase": run_config["phase_prefix"] + run_config["sub_label"],
            }
            Path(run_config["result_file"]).write_text(json.dumps(result))
            return result

        with mock.patch.object(rc, "_run_in_subprocess", side_effect=failing_subprocess):
            with mock.patch("builtins.__import__", side_effect=_make_import_patcher(mock_mpi)):
                rc.run_phase_6(args)

        summary = data_dir / "phase6_summary.json"
        assert summary.exists()


# ============================================================================
# Tests for parse_args --mem-limit-gb
# ============================================================================


class TestParseArgsMem:
    """Tests for the --mem-limit-gb CLI argument."""

    def test_default_mem_limit(self):
        """Default should be 12 GB."""
        from experiments.shinnecock_study.run_comparison import parse_args
        with mock.patch("sys.argv", ["prog", "--phase", "6"]):
            args = parse_args()
        assert args.mem_limit_gb == 12.0

    def test_custom_mem_limit(self):
        """Should accept custom values."""
        from experiments.shinnecock_study.run_comparison import parse_args
        with mock.patch("sys.argv", ["prog", "--phase", "6", "--mem-limit-gb", "8.5"]):
            args = parse_args()
        assert args.mem_limit_gb == 8.5

    def test_mem_limit_conversion_gb_to_mb(self):
        """Verify GB-to-MB conversion: 8 GB -> 8192 MB."""
        args = argparse.Namespace(mem_limit_gb=8.0)
        mem_limit_mb = float(getattr(args, 'mem_limit_gb', 12.0)) * 1024
        assert mem_limit_mb == 8192.0

    def test_default_mem_limit_fallback(self):
        """If mem_limit_gb is not set, default should be 12 GB."""
        args = argparse.Namespace()
        mem_limit_mb = float(getattr(args, 'mem_limit_gb', 12.0)) * 1024
        assert mem_limit_mb == 12288.0


# ============================================================================
# Tests for constants
# ============================================================================


class TestConstants:
    """Verify constants are sane."""

    def test_batch_size_is_positive_int(self):
        assert isinstance(SWEEP_BATCH_SIZE, int)
        assert SWEEP_BATCH_SIZE > 0

    def test_batch_size_reasonable(self):
        assert SWEEP_BATCH_SIZE <= 20

    def test_watchdog_interval_positive(self):
        assert MEM_WATCHDOG_INTERVAL_S > 0

    def test_sweep_baseline_has_required_keys(self):
        required = {"obs_noise_level", "obs_fraction", "obs_frequency",
                    "background_error_std",
                    "nt_da", "nt_ramp", "friction_scale_factor"}
        assert required.issubset(SWEEP_BASELINE.keys())

    def test_sweep_dims_all_have_param_and_values(self):
        for name, dim in SWEEP_DIMS.items():
            assert "param" in dim, f"SWEEP_DIMS['{name}'] missing 'param'"
            assert "values" in dim, f"SWEEP_DIMS['{name}'] missing 'values'"
            assert len(dim["values"]) > 0, f"SWEEP_DIMS['{name}'] has no values"
