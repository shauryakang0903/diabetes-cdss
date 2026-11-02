"""Run final software validation, a local server check, then build audit reports."""

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation.nhanes_cdss_integrated import OUTPUT_DIR, ROOT, build_validation_report, snapshot_inputs


def server_health_check():
    """Own a temporary headless server; do not disturb the user's port 8501."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [sys.executable, "-m", "streamlit", "run", str(ROOT/"app.py"),
               "--server.address=127.0.0.1", f"--server.port={port}", "--server.headless=true",
               "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
    report = dict(passed=False, command=command, url=f"http://127.0.0.1:{port}/_stcore/health")
    with (OUTPUT_DIR/"final_streamlit_server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Temporary Streamlit server exited; see final_streamlit_server.log")
                try:
                    with urlopen(report["url"], timeout=1) as response:
                        body = response.read().decode().strip()
                        if response.status == 200 and body == "ok":
                            report.update(passed=True, status_code=200, response=body)
                            break
                except (URLError, TimeoutError, OSError):
                    pass
                time.sleep(.2)
            if not report["passed"]:
                raise RuntimeError("Streamlit health check did not succeed within 45 seconds")
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    # Stop only this owned test process and its venv-launcher children.
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                else:
                    process.terminate()
                process.wait(timeout=15)
            report["owned_test_server_stopped"] = process.poll() is not None
    return report


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    before = snapshot_inputs()
    pytest_report, smoke_report = OUTPUT_DIR/"final_pytest_report.xml", OUTPUT_DIR/"final_streamlit_smoke.json"
    # Require a fresh smoke result; the test will write it only after all checks pass.
    env = os.environ.copy()
    env["NHANES_FINAL_SMOKE_REPORT"] = str(smoke_report)
    env["NHANES_FINAL_VALIDATION_RUN_ID"] = uuid.uuid4().hex
    env["PYTHONIOENCODING"] = "utf-8"
    command = [sys.executable, "-m", "pytest", "-q", "--ignore=tests/test_nhanes.py", "-p", "no:cacheprovider",
               "--basetemp=.test_tmp_final_cdss", f"--junitxml={pytest_report}"]
    print("Running regression tests (raw temporal integration module excluded)…", flush=True)
    with (OUTPUT_DIR/"final_pytest.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError("Regression failed; see results/nhanes/cdss/final_pytest.log")
    print("Regression passed. Checking a temporary localhost Streamlit server…", flush=True)
    health = server_health_check()
    smoke = json.loads(smoke_report.read_text(encoding="utf-8"))
    if smoke["run_id"] != env["NHANES_FINAL_VALIDATION_RUN_ID"]:
        raise RuntimeError("The Streamlit smoke report is stale or belongs to another validation run")
    smoke["server_health"] = health
    smoke_report.write_text(json.dumps(smoke, indent=2, allow_nan=False), encoding="utf-8")
    print("Server health passed. Building scenario, language, integrity and traceability reports…", flush=True)
    frame, trace, audit, summary, provenance = build_validation_report(pytest_report, smoke_report, health, before)
    print(frame[["scenario", "predicted_probability", "model_derived_stratum", "backend_deterministic"]].to_string(index=False))
    print("Tests:", provenance["pytest"])
    print("Traceability:", trace.status.value_counts().to_dict())
    print("Language flags:", int((audit.status == "FLAGGED").sum()))
    print("Final audit saved under", OUTPUT_DIR)


if __name__ == "__main__":
    main()
