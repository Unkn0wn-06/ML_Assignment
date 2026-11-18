"""Capture the default real-data experiment with the local Edge installation."""

import subprocess
import sys
import time
from pathlib import Path
from socket import socket
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
destination = root / "docs/ui"
destination.mkdir(parents=True, exist_ok=True)
with socket() as reservation:
    reservation.bind(("127.0.0.1", 0))
    port = reservation.getsockname()[1]
url = f"http://127.0.0.1:{port}"
server = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless=true",
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ],
    cwd=root,
)
try:
    for attempt in range(80):
        try:
            with urlopen(url + "/_stcore/health", timeout=1) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(0.25)
    with sync_playwright() as automation:
        browser = automation.chromium.launch(
            channel="msedge", args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader"]
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
        page.goto(url)
        page.get_by_role("button", name="Run experiment").click(timeout=60000)
        page.get_by_text("Scenario map", exact=True).wait_for(timeout=120000)
        for name in ["Overview", "Compare controllers", "Model diagnostics", "Source data"]:
            page.get_by_role("tab", name=name, exact=True).click()
            for height in [1000, 2800]:
                page.set_viewport_size({"width": 1600, "height": height})
                page.wait_for_timeout(1300)
                assert not page.locator('[data-testid="stException"]').count()
                page.screenshot(
                    path=destination / f"{name.lower().replace(' ', '-')}-{height}.png",
                    timeout=60000,
                )
        room = next(frame for frame in page.frames if frame.locator("#room").count())
        print(
            "Room canvases:",
            room.locator("#room canvas").count(),
            room.locator("#label").inner_text(),
        )
        browser.close()
finally:
    server.terminate()
    server.wait(timeout=20)
