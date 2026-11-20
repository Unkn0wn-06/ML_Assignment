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
        browser = automation.chromium.launch(channel="msedge", args=["--enable-unsafe-swiftshader"])
        viewport = {"width": 1600, "height": 900}
        page = browser.new_page(viewport=viewport)
        page.goto(url)
        page.get_by_role("spinbutton", name="Consecutive training seeds").fill("1")
        page.get_by_role("spinbutton", name="Consecutive training seeds").press("Enter")
        page.get_by_role("button", name="Run experiment").click(timeout=60000)
        picker = page.get_by_role("combobox", name="Replay scenario")
        picker.wait_for(timeout=300000)
        picker.fill("PlateWallTest_75percent")
        page.get_by_role("option", name="PlateWallTest_75percent", exact=False).click()
        frame_element = page.locator("iframe[srcdoc]").filter(visible=True).first
        page.wait_for_function(
            """Array.from(document.querySelectorAll('iframe[srcdoc]')).some(el=>el.srcdoc.includes('PlateWallTest_75percent') && el.contentDocument?.querySelector('#room canvas'))""",
            timeout=120000,
        )
        # Capture the very same component document, without cropping it behind Streamlit's toolbar.
        markup = frame_element.get_attribute("srcdoc")
        scene = browser.new_page(viewport=viewport)
        errors = []
        scene.on("pageerror", lambda error: errors.append(str(error)))
        scene.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        scene.set_content(markup)
        scene.wait_for_selector("#room canvas")
        assert scene.evaluate("""() => temperature(data.limit-5.01) === palette[0] &&
            temperature(data.limit-5) === palette[1] && temperature(data.limit+.01) === palette[6] &&
            document.querySelector('#keys').innerText.includes(String(data.limit-5))""")
        cold, hot = scene.evaluate("""() => {
            const temperatures=frames.map(row=>row.outlet_temperature);
            return [temperatures.indexOf(Math.min(...temperatures)),temperatures.indexOf(Math.max(...temperatures))];
        }""")
        images = []
        for index in (cold, hot):
            scene.locator("#step").evaluate(
                " (el,i)=>{el.value=i;el.dispatchEvent(new Event('input'))}", index
            )
            images.append(scene.locator("#room canvas").screenshot())
        assert images[0] != images[1], (
            "Rack colours did not change across the observed temperature range"
        )
        scene.locator("#step").evaluate("el=>{el.value=9;el.dispatchEvent(new Event('input'))}")
        scene.locator(".pin").first.click()
        assert not errors, errors
        assert scene.locator("#zone").is_visible(), scene.locator("#info").inner_text()
        assert scene.locator("#info").inner_text().find("Initial state") < 0
        scene.bring_to_front()
        fps = scene.evaluate("""async () => {
            const canvas=document.querySelector('#room canvas');
            let start;
            for(let i=0;i<60;i++){
                if(i===15) start=performance.now();
                await new Promise(requestAnimationFrame);
                canvas.dispatchEvent(new PointerEvent('pointermove',{buttons:1,movementX:2}));
            }
            const fps=45000/(performance.now()-start);
            canvas.dispatchEvent(new PointerEvent("pointerup"));
            return fps;
        }""")
        print(f"Headless Edge WebGL orbit: {fps:.1f} frames/s", flush=True)
        scene.wait_for_timeout(300)
        scene.locator("#home").click()
        scene.screenshot(path=destination / "viewer-3d.png")
        scene.get_by_role("button", name="PLAN", exact=True).click()
        scene.screenshot(path=destination / "viewer-plan.png")
        scene.locator("#compare").check()
        assert scene.locator("#peer").is_visible()
        scene.locator("#compare").uncheck()
        scene.get_by_role("button", name="View cooling loop", exact=True).click()
        assert scene.evaluate("""() => {
            const labels=[...document.querySelectorAll('#loop text')].map(el=>el.getBoundingClientRect());
            return labels.every((a,i)=>labels.slice(i+1).every(b=>a.right<=b.left || b.right<=a.left || a.bottom<=b.top || b.bottom<=a.top));
        }""")
        scene.screenshot(path=destination / "viewer-loop.png")
        scene.locator("#play").click()
        scene.wait_for_function("Number(document.querySelector('#step').value) > 9")
        scene.locator("#play").click()
        assert not errors, errors
        fallback = browser.new_page(viewport=viewport)
        fallback.evaluate("""() => { const original=HTMLCanvasElement.prototype.getContext;
            HTMLCanvasElement.prototype.getContext=function(type,...args){return type.includes('webgl')?null:original.call(this,type,...args)} }""")
        fallback.set_content(markup)
        assert fallback.locator(".fallback").is_visible()
        for name in ["Compare", "Diagnostics", "Source data"]:
            page.get_by_role("tab", name=name, exact=True).click()
            page.locator('[data-testid="stMain"]').evaluate("el=>el.scrollTop=0")
            page.wait_for_timeout(800)
            assert not page.locator('[data-testid="stException"]').count()
            page.screenshot(path=destination / (name.lower().replace(" ", "-") + "-900.png"))
        browser.close()
finally:
    server.terminate()
    server.wait(timeout=20)
