"""Take screenshots of the three app parts in dark and light mode (outputs/figures/app/).

Starts app.py on a free port, drives it with headless Chromium, then stops it.
Requires `pip install selenium` and a system chromium + chromedriver.
"""

from argparse import ArgumentParser
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/figures/app"


def free_port() -> int:
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def wait_for_plots(driver, container: str, timeout: float = 40) -> None:
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script(
            "const p = document.querySelector(arguments[0] + ' .js-plotly-plot'); return !!(p && p.data && p.data.length);", container
        )
    )
    time.sleep(1.5)


def capture(driver, path: Path, width: int) -> None:
    # Measure after each resize: the layout reflows when the viewport grows.
    for _ in range(3):
        height = driver.execute_script(
            "const main = document.querySelector('main');"
            "return Math.ceil(Math.max(document.documentElement.scrollHeight, document.body.scrollHeight, main ? main.scrollHeight : 0));"
        )
        # The window is taller than the viewport by the browser chrome.
        chrome = driver.execute_script("return window.outerHeight - window.innerHeight;")
        driver.set_window_size(width, max(height, 660) + chrome)
        time.sleep(1.0)
    driver.save_screenshot(str(path))
    print(path.relative_to(ROOT), f"{width}x{height}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--recording", default="211104B_D")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-c", f"import app; app.app.run(debug=False, host='127.0.0.1', port={port})"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    options = Options()
    options.binary_location = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
    for flag in ("--headless=new", "--no-sandbox", "--hide-scrollbars", "--force-device-scale-factor=1", f"--window-size={args.width},900"):
        options.add_argument(flag)
    options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    driver = webdriver.Chrome(options=options, service=Service(shutil.which("chromedriver")))
    base = f"http://127.0.0.1:{port}"
    problems = []
    try:
        for _ in range(60):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.5)
        for theme in ("dark", "light"):
            driver.set_window_size(args.width, 900)
            driver.get(f"{base}/?theme={theme}&page=comparison&recording={args.recording}")
            wait_for_plots(driver, "#timeline")
            capture(driver, OUTPUT / f"comparison_{theme}{args.suffix}.png", args.width)

            for page in ("visualisation", "prediction"):
                driver.set_window_size(args.width, 900)
                driver.get(f"{base}/?theme={theme}&page={page}&recording={args.recording}")
                wait_for_plots(driver, "#eeg")
                driver.find_element(By.ID, "next-seizure").click()
                time.sleep(2.0)
                # Click a trace inside the seizure to fill the spectrum card.
                plot = driver.find_element(By.CSS_SELECTOR, "#eeg .js-plotly-plot")
                driver.execute_script("arguments[0].scrollIntoView({block: 'start'});", plot)
                size = plot.size
                ActionChains(driver).move_to_element_with_offset(plot, 0, int(-size["height"] * 0.18)).pause(0.4).click().perform()
                time.sleep(2.0)
                driver.execute_script("window.scrollTo(0, 0);")
                capture(driver, OUTPUT / f"{page}_{theme}{args.suffix}.png", args.width)
        for entry in driver.get_log("browser"):
            if entry["level"] == "SEVERE":
                problems.append(entry["message"])
    finally:
        driver.quit()
        server.terminate()
        output = server.communicate(timeout=10)[0]
    errors = [line for line in output.splitlines() if "Traceback" in line or "Error" in line]
    print("Browser console errors:", problems or "none")
    print("Server errors:", errors or "none")
    if errors:
        print(output[-3000:])


if __name__ == "__main__":
    main()
