"""Take screenshots of the app as it appears on screen (outputs/figures/app/).

Three window sizes (1920x1080, 1440x900, 1280x800), both themes, the three parts, plus the
side panels, overlay mode, reference channels, the empty state and the all-model comparison.
Starts app.py on a free port, drives it with headless Chromium, then stops it.
Requires `pip install selenium` and a system chromium + chromedriver.
"""
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/figures/app"
OUT.mkdir(parents=True, exist_ok=True)
with socket.socket() as handle:
    handle.bind(("127.0.0.1", 0))
    PORT = handle.getsockname()[1]
LOG = Path(tempfile.gettempdir()) / "eeg_app_screenshots.log"
# The server log goes to a file: an unread pipe would fill up and block the server.
server = subprocess.Popen([sys.executable, "-c", f"import app; app.app.run(debug=False, port={PORT})"], cwd=ROOT, stdout=open(LOG, "w"), stderr=subprocess.STDOUT)
o = Options(); o.binary_location = shutil.which("chromium")
for f in ("--headless=new", "--no-sandbox", "--hide-scrollbars", "--window-size=1920,1080"): o.add_argument(f)
o.set_capability("goog:loggingPrefs", {"browser": "ALL"})
d = webdriver.Chrome(options=o, service=Service(shutil.which("chromedriver")))
click = lambda xpath: d.execute_script("arguments[0].click();", d.find_element(By.XPATH, xpath))
label = lambda text: click(f"//label[normalize-space()='{text}']")
def size(w, h):
    chrome = d.execute_script("return window.outerHeight - window.innerHeight;")
    d.set_window_size(w, h + chrome); time.sleep(1.5)
def rest():
    # Park the pointer on the brand so that no tooltip or hover label is captured.
    ActionChains(d).move_to_element(d.find_element(By.CSS_SELECTOR, ".brand")).perform(); time.sleep(0.9)
def shot(name, full=False):
    rest()
    if full:
        w = d.execute_script("return window.innerWidth"); h0 = d.execute_script("return window.innerHeight")
        h = d.execute_script("return Math.ceil(Math.max(document.documentElement.scrollHeight, document.querySelector('main').scrollHeight))")
        size(w, h); rest(); d.save_screenshot(str(OUT / name)); size(w, h0)
    else:
        d.save_screenshot(str(OUT / name))
    print(name, d.execute_script("const p=document.querySelector('#eeg .js-plotly-plot'); const r=p?p.getBoundingClientRect():null; return [window.innerWidth, window.innerHeight, r&&Math.round(r.top), r&&Math.round(r.bottom), document.documentElement.scrollHeight > window.innerHeight + 2]"))
try:
    time.sleep(8); d.get(f"http://127.0.0.1:{PORT}/"); time.sleep(8)
    for w, h in ((1920, 1080), (1440, 900), (1280, 800)):
        size(w, h)
        for theme in ("light", "dark"):
            for page in ("visualisation", "prediction"):
                d.get(f"http://127.0.0.1:{PORT}/?theme={theme}&page={page}&recording=211104B_D"); time.sleep(8)
                d.find_element(By.ID, "next-seizure").click(); time.sleep(3)
                shot(f"{page}_{theme}_{w}.png")
            d.get(f"http://127.0.0.1:{PORT}/?theme={theme}&page=comparison&recording=211104B_D"); time.sleep(8)
            shot(f"comparison_{theme}_{w}.png", full=(w == 1920))
    # states, at 1440x900 light
    size(1440, 900)
    d.get(f"http://127.0.0.1:{PORT}/?theme=light&page=prediction&recording=190304A_E"); time.sleep(8)
    d.find_element(By.ID, "next-seizure").click(); time.sleep(3)
    label("Reference channels (ECG, EMG, SLI)"); time.sleep(4); shot("state_reference_1440.png")
    label("Overlay"); time.sleep(4); shot("state_overlay_1440.png")
    label("All stacked"); label("Reference channels (ECG, EMG, SLI)"); time.sleep(4)
    plot = d.find_element(By.CSS_SELECTOR, "#eeg .js-plotly-plot")
    ActionChains(d).move_to_element_with_offset(plot, 40, int(-plot.size["height"] * 0.2)).pause(0.4).click().perform(); time.sleep(3)
    shot("state_spectrum_drawer_1440.png")
    for close in d.find_elements(By.CSS_SELECTOR, ".mantine-Drawer-close"): d.execute_script("arguments[0].click();", close)
    time.sleep(1); click("//button[@id='open-intervals']"); time.sleep(2); shot("state_intervals_drawer_1440.png")
    for close in d.find_elements(By.CSS_SELECTOR, ".mantine-Drawer-close"): d.execute_script("arguments[0].click();", close)
    for name in ["Fp1","Fp2","F7","F3","Fz","F4","F8","T3","C3","Cz","C4","T4","T5","P3","Pz","P4","T6","O1","O2"]:
        click(f"//div[contains(@class,'channel-chip')]//label[normalize-space()='{name}']")
    time.sleep(12); shot("state_no_channel_1440.png")
    label("Comparison"); time.sleep(4); label("All models"); time.sleep(6); d.execute_script("window.scrollTo(0,0)"); shot("state_all_models_1440.png", full=True)
    print("console:", [e["message"][:200] for e in d.get_log("browser") if e["level"] == "SEVERE"] or "none")
finally:
    d.quit(); server.terminate(); server.wait(timeout=10)
    print("server tracebacks:", LOG.read_text().count("Traceback"))
