from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

class ZoomBot:
    def __init__(self, meeting_url: str):
        self.meeting_url = meeting_url
        self.driver = None

    def join_meeting(self):
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--use-fake-ui-for-media-stream')
        options.add_argument('--no-sandbox')
        options.add_argument('--window-size=1920,1080')
        self.driver = webdriver.Chrome(options=options)
        self.driver.get(self.meeting_url)
        time.sleep(10)  # Wait for page to load
        # Optionally, automate UI to join as guest, mute mic/cam, etc.

    def close(self):
        if self.driver:
            self.driver.quit()
