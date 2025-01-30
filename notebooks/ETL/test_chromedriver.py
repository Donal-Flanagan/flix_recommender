# test_chromedriver.py
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
try:
    print("Attempting to install WebDriver...")
    driver_path = ChromeDriverManager().install()
    print(f"WebDriver installed at: {driver_path}")
except Exception as e:
    print(f"Error installing WebDriver: {e}")
