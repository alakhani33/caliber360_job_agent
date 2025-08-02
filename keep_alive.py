# keep_alive.py

import requests

# Replace with your actual Streamlit app URL
url = "https://caliber360ai.streamlit.app/"

try:
    response = requests.get(url)
    print(f"✅ Pinged {url} — Status Code: {response.status_code}")
except Exception as e:
    print(f"❌ Failed to ping {url}: {e}")


# To run in git bash:
# while true
# do
#   python keep_alive.py
#   sleep 600
# done


