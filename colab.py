import requests
import json

def run_code_in_colab(code, api_key):
  """Lähettää annetun Python-koodin suoritettavaksi Google Colabille.

  Args:
    code: Python-koodi merkkijonona.
    api_key: Sinun Google Colab API-avaimesi.

  Returns:
    str: Luodun notebookin nimi, jos pyyntö onnistui.
    None: Jos tapahtui virhe.
  """

  url = "https://colab.research.google.com/v2/notebooks"
  headers = {
      "Authorization": f"Bearer {api_key}",
      "Content-Type": "application/json"
  }

  data = {
      "content": [
          {
              "cell_type": "code",
              "source": code.splitlines()
          }
      ]
  }

  response = requests.post(url, headers=headers, json=data)

  if response.status_code == 200:
      notebook_id = response.json()['name']
      print(f"Notebook created successfully: {notebook_id}")
      return notebook_id
  else:
      print(f"Error creating notebook: {response.text}")
      return None

# Esimerkki käytöstä:
code_to_run = """
print("Hello from Colab!")
"""
api_key = "AIzaSyDEM6EVGdYR9AnglbZarF4wruSzKU2DhuU"  # oma api avain
notebook_id = run_code_in_colab(code_to_run, api_key)