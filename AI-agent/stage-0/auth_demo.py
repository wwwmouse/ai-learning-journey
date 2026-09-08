import os
import requests as rq

url="https://api.github.com/user"

r1 = rq.get(url,timeout=10)
print(f"不带token:{r1.status_code}")

token = os.environ['GITHUB_TOKEN']
r2 = rq.get(url, headers={'Authorization': f"Bearer {token}"},timeout=10)
print(f"带token:{r2.status_code}")
print(r2.json()['login'])