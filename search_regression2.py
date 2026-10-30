"""Search HF mirror for medical-image datasets with continuous numeric targets."""

import json
import urllib.parse
import urllib.request

HEADERS = {"User-Agent": "python-requests/2.31"}


def get_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


QUERIES = [
    "boneage",
    "bone age x-ray kaggle",
    "age estimation xray hand",
    "Alzheimer MMSE MRI",
    "OASIS dementia rating",
    "knee osteoarthritis grade KL",
    "Lung-RADS",
    "SIRAD severity",
    "COVID severity CT score",
    "intracranial hemorrhage volume",
]

if __name__ == "__main__":
    for q in QUERIES:
        print("==", q)
        try:
            data = get_json(
                "https://hf-mirror.com/api/datasets?search="
                + urllib.parse.quote(q)
                + "&limit=6&sort=downloads&direction=-1"
            )
            for d in data:
                print("   ", d["id"], d.get("downloads"))
        except Exception as e:
            print("   ERR", str(e)[:100])
