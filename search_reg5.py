"""Focused search for continuous-label medical imaging datasets on the mirror."""

import json
import urllib.parse
import urllib.request

HEADERS = {"User-Agent": "python-requests/2.31"}


def get_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


QUERIES = [
    "oasis alzheimer",
    "bone age months",
    "age prediction radiograph",
    "brain age MRI",
    "glaucoma cup ratio",
    "coronary calcification score",
    "TADPOLE",
    "hippocampal volume",
    "Alzheimer ordinal",
    "IDRiD",
    "DDR diabetic retinopathy",
    "REFUGE fundus",
]

if __name__ == "__main__":
    for q in QUERIES:
        print("==", q)
        try:
            data = get_json(
                "https://hf-mirror.com/api/datasets?search="
                + urllib.parse.quote(q)
                + "&limit=8&sort=downloads&direction=-1"
            )
            for d in data:
                print("   ", d["id"], d.get("downloads"))
        except Exception as e:
            print("   ERR", str(e)[:100])
