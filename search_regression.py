"""Search HF mirror for medical-image datasets that carry continuous labels."""

import json
import urllib.parse
import urllib.request

HEADERS = {"User-Agent": "python-requests/2.31"}


def get_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


QUERIES = [
    "bone age",
    "MMSE",
    "Alzheimer",
    "severity grading radiograph",
    "edema volume",
    "tumor volume segmentation",
    "Gleason grade",
    "BI-RADS",
    "cardiac ejection fraction",
    "lesion diameter",
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
