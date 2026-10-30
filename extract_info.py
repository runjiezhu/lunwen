import re

s = open("medmnist_info.py", encoding="utf-8").read()

# The info file is a dict of dicts; find the top-level key blocks by name.
for name in ["synapsemnist3d", "vesselmnist3d", "fracturemnist3d", "nodulemnist3d"]:
    m = re.search(r'^\s*"%s":\s*\{' % name, s, re.M)
    if not m:
        print(name, "NOT FOUND")
        continue
    start = m.end() - 1
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                print(s[m.start():i + 1])
                print("=" * 70)
                break
