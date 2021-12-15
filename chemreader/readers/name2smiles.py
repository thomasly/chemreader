from urllib.request import urlopen
from urllib.parse import quote
from urllib.error import HTTPError


def CIRconvert(ids):
    try:
        url = "http://cactus.nci.nih.gov/chemical/structure/" + quote(ids) + "/smiles"
        ans = urlopen(url).read().decode("utf8")
        return ans
    except HTTPError:
        return None


if __name__ == "__main__":
    identifiers = [
        "3-Methylheptane",
        "Aspirin",
        "Diethylsulfate",
        "Diethyl sulfate",
        "50-78-2",
        "Adamant",
    ]

    for ids in identifiers:
        print(ids, CIRconvert(ids))
