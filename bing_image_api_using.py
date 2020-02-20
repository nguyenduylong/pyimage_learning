from requests import exceptions
import requests
import cv2
import os

EXCEPTIONS = set([IOError, FileNotFoundError,
exceptions.RequestException, exceptions.HTTPError,
	exceptions.ConnectionError, exceptions.Timeout])

allow_image_extentions = [ '.png', '.jpg']

API_KEY = "f528aaf59df340039c3584e7cf92ed0a"
MAX_RESULTS = 250
GROUP_SIZE = 250
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
headers = {"Ocp-Apim-Subscription-Key" : API_KEY}

class PokemonImageDownload:
    def download(term, output_folder):
        downloaded_total = 0
        params = {"q": term, "offset": 0, "count": GROUP_SIZE}
        print("[INFO] searching Bing API for '{}'".format(term))
        search = requests.get(URL, headers=headers, params=params)
        search.raise_for_status()

        results = search.json()
        estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
        print("[INFO] {} total results for '{}'".format(estNumResults, term))

        for offset in range(0, estNumResults, GROUP_SIZE):
            print("[INFO] making request for group {}-{} of {}...".format(
                offset, offset + GROUP_SIZE, estNumResults))
            params["offset"] = offset
            search = requests.get(URL, headers=headers, params=params)
            search.raise_for_status()
            results = search.json()
            print("[INFO] saving images for group {}-{} of {}...".format(
                offset, offset + GROUP_SIZE, estNumResults))
        for v in results["value"]:
            p = ''
            try:
                print("[INFO] fetching: {}".format(v["contentUrl"]))
                r = requests.get(v["contentUrl"], timeout=30)
                ext = v["contentUrl"][v["contentUrl"].rfind("."):]
                if ext in allow_image_extentions:
                    p = os.path.sep.join([output_folder, "{}{}".format(str(downloaded_total).zfill(8), ext)])
                    f = open(p, "wb")
                    f.write(r.content)
                    f.close()
                    print("[INFO] saved file: {}".format(p))
            except Exception as e:
                if type(e) in EXCEPTIONS:
                    print("[INFO] skipping: {}".format(v["contentUrl"]))
                    continue
            if p is not '':
                image = cv2.imread(p)
                if image is None:
                    print("[INFO] deleting: {}".format(p))
                    os.remove(p)
                    continue
                downloaded_total += 1







