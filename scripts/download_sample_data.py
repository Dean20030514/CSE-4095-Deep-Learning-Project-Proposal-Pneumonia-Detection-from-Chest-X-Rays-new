import argparse, os, urllib.request, zipfile

# TODO: replace with a real public URL to a tiny sample set
URL = "https://storage.googleapis.com/quickstart-cxr/sample_10.zip"

parser = argparse.ArgumentParser()
parser.add_argument('--dest', default='data/sample_10')
args = parser.parse_args()

os.makedirs(args.dest, exist_ok=True)
zip_path = os.path.join(args.dest, 'sample_10.zip')
print("Downloading sample to", zip_path)
urllib.request.urlretrieve(URL, zip_path)
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(args.dest)
os.remove(zip_path)
print("Sample downloaded to", args.dest)
