import argparse
from antariksh.ingest.mast import download_all

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=None)
a  = p.parse_args()
download_all(limit=a.limit)
