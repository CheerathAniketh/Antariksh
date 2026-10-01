import argparse
from antariksh.ingest.targets import build_targets

p = argparse.ArgumentParser()
p.add_argument("--refresh", action="store_true", help="re-fetch TOI table (changes the snapshot!)")
build_targets(refresh=p.parse_args().refresh)