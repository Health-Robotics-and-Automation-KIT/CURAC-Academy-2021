import subprocess
from pathlib import Path
import zipfile

try:
    import requests
except ModuleNotFoundError:
    raise ModuleNotFoundError("[HERA ERROR]: please install the requests package before continuing.\n \t pip3 install requests")

try:
    from tqdm import tqdm as progress
except ModuleNotFoundError:
    raise ModuleNotFoundError("[HERA ERROR]: please install the tqdm package before continuing.\n \t pip3 install tqdm")


def download_url(url: str, save_path: Path, chunk_size=128):
    r = requests.get(url, stream=True)
    size = r.headers["content-length"]
    length = int(int(size) / chunk_size)
    with open(str(save_path.absolute()), "wb") as fd, progress(
        total=length * chunk_size, unit_scale=True, unit_divisor=1024, unit="B", desc=save_path.name
    ) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
            pbar.update(chunk_size)


if __name__ == "__main__":
    try:
        out = subprocess.check_output(["conda", "--version"])
    except OSError:
        raise RuntimeError("Conda (Miniconda or Anaconda) must be installed. https://docs.conda.io/en/latest/miniconda.html")

    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--cpu", help="Whether to install pythorch without gpu support.", action="store_true")
    args = parser.parse_args()
    cpu_only = args.cpu

    print("[HERA INFO]: Creating a new conda environment named curac and installing packages...")
    print(f"[HERA INFO]: Pytorch will be installed {'without' if cpu_only else 'with'} gpu support.")

    try:
        subprocess.check_call(
            ["conda", "env", "create", "--file", "conda_cpu.yml" if not cpu_only else "conda_gpu.yml"]
        )
    except:
        print("[HERA WARNING]: Conda environment already exists.")

    dataset_url = "https://cloud.ipr.kit.edu/s/CzrjNSqZrmGDFCk/download"
    dataset_dir = Path("data")
    tmp_file = Path("skin_data.zip")

    if not dataset_dir.is_dir():
        dataset_dir.mkdir()

    if not any(dataset_dir.iterdir()):
        print("[HERA INFO]: Dowloading datasets...")
        download_url(dataset_url, tmp_file)
        print("[HERA INFO]: Unpacking datasets...")
        with zipfile.ZipFile(str(tmp_file.absolute()), "r") as zip_ref:
            zip_ref.extractall(str(dataset_dir.absolute()))

        tmp_file.unlink()
    else:
        print(f"[HERA WARNING]: Data directory {dataset_dir} is not empty. Will not download data again.")

    print("[HERA INFO]: Done!")
