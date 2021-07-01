# CURAC Academy 2021 - Workshop
## System requirements
* Ubuntu 18.04 or MacOSX. Windows 10 might work as well, but was not tested.
* git ([installation instructions](https://git-scm.com/downloads))
* Python 3.7 or later ([download](https://www.python.org/downloads/))
* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://docs.anaconda.com/anaconda/install/)
* `requests` and `tqdm` package installed (`pip3 install requests tqdm`).

## Installation
1. Clone this repository. `git clone https://github.com/Health-Robotics-and-Automation-KIT/CURAC-Academy-2021.git`
2. Go into the cloned folder. `cd CURAC-Academy-2021` 
3. Execute the setup script `python3 setup_env.py`. This will create a new conda environment, install all required packages and download a dataset. If you do not want to use the gpu version of pytorch, call the script as `python3 setup_env.py --cpu`. This will be the case if you are on a laptop or desktop without a strong dedicated graphics card.
4. Activate the conda environment `conda activate curac` 
5. Start a jupyter notebook server `jupyter notebook`.
6. Open chrome or firefox and go to `http://localhost:8888/tree`.
7. Open the `Introduction Tutorial.ipynb` notebook to get started.
8. Continue to the `Skin Cancer Classification.ipynb` notebook.
