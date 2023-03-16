# `Jupyter notebook` setup Instructions

We recommend either:
1. Using Google Collab (if you have a Google account)
2. Setting up your own local environment


## Instructions for Google Collab
Due to slight differences in how jupyter notebooks are rendered in Google Collab, you'll need to *use the files located in the "Google Collab" folder*.

1. Download or clone the repo.
2. Make sure you're signed into your Google account and open [Google Collab](https://colab.research.google.com/)
3. Navigate to the `Upload` tab > click `choose file` > go under the "Google Collab" directory > select the `AI_assistance_Tutorial.ipynb` file
4. Run the first cell in the notebook (it might take a bit of time for the Google backend to activate your virtual machine)
5. When the cell successfully runs, click `choose file` again and select the following three files for upload:
    - `tutorialObjs.py`
    - `ba_mcts.py`
    - `ai-assistance-overview.png`
6. Run the subsequent import cell to verify you don't have any errors.

## Instructions for local environment
### Requirements
- Python 3 (we've testing on 3.8+)
- `jupyter notebook`
- packages in `requirements.txt` 

### For `pyenv` or `virtualenv`
1. Download or clone the repo.
2. Instructions can differ depending on your tool, so if you need help, reference [this external source](https://realpython.com/intro-to-pyenv/#virtual-environments-and-pyenv)

3. Once your environment is activated, remember to run `pip install -r requirements.txt`.

4. Run the `jupyter notebook` command

5. Open the `AI_assistance_Tutorial.ipynb` file and run the first cell to verify all imports were completed without error
  
### For `Anaconda` environments
1. Download or clone the repo.
2. To creat a new environment named "myenv" with Python 3.8 installed, run the following command:

    `conda create --name myenv python=3.8`

3. After creating the environment, you need to activate it before installing any packages. Do this with the following command:

    `conda activate myenv`

4. Run the following command to install all the packages in `requirements.txt`:

    `conda install --file requirements.txt`

5. Run the `jupyter notebook` command

6. Open the `AI_assistance_Tutorial.ipynb` file from jupyter and run the first cell to verify all imports were completed without error

