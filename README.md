# Tutorial on developing AI assistants

AI has recently shown impressive advances, from learning to play the Atari
games to defeating expert human players in the game of Go. Beyond games, AI has
also exploded in fields such as computer vision and natural language
processing, where vast amounts of labeled data are available. More broadly
speaking, technology as a whole has massively changed the landscape of most
fields. However, current approaches can only help in tasks where we either can
precisely specify the objective or already have plenty of observations of
solutions to learn from.

However, important real-world problems rarely have well-specified objectives or
solutions to learn from. Instead, most problems depend on the goals and
preferences of humans - the users - who are solving them. As a result, we need
approaches that explicitly consider the user. We, the [Artificial agents with
Theory Of Mind team of FCAI](https://fcai.fi/fcai-teams#atom), do exactly that
by developing techniques and methods that assist users in their tasks.

![AI-assistance diagram](figures/ai-assistance.png)

Here you can find a tutorial on creating AI-assistants.

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
5. When the cell successfully runs, click `choose file` again and select the following two files for upload:
    - `tutorialObjs.py`
    - `ai-assistance-overview.png`
6. Run the subsequent import cell to verify you don't have any errors.

## Instructions for local environment
### Requirements
- Python 3 (we've tested on 3.8 and 3.9)
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

