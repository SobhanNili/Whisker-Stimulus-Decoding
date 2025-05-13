![Static Badge](https://img.shields.io/badge/In%20Progress%20-%20orange)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Project Description
In this project, we decode the presence/non-presence of whisker stimulation from neural recordings in mice using a simple template matching model that is interpretable. Then we try to compare the decoded stimulus values with mice behavior that was trained on contextual operant conditioning task to see whether we can interpret the decoder outputs as the mouse's perception of stimulus or not.

Semester project in Labratory of Sensory Processing, EPFL 

Data from [Oryshchuk et al.](https://zenodo.org/records/10115924)

# Demo
This repo provides an intuative GUI to run the analysis and visualize the results:

# Installation
First [install conda/miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install). To access the GUI, create a conda env based on `environment.yml` file (using `conda env create -f environment.yml`). Then activate this env in terminal (`conda activate smm`) and lunch the gradio app by executing `python src/app.py`.
