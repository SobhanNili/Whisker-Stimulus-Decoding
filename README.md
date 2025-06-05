![Static Badge](https://img.shields.io/badge/In%20Progress%20-%20orange)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Project Topic
In this project, we aim to infer how the perception of whisker stimuli evolves in mice during an operant conditioning task.

Since perception is a latent variable, we estimate it by combining stimulus decoding from barrel cortex spiking activity with mice licking behavior. We then adjust the decoder parameters to capture behavioral patterns, thereby bringing the decoded representation closer to the underlying perceptual process.

For more information please refer to the abstract presentation included in the repository.

Semester project in Labratory of Sensory Processing, EPFL 

Data from [Oryshchuk et al.](https://zenodo.org/records/10115924)

# Demo
This repo provides an intuative GUI to run the analysis and visualize the results:

# Installation
First [install conda/miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install). To access the GUI, create a conda env based on `environment.yml` file (using `conda env create -f environment.yml`). Then activate this env in terminal (`conda activate smm`) and launce the gradio app by executing `python -m src.app` and viewing the provided web address in browser.
