![Static Badge](https://img.shields.io/badge/In%20Progress%20-%20orange)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Description
In this project, we decode the presence/non-presence of whisker stimulation from neural recordings in mice using a simple template matching model that is interpretable. Then we try to compare the decoded stimulus values with mice behavior that was trained on contextual operant conditioning task to see whether we can interpret the decoder outputs as the mouse's perception of stimulus or not.

Semester project in Labratory of Sensory Processing, EPFL 

Data from [Oryshchuk et al.](https://zenodo.org/records/10115924)

# Demo
This repo provides an intuative GUI to run the analysis and visualize the results:

# Installation
To access the GUI, create a conda env as mentioned in `requirements.txt` file, and after activating the env, lunch the gradio app by executing `python src/app.py`
