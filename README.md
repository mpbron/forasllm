[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14610767.svg)](https://doi.org/10.5281/zenodo.14610767)

# Combining Large Language Model Classifications and Active Learning for Improved Technology-Assisted Review

This repository contains the code to reproduce the results of the paper *Combining Large Language Model Classifications and Active Learning for Improved Technology-Assisted Review*.

The paper can be found on [https://ceur-ws.org/Vol-3770/paper8.pdf](https://ceur-ws.org/Vol-3770/paper8.pdf).

## Set up

We assume that you are working on a recent Linux system (for example Debian 12
or Ubuntu 24.04) with a recent Python version available (e.g. Python 3.10 or
higher). Older Linux systems and other operating systems may work, but these are
not tested. For rendering the PDFs, Quarto, R (with `knitr`, `tidyverse` and `reticulate` installed), and a working LaTeX distribution
should be installed (e.g., `texlive-full` on Debian / Ubuntu). 


Open a terminal and create a new virtual environment:

```bash
python3 -m venv .venv
# or 
python -m venv .venv

```
Activate it:
```bash
source .venv/bin/activate
```

Install the required Python dependencies.

```bash
pip install -r requirements.txt
```

Create an `.env` file that contains your OpenAI / Azure API key by using the `.env-example` file. 

## Download the data

The data is stored on [https://osf.io/6mjtd/](https://osf.io/6mjtd/). Download the data and place the data folder in the root of the repository. 


## Funding

The project was funded by the Dutch Research Council under grant no. [406.22.GO.048](https://app.dimensions.ai/details/grant/grant.13726450). 

