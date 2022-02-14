The repository comprises of the scripts used to generate the results in the paper. 

#### Download the dataset

The file `data/WASABI/urls.txt` contains the urls used to query the WASABI database. To download the dataset:

1. locate at `data/WASABI`

2. run the command 

```wget -i urls.txt```

It downloads a set of files. Each file contains 200 artists, and information on their albums and songs. For a comprehensive description of the dataset fields, we refer to the documentation of the WASABI API at https://wasabi.i3s.unice.fr/apidoc/ .

#### Organisation of the repository

The notebooks used to filter the dataset and generate the results are stored at `notebooks/` and `notebooks_colab/`. The flow of the process can be followed by running them according to the numbers in their titles.

- Notebooks contained in `notebooks/` were run under the environment specified by the file `environment.yml`

- Notebooks contained in `notebooks_colab/` were run in Google Colab, and results were saved in Google Drive.
