# Pydlosky

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19452068.svg)](https://doi.org/10.5281/zenodo.19452068)

Pydlosky is the companion software with the article:

* Revisiting the dynamical properties of the Pedlosky’s Two-Layer Model for Finite
Amplitude Baroclinic Waves, De Ro, N., Demaeyer, J., and Vannitsem, S. (2026). Submitted to ...

**Please cite this article if you use (a part of) this software for a publication.**

The article revisits the Pedlosky quasi-geostrophic model originally proposed by Joseph Pedlosky in the 1970s and 1980s:

* Pedlosky, J. (1970). Finite-amplitude baroclinic waves. [Journal of Atmospheric Sciences, 27(1), 15-30](https://doi.org/10.1175/1520-0469(1970)027%3C0015:FABW%3E2.0.CO;2).
* Pedlosky, J. (1971). Finite-amplitude baroclinic waves with small dissipation. [Journal of Atmospheric Sciences, 28(4), 587-597](https://doi.org/10.1175/1520-0469(1971)028%3C0587:FABWWS%3E2.0.CO;2).
* Pedlosky, J., and C. Frenzen (1980). Chaotic and Periodic Behavior of Finite-Amplitude Baroclinic Waves. [Journal of Atmospheric Sciences, 37(6), 1177–1196](https://doi.org/10.1175/1520-0469(1980)037%3C1177:CAPBOF%3E2.0.CO;2).

<p float="left">
  <img src="https://github.com/Climdyn/Pydlosky/blob/main/Animation/trajectory_animation.gif?raw=true" height="320" />
  <img src="https://github.com/Climdyn/Pydlosky/blob/main/Animation/basin_animation.gif?raw=true" height="320" /> 
</p>

## General information

Pydlosky includes Python classes to define the model in the [`PedloskySystem`](./PedloskySystem) folder. This folder also includes specific integrators which relies on [numbalsoda](https://github.com/Nicholaswogan/NumbaLSODA) to perform the various ordinary differential equations time integrations in the study.

The study itself is contained in the [project notebook](./Notebook/project_notebook.ipynb) found in the [`Notebook`](./Notebook) folder.
Fortran and AUTO definition files for this model can also be found in this folder.
Indeed, this study relies on [auto-07p](https://github.com/auto-07p/auto-07p) and the automation codebase [auto-AUTO](https://github.com/Climdyn/auto-AUTO) to produce the bifurcation diagrams of the model.
Therefore, in addition to creating an [Anaconda](https://www.anaconda.com/) environment with

    conda env create -f environment.yml
    
the users must follow the [installation instructions](https://github.com/Climdyn/auto-AUTO?tab=readme-ov-file#installation) found on the auto-AUTO webpage in order for this notebook to fully work.

Once the installation has been performed, the user can start `jupyter-notebook`:

    conda activate pydlosky
    cd Notebook
    jupyter-notebook

and load the project notebook.
