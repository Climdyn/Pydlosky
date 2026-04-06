# Pydlosky

Pydlosky is the companion software with the article:

* Revisiting the dynamical properties of the Pedlosky’s Two-Layer Model for Finite
Amplitude Baroclinic Waves, De Ro, N., Demaeyer, J., and Vannitsem, S. (2026). Submitted to ...

The article revisits the Pedlosky geophysical model proposed by Joseph Pedlosky in the 1970s:

* Pedlosky, J. (1970). Finite-amplitude baroclinic waves. [Journal of Atmospheric Sciences, 27(1), 15-30](https://doi.org/10.1175/1520-0469(1970)027%3C0015:FABW%3E2.0.CO;2).
* Pedlosky, J. (1971). Finite-amplitude baroclinic waves with small dissipation. [Journal of Atmospheric Sciences, 28(4), 587-597](https://doi.org/10.1175/1520-0469(1971)028%3C0587:FABWWS%3E2.0.CO;2).

Pydlosky includes Python classes to define the model in the [`PedloskySystem`](./PedloskySystem) folder. This folder also includes specific integrators which relies on [numbalsoda](https://github.com/Nicholaswogan/NumbaLSODA) to perform the various ordinary differential equations time integrations in the study.

The study itself is contained in the [project notebook](./Notebook/project_notebook.ipynb) found in the [`Notebook`](./Notebook) folder.
Fortran and AUTO definition files for this model can also be found in this folder.
Indeed, this study relies on [auto-07p](https://github.com/auto-07p/auto-07p) and the automation codebase [auto-AUTO](https://github.com/Climdyn/auto-AUTO) to produce the bifurcation diagrams of the model.
Therefore, the users must follow the [installation instructions](https://github.com/Climdyn/auto-AUTO?tab=readme-ov-file#installation) found on the latter webpage in order for this notebook to fully work.
