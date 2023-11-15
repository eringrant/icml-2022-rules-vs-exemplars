# icml-2022-rules-vs-exemplars

Code for our 2022 ICML paper "Distinguishing rule and exemplar-based generalization in learning systems."
To cite the work that this code is associated with, use:

```
@inproceedings{dasgupta20222distinguishing,
  title={Distinguishing rule and exemplar-based generalization in learning systems},
  author={Dasgupta, Ishita and Grant, Erin and Griffiths, Tom},
  booktitle={Proceedings of the 39th International Conference on Machine Learning},
  pages={4816--4830},
  year={2022},
  editor={Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume={162},
  series={Proceedings of Machine Learning Research},
  month={17--23 Jul},
  publisher={PMLR},
  url={https://proceedings.mlr.press/v162/dasgupta22b.html},
}
```

## tl;dr:

Install the [package](#package-installation), then run a command such as the following:

```bash
python scripts/main.py --gin_config='configs/static/celeba.gin'
```

Also see the analysis notebooks in [`analyses/`](analyses/).

## Package installation

### Option 1: Conda install

To install via [Conda](https://docs.conda.io/), do:

```bash
git clone git@github.com:eringrant/icml-2022-rules-vs-exemplars
cd icml-2022-rules-vs-exemplars
conda env create --file environment.yml
```

The Conda environment can then be activated via 
```bash
conda activate rules-vs-exemplars
```

### Option 2: pip install

To install via [pip](https://pip.pypa.io/), do:

```bash
git clone git@github.com:eringrant/icml-2022-rules-vs-exemplars
cd icml-2022-rules-vs-exemplars
pip install -e .
```
