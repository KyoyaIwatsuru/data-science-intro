data-science-intro
=============

Some sample codes used in my lecture

## Outline

1. [Getting Started](/code/ipynb/getting-started.ipynb)
1. [How to use scikit-learn](code/ipynb/sklearn-hands-on.ipynb)
1. [Eye gaze analysis](/code/ipynb/gaze-analysis.ipynb)

## Directory structure: the best practice

```
.
├── README.md
├── code
│   ├── ipynb   : codes for machine learning and visualizations
│   └── lib     : codes for feature calculation, filtering, etc.
└── data
    ├── input   : raw data from sensors. never change!
    ├── output  : calculated features, processing steps
    └── working : classification results, figures, etc.
```

## Using Jupyter Notebook in VCS

Please start jupyter notebook with this option

```
$ jupyter notebook --config=.ipynb_config.py
```

... or add the following code into ~/.jupyter/jupyter_notebook_config.py

```
def scrub_output_pre_save(model, **kwargs):
    """scrub output before saving notebooks"""
    # only run on notebooks
    if model['type'] != 'notebook':
        return
    # only run on nbformat v4
    if model['content']['nbformat'] != 4:
        return

    for cell in model['content']['cells']:
        if cell['cell_type'] != 'code':
            continue
        cell['execution_count'] = None

c.FileContentsManager.pre_save_hook = scrub_output_pre_save
```

## Useful links

* [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
* [An introduction to machine learning with scikit-learn](http://scikit-learn.org/stable/tutorial/basic/tutorial.html)
* [Reality Media Workshop -- Introduction to Data Analysis](https://github.com/kkai/kmd_data_intro/blob/master/1-intro.pdf)
