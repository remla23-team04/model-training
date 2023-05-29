# model-training


### mllint
```
pip install mllint
mllint run
```

### PyLint
```
pip install pylint==2.12.2
pylint src/model-training.py
```
Latest score: Your code has been rated at -5.09/10 (previous run: -5.26/10, +0.18)

### DS Linter
```
pip install dslinter
pylint --load-plugins=dslinter src/model-training.py
pylint src/model-training.py
```
Latest score: Your code has been rated at -5.61/10 (previous run: -5.09/10, -0.53)

### Data pipeline (DVC)
```
pip install dvc
dvc init
dvc run -n get_data -d src/get_data.py -o smsspamcollection python src/get_data.py
```


