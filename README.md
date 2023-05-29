# model-training


### mllint
```
pip install mllint
mllint run
```

### Data pipeline (DVC)
```
pip install dvc
dvc init
dvc run -n get_data -d src/get_data.py -o smsspamcollection python src/get_data.py
```
