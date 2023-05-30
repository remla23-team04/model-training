### DSLinter output for src/pre_process.py file
Run with `pylint --load-plugins=dslinter src/pre_process.py`

```
************* Module pre_process
src\pre_process.py:10:0: C0301: Line too long (104/100) (line-too-long)
src\pre_process.py:1:0: C0114: Missing module docstring (missing-module-docstring)
src\pre_process.py:6:0: C0413: Import "from nltk.corpus import stopwords" should be placed at the top of the module (wrong-import-position)
src\pre_process.py:7:0: C0413: Import "from nltk.stem.porter import PorterStemmer" should be placed at the top of the module (wrong-import-position)
src\pre_process.py:19:4: R5504: There is no column selection after the dataframe is imported. (column-selection-pandas)
src\pre_process.py:19:14: R5503: Datatype is not set when a dataframe is imported from data. (datatype-pandas)
src\pre_process.py:29:4: C0103: Variable name "ps" doesn't conform to snake_case naming style (invalid-name)

------------------------------------------------------------------
Your code has been rated at 7.41/10 (previous run: 8.15/10, -0.74)
```