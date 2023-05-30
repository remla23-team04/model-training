### DSLinter output for src/train.py file
Run with: `pylint --load-plugins=dslinter src/train.py`

```
************* Module train
src\train.py:57:0: C0305: Trailing newlines (trailing-newlines)
src\train.py:1:0: C0114: Missing module docstring (missing-module-docstring)
src\train.py:1:0: W5508: The np.random.seed() is not set in numpy program. (randomness-control-numpy)
src\train.py:13:0: C0103: Constant name "path_to_output" doesn't conform to UPPER_CASE naming style (invalid-name)
src\train.py:14:0: C0103: Constant name "path_to_model" doesn't conform to UPPER_CASE naming style (invalid-name)
src\train.py:15:0: C0103: Constant name "pkl_file_name" doesn't conform to UPPER_CASE naming style (invalid-name)
src\train.py:16:0: C0103: Constant name "classifier_name" doesn't conform to UPPER_CASE naming style (invalid-name)
src\train.py:18:0: C0413: Import "from evaluate import evaluation" should be placed at the top of the module (wrong-import-position)
src\train.py:19:0: C0413: Import "from pre_process import pre_process" should be placed at the top of the module (wrong-import-position)
src\train.py:28:4: W0621: Redefining name 'X_train' from outer scope (line 51) (redefined-outer-name)
src\train.py:28:13: W0621: Redefining name 'X_test' from outer scope (line 51) (redefined-outer-name)
src\train.py:28:21: W0621: Redefining name 'y_train' from outer scope (line 51) (redefined-outer-name)
src\train.py:28:30: W0621: Redefining name 'y_test' from outer scope (line 51) (redefined-outer-name)
src\train.py:31:4: W0621: Redefining name 'classifier' from outer scope (line 51) (redefined-outer-name)
src\train.py:28:13: C0103: Variable name "X_test" doesn't conform to snake_case naming style (invalid-name)
src\train.py:48:4: C0103: Constant name "bow_path" doesn't conform to UPPER_CASE naming style (invalid-name)
src\train.py:49:20: R1732: Consider using 'with' for resource-allocating operations (consider-using-with)
src\train.py:2:0: W0611: Unused numpy imported as np (unused-import)
src\train.py:3:0: C0411: standard import "import os" should be placed before "import numpy as np" (wrong-import-order)
src\train.py:5:0: C0411: standard import "import pickle" should be placed before "import numpy as np" (wrong-import-order)

------------------------------------------------------------------
Your code has been rated at 3.10/10 (previous run: 3.45/10, -0.34)
```