### DSLinter output for src/evaluate.py file
Run with `pylint --load-plugins=dslinter src/evaluate.py`

```
************* Module evaluate
src\evaluate.py:16:0: C0304: Final newline missing (missing-final-newline)
src\evaluate.py:1:0: C0114: Missing module docstring (missing-module-docstring)
src\evaluate.py:4:27: C0103: Argument name "X_test" doesn't conform to snake_case naming style (invalid-name)
src\evaluate.py:14:4: C0103: Variable name "cm" doesn't conform to snake_case naming style (invalid-name)

------------------------------------------------------------------
Your code has been rated at 3.33/10 (previous run: 3.33/10, +0.00)
```