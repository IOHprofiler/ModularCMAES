# Mixed-integer problems

`Settings(integer_variables=[...])` marks coordinates as integers: they are rounded before every
evaluation, and their step size has a lower bound so the search keeps moving between integers
([integer variables](../guide/integer-variables.md)).

```python
--8<-- "examples/mixed_integer.py"
```

Output:

```text
best x = [ 7.         12.          0.40540541 -1.25      ], f = 0.141892, after 4992 evaluations
```
