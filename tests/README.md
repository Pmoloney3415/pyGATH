# Tests

`unit/` contains focused tests of individual modules and numerical operations.
`regression/` contains slower end-to-end tests that load complete configuration
decks, initialize rays, trace them, and reconstruct or deposit the resulting
fields.

Run the groups separately with:

```console
uv run pytest tests/unit
uv run pytest tests/regression
```

Unit tests should verify the core functionality of important functions without
being overly fragile to plotting details or exact options in example decks.
