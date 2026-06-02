# DeepRetro Package `src` Dependency Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the remaining package-level dependency on `src` so `deepretro.utils.az` relies only on package-local modules and docs describe the package behavior accurately.

**Architecture:** Keep the change narrowly scoped to the package boundary. Add a package-local decorator in `deepretro.utils.cache` that mirrors the existing cached-call ergonomics used by `deepretro.utils.az`, switch `az.py` to import `BASIC_MOLECULES` and `cache_results` from `deepretro`, and update the package docs/tests to prove the package no longer imports from `src`.

**Tech Stack:** Python 3.12, pytest, AiZynthFinder package tests, Sphinx docs, package-local cache helpers in `deepretro.utils.cache`

---

### Task 1: Add a regression test for package-local caching in `az.py`

**Files:**
- Modify: `deepretro/tests/test_cache.py`
- Test: `deepretro/tests/test_cache.py`

- [ ] **Step 1: Write the failing test**

```python
def test_cache_results_decorator_caches_repeated_calls() -> None:
    from deepretro.utils.cache import cache_results

    calls = {"count": 0}

    @cache_results
    def compute(smiles: str, *, az_model: str = "USPTO") -> dict[str, object]:
        calls["count"] += 1
        return {"smiles": smiles, "az_model": az_model, "count": calls["count"]}

    first = compute("CCO", az_model="USPTO")
    second = compute("CCO", az_model="USPTO")
    third = compute("CCN", az_model="USPTO")

    assert first == second
    assert first["count"] == 1
    assert third["count"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH='C:\Users\gask4\DeepRetro\.worktrees\package-src-decoupling'
pytest deepretro/tests/test_cache.py -q
```

Expected: FAIL with `ImportError` or `AttributeError` because `cache_results` is not exported from `deepretro.utils.cache`.

- [ ] **Step 3: Write minimal implementation**

Add a module-level in-memory decorator backed by a private `CacheManager` instance in `deepretro/utils/cache.py`:

```python
_FUNCTION_CACHE = CacheManager()


def cache_results(func):
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        cache_key = make_cache_key(func.__name__, *args, **kwargs)
        cached = _FUNCTION_CACHE.get(cache_key, default=_MISS)
        if cached is not _MISS:
            return cached
        result = func(*args, **kwargs)
        _FUNCTION_CACHE.set(cache_key, result)
        return result

    return wrapper
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
$env:PYTHONPATH='C:\Users\gask4\DeepRetro\.worktrees\package-src-decoupling'
pytest deepretro/tests/test_cache.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```powershell
git add deepretro/tests/test_cache.py deepretro/utils/cache.py
git commit -m "port package cache decorator"
```

### Task 2: Prove `deepretro.utils.az` no longer depends on `src`

**Files:**
- Modify: `deepretro/tests/test_az.py`
- Modify: `deepretro/utils/az.py`
- Test: `deepretro/tests/test_az.py`

- [ ] **Step 1: Write the failing test**

Add a test that reads the module source and asserts package-local imports:

```python
def test_az_module_uses_package_local_imports() -> None:
    source = AZ_MODULE_PATH.read_text(encoding="utf-8")
    assert "from src.variables import BASIC_MOLECULES" not in source
    assert "from src.cache import cache_results" not in source
    assert "from deepretro.utils.variables import BASIC_MOLECULES" in source
    assert "from deepretro.utils.cache import cache_results" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH='C:\Users\gask4\DeepRetro\.worktrees\package-src-decoupling'
pytest deepretro/tests/test_az.py -q -k package_local_imports
```

Expected: FAIL because `az.py` still imports from `src`.

- [ ] **Step 3: Write minimal implementation**

Update `deepretro/utils/az.py` imports:

```python
from deepretro.utils.cache import cache_results
from deepretro.utils.variables import BASIC_MOLECULES
```

Leave runtime behavior unchanged beyond import ownership.

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
$env:PYTHONPATH='C:\Users\gask4\DeepRetro\.worktrees\package-src-decoupling'
pytest deepretro/tests/test_az.py -q -k package_local_imports
```

Expected: PASS

- [ ] **Step 5: Commit**

```powershell
git add deepretro/tests/test_az.py deepretro/utils/az.py
git commit -m "decouple package az imports from src"
```

### Task 3: Update package docs to describe package-local caching

**Files:**
- Modify: `docs/source/package/deepretro.utils_pkg.rst`
- Test: `docs/source/package/deepretro.utils_pkg.rst`

- [ ] **Step 1: Write the failing documentation assertion**

Use a text search as the failing check:

```powershell
rg -n "src\.cache\.cache_results" docs/source/package/deepretro.utils_pkg.rst
```

Expected: One match in the AiZynthFinder integration notes.

- [ ] **Step 2: Write minimal documentation update**

Replace the legacy wording with package-local wording:

```rst
- Caching via ``deepretro.utils.cache.cache_results`` decorator.
```

- [ ] **Step 3: Run the documentation assertion to verify it passes**

Run:

```powershell
rg -n "src\.cache\.cache_results" docs/source/package/deepretro.utils_pkg.rst
```

Expected: no matches

- [ ] **Step 4: Commit**

```powershell
git add docs/source/package/deepretro.utils_pkg.rst
git commit -m "update package cache docs"
```

### Task 4: Run focused verification and prepare branch for review

**Files:**
- Verify: `deepretro/tests/test_cache.py`
- Verify: `deepretro/tests/test_az.py`
- Verify: `deepretro/utils/az.py`
- Verify: `deepretro/utils/cache.py`
- Verify: `docs/source/package/deepretro.utils_pkg.rst`

- [ ] **Step 1: Run focused package tests**

Run:

```powershell
$env:PYTHONPATH='C:\Users\gask4\DeepRetro\.worktrees\package-src-decoupling'
pytest deepretro/tests/test_cache.py deepretro/tests/test_az.py -q
```

Expected: all selected tests pass, with slow/integration tests skipped only when optional dependencies are absent.

- [ ] **Step 2: Verify the package no longer imports from `src`**

Run:

```powershell
rg -n "from src\.variables import BASIC_MOLECULES|from src\.cache import cache_results|src\.cache\.cache_results" deepretro docs -g "!**/__pycache__/**"
```

Expected: no matches under `deepretro/` and no legacy docs reference in `docs/source/package/deepretro.utils_pkg.rst`

- [ ] **Step 3: Review git diff**

Run:

```powershell
git status --short
git diff -- deepretro/tests/test_cache.py deepretro/tests/test_az.py deepretro/utils/cache.py deepretro/utils/az.py docs/source/package/deepretro.utils_pkg.rst
```

Expected: only the planned package-decoupling changes are present.

- [ ] **Step 4: Push branch to fork**

```powershell
git push -u origin package-src-decoupling
```

- [ ] **Step 5: Produce compare URL for PR creation**

Use:

```text
https://github.com/deepforestsci/DeepRetro/compare/main...Darkxzie:DeepRetro:package-src-decoupling?expand=1
```
