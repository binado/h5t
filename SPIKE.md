# Spike: minimal annotation resolution (issue #33 §4)

Question: how much of `_compile.py`'s annotation machinery (bytecode name scanning,
weak scope snapshots, frame capture) is load-bearing, and what is the smallest mechanism
that keeps the behaviour users can see? **This branch is a prototype and is not meant
to be merged as-is.**

## Variants

| Variant | `_compile.py` lines | 3.11 (98 + 11 skipped) | 3.14 (109) |
| --- | --- | --- | --- |
| baseline (`main`) | 1095 | 98 pass | 109 pass |
| **B2** minimal: `get_type_hints` / `annotationlib`, `localns=`, no frame capture | 955 (-140) | 38 fail | 41 fail |
| **B3** = B2 + one *transient* `sys._getframe` read per decorator call, never stored | 975 (-120) | 2 fail | 4 fail |
| **B2'** = B3 without `_raised_by_annotation` (any `NameError` defers) | ~953 (-142) | 3 fail | 6 fail |

Deleted in every variant: `_code_names`, `_names_in_source`, `_referenced_names`,
`_quoted_names`, `_annotation_names`, `_record_annotation_names`, `_weak_scope`,
`_unpack_scope`, the `_h5t_scope` lookup, and `inspect.currentframe()` in both decorators
(about 190 lines of deletions against ~67 added). `_PendingRecord` keeps only
`kind/data/attrs/extras_raw/py_name/localns`.

Public API delta: `dataset(..., localns=None)` and `group(..., localns=None)`. Nothing else.

## What B2 breaks, and why B3 is enough

B2's 38 failures are almost all one cause: **a function-local record referencing a
function-local type that already exists** (`class Dep` then `class R: dep: Dep`). With
`from __future__ import annotations` the annotation is a string, and without the decorator's
frame there is nothing to resolve `Dep` against. Reading `f_locals` once, for the immediate
compile, recovers all of them (38 -> 2 on 3.11). Nothing needs to be retained for this
case, because a non-forward reference resolves on the first compile and never defers.

## Remaining failures, grouped by behaviour

| Group | Tests (B3) | Real behaviour lost? |
| --- | --- | --- |
| GC/retention of the scope snapshot (`test_deferred_path_snapshots_the_scope_weakly`, `..._does_not_retain_locals_the_annotations_never_name`, 3.14 `..._mixed_record_snapshots_the_scope...`) | 2 (3.11) / 3 (3.14) | **No** (the tests assert on the deleted `.scope`). Retention is now zero by construction. See "mixed record" below for the one behaviour they protected. |
| PEP 649 deferral test (`test_the_group_decorator_defers_under_pep_649`) | 1 (3.14) | **No.** It asserts `marker(record).scope == {}`; the probe confirms bare function-local forward refs still resolve on 3.14 via the `__annotate__` closure, identically to `main`. |
| Callee-`NameError` classification (B2' only: `test_a_broken_annotation_helper_...`, 3.14 `test_a_broken_lazy_annotation_helper_...`) | 1 (3.11) / 2 (3.14) | **Yes, if `_raised_by_annotation` is dropped.** A buggy helper defers instead of failing at the class statement, and it surfaces at load as `name 'helper' is not defined` (the *local* name, because no scope was kept), hiding the real `undefined_name_xyz`. |
| Class-body names in annotation namespace | 0 tests | Not covered by any test: `vars(cls)` is no longer in the namespace, because `get_type_hints(localns=...)` replaces it. Probably harmless (only matters for a nested class referenced by a string annotation); flag it in the changelog if kept. |

## Behaviour probes (not covered by the suite)

Run against `main` and this branch, on 3.11 and 3.14 (scripts were scratch, results below):

| Scenario | `main` | B3 |
| --- | --- | --- |
| function-local self-reference `child: Node \| None` (string annotations) | **SchemaError** | **works** |
| function-local forward reference (`A` -> `B` defined after) | SchemaError | SchemaError; **works with `localns={"B": B}`** |
| function-local mutual reference | SchemaError | SchemaError (works with `localns=`) |
| function-local non-forward reference | works | works |
| 3.14 bare forward / mutual / self reference in a function | works (closure) | works (closure), identical |
| 3.14 *quoted* forward reference in a function | SchemaError | SchemaError, identical |
| buggy annotation helper | SchemaError at decoration | same in B3, deferred + misleading in B2' |
| **"mixed" record**: deferred on a module-level name defined later *and* uses a function-local type, string annotations | **works** (snapshot) | **SchemaError** `name 'Dep' is not defined` (same on 3.11 and 3.14); works with `localns={"Dep": Dep}` or with bare annotations on 3.14 |

So B3 *gains* function-local self-reference and loses exactly one thing: the mixed record.
That requires a function-scope group record, stringized annotations, a deferral triggered by
a different (module-level, later-defined) name, and a function-local dependency, all at once.

## Findings worth knowing

1. **Seeding the class's own name breaks the decorator unless the class is already marked.**
   `{cls.__name__: cls}` makes `child: Node` resolve at decoration time, before
   `__h5t_record__` exists, so `_field_spec` rejects it as an unsupported annotation.
   The prototype plants the `_PendingRecord` marker *before* compiling and removes it on
   a non-deferral failure. Without this, module-level self-reference also breaks.
2. `typing.get_type_hints` evaluates the annotations of every class in the MRO, not just
   `cls`; the prototype filters to `inspect.get_annotations(cls)` keys, but a broken
   annotation on a base now fails the subclass's compile.
3. Under 3.14, `annotationlib.get_annotations(format=VALUE)` leaves quoted annotations as
   strings, so h5t still evaluates those itself (`eval` against module globals + localns).
4. The `_ANNOTATION_FILE` filename trick no longer covers the <=3.13 path, since
   `get_type_hints` compiles under `<string>`; the prototype accepts both. That is
   fragile (any `exec`'d `<string>` code matches) and is an argument for keeping the
   classification only where h5t does its own `eval`.

## Recommendations

| Behaviour | Recommendation |
| --- | --- |
| Bytecode scanning (`_code_names`, `_quoted_names`, `_annotation_names`, ...) and weak scope snapshots | **Drop.** Their only user-visible effect is the mixed record. |
| Function-local non-forward references (the 38 B2 failures) | **Keep, via the one-shot transient `f_locals` read (B3).** ~20 lines, retains nothing. |
| Mixed record (function scope + string annotations + deferral on another name) | **Require `localns=`**, or drop. If a smaller mechanism is wanted: evaluate string annotations *per field* at the first compile and store the already-resolved values on the `_PendingRecord`, so only the unresolved fields are retried. No bytecode scanning, no weak refs; costs a retained strong reference to resolved types, which is the retention the snapshot was built to avoid, so prefer `localns=`. |
| Callee-`NameError` classification | **Keep.** Without it errors become misleading (see B2'). Scope it to the code h5t evaluates itself (3.14 `__annotate__`, plus the `<string>` filename on <=3.13) and accept the `<string>` fragility, or document that a helper bug on <=3.13 may defer. |
| PEP 649 asymmetries pinned in `test_pep649.py` | **Keep the behaviour** (unchanged); rewrite the two `.scope` assertions to assert the load result instead. |
| `localns=` kwarg | **Keep**, additive. It also fixes forward/mutual references inside functions, which `main` cannot do at all. |

Suggested changelog wording if adopted: "Removed the annotation-scope snapshot: a
function-local `@h5t.group` record that defers on a later-defined name no longer sees
function-local types captured at decoration. Pass `localns={...}` to the decorator.
Function-local self-references now resolve."

## Reproducing

```bash
uv run pytest; uv run --python 3.14 pytest
```
Expected here: 2 and 4 failures (all four assert on the removed `_PendingRecord.scope`
or `_h5t_scope`).
