# What's next

Status as of this writing: **the plain-class-records plan (PR1-3, plus the 0.3 docs step) is
implemented, tested, and committed** on `feat/payload-foreign-records`. `@h5t.dataset` and
`@h5t.group` cover the whole schema surface — a root, a nested group, or a nested dataset can all
load into a plain class — and `Group`/`Dataset` inheritance is documented in `README.md` as the
legacy path. What remains is the deprecation rollout itself, which the plan always scoped as later
releases, not this sitting.

## Done

- **PR1** (commit `fcfa92c`) — closed the capability gap between `Payload` and `Dataset`:
  `Payload.data_attr` renamed to `data`; added `attrs: str | None = None`; `Eager` may combine
  with `Payload` when the payload field is `LazyArray`.
- **PR2** (commit `8638b8f`) — added `@h5t.dataset(data=, attrs=, extras=)`, sugar for
  `Annotated[T, Payload(...)]` that validates eagerly against the record's own definition.
- **PR3** (commit `3f761d4`) — added `@h5t.group(attrs=, extras=)` and `h5t.load(schema, path,
  root)`. `RecordKind` (`DATASET`/`GROUP`) splits `ForeignSpec` compilation: a `DATASET`-kind
  record (`Payload`, `@h5t.dataset`) stays eager since it may only declare attributes; a
  `GROUP`-kind record (`@h5t.group`) defers like `Group`/`Dataset`, since it may reference other
  schema types, including itself (`child: Node | None`). A `GROUP`-kind member resolves its
  `ForeignSpec` at load time rather than compile time — the same thing a `Group`-subclass member's
  `__h5spec__` already does — so a self-referential schema terminates instead of recursing forever
  at compile time. `Group.from_file` and `h5t check` are now thin wrappers around `h5t.load`.
- **0.3 docs step** (same sitting as PR3) — `README.md` now leads with `@h5t.dataset`/
  `@h5t.group`/`h5t.load`, documents `Group`/`Dataset` inheritance as a "Legacy" section, and no
  longer claims foreign annotations can't capture a defining frame or that foreign groups are out
  of scope — both were true before PR2/PR3 and are not anymore. **No runtime warning was added**;
  that is explicitly 0.4 work, below.

## Left to implement

### 0.4 (a later release) — deprecate `Group`/`Dataset` inheritance

Not started. Per the original plan this is deliberately a separate release from 0.3, so the
warning has a changelog entry of its own and users get a version where the message is new but
nothing breaks yet:

- Add `DeprecationWarning` from `_Record.__init_subclass__`, guarded by the existing
  `cls.__module__ == __name__` check so `Group`/`Dataset` themselves (and any class compiled
  before the warning is added, if that ever matters) stay silent. Fire it once per subclass
  declaration, not per instance/load, to avoid spamming a hot load loop.
- Also warn from `Group.from_file` (or fold that into `h5t.load`'s `Group`-subclass branch) so a
  schema declared before this release — already on disk in a user's own module — still warns when
  used, not just when redeclared.
- Add `filterwarnings` entries to `pyproject.toml`'s `[tool.pytest.ini_options]` so the existing
  suite doesn't fail under `-W error`: every `tests/*.py` file that still declares an
  `h5t.Group`/`h5t.Dataset` subclass (most of them — `conftest.py`'s `Result`/`Measurement`/
  `Nested`, `test_loading.py`, `test_dataset.py`'s inline classes, etc.) needs the warning
  suppressed rather than the tests themselves migrated, since the inheritance path is still
  supported and still worth testing directly through 1.0.
- Update `README.md`'s "Legacy" section to say the warning now fires, with a one-line migration
  example (inheritance form next to its `@h5t.dataset`/`@h5t.group` equivalent).

### 1.0 (a later release) — remove the legacy path

- Remove `Group`, `Dataset`, `Eager`-on-`Dataset`, and `_reserved_names`/`_check_fields`'s
  reserved-name half (a `@h5t.dataset`/`@h5t.group` record never needed it — `dir()` on a plain
  class has nothing to forbid).
- `_compile.py` loses `_load_dataset`, `_load_group`'s `object.__new__`/`object.__setattr__`
  construction path, `_compile_class`/`_merged_fields`/`_resolve_extras`/`SchemaMeta`'s deferral
  machinery for `Group`/`Dataset` specifically (the `@h5t.group` deferral path — `_PendingRecord`,
  `_ensure_record_compiled` — stays, since it isn't inheritance-specific).
- Decide then whether `RecordKind` can collapse away entirely (only meaningful if every schema is
  a decorated record) or whether it's still pulling weight for some other reason discovered along
  the way.
- Rewrite `README.md`/`AGENTS.md` to drop the "Legacy" section and every remaining inheritance
  example.

## Resuming

The design decisions behind PR3 (why `DATASET` stays eager and `GROUP` defers, why a `GROUP`-kind
member resolves at load time rather than compile time) are recorded in `AGENTS.md`'s "Foreign
record types" section — read that before touching `_foreign_spec`/`_compile_foreign_fields` again.
Nothing here for 0.4/1.0 has open design questions the way PR3 did; both are mechanical once
started, gated only on wanting to actually ship a breaking-ish change to users.
