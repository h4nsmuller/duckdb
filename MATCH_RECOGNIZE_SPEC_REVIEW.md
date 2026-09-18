# MATCH_RECOGNIZE vs ISO/IEC 19075-5 — handover

Reviewing DuckDB's `MATCH_RECOGNIZE` against the spec PDFs in the repo root
(`ISO_IEC_19075-5_2021(en).pdf` and its 2026 corrigendum), fixing what is wrong and testing
what is right. Everything below is on branch **`mr_classifier_nested`** (branched from
`a8e908a50e0`, i.e. main), 5 commits, working tree clean, no PR opened — Hannes opens those.

`test/sql/match_recognize/*` passes: 3659 assertions, 25 files.

## How to work on this

```bash
DUCKDB_EXTENSIONS='json' make reldebug          # json is needed by one test file
build/reldebug/test/unittest "test/sql/match_recognize/*"
```

Extract the spec text (needs `pypdf`):

```python
from pypdf import PdfReader
t = "\n".join(p.extract_text() or "" for p in PdfReader("ISO_IEC_19075-5_2021(en).pdf").pages)
```

Headings repeat (table of contents + body + running headers), so search for a distinctive
sentence rather than the heading, and strip the `Licensed to DuckDB Labs...prohibited.` and
`© ISO/IEC 2021...` boilerplate that is interleaved into the text.

**Cross-check every finding against Trino** before changing anything — it implements the spec
closely and disagreements are usually mine:

```bash
docker run --rm -d --name trino-mr -p 8080:8080 trinodb/trino:latest
trino --server localhost:8080 --output-format=TSV --execute "WITH t(i,v) AS (VALUES (1,5)) SELECT ..."
```

Trino uppercases classifier labels and prints doubles as `1.0E0`; normalise before diffing.

## Done on this branch

| commit | what |
|---|---|
| `a87f9a77a30` | `CLASSIFIER(variable)`; `FIRST`/`LAST` nested in `PREV`/`NEXT` |
| `9fc0d8342f1` | test: the composed shape a real query has |
| `9fdde0a3236` | expression as a navigation operand; refuse matcher state in a step |
| `deb8031ff44` | refuse `{- -}` with `ALL ROWS PER MATCH WITH UNMATCHED ROWS` |
| `4aaa9b41c15` | validate `PREV`/`NEXT` offsets; step from a pattern variable's row |

Details worth knowing:

- **`CLASSIFIER(V)`** (§5.9) reads the classifier as a prefixed column reference does — off the
  last row `V` matched, running semantics, NULL before it has matched. Implemented in MEASURES.
  In DEFINE it now says it is unsupported there rather than failing catalog lookup; Trino does
  support it in DEFINE, which would need the matcher to hand back another variable's classifier
  mid-decision.
- **Nesting** (§5.6.4): `PREV(LAST(A.x), n)` is bound by rewriting to `LAST(A.<x stepped back n>)` —
  the step is the same for every row, so taking it below the matcher and picking the row afterwards
  reaches the same row. The operand may be a compound expression (`A.Price + A.Tax`); every
  qualifier in it must name the same variable. The spec's worked example returns **11**, as it says.
- **Offsets**: `PREV(p,-1)` used to silently behave as `NEXT(p,1)` and `PREV(p,NULL)` to yield NULL,
  in both MEASURES and DEFINE. They now go through the same constant check `FIRST`/`LAST` used.

Two existing tests encoded behaviour the spec contradicts, and were changed — **worth a second
look**: `test_match_recognize_standard.test` asserted `{- -}` + `WITH UNMATCHED ROWS` working, and
`test_match_recognize_navigation.test` asserted `PREV(A.v)` was an error.

## Sections still to read

Covered so far: §4.18 prohibited nesting, §5.3 (partly), §5.5, §5.6, §5.9, §6 (skipped, see below),
§4.6.3 and §4.17.3 declared column lists, the corrigendum (purely editorial — example syntax, a
correlation name, a cross-reference).

Not yet read closely:

- **§5.3 / §5.4 running vs final** — only the opening paragraphs. Four pages, the subtlest area
  and the most likely place for a real semantic divergence. Start here.
- **§4.12.2 empty matches, §4.12.3 unmatched rows** — read the surrounding prose; the worked
  examples and their result tables are good test material and the corrigendum rewrites both.
- **§4.13 AFTER MATCH SKIP**, **§4.14.2 PERMUTE**, **§4.15 SUBSET**, **§4.16 DEFINE**,
  **§5.7 ordinary column references reconsidered**, **§5.8 MATCH_NUMBER**.
- **§4.3 / §4.4** the two headline worked examples with full result tables — these are directly
  usable as spec-derived tests and nothing in the suite is taken from them.

## Known gaps, deliberately not fixed

- **`PREV(CLASSIFIER())`** — §5.9 permits it, with a DEFINE example; Trino supports it. DuckDB now
  reports it is unsupported rather than failing catalog lookup. Implementing it needs a second
  hoisting level *above* the matcher: `HoistMeasureNavigation` computes a step below the matcher,
  where no row has been classified yet.
- **Feature R020**, `MATCH_RECOGNIZE` in a `WINDOW` clause (§6) — parser error. Looks like a
  deliberate R010-only scope, worth confirming with Hannes before touching.
- **`COUNT(A.*)`** (§5.5) — separate PR #25823 already covers it.

## Spec rules DuckDB is deliberately more permissive about

All return correct answers; reported rather than changed. A joined table as the pattern input,
outer references in MEASURES/DEFINE, `MATCH_RECOGNIZE` nested in MEASURES, recursive CTEs,
`PREV(1)`/`FIRST(1)` with no column reference, and `sum(Price + A.Tax)` mixing an unqualified
reference (implicitly the universal variable) with a qualified one — the last is a spec syntax
error that DuckDB accepts.

## Related work elsewhere

- **PR #25874** (branch `mr_hairy`) — a wide-measures test and benchmark, plus a binder fix letting
  `FIRST`/`LAST` take an expression with a pattern-variable qualifier anywhere inside. That fix
  turns out to satisfy a spec rule found later: `FIRST(A.Price + B.Tax)` is a syntax error, and on
  `mr_hairy` it reports so clearly. CI was green (44 passed) at last check.
- **PR #25823** — `COUNT(x.*)`.
- Unused-window-expression pruning was implemented here and then **dropped**: merged PR #25770 does
  it already on `v2.0-cyanoptera`, and better (it preserves volatile expressions such as `nextval`,
  which the version here pruned). Do not re-add it.
