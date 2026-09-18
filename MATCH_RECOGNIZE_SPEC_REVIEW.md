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

## Second pass (2026-09-18): sections read, deviations found

Read closely this round: **§4.3, §4.4** (both headline worked examples), **§4.12.2** (empty
matches, all three result tables), **§4.12.3** (unmatched rows), **§4.13** (AFTER MATCH SKIP),
**§4.14.1** (PATTERN syntax and precedence), **§5.3** (running vs final, all four pages),
**§5.4** (RUNNING/FINAL keywords), **§5.7** (ordinary column references), **§5.8**
(MATCH_NUMBER), **§7.2.6–7.2.8** (anchors, empty pattern, infinite repetitions of empty
matches).

### Fixed on this branch

Each was confirmed against **both Trino and Oracle** before changing anything — the differential
harness runs the same query on all three and diffs the answers.

| # | what | spec |
|---|---|---|
| 2 | `{- -}` was refused with ONE ROW PER MATCH; it is permitted and simply has no effect | §4.14.3 |
| 3 | `SELECT *` under ALL ROWS PER MATCH reported the input's own column order with the measures appended; it now reports partitioning columns, ordering columns, measures, then the rest of the input | §4.17.1 |
| 4 | a SUBSET could be built out of a single-member SUBSET (`U = (A), V = (U, C)`); unions of unions are refused whatever the inner one's size | §4.15 |
| 5 | PREV/NEXT over a pattern variable was refused in DEFINE — `PREV(A.c)`, `PREV(LAST(A.c), n)`, `PREV(FIRST(A.c), n)` — although the same nesting already worked in MEASURES | §5.6.4 |

Fix 5 applies the rewrite MEASURES already used: an ordinary qualified reference is the last row the
variable matched, so `PREV(A.c, n)` is `PREV(LAST(A.c), n)`, and `PREV(FIRST(X.c), n)` is
`FIRST(X.<c stepped back n>)` — the step is the same for every row, so taking it below the matcher
and letting FIRST pick the row afterwards reaches the same value. Trino and Oracle agree with the
new answers on all ten spellings tested.

Fix 3 changes user-visible output for `SELECT *`, which is why three existing tests had to be
re-baselined (`standard`, `names`, `shapes`). Worth a second look.

### Deviations still open

1. **Aggregates are not supported in DEFINE at all.** §5.4 is explicit: "Aggregates, FIRST, and
   LAST can occur in ... the DEFINE clause. When processing the DEFINE clause, the engine is
   still in the midst of recognizing a match; therefore, the only supported semantics is
   running." DuckDB's `MatchRecognizeDefineBinder::BindAggregate` rejects every aggregate with
   *"A MATCH_RECOGNIZE condition decides one row at a time, so it cannot be an aggregate"*.
   Three spec examples cannot run:
   - §5.3's central illustration, `PATTERN (A+) DEFINE A AS A.Price >= AVG (A.Price)`, whose
     evaluation the spec walks through row by row and whose result is Table 12.
   - §5.3's empty-set example, `DEFINE B AS B.Price > COUNT (A.*) * 50`.
   - §5.3's forward-reference example, `DEFINE X AS COUNT (Y.*) > 3`.
   This is the semantic divergence the first pass predicted would be in §5.3.

2. **`RUNNING` is rejected in DEFINE.** §5.4 rule 3: "In DEFINE, FINAL is not permitted; RUNNING
   may be used for added clarity if desired." DuckDB rejects both with *"RUNNING and FINAL are
   only meaningful in the MEASURES of a MATCH_RECOGNIZE"*. Refusing FINAL is right; refusing
   RUNNING is not. Reachable independently of finding 1, since RUNNING may also precede
   FIRST/LAST: `DEFINE A AS A.x >= RUNNING LAST(A.x)`.

3. **Quantifier bounds are more permissive than the grammar.** §4.14.1 gives `{n}` with n > 0
   and `{,m}` with m > 0; DuckDB accepts `A{0}` and `A{,0}`. Harmless (both behave as an empty
   match) and arguably better than erroring, so reported rather than changed.

4. **Low confidence, probably editorial.** §4.12.2 and §4.12.3 write `CLASSIFIER AS Classy` and
   `FIRST A.Price AS Firstp` without parentheses, which DuckDB rejects. §4.4 writes
   `CLASSIFIER()` and §4.3 writes `LAST (A.Price)`, so the bare forms look like slips in the
   explanatory text rather than grammar. Worth a look at ISO/IEC 9075-2 before acting.

5. **[FIXED] `{- -}` is refused with ONE ROW PER MATCH.** §4.14.3: "The exclusion syntax is permitted
   with ONE ROW PER MATCH, though it has no effect since in this case there is only a single
   summary row per match." DuckDB raises *"Pattern exclusion syntax {- -} requires ALL ROWS PER
   MATCH"*, for ONE ROW PER MATCH and for the default. `ValidateClauses` tests
   `!MatchRecognizeReportsRows(...)`; per the spec only the WITH UNMATCHED ROWS case should be
   refused, which the same function now also handles. `test_match_recognize_errors.test`
   currently pins the wrong behaviour.

6. **[FIXED] `SELECT *` column order under ALL ROWS PER MATCH.** §4.17.1 requires partitioning columns,
   then ordering columns, then measure columns, then the remaining input columns — "designed to
   facilitate comparing the output when the query is toggled between ONE ROW PER MATCH and ALL
   ROWS PER MATCH". DuckDB emits the input table's own column order with the measures appended
   last. On input `(price, extra, sym, day)` with `PARTITION BY sym ORDER BY day`:
   spec `sym, day, mno, price, extra`; DuckDB `price, extra, sym, day, mno`. ONE ROW PER MATCH
   is correct (partitioning then measures).

7. **[FIXED] Union of a union slips through when the inner union has one member.** §4.15: "the list of
   row pattern variables on the right hand side cannot include any union row pattern variables
   (there are no unions of unions)". `SUBSET U = (A, B), V = (U, C)` is correctly refused, but
   `SUBSET U = (A), V = (U, C)` is accepted — `BuildMeasureSymbols` rejects a member only when
   its expansion has more than one symbol.

8. **`SUBSET U = (A, A)` is accepted** where §4.15 requires a list of *distinct* primary
   variables. Harmless, a union is a set either way.

9. **Mixed qualifiers inside an aggregate are accepted.** §5.9 gives
   `ARRAY_AGG (CLASSIFIER () || A.Name)` as a syntax error, since CLASSIFIER() references the
   universal variable and `A.Name` references A; DuckDB evaluates it. Same family as the
   already-known `sum(Price + A.Tax)` permissiveness (§5.2), so one fix would cover both.

### Verified conforming

Every one of these was run against the implementation, not read off the source:

- §4.3 Table 2 and §4.4 Table 3 reproduce exactly.
- §4.12.2 Tables 6, 7 and 8 reproduce exactly, **including empty-match numbering** — matches 1
  through 11 with the gaps that OMIT EMPTY MATCHES creates, which is the subtle part.
- §4.12.3 Table 9 reproduces exactly, unmatched rows null-extended.
- §4.13: an empty match skips exactly one row and never raises the SKIP TO exception, even for
  `AFTER MATCH SKIP TO LAST A` over `PATTERN (A*)` where A never matches.
- §4.14.1: `A**` is refused, `(A*)*` is accepted, and all the reluctant range forms
  (`{n}?`, `{n,}?`, `{,m}?`) parse.
- §5.7: an ordinary column reference is `RUNNING LAST` under ALL ROWS PER MATCH and `FINAL LAST`
  under ONE ROW PER MATCH.
- §5.8: MATCH_NUMBER() in DEFINE reproduces the alternating `(A+ | B+)` example; MATCH_NUMBER()
  and CLASSIFIER() outside MEASURES/DEFINE are refused.
- §7.2.7/§7.2.8: the empty pattern `()` matches an empty set of rows, and Perl's stopping rule
  holds — `PATTERN ((A??)* B)` with an unmatchable A terminates and matches B alone, and
  `(A?){0,3}` and `(A?){2,3}` terminate on an unmatchable A.

- §4.14.3: excluded rows still drive AFTER MATCH SKIP — with `PATTERN ({- A -} B+ {- C -})`
  and SKIP PAST LAST ROW, matching resumes after the excluded C row, not after the last B.
- §4.15/§4.16: an unqualified column reference in DEFINE is the current row, i.e. `A AS price > 11`
  behaves as `A AS A.price > 11`.
- §5.9: `CLASSIFIER(AB)` over a SUBSET returns the classifier of the last row mapped to A or B and
  null before any, and `ARRAY_AGG(CLASSIFIER() ORDER BY ...)` under ONE ROW PER MATCH returns the
  whole classifier sequence.
- §7.2.2/§7.2.4 preferment, which is the subtlest part of the engine and all of it holds:
  `V{2,4}` takes 4 rows and `V{2,4}?` takes 2; `A{1,2} | B{2,3}` prefers the first alternative;
  and for `(A|B){1,2}` the greedy form gives `AA` while the reluctant form gives a singleton `A`
  rather than `AA` or `B` — the spec's rule that lexicographic order is the first priority and
  length only the second.

The spec-derived queries are in this branch's history only as a scratch script; turning §4.3,
§4.4, §4.12.2 and §4.12.3 into a `test_match_recognize_spec.test` is the obvious next step —
nothing in the suite is currently taken from the standard's own result tables.

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
