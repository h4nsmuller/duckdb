# NUMERIC: one exact numeric type with an inferred representation

Status 2026-10-12: design settled, implementation not started beyond reserving the enum slot.
Branch `numeric-type` of h4nsmuller/duckdb (worktree /home/ubuntu/duckdb/.claude/worktrees/numeric-type,
base upstream/v2.0-cyanoptera 2658ef961f6). No upstream PR (user decision: design first, not before Wednesday).
Arbitrary-length decimals (bignum-decimal) are a separate session (worktree `bignum-arithmetic`); this design
assumes they exist as the widest representation.

## The idea (user, 2026-10-12)

One exact numeric logical type, `NUMERIC`, that holds any signed/unsigned integer or decimal of any width.
**All other exact numeric types become aliases.** `INTEGER`, `DECIMAL(w,s)` etc. only add a catalog restriction
on the column, like a CHECK constraint. The engine decides **per table**, from statistics (min, max, max scale),
which physical type and kernel to use. Storage keeps each row group in the narrowest type that fits its values.

## Decisions

1. **The type decision lives in the binder, not the optimizer.** Result types are copied into the prepared
   statement before the optimizer runs (`src/main/client_context.cpp:497` vs the optimizer call at `:516`), and
   function overloads are resolved during binding. The optimizer's statistics propagator keeps its existing job:
   choose a cheaper *kernel* (`src/function/scalar/operator/arithmetic.cpp:200` swaps checked add for unchecked
   add when min/max prove no overflow). It never changes types, and must not.
2. **A representation is an existing concrete type.** A NUMERIC column resolves at bind time to one of
   TINYINT..HUGEINT, DECIMAL(w,s), or BIGNUM(-decimal). Nothing new flows through vectors, functions, casts,
   Arrow, Parquet, the C API or clients. DECIMAL is the existing proof: its kernels are picked by a bind
   callback from the argument's internal type (`BindDecimalAddSubtract`, `arithmetic.cpp:351`), and the C API
   already exposes `duckdb_decimal_internal_type`. The existing type names are the *names of representations*,
   so `typeof(x)` on a NUMERIC column returning `BIGINT` is coherent, not a leak.
3. **Restricted types resolve to themselves.** A declared INTEGER is NUMERIC + range restriction and always
   resolves to INT32; DECIMAL(10,2) always resolves to DECIMAL(10,2). The restriction must be "coerce to the
   declared representation, then check" (today's cast semantics: round to scale, error on overflow), not a bare
   CHECK, or `INSERT 1.005 INTO DECIMAL(10,2)` would reject instead of yielding 1.01. Consequence: existing
   tables, the 1803 test files that mention INTEGER/BIGINT, and every client see no change. Only a bare NUMERIC
   column infers from data. This is what makes the "all aliases" end state *smaller* than a new runtime type.
4. **Per-row-group storage type, chosen at checkpoint.** Storage survey result: neither GEOMETRY nor VARIANT
   does per-segment typing. Both choose one type per ColumnData (= per row group) at checkpoint, rebuild via
   scan + re-append, and record it once in `PersistentColumnData::extra_data` (field 99, forward compatible).
   True per-segment typing would need a DataPointer format change gated by storage version plus a hand-written
   deserializer, because segment statistics take their type from the deserializer context
   (`base_statistics.cpp:434`, `data_pointer.cpp:32`). Bitpacking already stores the narrowest width per 2048
   values inside a segment, so per-row-group type loses nothing measurable against per-segment.
5. **Inference never shrinks between checkpoints; widening is a catalog event.** Column stats only widen on
   append and are not shrunk by rollback (`ColumnData::RevertAppend`, `LocalStorage::Rollback` only truncate).
   Resolved type R = Resolve(main table stats merged with the transaction's local-storage stats); note
   `DataTable::GetStatistics` (`data_table.cpp:1813`) covers the main collection only, local appends live in
   `LocalTableStorage` and merge at commit. When an append widens R, bump the catalog version so prepared
   statements rebind through the existing `CheckCatalogIdentity` path (`prepared_statement_data.cpp:28,96`);
   prepared statements never rebind on data changes otherwise. The scan always casts storage type -> bound
   type; narrow -> wide is lossless; a wide storage type meeting a narrower bound type throws (unreachable after
   the version bump, kept as a safety net).
6. **Insert target type = Widen(R, source type)** via `LogicalType::MaxLogicalType` (INTEGER + DECIMAL(5,2) ->
   DECIMAL(12,2)). The appended vector arrives in that type; the row group's inner column is widened by rebuild
   (rare, at most a handful of steps per column lifetime). `x::NUMERIC` in expressions: identity for numeric
   inputs; non-numeric inputs wait for bignum-decimal (slice 1 throws a clear error). Affected existing tests:
   `test/sql/types/decimal/decimal_values_warning.test` (`NULL::NUMERIC`, today DECIMAL(18,3)),
   `test/sql/pg_catalog/pg_type.test` (type name listing).

## Resolution rule Resolve(stats)

- no stats / empty: INTEGER (friendliest default for clients; open question whether SMALLINT or BIGINT).
- max_scale == 0: narrowest *signed* integer type covering [min, max] (TINYINT..HUGEINT). Unsigned
  representations are never chosen (HUGEINT instead) to avoid unsigned arithmetic surprises.
- max_scale > 0: DECIMAL(width, max_scale), width = digits(max(|min|,|max|)) + max_scale clamped to
  [max_scale + 1, 38].
- beyond HUGEINT / DECIMAL(38): BIGNUM for integers, bignum-decimal later; slice 1 throws NotImplemented.

## Hook points (all verified in the 2.0 tree)

Type system
- `src/include/duckdb/common/types.hpp`: `LogicalTypeId::NUMERIC = 111` (TUPLE is 110) and
  `LogicalType::NUMERIC` constant. DONE on the branch (only change so far).
- `src/common/types.cpp`: `ToString` falls back to `EnumUtil::ToString(id())` so "NUMERIC" prints once
  `python3 scripts/generate_enum_util.py` is re-run; `InternalType` falls through
  `TryGetBuiltinPhysicalType` (`types.cpp:41,181`) to INVALID, which is what we want: no vector ever has type
  NUMERIC, so a NUMERIC-typed vector fails loudly. Leave `IsNumeric()` false for NUMERIC.
- `src/catalog/default/default_types.cpp`: BUILTIN_TYPES entry `{"numeric", DECIMAL, RegisterDecimalConstructors}`
  becomes NUMERIC with two constructors: identity (bare NUMERIC) and the DECIMAL `(width, scale)` constructor
  (`RegisterDecimalConstructors` pattern with `TypeConstructor::Signature`).
- `GetTypeIdSize(PhysicalType::INVALID)` throws (`types.cpp:495`) and is called on column types in
  `LocalTableStorage::EstimatedSize` (`local_storage.cpp:121`); special-case NUMERIC there.

Statistics (new `StatisticsType::UNBOUNDED_NUMERIC_STATS`, data {hugeint min, max at `scale`, uint8 scale,
flags has_min_max/overflow}; union member in `base_statistics.hpp:150-200`)
- `GetStatsType` special-case before the physical switch (`base_statistics.cpp:76`, like GEOMETRY).
- Switches to extend: Merge :154, CreateUnknownType :213, CreateEmptyType :234, Serialize :395 (fields 200+),
  Deserialize :428, ToStruct :474, Verify :513 (no-op), FromConstantType :567.
- `NumericStats::Merge` asserts equal types (`numeric_stats.cpp:41`), so inner typed stats must be converted
  (via Value, rescale) when merged into the NUMERIC column/table stats: `ColumnAppendState::FlushSegmentStats`
  (`column_data.cpp:523`), `FinalFlush` :540, `InitializeColumn` :885, `ColumnCheckpointState::FlushSegmentInternal`
  (`column_checkpoint_state.cpp:150`), `ColumnDataFinalizeAppendState` type switch (`column_data.cpp:550-574`).
- Zonemaps: filters are `ExpressionFilter`s evaluated against stats (`expression_filter.cpp:195`), so stats
  handed to a filter must be typed as the filter's bound column type R. Helper needed:
  ConvertNumericStats(source, R) -> NumericStats(R). Row-group level `RowGroup::CheckZonemap` (`row_group.cpp:732`)
  uses the column's own stats; segment level `ColumnData::CheckZonemap` (`column_data.cpp:435`) uses
  `state.current`, which a wrapper column does not have (GeoColumnData leaves this un-overridden and has a TODO).

Storage (`NumericColumnData`, modelled on `GeoColumnData`, `src/storage/table/geo_column_data.cpp`, 722 lines)
- Dispatch: `ColumnData::CreateColumn` (`column_data.cpp:1369-1391`), add a NUMERIC branch before the physical switch.
- Wrapper owns one inner typed `ColumnData` (inner type = current storage type of this row group), no segments
  of its own. Overrides: InitializeScan/Scan/ScanCount/Skip/Fetch/FetchRows (scan inner into a temp vector of the
  inner type, `VectorOperations::Cast` into the result vector of type R when they differ; GeoColumnData::Scan
  :58-78 is the pattern), InitializeAppend/Append/FinalizeAppend/RevertAppend (widen inner column by rebuild when
  the incoming vector type is wider, cast up when narrower; ColumnData::Append has no type check, the
  uncompressed append reads `sizeof(T)` of the segment's type, so a mismatch is a silent misread), CheckZonemap
  (both overloads), Filter/Select (defaults at `column_data.cpp:387,396` scan into `result` then
  `ColumnSegment::FilterSelection`; reuse after the cast), Update/UpdateColumn (cast to inner type when it fits,
  NotImplemented otherwise in slice 1; `UpdateSegment` is bound to one physical type, `update_segment.cpp:25-40`,
  `RowGroup::Update` asserts type ids at `row_group.cpp:1268`), Checkpoint (pick the narrowest type from the row
  group's stats, rebuild like `GeoColumnData::Checkpoint` :345-546, record in extra_data), InitializeColumn
  (`:570-606` pattern: create inner column of the stored type, convert stats), Serialize, IsPersistent,
  HasAnyChanges, GetMaxEntry, GetColumnSegmentInfo, Verify, VisitBlockIds.
- Persistence: new `ExtraPersistentColumnDataType::NUMERIC` (`column_data.hpp:299`) with a `LogicalType
  storage_type`; `GetPersistentColumnDataType` (`column_data.cpp:978-1065`) must return
  `PersistentColumnData(storage_type)` and `PersistentColumnData::Deserialize` must set the deserializer's
  LogicalType context to the storage type before reading `data_pointers` (the GEOMETRY branch at :1086 shows how).
  `ColumnSegment::CreatePersistentSegment` takes its type from the stats, so segments come back correctly typed.
- Checkpointer: `ColumnDataCheckpointer` analyzes/compresses one ColumnData with one physical type
  (`column_data_checkpointer.cpp:83,167`); checkpoint the inner column, which is already typed.
- Indexes: ART accepts any fixed-width numeric physical type at `art.cpp:50-77`; with InternalType INVALID a
  NUMERIC key is refused there ("Invalid type for index key"). `AppendToIndexes` scans the collection with
  `table.GetTypes()` (`local_storage.cpp:222,499`), so a table with NUMERIC columns and any index needs a typed
  scan or a NotImplemented in slice 1.

Binder
- `src/planner/binder/tableref/bind_basetableref.cpp:255-258`: `table_types` and `return_types` come from
  `col.Type()`; for NUMERIC columns use `DuckTableEntry::ResolveColumnType(context, col)` (new; merges main and
  local stats, calls Resolve). Non-DuckDB catalogs with NUMERIC columns: NotImplemented.
- `TableCatalogEntry::GetExpectedTypeForInsert(column)` is already virtual (`table_catalog_entry.cpp:361`) and
  used by `bind_insert.cpp:89` (VALUES), `bind_update.cpp:66`, `bind_merge_into.cpp:133`. It has no context
  parameter, so either add one or resolve from main-table stats only there and widen in the SELECT path.
- `bind_insert.cpp:681-691` (INSERT ... SELECT) rebuilds `target_types` from `column.Type()`; for NUMERIC use
  Widen(R, source type).
- `bound_cast_expression.cpp:32` `AddCastToTypeInternal`: rule for target NUMERIC (identity for numeric sources).
- CTAS/DESCRIBE/duckdb_columns keep NUMERIC as the catalog column type; nothing to change for slice 1.

## Slices

1. (3-5 days) Bare NUMERIC columns: type scaffolding, UNBOUNDED_NUMERIC_STATS, NumericColumnData with
   per-row-group storage type, bind-time resolution, insert widening, zonemap stat conversion, tests under
   test/sql/types/numeric/. Updates that need widening and indexes on NUMERIC throw NotImplemented.
2. Catalog version bump on widening; UPDATE with widening; indexes via typed scan.
3. Restricted types as NUMERIC + restriction (the "all aliases" end state). Storage for restricted columns can
   stay as it is at first; the restriction is enforced where the implicit cast is today.
4. bignum-decimal representation; arithmetic that widens instead of overflowing (NUMERIC[INT32] +
   NUMERIC[INT32] -> INT64), which is what finally removes the TryAdd checks.

## Why the full "one runtime type" version is weeks, not days (survey 2026-10-12)

1268 `LogicalTypeId::<int type>` references in 92 files, 1036 `LogicalType::<int type>` constant uses, 173
PhysicalType switch sites, 39 Parquet/Arrow mapping sites, 1803 test files mentioning INTEGER/BIGINT. A per-vector
physical type is incompatible with TupleDataLayout, aggregate states, sort layouts and Arrow/Parquet buffers.
Comparable merged PRs: "Make LogicalType immutable" 18 files in 14 h; VARINT 45 files in 3 days; both self-contained.
