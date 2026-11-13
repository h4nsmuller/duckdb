//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/semi_join_to_inner.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optional_ptr.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {

class LogicalComparisonJoin;
class LogicalOperator;
class TableCatalogEntry;

//! Converts a SEMI join whose right side is distinct on its equality keys into an INNER join:
//!   SEMI[l.k = r.k, ...](L, R) -> INNER[l.k = r.k, ...](L, R), if R has at most one row per value of {r.k, ...}
//! Every left row finds at most one match, so the INNER join returns the same rows. Unlike the right side of a SEMI
//! join, the right side of an INNER join takes part in join ordering.
class SemiJoinToInner {
public:
	void Optimize(LogicalOperator &plan);

private:
	void VisitOperator(LogicalOperator &op);
	//! Whether the AGGREGATE_REUSE optimizer can absorb the SEMI join: it needs the join to stay a SEMI join
	bool FeedsAggregateReuse(LogicalComparisonJoin &join);

private:
	//! The tables scanned by the plan, one entry per scan
	vector<optional_ptr<TableCatalogEntry>> scanned_tables;
};

} // namespace duckdb
