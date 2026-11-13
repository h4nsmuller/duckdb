//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/rank_window_topn.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optional_idx.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/expression/bound_parameter_data.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class BoundWindowExpression;
class LogicalGet;
class Optimizer;

//! Adds a per-partition threshold pre-filter below a window whose rank() / dense_rank() is filtered with rank <= k:
//!
//!   FILTER(rk <= k)                          FILTER(rk <= k)
//!     PROJECTION*                              PROJECTION*
//!       WINDOW(rank() OVER (p, o))               WINDOW(rank() OVER (p, o))
//!         <input>                     ==>          FILTER(key <= th OR th IS NULL)
//!                                                    JOIN(p IS NOT DISTINCT FROM g)
//!                                                      <input>
//!                                                      PROJECTION(g, th := list_extract(agg, k))
//!                                                        AGGREGATE(GROUP BY p; agg := min(key, k))
//!                                                          <input>
//!
//! th is the k-th smallest key of the partition (the k-th smallest distinct key for dense_rank), so that key <= th is
//! exactly rank <= k, ties included. A partition with fewer than k keys has no threshold and keeps all of its rows.
//! Removing a row whose rank exceeds k does not change the rank of any row that stays, so the window and the filter
//! are unchanged and compute the same ranks for the rows that survive.
class RankWindowTopN {
public:
	explicit RankWindowTopN(Optimizer &optimizer);

	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	//! One matched ranking: rk <= k on the window expression at index
	struct RankTerm {
		idx_t index;
		//! The number of keys per partition that the filter keeps
		int64_t limit;
		//! dense_rank() instead of rank()
		bool dense;
		//! The key is the order column itself (a single NULLS LAST ordering) instead of a sort key
		bool raw_column;
		//! Only for raw_column: the ordering is descending
		bool descending;
	};

	//! A matched FILTER -> PROJECTION* -> WINDOW chain: one ranking, or an OR of rankings of the same window
	struct RankMatch {
		vector<RankTerm> terms;
		//! The number of projections between the filter and the window
		idx_t depth = 0;
	};

	//! The estimates of the definition of a materialized CTE
	struct CTEEstimate {
		optional_idx rows;
		vector<optional_idx> distinct;
	};

private:
	unique_ptr<LogicalOperator> OptimizeInternal(unique_ptr<LogicalOperator> op, vector<ReplacementBinding> &pending);

	void CountPositionalRead(LogicalOperator &op);
	void CountReferences(LogicalOperator &op);
	bool IsReferenced(const ColumnBinding &binding) const;

	//! An upper bound of the number of rows produced by the operator, if known
	optional_idx EstimateRows(LogicalOperator &op);
	//! An upper bound of the number of distinct values of a column in the output of scope, if known
	optional_idx EstimateDistinct(LogicalOperator &scope, const ColumnBinding &binding);
	optional_idx EstimateDistinct(LogicalOperator &scope, const Expression &expr);
	optional_idx EstimateDistinctColumns(LogicalOperator &scope, const Expression &expr);
	bool PreFilterIsUseless(LogicalOperator &input, const vector<unique_ptr<Expression>> &partitions, int64_t kept);

	bool TryMatch(LogicalOperator &filter, RankMatch &result);
	bool MatchPredicates(LogicalOperator &filter, const vector<pair<ColumnBinding, int64_t>> &predicates,
	                     RankMatch &result);

	unique_ptr<Expression> CreateKey(const BoundWindowExpression &w_expr, const RankTerm &term);
	//! Rewrites the input of the window of the matched chain. Returns the binding replacements that the window and its
	//! ancestors need.
	vector<ReplacementBinding> Rewrite(LogicalOperator &filter, const RankMatch &match);

private:
	Optimizer &optimizer;
	optional_ptr<bound_parameter_map_t> parameter_data;
	//! How often the plan references each column binding
	column_binding_map_t<idx_t> references;
	//! The estimates of the materialized CTEs that have been visited, by CTE index
	unordered_map<idx_t, CTEEstimate> cte_estimates;
};

} // namespace duckdb
