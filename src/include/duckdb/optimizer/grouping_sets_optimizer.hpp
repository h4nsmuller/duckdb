//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/grouping_sets_optimizer.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"

namespace duckdb {
class AggregateInputEstimator;
class Optimizer;

//! The GroupingSetsOptimizer rewrites aggregates over multiple grouping sets (ROLLUP/CUBE/GROUPING SETS) into
//! explicit aggregate branches. Compatible aggregates use a cascade that combines exported states; other aggregates
//! use one branch per grouping set over a shared materialized input. The cascade is not used for aggregates with wide
//! states (sketches) whose coarser grouping sets are estimated to read almost as many rows as the input holds.
class GroupingSetsOptimizer : public LogicalOperatorVisitor {
public:
	explicit GroupingSetsOptimizer(Optimizer &optimizer, bool mandatory_aggregate_lowering = false);

	//! Rewrite the grouping sets of a plan (plan: the root, which holds the materialized CTEs the estimates look up)
	void Optimize(unique_ptr<LogicalOperator> &plan);
	void VisitOperator(unique_ptr<LogicalOperator> &op) override;

private:
	//! The estimated input cardinality of an aggregate and the estimated number of groups of each grouping set
	struct GroupingSetEstimates {
		idx_t input_cardinality;
		vector<idx_t> set_groups;
	};

	void EstimateGroupingSets(AggregateInputEstimator &estimator, LogicalOperator &op);
	bool TryRewriteGroupingSets(unique_ptr<LogicalOperator> &op);
	bool TryExpandGroupingSets(unique_ptr<LogicalOperator> &op);
	unique_ptr<Expression> VisitReplace(BoundColumnRefExpression &expr, unique_ptr<Expression> *expr_ptr) override;

private:
	Optimizer &optimizer;
	bool mandatory_aggregate_lowering;
	column_binding_map_t<ColumnBinding> replacement_map;
	//! The estimates of the aggregates with wide states over multiple grouping sets, by group index (aggregates
	//! whose input cannot be estimated from table statistics have no entry)
	unordered_map<TableIndex, GroupingSetEstimates> grouping_set_estimates;
};
} // namespace duckdb
