#include "duckdb/optimizer/semi_join_to_inner.hpp"

#include "duckdb/optimizer/aggregate_reuse_internal.hpp"
#include "duckdb/optimizer/distinctness.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

namespace duckdb {

namespace {

void CollectScannedTables(LogicalOperator &op, vector<optional_ptr<TableCatalogEntry>> &tables) {
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto table = op.Cast<LogicalGet>().GetTable();
		if (table) {
			tables.push_back(table);
		}
	}
	for (auto &child : op.children) {
		CollectScannedTables(*child, tables);
	}
}

idx_t CountScans(const vector<optional_ptr<TableCatalogEntry>> &tables, TableCatalogEntry &table) {
	idx_t count = 0;
	for (auto &entry : tables) {
		if (entry.get() == &table) {
			count++;
		}
	}
	return count;
}

//! Whether the plan reads a DELIM_GET or a CTE
bool ReadsDelimOrCTE(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_DELIM_GET || op.type == LogicalOperatorType::LOGICAL_CTE_REF) {
		return true;
	}
	for (auto &child : op.children) {
		if (ReadsDelimOrCTE(*child)) {
			return true;
		}
	}
	return false;
}

} // namespace

void SemiJoinToInner::Optimize(LogicalOperator &plan) {
	scanned_tables.clear();
	CollectScannedTables(plan, scanned_tables);
	VisitOperator(plan);
}

bool SemiJoinToInner::FeedsAggregateReuse(LogicalComparisonJoin &join) {
	// AGGREGATE_REUSE runs after join ordering. It matches a SEMI join against a projection over an aggregate that is
	// grouped by one column of a table that is joined (and aggregated) outside of the SEMI join as well, computes the
	// outer aggregate from the state of the inner one and turns the SEMI join into an INNER join itself.
	// If the join is converted here, the join order optimizer can move the aggregate and the shape is lost.
	auto &right = *join.children[1];
	if (right.type != LogicalOperatorType::LOGICAL_PROJECTION || right.children.size() != 1) {
		return false;
	}
	auto aggregate = FindUnaryAggregate(*right.children[0]);
	if (!aggregate || aggregate->groups.size() != 1 || aggregate->expressions.empty() ||
	    aggregate->children.size() != 1) {
		return false;
	}
	auto origin = GetTableColumnOrigin(*aggregate->children[0], *aggregate->groups[0]);
	if (!origin) {
		return false;
	}
	// the aggregated table has to be scanned outside of the right side as well
	vector<optional_ptr<TableCatalogEntry>> right_tables;
	CollectScannedTables(right, right_tables);
	return CountScans(scanned_tables, origin->table.get()) > CountScans(right_tables, origin->table.get());
}

void SemiJoinToInner::VisitOperator(LogicalOperator &op) {
	// bottom-up, so that a converted join below is used to derive the distinctness of the right side of a join above
	for (auto &child : op.children) {
		VisitOperator(*child);
	}
	if (op.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return;
	}
	auto &join = op.Cast<LogicalComparisonJoin>();
	if (join.join_type != JoinType::SEMI || join.HasProjectionMap() || join.filter_pushdown) {
		return;
	}
	column_binding_set_t keys;
	if (!Distinctness::GetJoinEqualityKeys(join, 1, keys)) {
		return;
	}
	// a right side that reads a DELIM_GET or a CTE is what is left of a decorrelated subquery: the SEMI join restricts
	// it to the correlated values, and the optimizers that work on that shape expect a SEMI join
	if (ReadsDelimOrCTE(*join.children[1]) || FeedsAggregateReuse(join)) {
		return;
	}
	if (!Distinctness::IsDistinctOn(*join.children[1], keys)) {
		return;
	}
	// the INNER join also emits the columns of the right side: nothing references them, they are pruned again by
	// the unused column removal
	join.join_type = JoinType::INNER;
	join.ResolveOperatorTypes();
}

} // namespace duckdb
