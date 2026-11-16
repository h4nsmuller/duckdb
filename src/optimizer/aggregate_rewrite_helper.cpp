#include "duckdb/optimizer/aggregate_rewrite_helper.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"

namespace duckdb {

optional_idx AggregateRewriteHelper::GetDirectReferenceIndex(const Expression &expression, LogicalOperator &input) {
	auto bindings = input.GetColumnBindings();
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		auto index = expression.Cast<BoundReferenceExpression>().Index();
		return index < bindings.size() ? optional_idx(index) : optional_idx();
	}
	if (expression.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
		return optional_idx();
	}
	auto binding = expression.Cast<BoundColumnRefExpression>().Binding();
	for (idx_t index = 0; index < bindings.size(); index++) {
		if (bindings[index] == binding) {
			return optional_idx(index);
		}
	}
	return optional_idx();
}

vector<Identifier> AggregateRewriteHelper::GenerateColumnNames(const string &prefix, idx_t column_count) {
	vector<Identifier> result;
	result.reserve(column_count);
	for (idx_t col_idx = 0; col_idx < column_count; col_idx++) {
		result.emplace_back(StringUtil::Format("%s_%llu", prefix, col_idx));
	}
	return result;
}

static void RebindExpression(unique_ptr<Expression> &expr, const column_binding_map_t<ColumnBinding> &replacement_map) {
	if (!expr || replacement_map.empty()) {
		return;
	}
	expr = ExpressionIterator::PreorderReplace<BoundColumnRefExpression>(
	    std::move(expr), [&](const BoundColumnRefExpression &colref) -> unique_ptr<Expression> {
		    auto entry = replacement_map.find(colref.Binding());
		    if (entry == replacement_map.end()) {
			    return nullptr;
		    }
		    return colref.WithBinding(entry->second);
	    });
}

unique_ptr<Expression>
AggregateRewriteHelper::CopyAndRebind(const Expression &expr,
                                      const column_binding_map_t<ColumnBinding> &replacement_map) {
	auto result = expr.Copy();
	RebindExpression(result, replacement_map);
	return result;
}

static void StageVolatileExpression(unique_ptr<Expression> &expr, TableIndex projection_index,
                                    vector<unique_ptr<Expression>> &projection_expressions) {
	if (!expr || !expr->IsVolatile()) {
		return;
	}
	auto return_type = expr->GetReturnType();
	const auto column_index = projection_expressions.size();
	projection_expressions.push_back(std::move(expr));
	expr = make_uniq<BoundColumnRefExpression>(return_type,
	                                           ColumnBinding(projection_index, ProjectionIndex(column_index)));
}

void AggregateRewriteHelper::StageVolatileAggregateInputs(Optimizer &optimizer, LogicalAggregate &aggr,
                                                          unique_ptr<LogicalOperator> &child) {
	// Branching rewrites must not multiply the evaluation of volatile aggregate inputs.
	bool has_volatile_input = false;
	for (auto &group : aggr.groups) {
		has_volatile_input |= group->IsVolatile();
	}
	for (auto &expr : aggr.expressions) {
		has_volatile_input |= expr->IsVolatile();
	}
	if (!has_volatile_input) {
		return;
	}

	auto child_bindings = child->GetColumnBindings();
	auto projection_index = optimizer.binder.GenerateTableIndex();
	auto projection = LogicalProjection::CreateIdentity(projection_index, std::move(child));

	column_binding_map_t<ColumnBinding> projection_replacements;
	for (idx_t col_idx = 0; col_idx < child_bindings.size(); col_idx++) {
		projection_replacements[child_bindings[col_idx]] = ColumnBinding(projection_index, ProjectionIndex(col_idx));
	}

	for (auto &group : aggr.groups) {
		StageVolatileExpression(group, projection_index, projection->expressions);
	}
	for (auto &expr : aggr.expressions) {
		auto &aggregate = expr->Cast<BoundAggregateExpression>();
		for (auto &child_expr : aggregate.GetChildrenMutable()) {
			StageVolatileExpression(child_expr, projection_index, projection->expressions);
		}
		if (aggregate.GetOrderBys()) {
			for (auto &order : aggregate.GetOrderBys()->orders) {
				StageVolatileExpression(order.expression, projection_index, projection->expressions);
			}
		}
		StageVolatileExpression(aggregate.GetFilterMutable(), projection_index, projection->expressions);
	}

	for (auto &group : aggr.groups) {
		RebindExpression(group, projection_replacements);
	}
	for (auto &expr : aggr.expressions) {
		RebindExpression(expr, projection_replacements);
	}

	child = std::move(projection);
}

unique_ptr<LogicalOperator> AggregateRewriteHelper::CreateCTERef(Optimizer &optimizer, TableIndex cte_index,
                                                                 const vector<LogicalType> &input_types,
                                                                 const vector<Identifier> &input_names,
                                                                 const vector<ColumnBinding> &input_bindings,
                                                                 column_binding_map_t<ColumnBinding> &replacement_map) {
	auto cte_ref_index = optimizer.binder.GenerateTableIndex();
	for (idx_t col_idx = 0; col_idx < input_bindings.size(); col_idx++) {
		replacement_map[input_bindings[col_idx]] = ColumnBinding(cte_ref_index, ProjectionIndex(col_idx));
	}
	return make_uniq<LogicalCTERef>(cte_ref_index, cte_index, input_types, input_names);
}

unique_ptr<BaseStatistics> AggregateRewriteHelper::GetScanColumnStatistics(ClientContext &context, LogicalGet &get,
                                                                           idx_t column_index) {
	auto &column_ids = get.GetColumnIds();
	if (column_index >= column_ids.size()) {
		return nullptr;
	}
	if (get.function.statistics_extended) {
		TableFunctionGetStatisticsInput input(get.bind_data.get(), column_ids[column_index]);
		return get.function.statistics_extended(context, input);
	}
	if (get.function.statistics) {
		return get.function.statistics(context, get.bind_data.get(), column_ids[column_index].GetPrimaryIndex());
	}
	return nullptr;
}

AggregateInputEstimator::AggregateInputEstimator(ClientContext &context_p, LogicalOperator &plan) : context(context_p) {
	CollectCTEDefinitions(plan);
}

void AggregateInputEstimator::CollectCTEDefinitions(LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE && op.children.size() == 2) {
		materialized_ctes.emplace(op.Cast<LogicalMaterializedCTE>().table_index, op);
	}
	for (auto &child : op.children) {
		CollectCTEDefinitions(*child);
	}
}

optional_idx AggregateInputEstimator::EstimateCardinality(LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_GET: {
		auto &get = op.Cast<LogicalGet>();
		if (!get.function.cardinality || !get.bind_data) {
			return optional_idx();
		}
		auto node_stats = get.function.cardinality(context, get.bind_data.get());
		if (!node_stats || !node_stats->has_estimated_cardinality) {
			return optional_idx();
		}
		return node_stats->estimated_cardinality;
	}
	case LogicalOperatorType::LOGICAL_CTE_REF: {
		auto cte_index = op.Cast<LogicalCTERef>().cte_index;
		auto entry = materialized_ctes.find(cte_index);
		if (entry == materialized_ctes.end()) {
			return optional_idx();
		}
		auto cached = cte_cardinalities.find(cte_index);
		if (cached != cte_cardinalities.end()) {
			return cached->second;
		}
		auto cardinality = EstimateCardinality(*entry->second.get().children[0]);
		cte_cardinalities[cte_index] = cardinality;
		return cardinality;
	}
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
		if (op.children.size() != 2) {
			return optional_idx();
		}
		return EstimateCardinality(*op.children[1]);
	case LogicalOperatorType::LOGICAL_UNION: {
		idx_t cardinality = 0;
		for (auto &child : op.children) {
			auto child_cardinality = EstimateCardinality(*child);
			if (!child_cardinality.IsValid()) {
				return optional_idx();
			}
			auto max_remaining = NumericLimits<idx_t>::Maximum() - 1 - cardinality;
			cardinality += MinValue<idx_t>(child_cardinality.GetIndex(), max_remaining);
		}
		return cardinality;
	}
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &aggr = op.Cast<LogicalAggregate>();
		if (aggr.groups.empty() && aggr.grouping_sets.size() <= 1) {
			return 1;
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		auto join_type = op.Cast<LogicalComparisonJoin>().join_type;
		if (join_type == JoinType::SEMI || join_type == JoinType::ANTI || join_type == JoinType::MARK ||
		    join_type == JoinType::SINGLE) {
			return EstimateCardinality(*op.children[0]);
		}
		break;
	}
	default:
		break;
	}
	optional_idx cardinality;
	for (auto &child : op.children) {
		auto child_cardinality = EstimateCardinality(*child);
		if (!child_cardinality.IsValid()) {
			return optional_idx();
		}
		if (!cardinality.IsValid() || child_cardinality.GetIndex() > cardinality.GetIndex()) {
			cardinality = child_cardinality;
		}
	}
	return cardinality;
}

bool AggregateInputEstimator::GetBaseColumn(LogicalOperator &op, ColumnBinding binding, BaseColumn &result) {
	bool owns_table_index = false;
	for (auto &table_index : op.GetTableIndex()) {
		if (table_index == binding.table_index) {
			owns_table_index = true;
			break;
		}
	}
	if (!owns_table_index) {
		for (auto &child : op.children) {
			if (GetBaseColumn(*child, binding, result)) {
				return true;
			}
		}
		return false;
	}
	const auto column_index = binding.column_index.GetIndex();
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		if (column_index >= op.expressions.size() ||
		    op.expressions[column_index]->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
			return false;
		}
		return GetBaseColumn(*op.children[0], op.expressions[column_index]->Cast<BoundColumnRefExpression>().Binding(),
		                     result);
	}
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &aggr = op.Cast<LogicalAggregate>();
		if (binding.table_index != aggr.group_index || column_index >= aggr.groups.size() ||
		    aggr.groups[column_index]->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
			return false;
		}
		return GetBaseColumn(*op.children[0], aggr.groups[column_index]->Cast<BoundColumnRefExpression>().Binding(),
		                     result);
	}
	case LogicalOperatorType::LOGICAL_CTE_REF: {
		auto entry = materialized_ctes.find(op.Cast<LogicalCTERef>().cte_index);
		if (entry == materialized_ctes.end()) {
			return false;
		}
		auto &definition = *entry->second.get().children[0];
		auto definition_bindings = definition.GetColumnBindings();
		if (column_index >= definition_bindings.size()) {
			return false;
		}
		return GetBaseColumn(definition, definition_bindings[column_index], result);
	}
	case LogicalOperatorType::LOGICAL_GET: {
		auto stats = AggregateRewriteHelper::GetScanColumnStatistics(context, op.Cast<LogicalGet>(), column_index);
		if (!stats) {
			return false;
		}
		auto distinct_count = stats->GetDistinctCount();
		if (distinct_count == 0) {
			if (stats->CanHaveNoNull()) {
				// no distinct count statistics
				return false;
			}
			// the column only holds NULL values
			distinct_count = 1;
		}
		auto table_cardinality = EstimateCardinality(op);
		if (!table_cardinality.IsValid()) {
			return false;
		}
		result.table_index = binding.table_index;
		result.distinct_count = distinct_count;
		result.table_cardinality = table_cardinality.GetIndex();
		return true;
	}
	default:
		return false;
	}
}

//! a * b, capped at limit (limit >= a, b >= 1)
static idx_t CappedMultiply(idx_t a, idx_t b, idx_t limit) {
	if (a > limit / b) {
		return limit;
	}
	return MinValue<idx_t>(a * b, limit);
}

optional_idx AggregateInputEstimator::EstimateGroupCount(LogicalOperator &input,
                                                         const vector<reference<Expression>> &groups,
                                                         idx_t input_cardinality) {
	column_binding_set_t bindings;
	for (auto &group : groups) {
		if (group.get().IsVolatile()) {
			return optional_idx();
		}
		ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
		    group.get(), [&](const BoundColumnRefExpression &ref) { bindings.insert(ref.Binding()); });
	}
	// the distinct value combinations of each scan
	struct ScanGroups {
		TableIndex table_index;
		idx_t group_count;
		idx_t table_cardinality;
	};
	vector<ScanGroups> scans;
	for (auto &binding : bindings) {
		BaseColumn column;
		if (!GetBaseColumn(input, binding, column)) {
			return optional_idx();
		}
		if (column.table_cardinality == 0) {
			// an empty table has no groups
			return 0;
		}
		bool found = false;
		for (auto &scan : scans) {
			if (scan.table_index == column.table_index) {
				scan.group_count = CappedMultiply(scan.group_count, column.distinct_count, scan.table_cardinality);
				found = true;
				break;
			}
		}
		if (!found) {
			scans.push_back({column.table_index, MinValue(column.distinct_count, column.table_cardinality),
			                 column.table_cardinality});
		}
	}
	idx_t group_count = 1;
	for (auto &scan : scans) {
		if (input_cardinality == 0) {
			return 0;
		}
		group_count = CappedMultiply(MinValue(group_count, input_cardinality), scan.group_count, input_cardinality);
	}
	return MinValue(group_count, MaxValue<idx_t>(input_cardinality, 1));
}

} // namespace duckdb
