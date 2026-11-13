#include "duckdb/optimizer/rank_window_topn.hpp"

#include "duckdb/function/aggregate/minmax_n_helpers.hpp"
#include "duckdb/function/builtin_function_lookup.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator_deep_copy.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_expression_get.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_prepare.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_window.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"

namespace duckdb {

namespace {

//! The alias of the threshold: marks a window input that has been rewritten
constexpr const char *THRESHOLD_ALIAS = "__rank_topn_threshold";

//! The aggregate keeps k keys per partition, which only pays off when k * partitions * factor is below the number of
//! input rows (a heap of k entries per partition against the sort of the input)
constexpr idx_t PARTITION_GUARD_FACTOR = 4;

//! How far below the window input the threshold projection of an earlier rewrite is searched for
constexpr idx_t REWRITTEN_SEARCH_DEPTH = 4;

idx_t SaturatingMultiply(idx_t left, idx_t right) {
	idx_t result;
	if (!TryMultiplyOperator::Operation(left, right, result)) {
		return NumericLimits<idx_t>::Maximum() - 1;
	}
	return MinValue<idx_t>(result, NumericLimits<idx_t>::Maximum() - 1);
}

idx_t SaturatingAdd(idx_t left, idx_t right) {
	idx_t result;
	if (!TryAddOperator::Operation(left, right, result)) {
		return NumericLimits<idx_t>::Maximum() - 1;
	}
	return MinValue<idx_t>(result, NumericLimits<idx_t>::Maximum() - 1);
}

bool TryGetBinding(const Expression &expr, ColumnBinding &result) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
		return false;
	}
	result = expr.Cast<BoundColumnRefExpression>().Binding();
	return true;
}

//! The operator in the subtree that produces the table index
optional_ptr<LogicalOperator> FindProducer(LogicalOperator &op, TableIndex table_index) {
	for (auto &index : op.GetTableIndex()) {
		if (index == table_index) {
			return op;
		}
	}
	for (auto &child : op.children) {
		auto result = FindProducer(*child, table_index);
		if (result) {
			return result;
		}
	}
	return nullptr;
}

//! rk <= c keeps c keys, rk < c keeps c - 1, rk = c keeps c (the filter above does the rest). The limit becomes the
//! n of the min / max aggregate, which requires n < MIN_MAX_N_MAX_VALUE.
bool TryGetLimit(ExpressionType comparison, const Value &constant, int64_t &limit) {
	if (constant.IsNull() || constant.type() != LogicalType::BIGINT) {
		return false;
	}
	auto value = constant.GetValue<int64_t>();
	switch (comparison) {
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
	case ExpressionType::COMPARE_EQUAL:
		limit = value;
		return value >= 1 && value < MIN_MAX_N_MAX_VALUE;
	case ExpressionType::COMPARE_LESSTHAN:
		limit = value - 1;
		return value >= 2 && value <= MIN_MAX_N_MAX_VALUE;
	default:
		return false;
	}
}

//! rk <= k with the constant on either side
bool TryGetRankPredicate(const Expression &expr, ColumnBinding &binding, int64_t &limit) {
	if (!BoundComparisonExpression::IsComparison(expr)) {
		return false;
	}
	auto &comparison = expr.Cast<BoundFunctionExpression>();
	auto &left = BoundComparisonExpression::Left(comparison);
	auto &right = BoundComparisonExpression::Right(comparison);
	if (right.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
		return TryGetBinding(left, binding) &&
		       TryGetLimit(expr.GetExpressionType(), right.Cast<BoundConstantExpression>().GetValue(), limit);
	}
	if (left.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
		return TryGetBinding(right, binding) && TryGetLimit(FlipComparisonExpression(expr.GetExpressionType()),
		                                                    left.Cast<BoundConstantExpression>().GetValue(), limit);
	}
	return false;
}

//! A rank() / dense_rank() over the whole partition: a frame, an argument ordering or a filter change what the
//! function counts
bool IsPlainRanking(const BoundWindowExpression &w_expr, bool &dense) {
	switch (w_expr.GetExpressionType()) {
	case ExpressionType::WINDOW_RANK:
		dense = false;
		break;
	case ExpressionType::WINDOW_RANK_DENSE:
		dense = true;
		break;
	default:
		return false;
	}
	if (w_expr.OrderBy().empty() || !w_expr.ArgOrders().empty() || w_expr.Filter() || !w_expr.GetChildren().empty() ||
	    w_expr.WindowStart() != WindowBoundary::UNBOUNDED_PRECEDING ||
	    w_expr.WindowEnd() != WindowBoundary::CURRENT_ROW_RANGE ||
	    w_expr.WindowExclude() != WindowExcludeMode::NO_OTHER) {
		return false;
	}
	for (auto &partition : w_expr.Partitions()) {
		if (partition->IsVolatile()) {
			return false;
		}
	}
	for (auto &order : w_expr.OrderBy()) {
		if (order.type != OrderType::ASCENDING && order.type != OrderType::DESCENDING) {
			return false;
		}
		if (order.null_order != OrderByNullType::NULLS_FIRST && order.null_order != OrderByNullType::NULLS_LAST) {
			return false;
		}
		if (order.expression->IsVolatile()) {
			return false;
		}
	}
	return true;
}

class VolatileExpressionFinder : public LogicalOperatorVisitor {
public:
	void VisitExpression(unique_ptr<Expression> *expression) override {
		found = found || (*expression)->IsVolatile();
		LogicalOperatorVisitor::VisitExpression(expression);
	}

	bool found = false;
};

bool IsScanChain(const LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_GET:
	case LogicalOperatorType::LOGICAL_PROJECTION:
	case LogicalOperatorType::LOGICAL_FILTER:
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
	case LogicalOperatorType::LOGICAL_CTE_REF:
		break;
	default:
		return false;
	}
	for (auto &child : op.children) {
		if (!IsScanChain(*child)) {
			return false;
		}
	}
	return true;
}

//! Whether computing the operator a second time for the aggregate costs no more than a second scan: a chain of scans,
//! projections and filters without volatile expressions
bool IsCheapToRecompute(LogicalOperator &op) {
	if (!IsScanChain(op)) {
		return false;
	}
	VolatileExpressionFinder finder;
	finder.VisitOperator(op);
	return !finder.found;
}

bool HasThresholdProjection(const LogicalOperator &op, idx_t depth) {
	if (depth == 0) {
		return false;
	}
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_PROJECTION:
		for (auto &expr : op.expressions) {
			if (expr->GetAlias() == THRESHOLD_ALIAS) {
				return true;
			}
		}
		return false;
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
		return HasThresholdProjection(*op.children[1], depth - 1);
	case LogicalOperatorType::LOGICAL_FILTER:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		for (auto &child : op.children) {
			if (HasThresholdProjection(*child, depth - 1)) {
				return true;
			}
		}
		return false;
	default:
		return false;
	}
}

//! Follows the binding from the filter through projections that pass it on unchanged to the window that produces it:
//! collects the binding of every level and returns the window
optional_ptr<LogicalOperator> FollowToWindow(LogicalOperator &filter, ColumnBinding binding,
                                             vector<ColumnBinding> &chain) {
	chain.push_back(binding);
	reference<LogicalOperator> child = *filter.children[0];
	while (child.get().type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &projection = child.get().Cast<LogicalProjection>();
		if (binding.table_index != projection.table_index ||
		    binding.column_index.GetIndex() >= projection.expressions.size()) {
			return nullptr;
		}
		if (!TryGetBinding(*projection.expressions[binding.column_index.GetIndex()], binding)) {
			return nullptr;
		}
		chain.push_back(binding);
		child = *projection.children[0];
	}
	if (child.get().type != LogicalOperatorType::LOGICAL_WINDOW) {
		return nullptr;
	}
	auto &window = child.get().Cast<LogicalWindow>();
	if (window.window_index != binding.table_index || binding.column_index.GetIndex() >= window.expressions.size()) {
		return nullptr;
	}
	return child.get();
}

} // namespace

RankWindowTopN::RankWindowTopN(Optimizer &optimizer) : optimizer(optimizer) {
}

unique_ptr<LogicalOperator> RankWindowTopN::Optimize(unique_ptr<LogicalOperator> op) {
	// every column that the plan returns is read
	CountPositionalRead(*op);
	CountReferences(*op);
	vector<ReplacementBinding> pending;
	return OptimizeInternal(std::move(op), pending);
}

//===--------------------------------------------------------------------===//
// Column references
//===--------------------------------------------------------------------===//
void RankWindowTopN::CountPositionalRead(LogicalOperator &op) {
	for (auto &binding : op.GetColumnBindings()) {
		references[binding]++;
	}
}

bool RankWindowTopN::IsReferenced(const ColumnBinding &binding) const {
	auto entry = references.find(binding);
	return entry != references.end() && entry->second > 0;
}

//! Counts the references to the columns of the plan, parents first. The unused column of a projection is dead (unused
//! columns have not been removed at this point) and its expression does not count, so that a rank that a subquery only
//! selects for its own filter does not look like it is read above the subquery.
void RankWindowTopN::CountReferences(LogicalOperator &op) {
	auto count_expression = [&](Expression &expr) {
		ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
		    expr, [&](const BoundColumnRefExpression &colref) { references[colref.Binding()]++; });
	};
	bool reads_by_expression = false;
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		auto &projection = op.Cast<LogicalProjection>();
		for (idx_t i = 0; i < op.expressions.size(); i++) {
			if (IsReferenced(ColumnBinding(projection.table_index, ProjectionIndex(i)))) {
				count_expression(*op.expressions[i]);
			}
		}
		reads_by_expression = true;
		break;
	}
	case LogicalOperatorType::LOGICAL_FILTER:
	case LogicalOperatorType::LOGICAL_WINDOW:
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
	case LogicalOperatorType::LOGICAL_ORDER_BY:
	case LogicalOperatorType::LOGICAL_TOP_N:
	case LogicalOperatorType::LOGICAL_LIMIT:
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
		reads_by_expression = true;
		DUCKDB_EXPLICIT_FALLTHROUGH;
	default:
		LogicalOperatorVisitor::EnumerateExpressions(op,
		                                             [&](unique_ptr<Expression> *expr) { count_expression(**expr); });
		break;
	}
	for (idx_t child_idx = 0; child_idx < op.children.size(); child_idx++) {
		auto &child = *op.children[child_idx];
		// any other operator reads the columns of its children by position, as do the scans of a materialized CTE
		const auto is_cte_definition = op.type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE && child_idx == 0;
		if (!reads_by_expression || is_cte_definition) {
			CountPositionalRead(child);
		}
		CountReferences(child);
	}
}

//===--------------------------------------------------------------------===//
// Estimates
//===--------------------------------------------------------------------===//
// Statistics have not been propagated at this point, so the statistics of the scans are followed through the operators
// that preserve the identity of a column

optional_idx RankWindowTopN::EstimateRows(LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_GET: {
		auto &get = op.Cast<LogicalGet>();
		if (!get.children.empty() || !get.function.cardinality || !get.bind_data) {
			return optional_idx();
		}
		auto node_stats = get.function.cardinality(optimizer.context, get.bind_data.get());
		if (!node_stats || !node_stats->has_estimated_cardinality) {
			return optional_idx();
		}
		return node_stats->estimated_cardinality;
	}
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &aggr = op.Cast<LogicalAggregate>();
		if (aggr.groups.empty()) {
			return 1;
		}
		auto &child = *op.children[0];
		auto child_rows = EstimateRows(child);
		if (!child_rows.IsValid()) {
			return optional_idx();
		}
		idx_t groups = 1;
		for (auto &group : aggr.groups) {
			auto distinct = EstimateDistinct(child, *group);
			if (!distinct.IsValid()) {
				groups = child_rows.GetIndex();
				break;
			}
			groups = SaturatingMultiply(groups, distinct.GetIndex());
		}
		groups = MinValue<idx_t>(groups, child_rows.GetIndex());
		return SaturatingMultiply(groups, MaxValue<idx_t>(aggr.grouping_sets.size(), 1));
	}
	case LogicalOperatorType::LOGICAL_UNION: {
		idx_t total = 0;
		for (auto &child : op.children) {
			auto child_rows = EstimateRows(*child);
			if (!child_rows.IsValid()) {
				return optional_idx();
			}
			total = SaturatingAdd(total, child_rows.GetIndex());
		}
		return total;
	}
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
		return EstimateRows(*op.children[0]);
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
		return EstimateRows(*op.children[1]);
	case LogicalOperatorType::LOGICAL_CTE_REF: {
		auto entry = cte_estimates.find(op.Cast<LogicalCTERef>().cte_index.index);
		return entry == cte_estimates.end() ? optional_idx() : entry->second.rows;
	}
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
		return 1;
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
		return op.Cast<LogicalExpressionGet>().expressions.size();
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
		return 0;
	default: {
		if (op.children.empty()) {
			return optional_idx();
		}
		idx_t result = 0;
		for (auto &child : op.children) {
			auto child_rows = EstimateRows(*child);
			if (!child_rows.IsValid()) {
				return optional_idx();
			}
			result = MaxValue<idx_t>(result, child_rows.GetIndex());
		}
		return result;
	}
	}
}

optional_idx RankWindowTopN::EstimateDistinct(LogicalOperator &scope, const ColumnBinding &binding) {
	auto producer = FindProducer(scope, binding.table_index);
	if (!producer) {
		return optional_idx();
	}
	auto &op = *producer;
	const auto column = binding.column_index.GetIndex();
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_PROJECTION:
		if (column >= op.expressions.size()) {
			return optional_idx();
		}
		return EstimateDistinct(*op.children[0], *op.expressions[column]);
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &aggr = op.Cast<LogicalAggregate>();
		if (binding.table_index != aggr.group_index || column >= aggr.groups.size()) {
			return optional_idx();
		}
		return EstimateDistinct(*op.children[0], *aggr.groups[column]);
	}
	case LogicalOperatorType::LOGICAL_GET: {
		auto &get = op.Cast<LogicalGet>();
		auto &column_ids = get.GetColumnIds();
		if (column >= column_ids.size() || column_ids[column].IsVirtualColumn() || !get.bind_data) {
			return optional_idx();
		}
		unique_ptr<BaseStatistics> stats;
		if (get.function.statistics_extended) {
			TableFunctionGetStatisticsInput input(get.bind_data.get(), column_ids[column]);
			stats = get.function.statistics_extended(optimizer.context, input);
		} else if (get.function.statistics) {
			stats =
			    get.function.statistics(optimizer.context, get.bind_data.get(), column_ids[column].GetPrimaryIndex());
		}
		if (!stats || stats->GetDistinctCount() == 0) {
			return optional_idx();
		}
		return stats->GetDistinctCount();
	}
	case LogicalOperatorType::LOGICAL_CTE_REF: {
		auto entry = cte_estimates.find(op.Cast<LogicalCTERef>().cte_index.index);
		if (entry == cte_estimates.end() || column >= entry->second.distinct.size()) {
			return optional_idx();
		}
		return entry->second.distinct[column];
	}
	case LogicalOperatorType::LOGICAL_UNION:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT: {
		idx_t total = 0;
		for (auto &child : op.children) {
			auto child_bindings = child->GetColumnBindings();
			if (column >= child_bindings.size()) {
				return optional_idx();
			}
			auto distinct = EstimateDistinct(*child, child_bindings[column]);
			if (!distinct.IsValid()) {
				return optional_idx();
			}
			total = SaturatingAdd(total, distinct.GetIndex());
			if (op.type != LogicalOperatorType::LOGICAL_UNION) {
				break;
			}
		}
		return total;
	}
	default:
		return optional_idx();
	}
}

//! A function of columns does not have more values than the combinations of the columns
optional_idx RankWindowTopN::EstimateDistinctColumns(LogicalOperator &scope, const Expression &expr) {
	column_binding_set_t bindings;
	ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
	    expr, [&](const BoundColumnRefExpression &colref) { bindings.insert(colref.Binding()); });
	idx_t result = 1;
	for (auto &binding : bindings) {
		auto distinct = EstimateDistinct(scope, binding);
		if (!distinct.IsValid()) {
			return optional_idx();
		}
		result = SaturatingMultiply(result, distinct.GetIndex());
	}
	return result;
}

optional_idx RankWindowTopN::EstimateDistinct(LogicalOperator &scope, const Expression &expr) {
	optional_idx estimate;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_COLUMN_REF:
		estimate = EstimateDistinct(scope, expr.Cast<BoundColumnRefExpression>().Binding());
		break;
	case ExpressionClass::BOUND_CASE: {
		// a CASE returns one of its result expressions
		auto &case_expr = expr.Cast<BoundCaseExpression>();
		estimate = EstimateDistinct(scope, case_expr.Else());
		for (auto &check : case_expr.CaseChecks()) {
			auto distinct = EstimateDistinct(scope, *check.then_expr);
			if (!estimate.IsValid() || !distinct.IsValid()) {
				return optional_idx();
			}
			estimate = SaturatingAdd(estimate.GetIndex(), distinct.GetIndex());
		}
		break;
	}
	case ExpressionClass::BOUND_FUNCTION: {
		estimate = EstimateDistinctColumns(scope, expr);
		// x % c has at most 2 * |c| + 1 values
		auto &function = expr.Cast<BoundFunctionExpression>();
		auto &children = function.GetChildren();
		if (function.Function().GetName().GetIdentifierName() != "%" || children.size() != 2 ||
		    children[1]->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
			break;
		}
		auto &modulus = children[1]->Cast<BoundConstantExpression>().GetValue();
		if (modulus.IsNull() || !modulus.type().IsIntegral()) {
			break;
		}
		auto modulus_bigint = modulus.DefaultTryCastAs(LogicalType::BIGINT);
		if (!modulus_bigint || modulus_bigint->IsNull()) {
			break;
		}
		const auto modulus_value = BigIntValue::Get(*modulus_bigint);
		if (modulus_value == NumericLimits<int64_t>::Minimum()) {
			break;
		}
		const auto values = SaturatingAdd(SaturatingMultiply(NumericCast<idx_t>(AbsValue(modulus_value)), 2), 1);
		estimate = estimate.IsValid() ? MinValue<idx_t>(estimate.GetIndex(), values) : values;
		break;
	}
	default:
		estimate = EstimateDistinctColumns(scope, expr);
		break;
	}
	if (!estimate.IsValid()) {
		return optional_idx();
	}
	// a small type does not have more values than its domain and NULL
	idx_t domain = NumericLimits<idx_t>::Maximum();
	switch (expr.GetReturnType().id()) {
	case LogicalTypeId::BOOLEAN:
		domain = 3;
		break;
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::UTINYINT:
		domain = idx_t(NumericLimits<uint8_t>::Maximum()) + 2;
		break;
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::USMALLINT:
		domain = idx_t(NumericLimits<uint16_t>::Maximum()) + 2;
		break;
	default:
		break;
	}
	return MinValue<idx_t>(estimate.GetIndex(), domain);
}

//! Whether the pre-filter cannot remove enough rows to pay for itself: every partition is a single row (the input is an
//! aggregate that groups by partition keys only), or the keys that are kept per partition cover most of the input
bool RankWindowTopN::PreFilterIsUseless(LogicalOperator &input, const vector<unique_ptr<Expression>> &partitions,
                                        int64_t kept) {
	// map the partition columns through the projections and filters below the window
	vector<ColumnBinding> bindings;
	for (auto &partition : partitions) {
		ColumnBinding binding;
		if (TryGetBinding(*partition, binding)) {
			bindings.push_back(binding);
		}
	}
	bool resolved = true;
	reference<LogicalOperator> bottom = input;
	while (true) {
		if (bottom.get().type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &projection = bottom.get().Cast<LogicalProjection>();
			for (auto &binding : bindings) {
				if (!resolved) {
					break;
				}
				resolved = binding.table_index == projection.table_index &&
				           binding.column_index.GetIndex() < projection.expressions.size() &&
				           TryGetBinding(*projection.expressions[binding.column_index.GetIndex()], binding);
			}
		} else if (bottom.get().type != LogicalOperatorType::LOGICAL_FILTER) {
			break;
		}
		bottom = *bottom.get().children[0];
	}
	if (resolved && bottom.get().type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &aggr = bottom.get().Cast<LogicalAggregate>();
		bool groups_are_partitions = aggr.grouping_sets.size() <= 1;
		for (idx_t group_idx = 0; group_idx < aggr.groups.size() && groups_are_partitions; group_idx++) {
			ColumnBinding group_binding(aggr.group_index, ProjectionIndex(group_idx));
			groups_are_partitions = std::find(bindings.begin(), bindings.end(), group_binding) != bindings.end();
		}
		if (groups_are_partitions) {
			return true;
		}
	}

	auto rows = EstimateRows(input);
	if (!rows.IsValid()) {
		return false;
	}
	idx_t partition_count = 1;
	for (auto &partition : partitions) {
		auto distinct = EstimateDistinct(input, *partition);
		if (!distinct.IsValid()) {
			return false;
		}
		partition_count = SaturatingMultiply(partition_count, distinct.GetIndex());
	}
	const auto kept_rows = SaturatingMultiply(partition_count, NumericCast<idx_t>(kept));
	return SaturatingMultiply(kept_rows, PARTITION_GUARD_FACTOR) > rows.GetIndex();
}

//===--------------------------------------------------------------------===//
// Matching
//===--------------------------------------------------------------------===//
bool RankWindowTopN::TryMatch(LogicalOperator &filter, RankMatch &result) {
	for (auto &expr : filter.expressions) {
		vector<pair<ColumnBinding, int64_t>> predicates;
		bool is_rank_filter = true;
		auto add_predicate = [&](const Expression &predicate) {
			ColumnBinding binding;
			int64_t limit;
			if (!TryGetRankPredicate(predicate, binding, limit)) {
				is_rank_filter = false;
				return;
			}
			predicates.emplace_back(binding, limit);
		};
		if (expr->GetExpressionType() == ExpressionType::CONJUNCTION_OR) {
			for (auto &child : expr->Cast<BoundConjunctionExpression>().GetChildren()) {
				add_predicate(*child);
			}
		} else {
			add_predicate(*expr);
		}
		if (!is_rank_filter || predicates.empty()) {
			continue;
		}
		RankMatch match;
		if (MatchPredicates(filter, predicates, match)) {
			result = std::move(match);
			return true;
		}
	}
	return false;
}

bool RankWindowTopN::MatchPredicates(LogicalOperator &filter, const vector<pair<ColumnBinding, int64_t>> &predicates,
                                     RankMatch &result) {
	optional_ptr<LogicalOperator> window;
	for (auto &predicate : predicates) {
		vector<ColumnBinding> chain;
		auto found = FollowToWindow(filter, predicate.first, chain);
		if (!found) {
			return false;
		}
		if (predicates.size() > 1) {
			// the pre-filter of one ranking can lower the rank that a surviving row has in another ranking: nothing
			// but the filter may read the ranks
			for (auto &binding : chain) {
				auto entry = references.find(binding);
				if (entry == references.end() || entry->second != 1) {
					return false;
				}
			}
		}
		if (!window) {
			window = found;
			result.depth = chain.size() - 1;
		} else if (window.get() != found.get()) {
			return false;
		}
		RankTerm term;
		term.index = chain.back().column_index.GetIndex();
		term.limit = predicate.second;
		auto &w_expr = found->expressions[term.index]->Cast<BoundWindowExpression>();
		if (!IsPlainRanking(w_expr, term.dense)) {
			return false;
		}
		auto &orders = w_expr.OrderBy();
		term.raw_column = orders.size() == 1 && orders[0].null_order == OrderByNullType::NULLS_LAST;
		term.descending = term.raw_column && orders[0].type == OrderType::DESCENDING;
		result.terms.push_back(term);
	}
	auto &first = window->expressions[result.terms[0].index]->Cast<BoundWindowExpression>();
	// all rankings of an OR have the same partitions: one aggregate and one join
	for (auto &term : result.terms) {
		if (!window->expressions[term.index]->Cast<BoundWindowExpression>().PartitionsAreEquivalent(first)) {
			return false;
		}
	}
	// any other function of the window observes the rows that the pre-filter removes
	for (auto &sibling : window->expressions) {
		bool is_matched = false;
		for (auto &term : result.terms) {
			is_matched = is_matched || sibling->Equals(*window->expressions[term.index]);
		}
		if (!is_matched) {
			return false;
		}
	}
	if (result.terms.size() > 1) {
		// an OR: one window function per disjunct and nothing else in the window. A second function of a matched
		// ranking would still see its ranks change for the rows that the other ranking keeps.
		unordered_set<idx_t> indexes;
		for (auto &term : result.terms) {
			indexes.insert(term.index);
		}
		if (indexes.size() != result.terms.size() || indexes.size() != window->expressions.size()) {
			return false;
		}
	}
	auto &input = *window->children[0];
	if (HasThresholdProjection(input, REWRITTEN_SEARCH_DEPTH)) {
		return false;
	}
	int64_t kept = 0;
	for (auto &term : result.terms) {
		kept += term.limit;
	}
	return !PreFilterIsUseless(input, first.Partitions(), kept);
}

//===--------------------------------------------------------------------===//
// Rewrite
//===--------------------------------------------------------------------===//
//! The ORDER BY as one comparable expression: the order column itself for a single NULLS LAST ordering (NULL keys rank
//! last and the aggregate, which ignores NULLs, agrees), or else a sort key
unique_ptr<Expression> RankWindowTopN::CreateKey(const BoundWindowExpression &w_expr, const RankTerm &term) {
	auto &orders = w_expr.OrderBy();
	if (term.raw_column) {
		return orders[0].expression->Copy();
	}
	vector<unique_ptr<Expression>> arguments;
	for (auto &order : orders) {
		arguments.push_back(order.expression->Copy());
		arguments.push_back(make_uniq<BoundConstantExpression>(Value(order.GetOrderModifier())));
	}
	FunctionBinder function_binder(optimizer.context);
	return function_binder.BindScalarFunction(Identifier::DefaultSchema(), "create_sort_key", std::move(arguments));
}

vector<ReplacementBinding> RankWindowTopN::Rewrite(LogicalOperator &filter, const RankMatch &match) {
	reference<LogicalOperator> window_ref = *filter.children[0];
	for (idx_t i = 0; i < match.depth; i++) {
		window_ref = *window_ref.get().children[0];
	}
	auto &window = window_ref.get();
	auto &first = window.expressions[match.terms[0].index]->Cast<BoundWindowExpression>();
	auto &context = optimizer.context;

	const auto group_index = optimizer.binder.GenerateTableIndex();
	const auto aggregate_index = optimizer.binder.GenerateTableIndex();
	const auto projection_index = optimizer.binder.GenerateTableIndex();

	// bind everything that can fail before the plan is changed
	vector<unique_ptr<Expression>> keys;
	vector<unique_ptr<Expression>> aggregates;
	vector<unique_ptr<Expression>> thresholds;
	vector<ExpressionType> comparisons;
	unique_ptr<LogicalOperator> build;
	auto &input = window.children[0];
	input->ResolveOperatorTypes();
	const auto input_bindings = input->GetColumnBindings();
	const auto input_types = input->types;
	const auto use_cte = !IsCheapToRecompute(*input);
	try {
		FunctionBinder function_binder(context);
		for (auto &term : match.terms) {
			auto &w_expr = window.expressions[term.index]->Cast<BoundWindowExpression>();
			auto key = CreateKey(w_expr, term);
			auto function = TryGetBuiltinAggregateFunction(context, term.descending ? "max" : "min",
			                                               {key->GetReturnType(), LogicalType::BIGINT});
			if (!function) {
				return {};
			}
			vector<unique_ptr<Expression>> arguments;
			arguments.push_back(key->Copy());
			arguments.push_back(make_uniq<BoundConstantExpression>(Value::BIGINT(term.limit)));
			// the k-th smallest key, or the k-th smallest distinct key for dense_rank
			auto aggregate = function_binder.BindAggregateFunction(function, std::move(arguments), nullptr,
			                                                       term.dense ? AggregateType::DISTINCT
			                                                                  : AggregateType::NON_DISTINCT);
			auto aggregate_ref = make_uniq<BoundColumnRefExpression>(
			    aggregate->GetReturnType(), ColumnBinding(aggregate_index, ProjectionIndex(aggregates.size())));
			auto threshold =
			    optimizer.BindScalarFunction("list_extract", std::move(aggregate_ref),
			                                 make_uniq<BoundConstantExpression>(Value::BIGINT(term.limit)));
			if (threshold->GetReturnType() != key->GetReturnType()) {
				return {};
			}
			threshold->SetAlias(THRESHOLD_ALIAS);
			comparisons.push_back(term.descending ? ExpressionType::COMPARE_GREATERTHANOREQUALTO
			                                      : ExpressionType::COMPARE_LESSTHANOREQUALTO);
			keys.push_back(std::move(key));
			aggregates.push_back(std::move(aggregate));
			thresholds.push_back(std::move(threshold));
		}
		if (!use_cte) {
			LogicalOperatorDeepCopy deep_copy(optimizer.binder, parameter_data);
			build = deep_copy.DeepCopy(input);
		}
	} catch (std::exception &) {
		// a function that does not bind or an input that cannot be copied: leave the plan alone
		return {};
	}

	// the two sides of the join: the input and its copy, or two scans of the materialized input
	vector<ReplacementBinding> probe_replacements;
	ColumnBindingReplacer build_replacer;
	unique_ptr<LogicalOperator> probe;
	unique_ptr<LogicalOperator> cte_definition;
	TableIndex cte_index;
	if (use_cte) {
		cte_index = optimizer.binder.GenerateTableIndex();
		const auto probe_index = optimizer.binder.GenerateTableIndex();
		const auto build_index = optimizer.binder.GenerateTableIndex();
		vector<Identifier> column_names;
		// the materialized input holds the columns that the plan reads. A column that nothing reads is a NULL constant,
		// which leaves the expression that computes it unused, as it is without the rewrite.
		vector<unique_ptr<Expression>> columns;
		for (idx_t i = 0; i < input_bindings.size(); i++) {
			column_names.emplace_back("c" + to_string(i));
			build_replacer.replacement_bindings.emplace_back(input_bindings[i],
			                                                 ColumnBinding(build_index, ProjectionIndex(i)));
			probe_replacements.emplace_back(input_bindings[i], ColumnBinding(probe_index, ProjectionIndex(i)));
			if (IsReferenced(input_bindings[i])) {
				columns.push_back(make_uniq<BoundColumnRefExpression>(input_types[i], input_bindings[i]));
			} else {
				columns.push_back(make_uniq<BoundConstantExpression>(Value(input_types[i])));
			}
		}
		cte_definition = make_uniq<LogicalProjection>(optimizer.binder.GenerateTableIndex(), std::move(columns));
		cte_definition->children.push_back(std::move(input));
		cte_definition->ResolveOperatorTypes();
		probe = make_uniq<LogicalCTERef>(probe_index, cte_index, input_types, column_names);
		build = make_uniq<LogicalCTERef>(build_index, cte_index, input_types, column_names);
	} else {
		build_replacer.AddReplacements(input_bindings, build->GetColumnBindings());
		probe = std::move(input);
	}
	ColumnBindingReplacer probe_replacer;
	probe_replacer.replacement_bindings = probe_replacements;

	// AGGREGATE(GROUP BY p; min(key, k)) over the build side
	auto &partitions = first.Partitions();
	auto aggregate_op = make_uniq<LogicalAggregate>(group_index, aggregate_index, std::move(aggregates));
	for (idx_t i = 0; i < partitions.size(); i++) {
		aggregate_op->groups.push_back(partitions[i]->Copy());
		auto &partition_stats = first.PartitionsStats();
		aggregate_op->group_stats.push_back(
		    i < partition_stats.size() && partition_stats[i] ? partition_stats[i]->ToUnique() : nullptr);
	}
	build_replacer.VisitOperatorBindings(*aggregate_op);
	aggregate_op->children.push_back(std::move(build));
	aggregate_op->ResolveOperatorTypes();

	// PROJECTION(g, th)
	vector<unique_ptr<Expression>> projections;
	for (idx_t i = 0; i < partitions.size(); i++) {
		projections.push_back(make_uniq<BoundColumnRefExpression>(partitions[i]->GetReturnType(),
		                                                          ColumnBinding(group_index, ProjectionIndex(i))));
	}
	for (auto &threshold : thresholds) {
		projections.push_back(std::move(threshold));
	}
	auto projection = make_uniq<LogicalProjection>(projection_index, std::move(projections));
	projection->children.push_back(std::move(aggregate_op));
	projection->ResolveOperatorTypes();

	// JOIN(p IS NOT DISTINCT FROM g): a NULL partition key is a partition like any other
	unique_ptr<LogicalOperator> join;
	if (partitions.empty()) {
		join = LogicalCrossProduct::Create(std::move(probe), std::move(projection));
	} else {
		auto comparison_join = make_uniq<LogicalComparisonJoin>(JoinType::INNER);
		for (idx_t i = 0; i < partitions.size(); i++) {
			auto left = partitions[i]->Copy();
			probe_replacer.VisitExpression(&left);
			auto right = make_uniq<BoundColumnRefExpression>(partitions[i]->GetReturnType(),
			                                                 ColumnBinding(projection_index, ProjectionIndex(i)));
			comparison_join->conditions.push_back(
			    JoinCondition(std::move(left), std::move(right), ExpressionType::COMPARE_NOT_DISTINCT_FROM));
		}
		comparison_join->children.push_back(std::move(probe));
		comparison_join->children.push_back(std::move(projection));
		join = std::move(comparison_join);
	}
	join->ResolveOperatorTypes();

	// FILTER(key <= th OR th IS NULL)
	vector<unique_ptr<Expression>> disjuncts;
	for (idx_t i = 0; i < keys.size(); i++) {
		auto key = std::move(keys[i]);
		probe_replacer.VisitExpression(&key);
		auto threshold_ref = make_uniq<BoundColumnRefExpression>(
		    key->GetReturnType(), ColumnBinding(projection_index, ProjectionIndex(partitions.size() + i)));
		auto is_null = make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NULL, LogicalType::BOOLEAN);
		is_null->GetChildrenMutable().push_back(threshold_ref->Copy());
		disjuncts.push_back(
		    BoundComparisonExpression::Create(comparisons[i], std::move(key), std::move(threshold_ref)));
		disjuncts.push_back(std::move(is_null));
	}
	unique_ptr<LogicalOperator> pre_filter = make_uniq<LogicalFilter>();
	pre_filter->expressions.push_back(
	    BoundConjunctionExpression::Create(ExpressionType::CONJUNCTION_OR, std::move(disjuncts)));
	pre_filter->children.push_back(std::move(join));
	pre_filter->ResolveOperatorTypes();

	if (use_cte) {
		auto cte = make_uniq<LogicalMaterializedCTE>(Identifier("__rank_topn_cte_" + to_string(cte_index.index)),
		                                             cte_index, input_types.size(), std::move(cte_definition),
		                                             std::move(pre_filter), CTEMaterialize::CTE_MATERIALIZE_DEFAULT);
		cte->ResolveOperatorTypes();
		window.children[0] = std::move(cte);
	} else {
		window.children[0] = std::move(pre_filter);
	}
	return probe_replacements;
}

unique_ptr<LogicalOperator> RankWindowTopN::OptimizeInternal(unique_ptr<LogicalOperator> op,
                                                             vector<ReplacementBinding> &pending) {
	if (op->type == LogicalOperatorType::LOGICAL_PREPARE) {
		parameter_data = op->Cast<LogicalPrepare>().prepared->value_map;
	}
	// children first. The input of a rewritten window may have been materialized: its columns are then only valid
	// inside the CTE definition and every operator above the window reads the CTE scan instead.
	for (idx_t child_idx = 0; child_idx < op->children.size(); child_idx++) {
		op->children[child_idx] = OptimizeInternal(std::move(op->children[child_idx]), pending);
		if (child_idx == 0 && op->type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
			// estimate the definition before the scans of the CTE are visited
			auto &definition = *op->children[0];
			CTEEstimate estimate;
			estimate.rows = EstimateRows(definition);
			for (auto &binding : definition.GetColumnBindings()) {
				estimate.distinct.push_back(EstimateDistinct(definition, binding));
			}
			cte_estimates[op->Cast<LogicalMaterializedCTE>().table_index.index] = std::move(estimate);
		}
	}
	if (!pending.empty()) {
		ColumnBindingReplacer replacer;
		replacer.replacement_bindings = pending;
		replacer.VisitOperatorBindings(*op);
	}
	if (op->type != LogicalOperatorType::LOGICAL_FILTER) {
		return op;
	}
	RankMatch match;
	if (!TryMatch(*op, match)) {
		return op;
	}
	auto replacements = Rewrite(*op, match);
	if (replacements.empty()) {
		return op;
	}
	// the filter, the projections and the window read the columns of the input
	ColumnBindingReplacer replacer;
	replacer.replacement_bindings = replacements;
	reference<LogicalOperator> current = *op;
	for (idx_t i = 0; i <= match.depth; i++) {
		replacer.VisitOperatorBindings(current);
		current = *current.get().children[0];
	}
	replacer.VisitOperatorBindings(current);
	for (auto &replacement : replacements) {
		auto entry = references.find(replacement.old_binding);
		if (entry != references.end()) {
			references[replacement.new_binding] = entry->second;
		}
		pending.push_back(replacement);
	}
	return op;
}

} // namespace duckdb
