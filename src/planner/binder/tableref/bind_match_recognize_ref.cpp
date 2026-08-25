
#include "duckdb/function/match_recognize.hpp"

#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/pattern_expression.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/tableref/match_recognize_ref.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"

#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/planner/expression_binder.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"

namespace duckdb {

// TODO make abstract superclass
// class BoundPatternExpression : public Expression {
//
// 	vector<unique_ptr<Expression>> children;
//
// };

BindResult ExpressionBinder::BindPatternExpression(unique_ptr<ParsedExpression> &expr, idx_t depth) {
	switch (expr->GetExpressionType()) {
	case ExpressionType::ALTERNATION: {
		auto &alternation = expr->Cast<AlternationExpression>();
		auto bound_left = BindExpression(alternation.child_left, depth);
		if (bound_left.HasError()) {
			return BindResult(bound_left.error);
		}
		auto bound_right = BindExpression(alternation.child_right, depth);
		if (bound_right.HasError()) {
			return BindResult(bound_right.error);
		}
		return BindResult(make_uniq_base<Expression, BoundAlternationExpression>(std::move(bound_left.expression),
		                                                                         std::move(bound_right.expression)));
	}
	case ExpressionType::CONCATENATION: {
		auto &concatenation = expr->Cast<ConcatenationExpression>();
		vector<unique_ptr<Expression>> bound_children;
		for (auto &child : concatenation.children) {
			auto child_bind_result = BindExpression(child, depth);
			if (child_bind_result.HasError()) {
				return BindResult(child_bind_result.error);
			}
			bound_children.push_back(std::move(child_bind_result.expression));
		}
		return BindResult(make_uniq_base<Expression, BoundConcatenationExpression>(std::move(bound_children)));
	}
	case ExpressionType::QUANTIFIER: {
		auto &quantifier = expr->Cast<QuantifiedExpression>();
		auto bound_child = BindExpression(quantifier.child, depth);
		if (bound_child.HasError()) {
			return BindResult(bound_child.error);
		}
		return BindResult(make_uniq_base<Expression, BoundQuantifierExpression>(
		    std::move(bound_child.expression), quantifier.min_count, quantifier.max_count));
	}
	default:
		throw NotImplementedException("Unimplemented pattern expression %s",
		                              ExpressionTypeToString(expr->GetExpressionType()));
	}
}

static void CheckAndZapQualifiers(ParsedExpression &root_expr, const string &define_name) {
	ParsedExpressionIterator::VisitExpressionMutable<ColumnRefExpression>(root_expr, [&](ColumnRefExpression &colref) {
		if (colref.IsQualified() && colref.ColumnNames()[0] != define_name) {
			throw NotImplementedException("Define references cannot refer to other defines just yet %s <> %s",
			                              colref.ColumnNames()[0].GetIdentifierName(), define_name);
		}
		colref.ColumnNamesMutable() = {colref.GetColumnName()};
	});
}

static void ReplaceFunctions(unique_ptr<ParsedExpression> &expr, const WindowExpression &pattern_window) {
	if (expr->GetExpressionType() == ExpressionType::FUNCTION) {
		auto &function = expr->Cast<FunctionExpression>();
		auto function_name = StringUtil::Upper(function.FunctionName().GetIdentifierName());

		string window_function;
		if (function_name == "PREV") {
			window_function = "lag";
		} else if (function_name == "NEXT") {
			window_function = "lead";
		} else if (function_name == "FIRST") {
			window_function = "first_value";
		} else if (function_name == "LAST") {
			window_function = "last_value";
		} else if (function_name == "MATCH_NUMBER") {
			throw NotImplementedException("MATCH_NUMBER");
		} else if (function_name == "CLASSIFIER") {
			throw NotImplementedException("CLASSIFIER");
		}

		if (!window_function.empty()) {
			auto new_expr =
			    pattern_window.Copy(); // we copy here because we need to keep all the partitioning and stuff
			auto &new_window = new_expr->Cast<WindowExpression>();
			new_window.SetFunctionName(window_function);
			new_window.GetArgumentsMutable() = std::move(function.GetArgumentsMutable());
			expr = std::move(new_expr);
		}
		// we do nothing if it's something else
	}
	ParsedExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<ParsedExpression> &child) { ReplaceFunctions(child, pattern_window); });
}

//! Pattern symbols live in the same namespace as the input columns, so they are qualified with an
//! internal prefix to keep a DEFINE from resolving to a base table column of the same name.
static string DefineColumnName(const string &symbol) {
	return MATCH_RECOGNIZE_DEFINE_PREFIX + symbol;
}

static unique_ptr<ParsedExpression> CreateStructExtract(const string &column_name, const string &child_name) {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(make_uniq<ColumnRefExpression>(Identifier(column_name)));
	children.push_back(make_uniq<ConstantExpression>(child_name));
	return make_uniq<FunctionExpression>("struct_extract", std::move(children));
}

BoundStatement Binder::Bind(MatchRecognizeRef &ref) {
	// Steps:
	// binding+planning input table
	// apply pattern matching window (todo: complete implementation)
	// filter out only complete matches
	// apply after match window
	// filter out only non-overlapping matches
	// todo: all rows per match, measures, final, ...
	// final projection that hides intermediate structures

	auto select_node = make_uniq<SelectNode>(std::move(ref.input));
	select_node->select_list.push_back(make_uniq<StarExpression>());

	// Pattern Matching Window: placeholder window expression
	auto pattern_window = make_uniq<WindowExpression>("", "", "match_recognize");

	pattern_window->WindowStartMutable() = WindowBoundary::UNBOUNDED_PRECEDING;
	pattern_window->WindowEndMutable() = WindowBoundary::UNBOUNDED_FOLLOWING;

	// copy partitions to bind them twice in different places
	vector<unique_ptr<ParsedExpression>> partitions;
	for (auto &expr : ref.config->partition_expressions) {
		partitions.push_back(expr->Copy());
	}
	pattern_window->PartitionsMutable() = std::move(partitions);
	pattern_window->OrderByMutable() = std::move(ref.config->order_by_expressions);

	// another select node
	// all the inputs for the defines go in their own select node

	auto define_select_node = make_uniq<SelectNode>(std::move(select_node->from_table));

	vector<unique_ptr<WindowExpression>> child_windows;
	define_select_node->select_list.push_back(make_uniq<StarExpression>());

	// we use this window function as a template for order, partition, and boundaries
	D_ASSERT(pattern_window->GetArguments().empty()); // for now
	auto window_template = pattern_window->Copy();

	// case_insensitive_set_t define_names;

	case_insensitive_map_t<unique_ptr<ParsedExpression>> pattern_window_child_entries;

	for (auto &expr : ref.config->defines_expression_list) {
		auto define_name = expr->GetAlias().GetIdentifierName();
		// TODO can this happen?
		D_ASSERT(!define_name.empty());
		auto column_name = DefineColumnName(define_name);
		D_ASSERT(pattern_window_child_entries.find(column_name) == pattern_window_child_entries.end());

		CheckAndZapQualifiers(*expr, define_name);
		ReplaceFunctions(expr, window_template->Cast<WindowExpression>());
		expr->SetAlias(Identifier(column_name));
		define_select_node->select_list.push_back(std::move(expr));
		pattern_window_child_entries[column_name] = make_uniq<ColumnRefExpression>(Identifier(column_name));
	}

	// rewrite the pattern symbols to the same internal names
	ParsedExpressionIterator::VisitExpressionMutable<ColumnRefExpression>(
	    *ref.config->pattern, [&](ColumnRefExpression &colref) {
		    D_ASSERT(colref.ColumnNames().size() == 1);
		    colref.ColumnNamesMutable() = {Identifier(DefineColumnName(colref.GetColumnName().GetIdentifierName()))};
	    });

	// push computation of measures into the lowest window.
	// for (auto &expr : ref.config->measures_expression_list) {
	// 	D_ASSERT(!expr->alias.empty());
	// 	CheckAndZapQualifiers(*expr, expr->alias);
	// 	define_select_node->select_list.push_back(expr->Copy());
	// 	pattern_window->children.push_back(make_uniq<ColumnRefExpression>(expr->alias));
	// }

	// for any symbols that *are* in the pattern but are *not* in the defines,
	// we just push a dummy column so this can bind
	ParsedExpressionIterator::VisitExpression<ColumnRefExpression>(
	    *ref.config->pattern, [&](const ColumnRefExpression &colref) {
		    D_ASSERT(colref.ColumnNames().size() == 1);
		    auto symbol_name = colref.ColumnNames()[0].GetIdentifierName();
		    if (pattern_window_child_entries.find(symbol_name) ==
		        pattern_window_child_entries.end()) { // TODO can those even occur multiple times?
			    // not in define list, implicitly created symbol yay
			    auto define_expression = make_uniq<ConstantExpression>(Value::BOOLEAN(true));
			    define_expression->SetAlias(Identifier(symbol_name));
			    define_select_node->select_list.push_back(std::move(define_expression));
			    pattern_window_child_entries[symbol_name] = make_uniq<ColumnRefExpression>(Identifier(symbol_name));
		    }
	    });

	vector<unique_ptr<ParsedExpression>> pattern_window_children;
	for (auto &pattern_window_child_entry : pattern_window_child_entries) {
		pattern_window_children.push_back(std::move(pattern_window_child_entry.second));
	}

	auto struct_pack_expr = make_uniq<FunctionExpression>("struct_pack", std::move(pattern_window_children));
	pattern_window->GetArgumentsMutable().emplace_back(std::move(struct_pack_expr));

	// TODO deal with measures, what on earth is in the output??
	// TODO likely measures have to be pushed into the way-down window, too

	// the AFTER MATCH SKIP option and the pattern are configuration rather than per-row arguments;
	// the bind callback moves them into the function data
	pattern_window->GetArgumentsMutable().emplace_back(
	    make_uniq<ConstantExpression>(Value::UTINYINT(static_cast<uint8_t>(ref.config->after_match))));
	auto skip_variable = Value(LogicalType::VARCHAR);
	if (ref.config->after_match_variable) {
		skip_variable = Value(DefineColumnName(ref.config->after_match_variable->GetValue().ToString()));
	}
	pattern_window->GetArgumentsMutable().emplace_back(make_uniq<ConstantExpression>(std::move(skip_variable)));
	pattern_window->GetArgumentsMutable().emplace_back(std::move(ref.config->pattern));

	auto define_select = make_uniq<SelectStatement>(std::move(define_select_node));
	select_node->from_table = make_uniq<SubqueryRef>(std::move(define_select));
	pattern_window->SetAlias("__pattern_window");
	select_node->select_list.push_back(std::move(pattern_window));

	// ONE ROW PER MATCH keeps only the row a match starts on, ALL ROWS PER MATCH keeps every row of
	// the match. Rows outside a match have a NULL struct, so both filters discard them.
	const auto all_rows = ref.config->rows_per_match == MatchRecognizeRows::MATCH_RECOGNIZE_ROWS_ALL;
	select_node->qualify = CreateStructExtract("__pattern_window", all_rows ? "in_match" : "is_match_start");

	auto child_binder = Binder::CreateBinder(context, this);
	auto result = child_binder->Bind(*select_node);
	bind_context.AddGenericBinding(result.plan->GetRootIndex(),
	                               !ref.alias.empty() ? ref.alias : "__match_recognize_table", result.names,
	                               result.types);
	return result;
}

} // namespace duckdb
