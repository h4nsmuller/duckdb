
#include "duckdb/function/match_recognize.hpp"

#include "duckdb/parser/expression/case_expression.hpp"
#include "duckdb/parser/expression/comparison_expression.hpp"
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

static void ReplaceFunctions(unique_ptr<ParsedExpression> &expr, const WindowExpression &pattern_window,
                             const string &define_name) {
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
			// DEFINE predicates are evaluated for every row before matching starts, so they cannot
			// depend on the match that is being assembled
			throw NotImplementedException(
			    "MATCH_NUMBER() is not supported in DEFINE, because a DEFINE condition is evaluated before "
			    "the match it would belong to exists. It is supported in MEASURES.");
		} else if (function_name == "CLASSIFIER" && function.GetArguments().empty()) {
			// the row being tested is the one this DEFINE is deciding on, so it classifies as this symbol
			expr = make_uniq<ConstantExpression>(Value(define_name));
			return;
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
	    *expr, [&](unique_ptr<ParsedExpression> &child) { ReplaceFunctions(child, pattern_window, define_name); });
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

//! CASE WHEN <classifier> = 'symbol' THEN <column> END - NULL on every row the variable did not match
static unique_ptr<ParsedExpression> ClassifiedValue(const string &symbol, unique_ptr<ParsedExpression> value) {
	auto classifier = CreateStructExtract("__pattern_window", "classifier");
	auto matches_symbol = make_uniq<ComparisonExpression>(ExpressionType::COMPARE_EQUAL, std::move(classifier),
	                                                      make_uniq<ConstantExpression>(Value(symbol)));
	auto result = make_uniq<CaseExpression>();
	CaseCheck check;
	check.when_expr = std::move(matches_symbol);
	check.then_expr = std::move(value);
	result->CaseChecksMutable().push_back(std::move(check));
	result->ElseMutable() = make_uniq<ConstantExpression>(Value());
	return std::move(result);
}

//! A reference to <symbol>.<column> resolves to that column on the last row the variable matched.
//! ONE ROW PER MATCH reports a finished match, so it sees the whole match (FINAL semantics); ALL ROWS
//! PER MATCH reports progress, so it only sees the match up to the current row (RUNNING semantics).
static void ScopeToMatch(WindowExpression &window, const MatchRecognizeConfig &config, bool running) {
	window.WindowStartMutable() = WindowBoundary::UNBOUNDED_PRECEDING;
	window.WindowEndMutable() = running ? WindowBoundary::CURRENT_ROW_ROWS : WindowBoundary::UNBOUNDED_FOLLOWING;

	// matches are numbered within a partition, so both are needed to identify one
	for (auto &expr : config.partition_expressions) {
		window.PartitionsMutable().push_back(expr->Copy());
	}
	window.PartitionsMutable().push_back(CreateStructExtract("__pattern_window", "match_number"));
	for (auto &order : config.order_by_expressions) {
		window.OrderByMutable().emplace_back(order.type, order.null_order, order.expression->Copy());
	}
}

static unique_ptr<ParsedExpression> MatchScopedValue(const MatchRecognizeConfig &config,
                                                     unique_ptr<ParsedExpression> value, bool running) {
	auto window = make_uniq<WindowExpression>("", "", "last_value");
	window->GetArgumentsMutable().emplace_back(std::move(value));
	window->HasIgnoreNullsMutable() = true;
	window->IgnoreNullsMutable() = true;
	ScopeToMatch(*window, config, running);
	return std::move(window);
}

//! Rewrite a MEASURES expression into something evaluable next to the pattern window
static void RewriteMeasure(Binder &binder, unique_ptr<ParsedExpression> &expr, const MatchRecognizeConfig &config,
                           const case_insensitive_set_t &symbols, bool running, bool inside_aggregate = false) {
	if (expr->GetExpressionType() == ExpressionType::FUNCTION) {
		auto &function = expr->Cast<FunctionExpression>();
		auto function_name = StringUtil::Upper(function.FunctionName().GetIdentifierName());
		if (function_name == "CLASSIFIER" && function.GetArguments().empty()) {
			expr = CreateStructExtract("__pattern_window", "classifier");
			return;
		}
		if (function_name == "MATCH_NUMBER" && function.GetArguments().empty()) {
			expr = CreateStructExtract("__pattern_window", "match_number");
			return;
		}
		// an aggregate in MEASURES aggregates the rows of the match
		EntryLookupInfo lookup(CatalogType::AGGREGATE_FUNCTION_ENTRY, QualifiedName(function.FunctionName()));
		auto entry = binder.GetCatalogEntry(function.GetQualifiedName().Catalog(), function.GetQualifiedName().Schema(),
		                                    lookup, OnEntryNotFound::RETURN_NULL);
		if (entry && entry->type == CatalogType::AGGREGATE_FUNCTION_ENTRY) {
			// the aggregate already spans the match, so inside it a variable only masks its rows
			for (auto &argument : function.GetArgumentsMutable()) {
				RewriteMeasure(binder, argument.GetExpressionMutable(), config, symbols, running, true);
			}
			auto &qualified = function.GetQualifiedName();
			auto window = make_uniq<WindowExpression>(qualified.Catalog().GetIdentifierName(),
			                                          qualified.Schema().GetIdentifierName(),
			                                          qualified.Name().GetIdentifierName());
			window->GetArgumentsMutable() = std::move(function.GetArgumentsMutable());
			window->DistinctMutable() = function.Distinct();
			ScopeToMatch(*window, config, running);
			expr = std::move(window);
			return;
		}
	}
	if (expr->GetExpressionType() == ExpressionType::COLUMN_REF) {
		auto &colref = expr->Cast<ColumnRefExpression>();
		auto &names = colref.ColumnNames();
		if (names.size() == 2 && symbols.find(names[0].GetIdentifierName()) != symbols.end()) {
			// a known pattern variable scopes the column to the rows it matched
			auto symbol = names[0].GetIdentifierName();
			auto column = make_uniq<ColumnRefExpression>(colref.GetColumnName());
			auto masked = ClassifiedValue(symbol, std::move(column));
			expr = inside_aggregate ? std::move(masked) : MatchScopedValue(config, std::move(masked), running);
			return;
		}
		if (colref.IsQualified()) {
			// the input is reached through a subquery here, so a table qualifier no longer resolves
			colref.ColumnNamesMutable() = {colref.GetColumnName()};
			return;
		}
	}
	ParsedExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<ParsedExpression> &child) {
		RewriteMeasure(binder, child, config, symbols, running, inside_aggregate);
	});
}

BoundStatement Binder::Bind(MatchRecognizeRef &ref) {
	// MATCH_RECOGNIZE is planned as a stack of select nodes:
	//   1. the input, plus one boolean column per DEFINE
	//   2. the pattern window, which classifies every row of every match; rows outside a match are
	//      dropped here
	//   3. the MEASURES, computed across the match a row belongs to
	//   4. for ONE ROW PER MATCH, a filter down to the row each match starts on

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
	vector<string> define_column_names;
	case_insensitive_set_t pattern_symbols;

	for (auto &expr : ref.config->defines_expression_list) {
		auto define_name = expr->GetAlias().GetIdentifierName();
		// TODO can this happen?
		D_ASSERT(!define_name.empty());
		auto column_name = DefineColumnName(define_name);
		D_ASSERT(pattern_window_child_entries.find(column_name) == pattern_window_child_entries.end());

		CheckAndZapQualifiers(*expr, define_name);
		ReplaceFunctions(expr, window_template->Cast<WindowExpression>(), define_name);
		expr->SetAlias(Identifier(column_name));
		define_select_node->select_list.push_back(std::move(expr));
		pattern_window_child_entries[column_name] = make_uniq<ColumnRefExpression>(Identifier(column_name));
		define_column_names.push_back(column_name);
		pattern_symbols.insert(define_name);
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
			    define_column_names.push_back(symbol_name);
			    pattern_symbols.insert(MatchRecognizeSymbolName(symbol_name));
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

	// keep every row of every match: the measures below are computed across the match. Rows outside a
	// match have a NULL struct, which discards them.
	select_node->qualify = CreateStructExtract("__pattern_window", "in_match");

	// MEASURES are projected on top of the pattern window, where the match a row belongs to is known
	const auto all_rows = ref.config->rows_per_match == MatchRecognizeRows::MATCH_RECOGNIZE_ROWS_ALL;
	auto pattern_select = make_uniq<SelectStatement>(std::move(select_node));
	auto measures_node = make_uniq<SelectNode>(make_uniq<SubqueryRef>(std::move(pattern_select)));

	// the DEFINE columns are an implementation detail, so they do not reach the output
	auto star = make_uniq<StarExpression>();
	for (auto &entry : define_column_names) {
		star->ExcludeListMutable().insert(QualifiedColumnName(Identifier(entry)));
	}
	measures_node->select_list.push_back(std::move(star));

	for (auto &expr : ref.config->measures_expression_list) {
		D_ASSERT(!expr->GetAlias().empty());
		// rewriting can replace the expression wholesale, which would drop the MEASURES alias
		auto alias = expr->GetAlias();
		RewriteMeasure(*this, expr, *ref.config, pattern_symbols, all_rows);
		expr->SetAlias(std::move(alias));
		measures_node->select_list.push_back(std::move(expr));
	}

	select_node = std::move(measures_node);

	// ONE ROW PER MATCH reports one row per match. This has to filter above the measures rather than
	// beside them: the measures are computed across the match, so the other rows have to still be there.
	if (!all_rows) {
		auto measures_select = make_uniq<SelectStatement>(std::move(select_node));
		auto filter_node = make_uniq<SelectNode>(make_uniq<SubqueryRef>(std::move(measures_select)));
		filter_node->select_list.push_back(make_uniq<StarExpression>());
		filter_node->where_clause = CreateStructExtract("__pattern_window", "is_match_start");
		select_node = std::move(filter_node);
	}

	auto child_binder = Binder::CreateBinder(context, this);
	auto result = child_binder->Bind(*select_node);
	bind_context.AddGenericBinding(result.plan->GetRootIndex(),
	                               !ref.alias.empty() ? ref.alias : "__match_recognize_table", result.names,
	                               result.types);
	return result;
}

} // namespace duckdb
