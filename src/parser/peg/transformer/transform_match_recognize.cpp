#include "duckdb/function/match_recognize.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/pattern_expression.hpp"
#include "duckdb/parser/peg/transformer/peg_transformer.hpp"
#include "duckdb/parser/tableref/match_recognize_ref.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// MATCH_RECOGNIZE clause
//===--------------------------------------------------------------------===//
unique_ptr<TableRef> PEGTransformerFactory::TransformMatchRecognizeBody(
    PEGTransformer &transformer, optional<vector<unique_ptr<ParsedExpression>>> window_partition,
    optional<vector<OrderByNode>> order_by_clause, vector<unique_ptr<ParsedExpression>> measures_clause,
    optional<MatchRecognizeRows> rows_per_match, optional<MatchRecognizeAfterMatchClause> after_match_skip,
    unique_ptr<ParsedExpression> pattern_clause, optional<vector<MatchRecognizeSubset>> subset_clause,
    vector<unique_ptr<ParsedExpression>> define_clause) {
	auto config = make_uniq<MatchRecognizeConfig>();

	if (window_partition) {
		config->partition_expressions = std::move(*window_partition);
	}
	if (order_by_clause) {
		config->order_by_expressions = std::move(*order_by_clause);
	}
	config->measures_expression_list = std::move(measures_clause);
	config->defines_expression_list = std::move(define_clause);

	config->rows_per_match = rows_per_match ? *rows_per_match : MatchRecognizeRows::MATCH_RECOGNIZE_ROWS_DEFAULT;

	if (after_match_skip) {
		config->after_match = after_match_skip->after_match;
		if (!after_match_skip->variable.empty()) {
			config->after_match_variable = make_uniq<ConstantExpression>(Value(after_match_skip->variable));
		}
	} else {
		config->after_match = MatchRecognizeAfterMatch::MATCH_RECOGNIZE_AFTER_MATCH_DEFAULT;
	}

	config->pattern = std::move(pattern_clause);
	if (subset_clause) {
		config->subsets = std::move(*subset_clause);
	}

	// the input table is attached by TransformTableRef
	return make_uniq<MatchRecognizeRef>(nullptr, std::move(config));
}

unique_ptr<TableRef> PEGTransformerFactory::TransformTableMatchRecognizeClause(
    PEGTransformer &transformer, unique_ptr<TableRef> match_recognize_body, const optional<TableAlias> &table_alias) {
	if (table_alias) {
		match_recognize_body->alias = table_alias->name;
		match_recognize_body->column_name_alias = table_alias->column_name_alias;
	}
	return match_recognize_body;
}

bool PEGTransformerFactory::TransformRunningSemantics(PEGTransformer &transformer) {
	return false;
}

bool PEGTransformerFactory::TransformFinalSemantics(PEGTransformer &transformer) {
	return true;
}

unique_ptr<ParsedExpression> PEGTransformerFactory::TransformMeasuresElement(PEGTransformer &transformer,
                                                                             optional<bool> measure_semantics,
                                                                             unique_ptr<ParsedExpression> expression,
                                                                             const Identifier &col_label_or_string) {
	if (measure_semantics) {
		// carry the choice to the binder, which knows the frame it turns into
		vector<unique_ptr<ParsedExpression>> wrapped;
		wrapped.push_back(std::move(expression));
		expression = make_uniq<FunctionExpression>(
		    *measure_semantics ? MATCH_RECOGNIZE_FINAL_MARKER : MATCH_RECOGNIZE_RUNNING_MARKER, std::move(wrapped));
	}
	expression->SetAlias(col_label_or_string);
	return expression;
}

unique_ptr<ParsedExpression> PEGTransformerFactory::TransformDefineElement(PEGTransformer &transformer,
                                                                           const Identifier &col_label_or_string,
                                                                           unique_ptr<ParsedExpression> expression) {
	expression->SetAlias(col_label_or_string);
	return expression;
}

vector<unique_ptr<ParsedExpression>>
PEGTransformerFactory::TransformMeasuresClause(PEGTransformer &transformer,
                                               vector<unique_ptr<ParsedExpression>> measures_element) {
	return measures_element;
}

MatchRecognizeSubset PEGTransformerFactory::TransformSubsetElement(PEGTransformer &transformer,
                                                                   const Identifier &col_label_or_string,
                                                                   const vector<Identifier> &col_label_or_string_1) {
	MatchRecognizeSubset result;
	result.name = col_label_or_string.GetIdentifierName();
	for (auto &member : col_label_or_string_1) {
		result.members.push_back(member.GetIdentifierName());
	}
	return result;
}

vector<MatchRecognizeSubset> PEGTransformerFactory::TransformSubsetClause(PEGTransformer &transformer,
                                                                          vector<MatchRecognizeSubset> subset_element) {
	return subset_element;
}

vector<unique_ptr<ParsedExpression>>
PEGTransformerFactory::TransformDefineClause(PEGTransformer &transformer,
                                             vector<unique_ptr<ParsedExpression>> define_element) {
	return define_element;
}

//===--------------------------------------------------------------------===//
// ROWS PER MATCH
//===--------------------------------------------------------------------===//
MatchRecognizeRows PEGTransformerFactory::TransformOneRowPerMatch(PEGTransformer &transformer) {
	return MatchRecognizeRows::MATCH_RECOGNIZE_ROWS_ONE;
}

MatchRecognizeRows PEGTransformerFactory::TransformAllRowsPerMatch(PEGTransformer &transformer) {
	return MatchRecognizeRows::MATCH_RECOGNIZE_ROWS_ALL;
}

//===--------------------------------------------------------------------===//
// AFTER MATCH SKIP
//===--------------------------------------------------------------------===//
MatchRecognizeAfterMatchClause PEGTransformerFactory::TransformSkipToNextRow(PEGTransformer &transformer) {
	return MatchRecognizeAfterMatchClause {MatchRecognizeAfterMatch::MATCH_RECOGNIZE_AFTER_MATCH_NEXT_ROW, ""};
}

MatchRecognizeAfterMatchClause PEGTransformerFactory::TransformSkipPastLastRow(PEGTransformer &transformer) {
	return MatchRecognizeAfterMatchClause {MatchRecognizeAfterMatch::MATCH_RECOGNIZE_AFTER_MATCH_LAST_ROW, ""};
}

MatchRecognizeAfterMatchClause PEGTransformerFactory::TransformSkipToFirstVar(PEGTransformer &transformer,
                                                                              const Identifier &col_label_or_string) {
	return MatchRecognizeAfterMatchClause {MatchRecognizeAfterMatch::MATCH_RECOGNIZE_AFTER_MATCH_FIRST_VAR,
	                                       col_label_or_string.GetIdentifierName()};
}

MatchRecognizeAfterMatchClause PEGTransformerFactory::TransformSkipToLastVar(PEGTransformer &transformer,
                                                                             const Identifier &col_label_or_string) {
	return MatchRecognizeAfterMatchClause {MatchRecognizeAfterMatch::MATCH_RECOGNIZE_AFTER_MATCH_LAST_VAR,
	                                       col_label_or_string.GetIdentifierName()};
}

//===--------------------------------------------------------------------===//
// PATTERN
//===--------------------------------------------------------------------===//
unique_ptr<ParsedExpression>
PEGTransformerFactory::TransformRowPattern(PEGTransformer &transformer, unique_ptr<ParsedExpression> row_pattern_term,
                                           optional<vector<unique_ptr<ParsedExpression>>> row_pattern_alternative) {
	auto result = std::move(row_pattern_term);
	if (!row_pattern_alternative) {
		return result;
	}
	// alternation is left-associative: A | B | C becomes ((A | B) | C)
	for (auto &alternative : *row_pattern_alternative) {
		result = make_uniq_base<ParsedExpression, AlternationExpression>(std::move(result), std::move(alternative));
	}
	return result;
}

unique_ptr<ParsedExpression>
PEGTransformerFactory::TransformRowPatternTerm(PEGTransformer &transformer,
                                               vector<unique_ptr<ParsedExpression>> row_pattern_factor) {
	D_ASSERT(!row_pattern_factor.empty());
	if (row_pattern_factor.size() == 1) {
		return std::move(row_pattern_factor[0]);
	}
	return make_uniq_base<ParsedExpression, ConcatenationExpression>(std::move(row_pattern_factor));
}

unique_ptr<ParsedExpression>
PEGTransformerFactory::TransformRowPatternFactor(PEGTransformer &transformer,
                                                 unique_ptr<ParsedExpression> row_pattern_primary,
                                                 optional<MatchRecognizeQuantifier> row_pattern_quantifier) {
	MatchRecognizeQuantifier quantifier;
	if (row_pattern_quantifier) {
		quantifier = *row_pattern_quantifier;
	} else {
		// an unquantified symbol or group matches exactly once
		quantifier.min_count = 1;
		quantifier.max_count = 1;
	}
	return make_uniq_base<ParsedExpression, QuantifiedExpression>(std::move(row_pattern_primary), quantifier.min_count,
	                                                              quantifier.max_count);
}

unique_ptr<ParsedExpression> PEGTransformerFactory::TransformRowPatternLabel(PEGTransformer &transformer,
                                                                             const Identifier &col_label_or_string) {
	return make_uniq_base<ParsedExpression, ColumnRefExpression>(col_label_or_string);
}

//===--------------------------------------------------------------------===//
// PATTERN quantifiers
//===--------------------------------------------------------------------===//
static idx_t QuantifierCount(const unique_ptr<ParsedExpression> &number_literal) {
	if (number_literal->GetExpressionClass() != ExpressionClass::CONSTANT) {
		throw ParserException("Pattern quantifier bounds must be integer constants");
	}
	auto value = number_literal->Cast<ConstantExpression>().GetValue();
	if (!value.DefaultTryCastAs(LogicalType::UBIGINT)) {
		throw ParserException("Pattern quantifier bounds must be non-negative integers");
	}
	return NumericCast<idx_t>(value.GetValue<uint64_t>());
}

MatchRecognizeQuantifier PEGTransformerFactory::TransformQuantifierStar(PEGTransformer &transformer) {
	MatchRecognizeQuantifier result;
	result.min_count = 0;
	return result;
}

MatchRecognizeQuantifier PEGTransformerFactory::TransformQuantifierPlus(PEGTransformer &transformer) {
	MatchRecognizeQuantifier result;
	result.min_count = 1;
	return result;
}

MatchRecognizeQuantifier
PEGTransformerFactory::TransformQuantifierMinMax(PEGTransformer &transformer,
                                                 unique_ptr<ParsedExpression> number_literal,
                                                 unique_ptr<ParsedExpression> number_literal_1) {
	MatchRecognizeQuantifier result;
	result.min_count = QuantifierCount(number_literal);
	result.max_count = QuantifierCount(number_literal_1);
	if (result.min_count.GetIndex() > result.max_count.GetIndex()) {
		throw ParserException("Min count cannot be larger than max count");
	}
	return result;
}

MatchRecognizeQuantifier PEGTransformerFactory::TransformQuantifierMin(PEGTransformer &transformer,
                                                                       unique_ptr<ParsedExpression> number_literal) {
	MatchRecognizeQuantifier result;
	result.min_count = QuantifierCount(number_literal);
	return result;
}

MatchRecognizeQuantifier PEGTransformerFactory::TransformQuantifierMax(PEGTransformer &transformer,
                                                                       unique_ptr<ParsedExpression> number_literal) {
	MatchRecognizeQuantifier result;
	result.max_count = QuantifierCount(number_literal);
	return result;
}

MatchRecognizeQuantifier PEGTransformerFactory::TransformQuantifierExact(PEGTransformer &transformer,
                                                                         unique_ptr<ParsedExpression> number_literal) {
	MatchRecognizeQuantifier result;
	result.min_count = QuantifierCount(number_literal);
	result.max_count = result.min_count;
	return result;
}

} // namespace duckdb
