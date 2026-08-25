//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/match_recognize.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/function.hpp"
#include "duckdb/parser/tableref/match_recognize_ref.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

//! Pattern symbols share a namespace with the input columns, so they are qualified with this prefix in
//! the plan to keep a DEFINE from resolving to a base table column of the same name.
constexpr const char *MATCH_RECOGNIZE_DEFINE_PREFIX = "__mr_define_";

//! The user facing pattern variable for a prefixed plan column
inline string MatchRecognizeSymbolName(const string &column_name) {
	const auto prefix_size = strlen(MATCH_RECOGNIZE_DEFINE_PREFIX);
	if (StringUtil::StartsWith(column_name, MATCH_RECOGNIZE_DEFINE_PREFIX)) {
		return column_name.substr(prefix_size);
	}
	return column_name;
}

class BoundAlternationExpression : public Expression {
public:
	static constexpr const ExpressionClass TYPE = ExpressionClass::PATTERN;

	BoundAlternationExpression(unique_ptr<Expression> child_left_p, unique_ptr<Expression> child_right_p)
	    : Expression(ExpressionType::ALTERNATION, ExpressionClass::PATTERN, LogicalType::INVALID),
	      child_left(std::move(child_left_p)), child_right(std::move(child_right_p)) {
	}

	unique_ptr<Expression> child_left;
	unique_ptr<Expression> child_right;

	string ToString() const override {
		return StringUtil::Format("(%s|%s)", child_left->ToString(), child_right->ToString());
	}

	unique_ptr<Expression> Copy() const override {
		auto child_left_copy = child_left->Copy();
		auto child_right_copy = child_right->Copy();
		return make_uniq<BoundAlternationExpression>(std::move(child_left_copy), std::move(child_right_copy));
	}
};

class BoundConcatenationExpression : public Expression {
public:
	static constexpr const ExpressionClass TYPE = ExpressionClass::PATTERN;

	BoundConcatenationExpression(vector<unique_ptr<Expression>> children_p)
	    : Expression(ExpressionType::CONCATENATION, ExpressionClass::PATTERN, LogicalType::INVALID),
	      children(std::move(children_p)) {
	}

	vector<unique_ptr<Expression>> children;

	string ToString() const override {
		return StringUtil::Join(children, children.size(), ", ",
		                        [](const unique_ptr<Expression> &expr) { return expr->ToString(); });
	}

	unique_ptr<Expression> Copy() const override {
		vector<unique_ptr<Expression>> children_copy;
		for (auto &child : children) {
			children_copy.push_back(child->Copy());
		}
		return make_uniq<BoundConcatenationExpression>(std::move(children_copy));
	}
};

class BoundQuantifierExpression : public Expression {
public:
	static constexpr const ExpressionClass TYPE = ExpressionClass::PATTERN;

	BoundQuantifierExpression(unique_ptr<Expression> child_p, optional_idx min_count_p, optional_idx max_count_p)
	    : Expression(ExpressionType::QUANTIFIER, ExpressionClass::PATTERN, LogicalType::INVALID),
	      child(std::move(child_p)), min_count(min_count_p), max_count(max_count_p) {
	}

	unique_ptr<Expression> child;

	optional_idx min_count;
	optional_idx max_count;

	static string QuantifierToString(optional_idx min_count, optional_idx max_count) {
		return StringUtil::Format("{%s,%s}", min_count.IsValid() ? to_string(min_count.GetIndex()) : "",
		                          max_count.IsValid() ? to_string(max_count.GetIndex()) : "");
	}

	string ToString() const override {
		return child->ToString() + QuantifierToString(min_count, max_count);
	}

	unique_ptr<Expression> Copy() const override {
		auto child_copy = child->Copy();
		return make_uniq<BoundQuantifierExpression>(std::move(child_copy), min_count, max_count);
	}
};

// TODO this needs to live somewhere else!!
struct MatchRecognizeFunctionData : FunctionData {
	unique_ptr<Expression> pattern;
	//! How to resume scanning after a match has been found
	MatchRecognizeAfterMatch after_match = MatchRecognizeAfterMatch::MATCH_RECOGNIZE_AFTER_MATCH_DEFAULT;
	//! The target pattern variable for the SKIP TO FIRST/LAST forms
	string after_match_variable;

	unique_ptr<FunctionData> Copy() const override {
		auto res = make_uniq<MatchRecognizeFunctionData>();

		res->pattern = pattern->Copy();
		res->after_match = after_match;
		res->after_match_variable = after_match_variable;
		return res;
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<MatchRecognizeFunctionData>();
		return other.pattern->Equals(*pattern) && other.after_match == after_match &&
		       other.after_match_variable == after_match_variable;
	}
};

} // namespace duckdb
