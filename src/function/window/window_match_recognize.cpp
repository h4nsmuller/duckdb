#include "duckdb/function/window/window_match_recognize.hpp"

#include "duckdb/function/match_recognize.hpp"
#include "duckdb/function/window/match_recognize_functions.hpp"
#include "duckdb/function/window/window_shared_expressions.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"

namespace duckdb {

//	Column indexes into the result struct
enum MatchRecognizeResult : idx_t { CLASSIFIER = 0, MATCH_NUMBER, IS_MATCH_START, MATCH_START, MATCH_END };

struct WindowMatchRecognizeGlobalState : WindowExecutorGlobalState {
	WindowMatchRecognizeGlobalState(ClientContext &client, const WindowExecutor &executor, const idx_t payload_count,
	                                const ValidityMask &partition_mask, const ValidityMask &order_mask)
	    : WindowExecutorGlobalState(client, executor, payload_count, partition_mask, order_mask),
	      result_vec(executor.wexpr.GetReturnType(), payload_count), spans(payload_count) {
		auto &config = executor.wexpr.BindInfo()->Cast<MatchRecognizeFunctionData>();
		condition_values.resize(config.conditions.size());
		for (auto &values : condition_values) {
			values.assign(payload_count, 0);
		}
		classifiers.resize(payload_count);
		D_ASSERT(result_vec.GetType().id() == LogicalTypeId::LIST);
	}

	//! One row of the result list
	struct Span {
		idx_t symbol;
		idx_t match_number;
		bool is_match_start;
		idx_t match_start;
		idx_t match_end;
	};

	//! Build the list vector the operator reads from the spans collected during matching
	void MaterializeSpans(const vector<string> &symbols) {
		idx_t total = 0;
		for (auto &row : spans) {
			total += row.size();
		}
		ListVector::Reserve(result_vec, total);
		ListVector::SetListSize(result_vec, total);
		auto list_data = FlatVector::GetDataMutable<list_entry_t>(result_vec);
		auto &child = ListVector::GetEntry(result_vec);
		auto &fields = StructVector::GetEntries(child);

		idx_t offset = 0;
		for (idx_t row = 0; row < spans.size(); row++) {
			list_data[row].offset = offset;
			list_data[row].length = spans[row].size();
			for (auto &span : spans[row]) {
				fields[CLASSIFIER].SetValue(offset, Value(MatchRecognizeSymbolName(symbols[span.symbol])));
				fields[MATCH_NUMBER].SetValue(offset, Value::UBIGINT(span.match_number));
				fields[IS_MATCH_START].SetValue(offset, Value::BOOLEAN(span.is_match_start));
				fields[MATCH_START].SetValue(offset, Value::UBIGINT(span.match_start));
				fields[MATCH_END].SetValue(offset, Value::UBIGINT(span.match_end));
				offset++;
			}
		}
	}

	// TODO can we get away with putting this into the local state?
	mutex state_lock;
	//! Set up once; the threads then take partitions from the cursor below
	bool prepared = false;
	//! Partitions are independent, so the threads that reach Finalize share them out
	vector<pair<idx_t, idx_t>> partitions;
	atomic<idx_t> next_partition {0};
	atomic<idx_t> completed_partitions {0};
	//! The variable that classified each row, written only by the thread that owns the partition
	vector<idx_t> classifiers;
	//! Materialised only when a condition has to be settled per row, and then shared by the threads
	DataChunk rows;
	//! One boolean per symbol per row. Sink fills these as rows arrive, over disjoint ranges, so the
	//! threads do not need to coordinate.
	vector<vector<uint8_t>> condition_values;

	Vector result_vec;
	vector<vector<Span>> spans;
};

//	Column indexes into the result struct

LogicalType WindowMatchRecognizeExecutor::ResultType() {
	// One entry per match a row takes part in: overlapping matches each keep their own, and the plan
	// unnests the list. Rows that matched nothing get an empty list, which unnest drops.
	return LogicalType::LIST(LogicalType::STRUCT({{"classifier", LogicalType::VARCHAR},
	                                              {"match_number", LogicalType::UBIGINT},
	                                              {"is_match_start", LogicalType::BOOLEAN},
	                                              {"match_start", LogicalType::UBIGINT},
	                                              {"match_end", LogicalType::UBIGINT}}));
}

//===--------------------------------------------------------------------===//
// Binding
//===--------------------------------------------------------------------===//
//! Point a condition's column references at the window's argument list
static void RebindToArguments(unique_ptr<Expression> &expr, const expression_map_t<idx_t> &argument_index,
                              idx_t match_number_index, bool &reads_match_number) {
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		auto entry = argument_index.find(*expr);
		if (entry == argument_index.end()) {
			throw BinderException("A DEFINE condition may only reference columns of the MATCH_RECOGNIZE input");
		}
		if (entry->second == match_number_index) {
			reads_match_number = true;
		}
		expr = make_uniq<BoundReferenceExpression>(expr->GetReturnType(), entry->second);
		return;
	}
	ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
		RebindToArguments(child, argument_index, match_number_index, reads_match_number);
	});
}

//! Replace each pattern leaf's symbol name with its index
static void ResolvePatternSymbols(unique_ptr<Expression> &pattern, const unordered_map<string, idx_t> &symbol_index) {
	if (pattern->GetExpressionType() == ExpressionType::VALUE_CONSTANT) {
		auto &constant = pattern->Cast<BoundConstantExpression>();
		if (constant.GetValue().type().id() == LogicalTypeId::VARCHAR) {
			auto entry = symbol_index.find(constant.GetValue().GetValue<string>());
			D_ASSERT(entry != symbol_index.end());
			pattern = make_uniq<BoundConstantExpression>(Value::UBIGINT(entry->second));
		}
		return;
	}
	switch (pattern->GetExpressionType()) {
	case ExpressionType::ALTERNATION: {
		auto &alternation = pattern->Cast<BoundAlternationExpression>();
		ResolvePatternSymbols(alternation.child_left, symbol_index);
		ResolvePatternSymbols(alternation.child_right, symbol_index);
		break;
	}
	case ExpressionType::CONCATENATION:
		for (auto &child : pattern->Cast<BoundConcatenationExpression>().children) {
			ResolvePatternSymbols(child, symbol_index);
		}
		break;
	case ExpressionType::QUANTIFIER:
		ResolvePatternSymbols(pattern->Cast<BoundQuantifierExpression>().child, symbol_index);
		break;
	default:
		break;
	}
}

unique_ptr<FunctionData> WindowMatchRecognizeExecutor::Bind(BindWindowFunctionInput &input) {
	auto &arguments = input.GetArguments();
	// Deserialization rebinds the function after the configuration arguments are gone, filling the
	// optional parameters with NULL. The bind data is restored by the deserialize callback instead.
	const auto configured = arguments.size() == 7 &&
	                        arguments[3]->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT &&
	                        !arguments[3]->Cast<BoundConstantExpression>().GetValue().IsNull();
	if (!configured) {
		input.GetBoundFunction().GetArguments().resize(MinValue<idx_t>(1, arguments.size()));
		input.GetBoundFunction().SetReturnType(ResultType());
		return nullptr;
	}

	auto bind_data = make_uniq<MatchRecognizeFunctionData>();
	bind_data->after_match = static_cast<MatchRecognizeAfterMatch>(
	    arguments[5]->Cast<BoundConstantExpression>().GetValue().GetValue<uint8_t>());
	auto &skip_variable = arguments[4]->Cast<BoundConstantExpression>().GetValue();
	if (!skip_variable.IsNull()) {
		bind_data->after_match_variable = skip_variable.GetValue<string>();
	}
	for (auto &symbol : ListValue::GetChildren(arguments[3]->Cast<BoundConstantExpression>().GetValue())) {
		bind_data->symbols.push_back(symbol.GetValue<string>());
	}
	bind_data->pattern = std::move(arguments[2]);

	// the columns are packed in argument order, which is the order the conditions address them in
	expression_map_t<idx_t> argument_index;
	auto &column_pack = arguments[0]->Cast<BoundFunctionExpression>();
	for (idx_t i = 0; i < column_pack.GetChildren().size(); i++) {
		argument_index[*column_pack.GetChildren()[i]] = i;
	}

	// the conditions are only packed so that they get bound; they are evaluated by the matcher
	unordered_set<idx_t> navigation_fields;
	for (auto &navigation : ListValue::GetChildren(arguments[6]->Cast<BoundConstantExpression>().GetValue())) {
		auto &fields = StructValue::GetChildren(navigation);
		MatchRecognizeFunctionData::Navigation spec;
		spec.last = fields[0].GetValue<bool>();
		spec.symbol = fields[1].IsNull() ? string() : fields[1].GetValue<string>();
		spec.field = NumericCast<idx_t>(fields[2].GetValue<uint64_t>());
		spec.offset = NumericCast<idx_t>(fields[3].GetValue<uint64_t>());
		navigation_fields.insert(spec.field);
		bind_data->navigations.push_back(spec);
	}

	auto &condition_pack = arguments[1]->Cast<BoundFunctionExpression>();
	for (auto &condition : condition_pack.GetChildrenMutable()) {
		bool reads_match_number = false;
		RebindToArguments(condition, argument_index, 0, reads_match_number);
		bool reads_navigation = false;
		ExpressionIterator::VisitExpression<BoundReferenceExpression>(
		    *condition, [&](const BoundReferenceExpression &bound_ref) {
			    reads_navigation = reads_navigation || navigation_fields.count(bound_ref.Index()) > 0;
		    });
		// Both kinds depend on the match being assembled, so both are settled per candidate row.
		// Re-deciding them for a whole partition after every match would be quadratic.
		bind_data->row_scoped.push_back(reads_navigation || reads_match_number);
		bind_data->depends_on_match_number = bind_data->depends_on_match_number || reads_match_number;
		bind_data->conditions.push_back(std::move(condition));
	}
	if (bind_data->conditions.size() != bind_data->symbols.size()) {
		throw BinderException("MATCH_RECOGNIZE has a condition for every pattern symbol");
	}

	// the matcher compares symbols on every candidate row, so the leaves carry an index into
	// symbols rather than the name itself
	unordered_map<string, idx_t> symbol_index;
	for (idx_t i = 0; i < bind_data->symbols.size(); i++) {
		symbol_index[bind_data->symbols[i]] = i;
	}
	ResolvePatternSymbols(bind_data->pattern, symbol_index);

	auto &bound_function = input.GetBoundFunction();
	bound_function.GetArguments().resize(1);
	bound_function.SetReturnType(ResultType());

	return std::move(bind_data);
}

//===--------------------------------------------------------------------===//
// Serialization
//===--------------------------------------------------------------------===//
// The pattern is built from expression types that only exist here, so it is written out directly
// rather than through the expression serializer.
static void SerializePattern(Serializer &serializer, const Expression &pattern) {
	serializer.WriteProperty(100, "type", pattern.GetExpressionType());
	switch (pattern.GetExpressionType()) {
	case ExpressionType::ALTERNATION: {
		auto &alternation = pattern.Cast<BoundAlternationExpression>();
		serializer.WriteObject(101, "left",
		                       [&](Serializer &child) { SerializePattern(child, *alternation.child_left); });
		serializer.WriteObject(102, "right",
		                       [&](Serializer &child) { SerializePattern(child, *alternation.child_right); });
		break;
	}
	case ExpressionType::CONCATENATION: {
		auto &concatenation = pattern.Cast<BoundConcatenationExpression>();
		serializer.WriteList(101, "children", concatenation.children.size(), [&](Serializer::List &list, idx_t i) {
			list.WriteObject([&](Serializer &child) { SerializePattern(child, *concatenation.children[i]); });
		});
		break;
	}
	case ExpressionType::QUANTIFIER: {
		auto &quantifier = pattern.Cast<BoundQuantifierExpression>();
		serializer.WriteObject(101, "child", [&](Serializer &child) { SerializePattern(child, *quantifier.child); });
		serializer.WriteProperty(102, "min_count", quantifier.min_count);
		serializer.WriteProperty(103, "max_count", quantifier.max_count);
		break;
	}
	case ExpressionType::VALUE_CONSTANT:
		serializer.WriteProperty(101, "symbol", pattern.Cast<BoundConstantExpression>().GetValue());
		break;
	default:
		throw SerializationException("Unsupported MATCH_RECOGNIZE pattern node");
	}
}

static unique_ptr<Expression> DeserializePattern(Deserializer &deserializer) {
	auto type = deserializer.ReadProperty<ExpressionType>(100, "type");
	switch (type) {
	case ExpressionType::ALTERNATION: {
		unique_ptr<Expression> left;
		unique_ptr<Expression> right;
		deserializer.ReadObject(101, "left", [&](Deserializer &child) { left = DeserializePattern(child); });
		deserializer.ReadObject(102, "right", [&](Deserializer &child) { right = DeserializePattern(child); });
		return make_uniq_base<Expression, BoundAlternationExpression>(std::move(left), std::move(right));
	}
	case ExpressionType::CONCATENATION: {
		vector<unique_ptr<Expression>> children;
		deserializer.ReadList(101, "children", [&](Deserializer::List &list, idx_t i) {
			list.ReadObject([&](Deserializer &child) { children.push_back(DeserializePattern(child)); });
		});
		return make_uniq_base<Expression, BoundConcatenationExpression>(std::move(children));
	}
	case ExpressionType::QUANTIFIER: {
		unique_ptr<Expression> child;
		deserializer.ReadObject(101, "child", [&](Deserializer &inner) { child = DeserializePattern(inner); });
		auto min_count = deserializer.ReadProperty<optional_idx>(102, "min_count");
		auto max_count = deserializer.ReadProperty<optional_idx>(103, "max_count");
		return make_uniq_base<Expression, BoundQuantifierExpression>(std::move(child), min_count, max_count);
	}
	case ExpressionType::VALUE_CONSTANT:
		return make_uniq_base<Expression, BoundConstantExpression>(deserializer.ReadProperty<Value>(101, "symbol"));
	default:
		throw SerializationException("Unsupported MATCH_RECOGNIZE pattern node");
	}
}

void WindowMatchRecognizeExecutor::Serialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data,
                                             const BoundWindowFunction &function) {
	auto &config = bind_data->Cast<MatchRecognizeFunctionData>();
	serializer.WriteObject(100, "pattern", [&](Serializer &child) { SerializePattern(child, *config.pattern); });
	serializer.WriteProperty(101, "conditions", config.conditions);
	serializer.WriteProperty(102, "symbols", config.symbols);
	serializer.WriteProperty(103, "after_match", config.after_match);
	serializer.WriteProperty(104, "after_match_variable", config.after_match_variable);
	serializer.WriteProperty(105, "depends_on_match_number", config.depends_on_match_number);
	serializer.WriteProperty(106, "row_scoped", config.row_scoped);
	serializer.WriteList(107, "navigations", config.navigations.size(), [&](Serializer::List &list, idx_t i) {
		auto &navigation = config.navigations[i];
		list.WriteObject([&](Serializer &child) {
			child.WriteProperty(100, "last", navigation.last);
			child.WriteProperty(101, "symbol", navigation.symbol);
			child.WriteProperty(102, "field", navigation.field);
			child.WriteProperty(103, "offset", navigation.offset);
		});
	});
}

unique_ptr<FunctionData> WindowMatchRecognizeExecutor::Deserialize(Deserializer &deserializer,
                                                                   BoundWindowFunction &function) {
	auto result = make_uniq<MatchRecognizeFunctionData>();
	deserializer.ReadObject(100, "pattern", [&](Deserializer &child) { result->pattern = DeserializePattern(child); });
	deserializer.ReadProperty(101, "conditions", result->conditions);
	deserializer.ReadProperty(102, "symbols", result->symbols);
	deserializer.ReadProperty(103, "after_match", result->after_match);
	deserializer.ReadProperty(104, "after_match_variable", result->after_match_variable);
	deserializer.ReadProperty(105, "depends_on_match_number", result->depends_on_match_number);
	deserializer.ReadProperty(106, "row_scoped", result->row_scoped);
	deserializer.ReadList(107, "navigations", [&](Deserializer::List &list, idx_t i) {
		list.ReadObject([&](Deserializer &child) {
			MatchRecognizeFunctionData::Navigation navigation;
			navigation.last = child.ReadProperty<bool>(100, "last");
			navigation.symbol = child.ReadProperty<string>(101, "symbol");
			navigation.field = child.ReadProperty<idx_t>(102, "field");
			navigation.offset = child.ReadProperty<idx_t>(103, "offset");
			result->navigations.push_back(navigation);
		});
	});
	function.SetReturnType(ResultType());
	return std::move(result);
}

//===--------------------------------------------------------------------===//
// WindowMatchRecognizeExecutor
//===--------------------------------------------------------------------===//
void WindowMatchRecognizeExecutor::GetBounds(WindowBoundsSet &required, const BoundWindowExpression &wexpr) {
	// matching spans a whole partition rather than a frame, so no frame boundaries are needed
}

void WindowMatchRecognizeExecutor::GetSharing(WindowExecutor &executor, WindowSharedExpressions &shared) {
	auto &config = executor.wexpr.BindInfo()->Cast<MatchRecognizeFunctionData>();
	// the conditions are evaluated per chunk as the rows arrive
	for (auto &child : executor.wexpr.GetChildren()) {
		executor.child_idx.emplace_back(shared.RegisterSink(child));
	}
	// navigation resolves to arbitrary rows, and MATCH_NUMBER() forces a re-evaluation per match;
	// both need the group kept around
	auto per_row = !config.navigations.empty();
	for (auto scoped : config.row_scoped) {
		per_row = per_row || scoped;
	}
	if (per_row) {
		for (auto &child : executor.wexpr.GetChildren()) {
			executor.aux_idx.emplace_back(shared.RegisterCollection(child, false));
		}
	}
}

unique_ptr<GlobalSinkState> WindowMatchRecognizeExecutor::GetGlobal(ClientContext &client,
                                                                    const WindowExecutor &executor,
                                                                    const idx_t payload_count,
                                                                    const ValidityMask &partition_mask,
                                                                    const ValidityMask &order_mask) {
	return make_uniq<WindowMatchRecognizeGlobalState>(client, executor, payload_count, partition_mask, order_mask);
}

//! Holds the per thread machinery Sink needs to evaluate the conditions
class MatchRecognizeLocalState : public WindowExecutorLocalState {
public:
	MatchRecognizeLocalState(ExecutionContext &context, const WindowMatchRecognizeGlobalState &gstate)
	    : WindowExecutorLocalState(context, gstate) {
		auto &config = gstate.executor.wexpr.BindInfo()->Cast<MatchRecognizeFunctionData>();
		for (auto &condition : config.conditions) {
			auto copied = condition->Copy();
			types.push_back(copied->GetReturnType());
			conditions.push_back(std::move(copied));
		}
		if (!conditions.empty()) {
			executor = make_uniq<ExpressionExecutor>(context.client, conditions);
			result.Initialize(context.client, types);
		}
	}

	vector<unique_ptr<Expression>> conditions;
	vector<LogicalType> types;
	unique_ptr<ExpressionExecutor> executor;
	DataChunk result;
};

unique_ptr<LocalSinkState> WindowMatchRecognizeExecutor::GetLocal(ExecutionContext &context,
                                                                  const GlobalSinkState &gstate) {
	return make_uniq<MatchRecognizeLocalState>(context, gstate.Cast<WindowMatchRecognizeGlobalState>());
}

void WindowMatchRecognizeExecutor::Sink(ExecutionContext &context, DataChunk &sink_chunk, DataChunk &coll_chunk,
                                        idx_t input_idx, OperatorSinkInput &sink) {
	auto &gstate = sink.global_state.Cast<WindowMatchRecognizeGlobalState>();
	auto &lstate = sink.local_state.Cast<MatchRecognizeLocalState>();
	if (!lstate.executor) {
		return;
	}

	const auto count = sink_chunk.size();
	auto &columns = StructVector::GetEntries(sink_chunk.data[gstate.executor.child_idx[0]]);
	vector<LogicalType> column_types;
	for (auto &column : columns) {
		column_types.push_back(column.GetType());
	}
	DataChunk slice;
	slice.InitializeEmpty(column_types);
	for (idx_t col = 0; col < columns.size(); col++) {
		slice.data[col].Reference(columns[col]);
	}
	slice.SetCardinality(count);

	lstate.result.Reset();
	lstate.executor->Execute(slice, lstate.result);
	for (idx_t i = 0; i < lstate.conditions.size(); i++) {
		UnifiedVectorFormat condition_data;
		lstate.result.data[i].ToUnifiedFormat(count, condition_data);
		auto values = UnifiedVectorFormat::GetData<bool>(condition_data);
		for (idx_t row = 0; row < count; row++) {
			const auto row_idx = condition_data.sel->get_index(row);
			gstate.condition_values[i][input_idx + row] =
			    condition_data.validity.RowIsValid(row_idx) && values[row_idx] ? 1 : 0;
		}
	}
}

//! Materialise the whole hash group. Row offsets are global to it, which is what the matcher and the
//! condition evaluation both index by.
static void FetchHashGroup(ColumnDataCollection &input, DataChunk &result_chunk) {
	ColumnDataScanState scan_state;
	DataChunk scan_chunk;
	input.InitializeScanChunk(scan_chunk);
	input.InitializeScan(scan_state);
	while (input.Scan(scan_state, scan_chunk)) {
		result_chunk.Append(scan_chunk);
	}
}

struct Match {
	Match(bool success_p, optional_idx end_idx_p = optional_idx::Invalid(), bool optional_p = false)
	    : success(success_p), end_idx(end_idx_p), optional(optional_p) {
	}

	bool success;
	optional_idx end_idx;
	bool optional;
};

// simplistic backtracking-based pattern executor
// FIXME pretty naive this, and an allocation-fest.
// Successful symbol matches record the matching variable at their row offset. Failed branches may leave
// stale entries behind, but every offset a successful match covers is written by that match, so reading
// the range of a successful match is well defined.
using SymbolMatcher = std::function<bool(idx_t symbol, idx_t row)>;

static vector<Match> MatchPattern(const Expression &pattern, idx_t input_size, const idx_t offset,
                                  const SymbolMatcher &symbol_matches, vector<idx_t> &classifiers) {
	if (offset >= input_size) {
		return {Match(false)};
	}
	switch (pattern.GetExpressionType()) {
	case ExpressionType::CONCATENATION: {
		auto &concatenation_expr = pattern.Cast<BoundConcatenationExpression>();

		auto child_start_idx = offset;
		idx_t token_idx = 0;
		vector<Match> matches;

		while (token_idx < concatenation_expr.children.size()) {
			auto &child_pattern = *concatenation_expr.children[token_idx];
			auto res = MatchPattern(child_pattern, input_size, child_start_idx, symbol_matches, classifiers);
			if (res.back().success) {
				matches.insert(matches.end(), res.begin(), res.end());
				child_start_idx = res.back().end_idx.GetIndex();
				token_idx++;
			} else {
				auto token_is_optional = false;
				if (child_pattern.GetExpressionType() == ExpressionType::QUANTIFIER) {
					auto &quantifier_expr = child_pattern.Cast<BoundQuantifierExpression>();
					if (!quantifier_expr.min_count.IsValid() || quantifier_expr.min_count.GetIndex() == 0) {
						token_is_optional = true;
					}
				}
				if (token_is_optional) {
					token_idx++;
					continue;
				}
				if (!matches.empty() && matches.back().optional) {
					child_start_idx = matches.back().end_idx.GetIndex();
					matches.pop_back();
					continue;
				}
				break;
			}
		}
		return {Match(token_idx == concatenation_expr.children.size(), child_start_idx)};
	}
	case ExpressionType::ALTERNATION: {
		// ordered choice: the left alternative wins if it matches
		auto &alternation_expr = pattern.Cast<BoundAlternationExpression>();
		auto left = MatchPattern(*alternation_expr.child_left, input_size, offset, symbol_matches, classifiers);
		if (left.back().success) {
			return left;
		}
		return MatchPattern(*alternation_expr.child_right, input_size, offset, symbol_matches, classifiers);
	}
	case ExpressionType::QUANTIFIER: {
		auto &quantifier_expr = pattern.Cast<BoundQuantifierExpression>();
		idx_t match_count = 0;
		auto child_start_idx = offset;
		auto max_offset = quantifier_expr.max_count.IsValid()
		                      ? MinValue(input_size, quantifier_expr.max_count.GetIndex())
		                      : input_size;
		vector<Match> matches;
		while (match_count < max_offset) {
			auto res = MatchPattern(*quantifier_expr.child, input_size, child_start_idx, symbol_matches, classifiers);
			if (!res.back().success) {
				break;
			}
			child_start_idx = res.back().end_idx.GetIndex();
			match_count++;
			// a repetition may only be given up if doing so still satisfies the minimum
			auto is_optional =
			    quantifier_expr.min_count.IsValid() ? match_count > quantifier_expr.min_count.GetIndex() : true;
			matches.emplace_back(Match(true, child_start_idx, is_optional));
		}
		if (!matches.empty() &&
		    (!quantifier_expr.min_count.IsValid() || match_count >= quantifier_expr.min_count.GetIndex())) {
			return matches;
		}
		return {Match(false)};
	}
	case ExpressionType::VALUE_CONSTANT: {
		// TODO cache those pointers in the map instead of the vector
		const auto symbol = NumericCast<idx_t>(pattern.Cast<BoundConstantExpression>().GetValue().GetValue<uint64_t>());
		// the row is tentatively this symbol while its condition is evaluated, which is what lets
		// LAST(X.c) see the row being tested
		auto previous = classifiers[offset];
		classifiers[offset] = symbol;
		if (!symbol_matches(symbol, offset)) {
			classifiers[offset] = previous;
			return {Match(false)};
		}
		return {Match(true, offset + 1)};
	}
	default:
		throw InternalException("Unsupported pattern type");
	}
}

//! Where to resume scanning after a match spanning [match_start, match_end]
static idx_t SkipTo(const MatchRecognizeFunctionData &config, idx_t skip_symbol, idx_t match_start, idx_t match_end,
                    const vector<idx_t> &classifiers) {
	auto resume = match_end + 1;
	switch (config.after_match) {
	case MatchRecognizeAfterMatch::MATCH_RECOGNIZE_AFTER_MATCH_NEXT_ROW:
		resume = match_start + 1;
		break;
	case MatchRecognizeAfterMatch::MATCH_RECOGNIZE_AFTER_MATCH_FIRST_VAR:
	case MatchRecognizeAfterMatch::MATCH_RECOGNIZE_AFTER_MATCH_LAST_VAR: {
		const auto first = config.after_match == MatchRecognizeAfterMatch::MATCH_RECOGNIZE_AFTER_MATCH_FIRST_VAR;
		optional_idx target;
		for (idx_t row = match_start; row <= match_end; row++) {
			if (classifiers[row] != skip_symbol) {
				continue;
			}
			target = row;
			if (first) {
				break;
			}
		}
		if (target.IsValid()) {
			resume = target.GetIndex();
		}
		break;
	}
	case MatchRecognizeAfterMatch::MATCH_RECOGNIZE_AFTER_MATCH_LAST_ROW:
	case MatchRecognizeAfterMatch::MATCH_RECOGNIZE_AFTER_MATCH_DEFAULT:
		break;
	}
	// never resume at or before the row the match started on, that would not terminate
	return MaxValue(resume, match_start + 1);
}

// this gets called per partition
void WindowMatchRecognizeExecutor::Finalize(ExecutionContext &context, optional_ptr<WindowCollection> collection,
                                            OperatorSinkInput &sink) {
	auto &gstate = sink.global_state.Cast<WindowMatchRecognizeGlobalState>();
	auto &config = gstate.executor.wexpr.BindInfo()->Cast<MatchRecognizeFunctionData>();

	// we always start with a new partition
	D_ASSERT(gstate.partition_mask.RowIsValid(0));

	// One time setup: work out where the partitions are, and materialise the rows if a condition has
	// to be settled per row. Both are shared by every thread that gets here.
	{
		lock_guard<mutex> lock(gstate.state_lock);
		if (!gstate.prepared) {
			gstate.prepared = true;
			idx_t partition_start = 0;
			for (idx_t payload_idx = 1; payload_idx <= gstate.payload_count; payload_idx++) {
				const auto at_end = payload_idx == gstate.payload_count;
				if (!at_end && !gstate.partition_mask.RowIsValid(payload_idx)) {
					continue;
				}
				gstate.partitions.emplace_back(partition_start, payload_idx - 1);
				partition_start = payload_idx;
			}
			auto per_row = !config.navigations.empty();
			for (auto scoped : config.row_scoped) {
				per_row = per_row || scoped;
			}
			if (per_row) {
				gstate.rows.Initialize(context.client, collection->inputs->Types(), gstate.payload_count);
				FetchHashGroup(*collection->inputs, gstate.rows);
			}
		}
	}

	auto &partition_chunk = gstate.rows;
	auto &classifiers = gstate.classifiers;

	// The conditions reference the window arguments; expression sharing decided where those landed in
	// the collection, so rebind them against it before evaluating.
	const auto columns_idx = gstate.executor.aux_idx.empty() ? 0 : gstate.executor.aux_idx[0];
	vector<LogicalType> condition_types;
	vector<unique_ptr<Expression>> conditions;
	for (auto &condition : config.conditions) {
		auto copied = condition->Copy();
		condition_types.push_back(copied->GetReturnType());
		conditions.push_back(std::move(copied));
	}
	// the conditions address the fields of the packed column struct, and MATCH_NUMBER() is field 0
	//	MATCH_NUMBER() is the first field of the packed column struct
	static constexpr idx_t MATCH_NUMBER_FIELD = 0;

	auto &condition_values = gstate.condition_values;
	unordered_map<string, idx_t> symbol_index;
	for (idx_t i = 0; i < config.symbols.size(); i++) {
		symbol_index[config.symbols[i]] = i;
	}
	// the skip target and the navigations name symbols too, resolved here so the matcher compares
	// indices rather than strings
	auto lookup_symbol = [&](const string &name) -> idx_t {
		auto entry = symbol_index.find(name);
		return entry == symbol_index.end() ? DConstants::INVALID_INDEX : entry->second;
	};
	const auto skip_symbol = lookup_symbol(config.after_match_variable);
	vector<idx_t> navigation_symbols;
	for (auto &navigation : config.navigations) {
		navigation_symbols.push_back(lookup_symbol(navigation.symbol));
	}

	// state of the attempt in progress, needed to resolve FIRST()/LAST()
	idx_t current_match_start = 0;
	idx_t current_match_number = 1;
	DataChunk row_chunk;
	DataChunk row_result;
	bool row_chunk_ready = false;
	// one executor per condition, reused across rows
	vector<unique_ptr<ExpressionExecutor>> row_executors(config.conditions.size());

	//	The row FIRST()/LAST() navigates to, or an invalid index when the match has no such row
	auto navigate = [&](const MatchRecognizeFunctionData::Navigation &navigation, idx_t row,
	                    const vector<idx_t> &assignment) -> optional_idx {
		if (navigation.symbol.empty()) {
			// the match as a whole, counted from whichever end
			if (navigation.last) {
				return row < current_match_start + navigation.offset ? optional_idx()
				                                                     : optional_idx(row - navigation.offset);
			}
			const auto target = current_match_start + navigation.offset;
			return target > row ? optional_idx() : optional_idx(target);
		}
		idx_t seen = 0;
		if (navigation.last) {
			for (idx_t candidate = row + 1; candidate > current_match_start; candidate--) {
				if (assignment[candidate - 1] == navigation_symbols[&navigation - config.navigations.data()] &&
				    seen++ == navigation.offset) {
					return candidate - 1;
				}
			}
		} else {
			for (idx_t candidate = current_match_start; candidate <= row; candidate++) {
				if (assignment[candidate] == navigation_symbols[&navigation - config.navigations.data()] &&
				    seen++ == navigation.offset) {
					return candidate;
				}
			}
		}
		return optional_idx();
	};

	SymbolMatcher symbol_matches = [&](idx_t index, idx_t row) -> bool {
		D_ASSERT(index < config.symbols.size());
		if (index >= config.row_scoped.size() || !config.row_scoped[index]) {
			return condition_values[index][row] != 0;
		}

		auto &columns = StructVector::GetEntries(partition_chunk.data[columns_idx]);
		if (!row_chunk_ready) {
			vector<LogicalType> column_types;
			for (auto &column : columns) {
				column_types.push_back(column.GetType());
			}
			row_chunk.Initialize(context.client, column_types, 1);
			// one expression is evaluated at a time here, so the result holds a single column
			row_result.Initialize(context.client, vector<LogicalType> {LogicalType::BOOLEAN}, 1);
			row_chunk_ready = true;
		}

		row_chunk.Reset();
		for (idx_t col = 0; col < columns.size(); col++) {
			row_chunk.data[col].Slice(columns[col], row, row + 1);
		}
		row_chunk.data[MATCH_NUMBER_FIELD].Reference(Value::UBIGINT(current_match_number), count_t(1));
		for (auto &navigation : config.navigations) {
			auto target = navigate(navigation, row, classifiers);
			if (target.IsValid()) {
				row_chunk.data[navigation.field].Slice(columns[navigation.field], target.GetIndex(),
				                                       target.GetIndex() + 1);
			} else {
				row_chunk.data[navigation.field].Reference(Value(columns[navigation.field].GetType()), count_t(1));
			}
		}
		row_chunk.SetCardinality(1);

		row_result.Reset();
		if (!row_executors[index]) {
			row_executors[index] = make_uniq<ExpressionExecutor>(context.client, *conditions[index]);
		}
		row_executors[index]->Execute(row_chunk, row_result);
		UnifiedVectorFormat result_data;
		row_result.data[0].ToUnifiedFormat(1, result_data);
		const auto result_idx = result_data.sel->get_index(0);
		return result_data.validity.RowIsValid(result_idx) &&
		       UnifiedVectorFormat::GetData<bool>(result_data)[result_idx];
	};

	// Partitions are independent, so every thread that reaches Finalize takes them from a shared
	// cursor rather than one thread doing the whole hash group.
	while (true) {
		const auto partition_idx = gstate.next_partition++;
		if (partition_idx >= gstate.partitions.size()) {
			break;
		}
		const auto partition_start = gstate.partitions[partition_idx].first;
		const auto partition_end = gstate.partitions[partition_idx].second;

		// scan the partition left to right, applying AFTER MATCH SKIP after every match. Rows that are
		// not part of any match keep a NULL struct, which filters them out downstream.
		idx_t match_number = 0;
		current_match_number = 1;
		auto row = partition_start;
		while (row <= partition_end) {
			current_match_start = row;
			auto match = MatchPattern(*config.pattern, partition_end + 1, row, symbol_matches, classifiers).back();
			// a zero length match would never advance the scan
			if (!match.success || !match.end_idx.IsValid() || match.end_idx.GetIndex() <= row) {
				row++;
				continue;
			}
			// a match can never reach beyond its own partition
			const auto match_end = MinValue(match.end_idx.GetIndex() - 1, partition_end);
			match_number++;

			for (idx_t match_row = row; match_row <= match_end; match_row++) {
				gstate.spans[match_row].push_back(WindowMatchRecognizeGlobalState::Span {
				    classifiers[match_row], match_number, match_row == row, row, match_end});
			}
			row = SkipTo(config, skip_symbol, row, match_end, classifiers);
			current_match_number = match_number + 1;
		}

		// the thread that finishes the last partition publishes the result
		if (++gstate.completed_partitions == gstate.partitions.size()) {
			gstate.MaterializeSpans(config.symbols);
		}
	}
}

// this should actually be it yay!
void WindowMatchRecognizeExecutor::GetData(ExecutionContext &context, DataChunk &eval_chunk, DataChunk &bounds,
                                           Vector &result, idx_t row_idx, OperatorSinkInput &sink) {
	auto &gstate = sink.global_state.Cast<WindowMatchRecognizeGlobalState>();
	lock_guard<mutex> lock(gstate.state_lock);
	result.Slice(gstate.result_vec, row_idx, row_idx + bounds.size());
}

WindowFunction MatchRecognizeFun::GetFunction() {
	// Everything after the columns is configuration that Bind() moves into the function data, so a
	// bound call carries only the first argument. Declaring the rest optional keeps the signature
	// resolvable both before and after that.
	WindowFunction fun(Name, {LogicalType::ANY}, WindowMatchRecognizeExecutor::ResultType(),
	                   ExpressionType::WINDOW_FUNCTION, WindowMatchRecognizeExecutor::Bind,
	                   WindowMatchRecognizeExecutor::GetBounds, WindowMatchRecognizeExecutor::GetSharing,
	                   WindowMatchRecognizeExecutor::GetGlobal, WindowMatchRecognizeExecutor::GetLocal,
	                   WindowMatchRecognizeExecutor::Sink, WindowMatchRecognizeExecutor::Finalize,
	                   WindowMatchRecognizeExecutor::GetData);

	auto &signature = fun.GetSignature();
	signature = FunctionSignature(vector<FunctionParameter>(), WindowMatchRecognizeExecutor::ResultType());
	signature.AddParameter(Identifier("columns"), LogicalType::ANY);
	signature.AddParameter(Identifier("conditions"), LogicalType::ANY, Value());
	signature.AddParameter(Identifier("pattern"), LogicalType::ANY, Value());
	signature.AddParameter(Identifier("symbols"), LogicalType::LIST(LogicalType::VARCHAR), Value());
	signature.AddParameter(Identifier("after_match_variable"), LogicalType::VARCHAR, Value());
	signature.AddParameter(Identifier("after_match"), LogicalType::UTINYINT, Value());
	signature.AddParameter(Identifier("navigations"), LogicalType::ANY, Value());

	fun.SetSerializeCallback(WindowMatchRecognizeExecutor::Serialize);
	fun.SetDeserializeCallback(WindowMatchRecognizeExecutor::Deserialize);

	return fun;
}

} // namespace duckdb
