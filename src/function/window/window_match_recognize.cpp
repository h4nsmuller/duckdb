#include "duckdb/function/window/window_match_recognize.hpp"

#include "duckdb/function/match_recognize.hpp"
#include "duckdb/function/window/match_recognize_functions.hpp"
#include "duckdb/function/window/window_shared_expressions.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"

namespace duckdb {

struct WindowMatchRecognizeGlobalState : WindowExecutorGlobalState {
	WindowMatchRecognizeGlobalState(ClientContext &client, const WindowExecutor &executor, const idx_t payload_count,
	                                const ValidityMask &partition_mask, const ValidityMask &order_mask)
	    : WindowExecutorGlobalState(client, executor, payload_count, partition_mask, order_mask),
	      result_vec(executor.wexpr.GetReturnType(), payload_count) {
		FlatVector::ValidityMutable(result_vec).SetAllInvalid(payload_count);
		D_ASSERT(result_vec.GetType().id() == LogicalTypeId::STRUCT);
		for (auto &struct_entry : StructVector::GetEntries(result_vec)) {
			FlatVector::ValidityMutable(struct_entry).SetAllInvalid(payload_count);
		}
	}

	// TODO can we get away with putting this into the local state?
	mutex state_lock;

	Vector result_vec;
};

//	Column indexes into the result struct
enum MatchRecognizeResult : idx_t { CLASSIFIER = 0, MATCH_NUMBER, IN_MATCH, IS_MATCH_START, MATCH_START, MATCH_END };

LogicalType WindowMatchRecognizeExecutor::ResultType() {
	return LogicalType::STRUCT({{"classifier", LogicalType::VARCHAR},
	                            {"match_number", LogicalType::UBIGINT},
	                            {"in_match", LogicalType::BOOLEAN},
	                            {"is_match_start", LogicalType::BOOLEAN},
	                            {"match_start", LogicalType::UBIGINT},
	                            {"match_end", LogicalType::UBIGINT}});
}

//===--------------------------------------------------------------------===//
// Binding
//===--------------------------------------------------------------------===//
unique_ptr<FunctionData> WindowMatchRecognizeExecutor::Bind(BindWindowFunctionInput &input) {
	auto &arguments = input.GetArguments();
	if (arguments.size() != 4) {
		throw BinderException("MATCH_RECOGNIZE takes a DEFINE struct, an AFTER MATCH SKIP option and a pattern");
	}

	// only the DEFINE struct is evaluated per row - the rest is configuration that moves into the bind
	// data, and the binder shrinks the argument list to match.
	auto bind_data = make_uniq<MatchRecognizeFunctionData>();
	bind_data->after_match = static_cast<MatchRecognizeAfterMatch>(
	    arguments[1]->Cast<BoundConstantExpression>().GetValue().GetValue<uint8_t>());
	auto &skip_variable = arguments[2]->Cast<BoundConstantExpression>().GetValue();
	if (!skip_variable.IsNull()) {
		bind_data->after_match_variable = skip_variable.GetValue<string>();
	}
	bind_data->pattern = std::move(arguments[3]);

	auto &bound_function = input.GetBoundFunction();
	bound_function.GetArguments().resize(1);
	bound_function.SetReturnType(ResultType());

	return std::move(bind_data);
}

//===--------------------------------------------------------------------===//
// WindowMatchRecognizeExecutor
//===--------------------------------------------------------------------===//
void WindowMatchRecognizeExecutor::GetBounds(WindowBoundsSet &required, const BoundWindowExpression &wexpr) {
	required.insert(FRAME_BEGIN);
	required.insert(FRAME_END);
}

void WindowMatchRecognizeExecutor::GetSharing(WindowExecutor &executor, WindowSharedExpressions &shared) {
	// RegisterCollection deduplicates child expressions - the defines are read back out of the collection
	auto &child_idx = executor.child_idx;
	for (auto &child : executor.wexpr.GetChildren()) {
		child_idx.emplace_back(shared.RegisterCollection(child, false));
	}
}

unique_ptr<GlobalSinkState> WindowMatchRecognizeExecutor::GetGlobal(ClientContext &client,
                                                                    const WindowExecutor &executor,
                                                                    const idx_t payload_count,
                                                                    const ValidityMask &partition_mask,
                                                                    const ValidityMask &order_mask) {
	return make_uniq<WindowMatchRecognizeGlobalState>(client, executor, payload_count, partition_mask, order_mask);
}

unique_ptr<LocalSinkState> WindowMatchRecognizeExecutor::GetLocal(ExecutionContext &context,
                                                                  const GlobalSinkState &gstate) {
	return make_uniq<WindowExecutorLocalState>(context, gstate.Cast<WindowMatchRecognizeGlobalState>());
}

static void FetchPartition(ClientContext &context, ColumnDataCollection &input, DataChunk &result_chunk,
                           idx_t partition_start, idx_t partition_end) {
	ColumnDataScanState scan_state;
	DataChunk scan_chunk;

	// TODO cache those allocations
	input.InitializeScanChunk(scan_chunk);

	auto partition_size = partition_end - partition_start + 1;

	D_ASSERT(result_chunk.GetCapacity() <= partition_size);

	input.InitializeScan(scan_state);
	// we do one Scan() because Seek() does nothing if its already on the right chunk
	input.Scan(scan_state, scan_chunk);
	input.Seek(partition_start, scan_state, scan_chunk);

	// we may have to slice the first chunk because the partition may start somewhere halfway into the chunk
	auto chunk_offset = partition_start - scan_state.current_row_index;
	scan_chunk.Slice(chunk_offset, scan_chunk.size() - chunk_offset);

	result_chunk.Append(scan_chunk);

	while (partition_end <= scan_state.next_row_index) {
		if (!input.Scan(scan_state, scan_chunk)) {
			// we need to get out here because otherwise the check below goes wrong
			break;
		}
		// we may have too many rows, so slice again
		if (scan_state.next_row_index > partition_end) {
			scan_chunk.Slice(0, scan_state.next_row_index - partition_end); // TODO verify this very complex math
		}
		result_chunk.Append(scan_chunk);
	}
	D_ASSERT(result_chunk.size() == partition_size);
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
static vector<Match> MatchPattern(const Expression &pattern, const DataChunk &input, const idx_t offset,
                                  unordered_map<string, uint8_t *> &define_child_mapping, vector<string> &classifiers) {
	if (offset >= input.size()) {
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
			auto res = MatchPattern(child_pattern, input, child_start_idx, define_child_mapping, classifiers);
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
		auto left = MatchPattern(*alternation_expr.child_left, input, offset, define_child_mapping, classifiers);
		if (left.back().success) {
			return left;
		}
		return MatchPattern(*alternation_expr.child_right, input, offset, define_child_mapping, classifiers);
	}
	case ExpressionType::QUANTIFIER: {
		auto &quantifier_expr = pattern.Cast<BoundQuantifierExpression>();
		idx_t match_count = 0;
		auto child_start_idx = offset;
		auto max_offset = quantifier_expr.max_count.IsValid()
		                      ? MinValue(input.size(), quantifier_expr.max_count.GetIndex())
		                      : input.size();
		vector<Match> matches;
		while (match_count < max_offset) {
			auto res = MatchPattern(*quantifier_expr.child, input, child_start_idx, define_child_mapping, classifiers);
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
	case ExpressionType::BOUND_COLUMN_REF: {
		// TODO cache those pointers in the map instead of the vector
		auto symbol = pattern.GetAlias().GetIdentifierName();
		D_ASSERT(define_child_mapping.find(symbol) != define_child_mapping.end());
		if (!define_child_mapping[symbol][offset]) {
			return {Match(false)};
		}
		classifiers[offset] = symbol;
		return {Match(true, offset + 1)};
	}
	default:
		throw InternalException("Unsupported pattern type");
	}
}

//! Where to resume scanning after a match spanning [match_start, match_end]
static idx_t SkipTo(const MatchRecognizeFunctionData &config, idx_t match_start, idx_t match_end,
                    const vector<string> &classifiers) {
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
			if (classifiers[row] != config.after_match_variable) {
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
	lock_guard<mutex> lock(gstate.state_lock);

	auto &config = gstate.executor.wexpr.BindInfo()->Cast<MatchRecognizeFunctionData>();

	idx_t partition_start = 0;
	// we always start with a new partition
	D_ASSERT(gstate.partition_mask.RowIsValid(0));

	// TODO this should probably be kept elsewhere
	DataChunk partition_chunk;
	partition_chunk.Initialize(context.client, collection->inputs->Types(), gstate.payload_count);
	// we keep a map with symbol names to pointers
	// TODO we could also put them into the boundref expression? maybe they're already there??
	// figure out which define has which child offset

	// the pattern variable that classified each row, indexed by row offset within the hash group
	vector<string> classifiers(gstate.payload_count);

	unordered_map<string, uint8_t *> define_child_mapping;
	// the DEFINE struct is wherever expression sharing put it, which is not necessarily column 0
	const auto defines_idx = gstate.executor.child_idx[0];
	auto defines_struct_child = partition_chunk.GetTypes()[defines_idx];
	for (idx_t struct_child_idx = 0; struct_child_idx < StructType::GetChildCount(defines_struct_child);
	     struct_child_idx++) {
		auto &child_vector = StructVector::GetEntries(partition_chunk.data[defines_idx])[struct_child_idx];
		D_ASSERT(child_vector.GetVectorType() == VectorType::FLAT_VECTOR);
		define_child_mapping[StructType::GetChildName(defines_struct_child, struct_child_idx).GetIdentifierName()] =
		    FlatVector::GetDataMutable<uint8_t>(child_vector);
	}

	// iterate over entire input, but there can be many partitions in the input
	for (idx_t payload_idx = 1; payload_idx <= gstate.payload_count; payload_idx++) {
		// a partition is closed either by the start of the next one or by the end of the input
		const auto at_end = payload_idx == gstate.payload_count;
		if (!at_end && !gstate.partition_mask.RowIsValid(payload_idx)) {
			continue;
		}
		const idx_t partition_end = payload_idx - 1;

		// FIXME
		FetchPartition(context.client, *collection->inputs, partition_chunk, partition_start, partition_end);

		// scan the partition left to right, applying AFTER MATCH SKIP after every match. Rows that are
		// not part of any match keep a NULL struct, which filters them out downstream.
		auto &struct_entries = StructVector::GetEntries(gstate.result_vec);
		idx_t match_number = 0;
		auto row = partition_start;
		while (row <= partition_end) {
			auto match = MatchPattern(*config.pattern, partition_chunk, row, define_child_mapping, classifiers).back();
			// a zero length match would never advance the scan
			if (!match.success || !match.end_idx.IsValid() || match.end_idx.GetIndex() <= row) {
				row++;
				continue;
			}
			// a match can never reach beyond its own partition
			const auto match_end = MinValue(match.end_idx.GetIndex() - 1, partition_end);
			match_number++;

			for (idx_t match_row = row; match_row <= match_end; match_row++) {
				FlatVector::ValidityMutable(gstate.result_vec).SetValid(match_row);
				struct_entries[CLASSIFIER].SetValue(match_row, Value(MatchRecognizeSymbolName(classifiers[match_row])));
				struct_entries[MATCH_NUMBER].SetValue(match_row, Value::UBIGINT(match_number));
				struct_entries[IN_MATCH].SetValue(match_row, Value::BOOLEAN(true));
				struct_entries[IS_MATCH_START].SetValue(match_row, Value::BOOLEAN(match_row == row));
				struct_entries[MATCH_START].SetValue(match_row, Value::UBIGINT(row));
				struct_entries[MATCH_END].SetValue(match_row, Value::UBIGINT(match_end));
			}
			row = SkipTo(config, row, match_end, classifiers);
		}
		partition_start = payload_idx;
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
	WindowFunction fun(Name, {LogicalType::ANY, LogicalType::UTINYINT, LogicalType::VARCHAR, LogicalType::ANY},
	                   WindowMatchRecognizeExecutor::ResultType(), ExpressionType::WINDOW_FUNCTION,
	                   WindowMatchRecognizeExecutor::Bind, WindowMatchRecognizeExecutor::GetBounds,
	                   WindowMatchRecognizeExecutor::GetSharing, WindowMatchRecognizeExecutor::GetGlobal,
	                   WindowMatchRecognizeExecutor::GetLocal, nullptr, WindowMatchRecognizeExecutor::Finalize,
	                   WindowMatchRecognizeExecutor::GetData);
	return fun;
}

} // namespace duckdb
