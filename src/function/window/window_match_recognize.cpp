#include "duckdb/function/window/window_match_recognize.hpp"

#include "duckdb/function/match_recognize.hpp"
#include "duckdb/function/window/match_recognize_functions.hpp"
#include "duckdb/function/window/window_shared_expressions.hpp"
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

LogicalType WindowMatchRecognizeExecutor::ResultType() {
	return LogicalType::STRUCT({{"classifiers", LogicalType::LIST(LogicalType::VARCHAR)},
	                            {"complete", LogicalType::BOOLEAN},
	                            {"match_start", LogicalType::UBIGINT},
	                            {"match_end", LogicalType::UBIGINT},
	                            {"skip_to", LogicalType::UBIGINT}});
}

//===--------------------------------------------------------------------===//
// Binding
//===--------------------------------------------------------------------===//
unique_ptr<FunctionData> WindowMatchRecognizeExecutor::Bind(BindWindowFunctionInput &input) {
	auto &arguments = input.GetArguments();
	if (arguments.size() != 2) {
		throw BinderException("MATCH_RECOGNIZE takes a DEFINE struct and a pattern");
	}

	// the pattern is passed as the trailing argument - move it into the bind data so that it is not
	// evaluated as a regular argument. The binder shrinks the argument list to match.
	auto bind_data = make_uniq<MatchRecognizeFunctionData>();
	bind_data->pattern = std::move(arguments.back());

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
static vector<Match> MatchPattern(const Expression &pattern, const DataChunk &input, const idx_t offset,
                                  unordered_map<string, uint8_t *> &define_child_mapping) {
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
			auto res = MatchPattern(child_pattern, input, child_start_idx, define_child_mapping);
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
		auto left = MatchPattern(*alternation_expr.child_left, input, offset, define_child_mapping);
		if (left.back().success) {
			return left;
		}
		return MatchPattern(*alternation_expr.child_right, input, offset, define_child_mapping);
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
			auto res = MatchPattern(*quantifier_expr.child, input, child_start_idx, define_child_mapping);
			if (!res.back().success) {
				break;
			}
			child_start_idx = res.back().end_idx.GetIndex();
			match_count++;
			auto is_optional =
			    quantifier_expr.min_count.IsValid() ? match_count >= quantifier_expr.min_count.GetIndex() : true;
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
		return {Match(define_child_mapping[symbol][offset], offset + 1)};
	}
	default:
		throw InternalException("Unsupported pattern type");
	}
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

	unordered_map<string, uint8_t *> define_child_mapping;
	auto defines_struct_child = partition_chunk.GetTypes()[0];
	for (idx_t struct_child_idx = 0; struct_child_idx < StructType::GetChildCount(defines_struct_child);
	     struct_child_idx++) {
		auto &child_vector = StructVector::GetEntries(partition_chunk.data[0])[struct_child_idx];
		D_ASSERT(child_vector.GetVectorType() == VectorType::FLAT_VECTOR);
		define_child_mapping[StructType::GetChildName(defines_struct_child, struct_child_idx).GetIdentifierName()] =
		    FlatVector::GetDataMutable<uint8_t>(child_vector);
	}

	// iterate over entire input, but there can be many partitions in the input
	for (idx_t payload_idx = 1; payload_idx < gstate.payload_count; payload_idx++) {
		if (!gstate.partition_mask.RowIsValid(payload_idx) && payload_idx + 1 < gstate.payload_count) {
			continue;
		}
		// the partition end offset depends on whether we found a next partition or if we are at the end
		auto partition_end =
		    payload_idx + 1 == gstate.partition_mask.RowIsValid(payload_idx) ? payload_idx - 1 : payload_idx;

		// FIXME
		FetchPartition(context.client, *collection->inputs, partition_chunk, partition_start, partition_end);

		for (idx_t partition_idx = partition_start; partition_idx < partition_end; partition_idx++) {
			auto match = MatchPattern(*config.pattern, partition_chunk, partition_idx, define_child_mapping).back();

			FlatVector::ValidityMutable(gstate.result_vec).SetValid(partition_idx);
			auto &struct_entries = StructVector::GetEntries(gstate.result_vec);
			// a failed match has no end index - the row is filtered out on the "complete" flag
			const idx_t match_end =
			    match.end_idx.IsValid() && match.end_idx.GetIndex() > 0 ? match.end_idx.GetIndex() - 1 : 0;
			// first entry is list of classifiers, TODO
			struct_entries[1].SetValue(partition_idx, Value::BOOLEAN(match.success));
			struct_entries[2].SetValue(partition_idx, Value::UBIGINT(partition_idx));
			struct_entries[3].SetValue(partition_idx, Value::UBIGINT(match_end));
			struct_entries[4].SetValue(partition_idx, Value::UBIGINT(match_end));
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
	WindowFunction fun(Name, {LogicalType::ANY, LogicalType::ANY}, WindowMatchRecognizeExecutor::ResultType(),
	                   ExpressionType::WINDOW_FUNCTION, WindowMatchRecognizeExecutor::Bind,
	                   WindowMatchRecognizeExecutor::GetBounds, WindowMatchRecognizeExecutor::GetSharing,
	                   WindowMatchRecognizeExecutor::GetGlobal, WindowMatchRecognizeExecutor::GetLocal, nullptr,
	                   WindowMatchRecognizeExecutor::Finalize, WindowMatchRecognizeExecutor::GetData);
	return fun;
}

} // namespace duckdb
