#include "duckdb/function/window/window_nonoverlap_intervals.hpp"

#include "duckdb/function/window/match_recognize_functions.hpp"
#include "duckdb/function/window/window_shared_expressions.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"

namespace duckdb {

enum class IntervalCompareMode : uint8_t {
	INCLUSIVE, // start >= prev_end
	EXCLUSIVE  // start > prev_end
};

class WindowNonOverlapIntervalsGlobalState : public WindowExecutorGlobalState {
public:
	WindowNonOverlapIntervalsGlobalState(ClientContext &client, const WindowExecutor &executor,
	                                     const idx_t payload_count, const ValidityMask &partition_mask,
	                                     const ValidityMask &order_mask)
	    : WindowExecutorGlobalState(client, executor, payload_count, partition_mask, order_mask),
	      compare_mode(IntervalCompareMode::EXCLUSIVE) {
	}

	IntervalCompareMode compare_mode;
	unordered_map<idx_t, uint64_t> last_end;
};

//===--------------------------------------------------------------------===//
// WindowNonOverlapIntervalsExecutor
//===--------------------------------------------------------------------===//
void WindowNonOverlapIntervalsExecutor::GetBounds(WindowBoundsSet &required, const BoundWindowExpression &wexpr) {
	required.insert(PARTITION_BEGIN);
	required.insert(FRAME_BEGIN);
	required.insert(FRAME_END);
}

void WindowNonOverlapIntervalsExecutor::GetSharing(WindowExecutor &executor, WindowSharedExpressions &shared) {
	auto &child_idx = executor.child_idx;
	for (auto &child : executor.wexpr.GetChildren()) {
		child_idx.emplace_back(shared.RegisterEvaluate(child));
	}
}

unique_ptr<GlobalSinkState> WindowNonOverlapIntervalsExecutor::GetGlobal(ClientContext &client,
                                                                         const WindowExecutor &executor,
                                                                         const idx_t payload_count,
                                                                         const ValidityMask &partition_mask,
                                                                         const ValidityMask &order_mask) {
	return make_uniq<WindowNonOverlapIntervalsGlobalState>(client, executor, payload_count, partition_mask, order_mask);
}

unique_ptr<LocalSinkState> WindowNonOverlapIntervalsExecutor::GetLocal(ExecutionContext &context,
                                                                       const GlobalSinkState &gstate) {
	return make_uniq<WindowExecutorLocalState>(context, gstate.Cast<WindowNonOverlapIntervalsGlobalState>());
}

void WindowNonOverlapIntervalsExecutor::GetData(ExecutionContext &context, DataChunk &eval_chunk, DataChunk &bounds,
                                                Vector &result, idx_t row_idx, OperatorSinkInput &sink) {
	auto &gstate = sink.global_state.Cast<WindowNonOverlapIntervalsGlobalState>();
	auto &executor = gstate.executor;
	const auto count = bounds.size();
	auto rdata = FlatVector::GetDataMutable<bool>(result);

	// Bounds for current partition
	auto partition_begin = FlatVector::GetData<const idx_t>(bounds.data[PARTITION_BEGIN]);

	// Data from our arguments: intervals(low, high, inclusive)
	auto low = FlatVector::GetData<uint64_t>(eval_chunk.data[executor.child_idx[0]]);
	auto high = FlatVector::GetData<uint64_t>(eval_chunk.data[executor.child_idx[1]]);

	auto incl = false;
	if (executor.child_idx.size() > 2) {
		UnifiedVectorFormat incl_data;
		eval_chunk.data[executor.child_idx[2]].ToUnifiedFormat(count, incl_data);
		if (count > 0) {
			const auto incl_idx = incl_data.sel->get_index(0);
			incl = incl_data.validity.RowIsValid(incl_idx) && UnifiedVectorFormat::GetData<bool>(incl_data)[incl_idx];
		}
	}
	if (incl || gstate.compare_mode == IntervalCompareMode::INCLUSIVE) {
		gstate.compare_mode = IntervalCompareMode::INCLUSIVE;
	} else {
		gstate.compare_mode = IntervalCompareMode::EXCLUSIVE;
	}

	// i = index within chunk, row_idx = global index/offset
	for (idx_t i = 0; i < count; ++i) {
		auto pbegin = partition_begin[i];

		// pick first interval in partition
		bool keep = false;
		if (row_idx + i == pbegin) {
			keep = true;
			gstate.last_end[pbegin] = high[i];
		} else {
			// pick next interval that does not overlap
			auto prev_end = gstate.last_end[pbegin];
			switch (gstate.compare_mode) {
			case IntervalCompareMode::INCLUSIVE:
				if (low[i] >= prev_end) {
					keep = true;
					gstate.last_end[pbegin] = high[i];
				}
				break;
			case IntervalCompareMode::EXCLUSIVE:
				if (low[i] > prev_end) {
					keep = true;
					gstate.last_end[pbegin] = high[i];
				}
				break;
			}
		}
		// write result into result vector
		rdata[i] = keep;
	}
}

WindowFunction NonOverlapIntervalsFun::GetFunction() {
	WindowFunction fun(Name, {LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::BOOLEAN}, LogicalType::BOOLEAN,
	                   ExpressionType::WINDOW_FUNCTION, nullptr, WindowNonOverlapIntervalsExecutor::GetBounds,
	                   WindowNonOverlapIntervalsExecutor::GetSharing, WindowNonOverlapIntervalsExecutor::GetGlobal,
	                   WindowNonOverlapIntervalsExecutor::GetLocal, nullptr, nullptr,
	                   WindowNonOverlapIntervalsExecutor::GetData);
	return fun;
}

} // namespace duckdb
