//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/aggregate_rewrite_helper.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/common/optional_idx.hpp"

namespace duckdb {
class BaseStatistics;
class ClientContext;
class LogicalAggregate;
class LogicalGet;
class Optimizer;

struct AggregateRewriteHelper {
	static optional_idx GetDirectReferenceIndex(const Expression &expression, LogicalOperator &input);
	static vector<Identifier> GenerateColumnNames(const string &prefix, idx_t column_count);
	static unique_ptr<Expression> CopyAndRebind(const Expression &expr,
	                                            const column_binding_map_t<ColumnBinding> &replacement_map);
	static void StageVolatileAggregateInputs(Optimizer &optimizer, LogicalAggregate &aggr,
	                                         unique_ptr<LogicalOperator> &child);
	static unique_ptr<LogicalOperator> CreateCTERef(Optimizer &optimizer, TableIndex cte_index,
	                                                const vector<LogicalType> &input_types,
	                                                const vector<Identifier> &input_names,
	                                                const vector<ColumnBinding> &input_bindings,
	                                                column_binding_map_t<ColumnBinding> &replacement_map);
	//! The table statistics of a column produced by a scan (nullptr if the scan has none)
	static unique_ptr<BaseStatistics> GetScanColumnStatistics(ClientContext &context, LogicalGet &get,
	                                                          idx_t column_index);
};

//! Row and group count estimates for the input of an aggregate, for the aggregate rewrites that run before join
//! ordering and statistics propagation. Only the table statistics are available at that point: the cardinality of a
//! scan is the cardinality of its table (filters are ignored), an operator has the cardinality of its largest input,
//! and a column has the distinct count of the base table column it passes on unchanged (through projections, group
//! columns and references to materialized CTEs).
class AggregateInputEstimator {
public:
	//! The input cardinality below which the estimates should not gate a rewrite: either plan is cheap
	static constexpr idx_t MIN_GATED_INPUT_CARDINALITY = 122880;

public:
	AggregateInputEstimator(ClientContext &context, LogicalOperator &plan);

	//! The estimated row count of an operator (invalid if a source below it has no estimate)
	optional_idx EstimateCardinality(LogicalOperator &op);
	//! An upper estimate of the number of groups of the given group expressions over the input: the product of the
	//! distinct counts of the referenced base table columns, where the columns of one scan contribute at most the
	//! cardinality of that scan, capped by the input cardinality. Invalid if a group is volatile or references a
	//! column without distinct count statistics.
	optional_idx EstimateGroupCount(LogicalOperator &input, const vector<reference<Expression>> &groups,
	                                idx_t input_cardinality);

private:
	struct BaseColumn {
		//! The table index of the scan producing the column
		TableIndex table_index;
		//! The distinct count of the column in the table statistics
		idx_t distinct_count;
		//! The cardinality of the scan
		idx_t table_cardinality;
	};

	void CollectCTEDefinitions(LogicalOperator &op);
	bool GetBaseColumn(LogicalOperator &op, ColumnBinding binding, BaseColumn &result);

private:
	ClientContext &context;
	//! The materialized CTEs of the plan, by CTE index
	unordered_map<TableIndex, reference<LogicalOperator>> materialized_ctes;
	//! The estimated cardinality of the materialized CTEs that were estimated already
	unordered_map<TableIndex, optional_idx> cte_cardinalities;
};

} // namespace duckdb
