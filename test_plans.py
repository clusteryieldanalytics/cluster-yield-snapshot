"""Tests for plan parsing — pure Python, no Spark."""

from cluster_yield_snapshot.plans import (
    operators_from_text, operators_from_json, operators_from_entry,
    extract_physical_plan, parse_operator_line, fingerprint_operators,
    tables_from_text, tables_from_json, tables_from_entry,
)
from cluster_yield_snapshot._util import fmt_bytes, parse_int

EXPLAIN_EXTENDED = """\
== Parsed Logical Plan ==
Project [region#10, amount#11]
+- UnresolvedRelation [orders]

== Analyzed Logical Plan ==
region: string
Project [region#10]
+- SubqueryAlias spark_catalog.default.orders

== Optimized Logical Plan ==
Project [region#10]
+- Relation default.orders[region#10] parquet

== Physical Plan ==
AdaptiveSparkPlan isFinalPlan=false
+- HashAggregate(keys=[region#10], functions=[sum(amount#11)])
   +- Exchange hashpartitioning(region#10, 200)
      +- HashAggregate(keys=[region#10], functions=[partial_sum(amount#11)])
         +- FileScan parquet default.orders[region#10,amount#11]
"""

JOIN_EXPLAIN = """\
== Physical Plan ==
AdaptiveSparkPlan isFinalPlan=false
+- SortMergeJoin [user_id#1L], [user_id#2L], Inner
   :- Sort [user_id#1L ASC]
   :  +- Exchange hashpartitioning(user_id#1L, 200)
   :     +- FileScan parquet spark_catalog.analytics.orders[user_id#1L]
   +- Sort [user_id#2L ASC]
      +- Exchange hashpartitioning(user_id#2L, 200)
         +- FileScan parquet spark_catalog.analytics.users[user_id#2L]
"""

SIMPLE = """\
*(2) HashAggregate(keys=[region#10])
+- Exchange hashpartitioning(region#10, 200)
   +- *(1) HashAggregate(keys=[region#10])
      +- *(1) FileScan parquet default.orders[region#10]
"""

JSON_PLAN = [
    {"class": "org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanExec"},
    {"class": "org.apache.spark.sql.execution.aggregate.HashAggregateExec"},
    {"class": "org.apache.spark.sql.execution.exchange.ShuffleExchangeExec"},
    {"class": "org.apache.spark.sql.execution.aggregate.HashAggregateExec"},
    {"class": "org.apache.spark.sql.execution.FileSourceScanExec",
     "tableName": "default.orders", "location": "dbfs:/warehouse/orders"},
]

def test_extract_physical(): assert "AdaptiveSparkPlan" in extract_physical_plan(EXPLAIN_EXTENDED) and "Parsed" not in extract_physical_plan(EXPLAIN_EXTENDED)
def test_extract_no_sections(): assert "HashAggregate" in extract_physical_plan(SIMPLE)
def test_parse_smj(): assert parse_operator_line("+- SortMergeJoin [k1], Inner") == "SortMergeJoin"
def test_parse_filescan(): assert parse_operator_line("   +- FileScan parquet db.t") == "FileScan"
def test_parse_codegen(): assert parse_operator_line("*(2) HashAggregate(keys=[])") == "HashAggregate"
def test_parse_tree(): assert parse_operator_line(":  +- Exchange hash(...)") == "Exchange"
def test_parse_empty(): assert parse_operator_line("") is None and parse_operator_line("   ") is None
def test_ops_text_extended():
    ops = operators_from_text(EXPLAIN_EXTENDED)
    assert "HashAggregate" in ops and "FileScan" in ops and "Project" not in ops
def test_ops_text_simple(): assert len(operators_from_text(SIMPLE)) == 4
def test_ops_text_join():
    ops = operators_from_text(JOIN_EXPLAIN)
    assert "SortMergeJoin" in ops and ops.count("FileScan") == 2
def test_ops_json(): assert operators_from_json(JSON_PLAN)[-1] == "FileSourceScanExec"
def test_ops_json_empty(): assert operators_from_json([]) == [] and operators_from_json("x") == []
def test_ops_entry_json(): assert any("Exec" in o for o in operators_from_entry({"plan": JSON_PLAN}))
def test_ops_entry_text(): assert "HashAggregate" in operators_from_entry({"planText": EXPLAIN_EXTENDED})
def test_fp_deterministic():
    o = ["HashAggregate", "Exchange"]
    assert fingerprint_operators(o) == fingerprint_operators(o)
def test_fp_different(): assert fingerprint_operators(["A"]) != fingerprint_operators(["B"])
def test_fp_empty(): assert fingerprint_operators([]) == "empty"
def test_fp_length(): assert len(fingerprint_operators(["A"])) == 16
def test_tables_text(): assert "default.orders" in tables_from_text(EXPLAIN_EXTENDED)
def test_tables_text_catalog():
    t = tables_from_text(JOIN_EXPLAIN)
    assert "analytics.orders" in t and "analytics.users" in t
def test_tables_json(): assert "default.orders" in tables_from_json(JSON_PLAN)
def test_tables_entry(): assert "default.orders" in tables_from_entry({"plan": JSON_PLAN})
def test_tables_entry_empty(): assert tables_from_entry({}) == set()
def test_fmt_bytes(): assert fmt_bytes(500) == "500 B" and fmt_bytes(5*1024**3) == "5.0 GB" and fmt_bytes(None) == "?"
def test_parse_int(): assert parse_int("200") == 200 and parse_int(None) is None

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn(); print(f"  ✓ {fn.__name__}"); passed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed: exit(1)
