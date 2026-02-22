"""Tests for plan parsing — pure Python, no Spark."""

from cluster_yield_snapshot.plans import (
    operators_from_text, operators_from_json, operators_from_entry,
    extract_physical_plan, parse_operator_line, fingerprint_operators,
    tables_from_text, tables_from_json, tables_from_entry,
    has_metrics, metrics_summary,
    scan_bytes_from_entry, scan_rows_from_entry, scan_files_from_entry,
    shuffle_bytes_from_entry, broadcast_bytes_from_entry,
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

# Photon plan with numbered detail section (from Databricks)
PHOTON_PLAN = """\
== Physical Plan ==
AdaptiveSparkPlan (22)
+- == Initial Plan ==
   ColumnarToRow (21)
   +- PhotonResultStage (20)
      +- PhotonGroupingAgg (19)
         +- PhotonShuffleExchangeSource (18)
            +- PhotonShuffleMapStage (17)
               +- PhotonShuffleExchangeSink (16)
                  +- PhotonGroupingAgg (15)
                     +- PhotonProject (14)
                        +- PhotonBroadcastHashJoin Inner (13)
                           :- PhotonProject (8)
                           :  +- PhotonBroadcastHashJoin Inner (7)
                           :     :- PhotonProject (2)
                           :     :  +- PhotonScan parquet workspace.test_cy.orders (1)
                           :     +- PhotonShuffleExchangeSource (6)
                           :        +- PhotonShuffleMapStage (5)
                           :           +- PhotonShuffleExchangeSink (4)
                           :              +- PhotonScan parquet workspace.test_cy.users (3)
                           +- PhotonShuffleExchangeSource (12)
                              +- PhotonShuffleMapStage (11)
                                 +- PhotonShuffleExchangeSink (10)
                                    +- PhotonScan parquet workspace.test_cy.products (9)

(1) PhotonScan parquet workspace.test_cy.orders
Output [5]: [user_id#13624L, product_id#13625L, amount#13627, status#13630, order_date#13629]
DictionaryFilters: [(status#13630 = completed)]
PartitionFilters: [isnotnull(order_date#13629), (order_date#13629 >= 2024-06-01)]
ReadSchema: struct<user_id:bigint,product_id:bigint,amount:decimal(10,2),status:string>
RequiredDataFilters: [isnotnull(user_id#13624L)]

(3) PhotonScan parquet workspace.test_cy.users
Output [2]: [user_id#13643L, country#13647]
ReadSchema: struct<user_id:bigint,country:string>

(9) PhotonScan parquet workspace.test_cy.products
Output [2]: [product_id#13650L, category#13652]
ReadSchema: struct<product_id:bigint,category:string>

== Photon Explanation ==
The query is fully supported by Photon.
"""

JSON_PLAN = [
    {"class": "org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanExec"},
    {"class": "org.apache.spark.sql.execution.aggregate.HashAggregateExec"},
    {"class": "org.apache.spark.sql.execution.exchange.ShuffleExchangeExec"},
    {"class": "org.apache.spark.sql.execution.aggregate.HashAggregateExec"},
    {"class": "org.apache.spark.sql.execution.FileSourceScanExec",
     "tableName": "default.orders", "location": "dbfs:/warehouse/orders"},
]

PHOTON_JSON_PLAN = [
    {"class": "com.databricks.photon.PhotonFileSourceScanExec",
     "tableName": "workspace.test_cy.orders"},
    {"class": "com.databricks.photon.PhotonFileSourceScanExec",
     "tableName": "workspace.test_cy.users"},
]

# ── Fixtures for runtime metric tests ────────────────────────────────────

PLAN_WITH_METRICS = {
    "label": "test-query",
    "nodeCount": 5,
    "plan": JSON_PLAN,
    "metrics": [
        {
            "nodeName": "Scan parquet default.orders",
            "simpleClassName": "FileSourceScanExec",
            "metrics": {
                "number of output rows": 1847293,
                "number of files read": 47,
                "size of files read": 523190272,
            },
        },
        {
            "nodeName": "Scan parquet default.users",
            "simpleClassName": "FileSourceScanExec",
            "metrics": {
                "number of output rows": 52000,
                "number of files read": 3,
                "size of files read": 4190208,
            },
        },
        {
            "nodeName": "HashAggregate",
            "simpleClassName": "HashAggregateExec",
            "metrics": {"number of output rows": 200},
        },
        {
            "nodeName": "Exchange",
            "simpleClassName": "ShuffleExchangeExec",
            "metrics": {"data size": 24576},
        },
        {
            "nodeName": "BroadcastExchange",
            "simpleClassName": "BroadcastExchangeExec",
            "metrics": {"data size": 4190208},
        },
    ],
}

PLAN_WITHOUT_METRICS = {
    "label": "pre-exec",
    "nodeCount": 5,
    "plan": JSON_PLAN,
}


# ── Existing tests (operator extraction, fingerprinting, tables) ─────────

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


# ── Photon text plan: boundary fix ───────────────────────────────────────

def test_photon_boundary_stops_at_details():
    """extract_physical_plan should NOT include numbered detail lines."""
    physical = extract_physical_plan(PHOTON_PLAN)
    # Tree operators should be present
    assert "PhotonGroupingAgg" in physical
    assert "PhotonScan" in physical
    # Detail section lines should NOT be present
    assert "Output [5]" not in physical
    assert "DictionaryFilters" not in physical
    assert "ReadSchema" not in physical
    assert "PartitionFilters" not in physical


def test_photon_operator_count():
    """Operator count should match actual tree nodes, not detail attributes."""
    ops = operators_from_text(PHOTON_PLAN)
    # The tree has 22 nodes. None of Output, ReadSchema, etc. should appear.
    assert len(ops) == 22
    assert "Output" not in ops
    assert "ReadSchema" not in ops
    assert "DictionaryFilters" not in ops
    assert "Arguments" not in ops


def test_photon_operators_present():
    """Photon operator names should be extracted correctly."""
    ops = operators_from_text(PHOTON_PLAN)
    assert "PhotonScan" in ops
    assert "PhotonBroadcastHashJoin" in ops
    assert "PhotonGroupingAgg" in ops
    assert "PhotonShuffleExchangeSink" in ops


def test_photon_tables_from_text():
    """PhotonScan table names should be extracted from tree lines."""
    tables = tables_from_text(PHOTON_PLAN)
    assert "workspace.test_cy.orders" in tables
    assert "workspace.test_cy.users" in tables
    assert "workspace.test_cy.products" in tables


def test_photon_json_scan_classes():
    """Photon scan class names should be recognized in JSON plans."""
    tables = tables_from_json(PHOTON_JSON_PLAN)
    assert "workspace.test_cy.orders" in tables
    assert "workspace.test_cy.users" in tables


def test_photon_stops_before_explanation():
    """Parser should stop before '== Photon Explanation ==' section."""
    physical = extract_physical_plan(PHOTON_PLAN)
    assert "Photon Explanation" not in physical
    assert "fully supported" not in physical


# ── Runtime metric convenience functions ─────────────────────────────────

def test_has_metrics_true():
    assert has_metrics(PLAN_WITH_METRICS) is True


def test_has_metrics_false():
    assert has_metrics(PLAN_WITHOUT_METRICS) is False


def test_has_metrics_empty():
    assert has_metrics({}) is False


def test_scan_bytes():
    assert scan_bytes_from_entry(PLAN_WITH_METRICS) == 523190272 + 4190208


def test_scan_rows():
    assert scan_rows_from_entry(PLAN_WITH_METRICS) == 1847293 + 52000


def test_scan_files():
    assert scan_files_from_entry(PLAN_WITH_METRICS) == 47 + 3


def test_shuffle_bytes():
    """Should include ShuffleExchange but NOT BroadcastExchange."""
    assert shuffle_bytes_from_entry(PLAN_WITH_METRICS) == 24576


def test_broadcast_bytes():
    assert broadcast_bytes_from_entry(PLAN_WITH_METRICS) == 4190208


def test_metrics_no_metrics():
    """All accessors return 0 when no metrics present."""
    assert scan_bytes_from_entry(PLAN_WITHOUT_METRICS) == 0
    assert shuffle_bytes_from_entry(PLAN_WITHOUT_METRICS) == 0
    assert broadcast_bytes_from_entry(PLAN_WITHOUT_METRICS) == 0


def test_metrics_summary_present():
    summary = metrics_summary(PLAN_WITH_METRICS)
    assert summary["scanBytes"] == 523190272 + 4190208
    assert summary["shuffleBytes"] == 24576
    assert summary["broadcastBytes"] == 4190208
    assert summary["scanFiles"] == 50


def test_metrics_summary_absent():
    assert metrics_summary(PLAN_WITHOUT_METRICS) == {}


def test_photon_metric_classnames():
    """Photon class names should be recognized by metric accessors."""
    photon_entry = {
        "metrics": [
            {
                "nodeName": "PhotonScan",
                "simpleClassName": "PhotonFileSourceScanExec",
                "metrics": {
                    "number of output rows": 500000,
                    "size of files read": 100_000_000,
                    "number of files read": 10,
                },
            },
            {
                "nodeName": "PhotonShuffleExchangeSink",
                "simpleClassName": "PhotonShuffleExchangeSinkExec",
                "metrics": {"data size": 5_000_000},
            },
        ]
    }
    assert scan_bytes_from_entry(photon_entry) == 100_000_000
    assert scan_rows_from_entry(photon_entry) == 500_000
    assert shuffle_bytes_from_entry(photon_entry) == 5_000_000

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