"""Tests for plan parsing — pure Python, no Spark."""

from cluster_yield_snapshot.plans import (
    operators_from_text, operators_from_json, operators_from_entry,
    extract_physical_plan, parse_operator_line, fingerprint_operators,
    tables_from_text, tables_from_json, tables_from_entry,
    has_metrics, metrics_summary,
    scan_bytes_from_entry, scan_rows_from_entry, scan_files_from_entry,
    shuffle_bytes_from_entry, broadcast_bytes_from_entry,
    parse_plan_details, has_plan_details,
    scans_from_entry, joins_from_entry, exchanges_from_entry,
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


# ── parse_plan_details tests ─────────────────────────────────────────────

def test_details_photon_scans():
    """Photon plan should produce scan details with table, format, filters."""
    d = parse_plan_details(PHOTON_PLAN)
    assert d["nodes"], "Should have parsed nodes"
    scans = d["scans"]
    assert len(scans) == 3
    tables = {s["table"] for s in scans}
    assert "workspace.test_cy.orders" in tables
    assert "workspace.test_cy.users" in tables
    assert "workspace.test_cy.products" in tables


def test_details_photon_orders_scan_metadata():
    """Orders scan should have partition filter, data filter, readSchema."""
    d = parse_plan_details(PHOTON_PLAN)
    orders = [s for s in d["scans"] if "orders" in s.get("table", "")][0]
    assert orders["format"] == "parquet"
    assert orders["hasPartitionFilter"] is True
    assert orders["hasDataFilter"] is True
    assert orders.get("readColumnCount", 0) == 4  # ReadSchema has 4 cols


def test_details_photon_users_scan_no_partition_filter():
    """Users scan should NOT have a partition filter."""
    d = parse_plan_details(PHOTON_PLAN)
    users = [s for s in d["scans"] if "users" in s.get("table", "")][0]
    assert users["hasPartitionFilter"] is False


def test_details_photon_enabled():
    """Photon plan should detect photonEnabled."""
    d = parse_plan_details(PHOTON_PLAN)
    assert d["photonEnabled"] is True


def test_details_no_detail_section():
    """Plan without numbered details should return empty nodes."""
    d = parse_plan_details(EXPLAIN_EXTENDED)
    assert d["nodes"] == {}


def test_details_scans_from_entry():
    """scans_from_entry should return scans from planDetails."""
    entry = {"planDetails": parse_plan_details(PHOTON_PLAN)}
    scans = scans_from_entry(entry)
    assert len(scans) == 3


def test_details_has_plan_details():
    """has_plan_details should check for non-empty nodes."""
    entry_with = {"planDetails": parse_plan_details(PHOTON_PLAN)}
    entry_without = {"planDetails": {"nodes": {}}}
    assert has_plan_details(entry_with) is True
    assert has_plan_details(entry_without) is False
    assert has_plan_details({}) is False


def test_details_read_schema_parsing():
    """ReadSchema should be parsed into column name:type dict."""
    d = parse_plan_details(PHOTON_PLAN)
    # Node 1 is orders scan
    orders_node = d["nodes"].get("1", {})
    schema = orders_node.get("readSchema", {})
    assert "user_id" in schema
    assert schema["user_id"] == "bigint"
    assert "amount" in schema
    assert "decimal" in schema["amount"]


def test_details_output_columns():
    """Output columns should be parsed from detail section."""
    d = parse_plan_details(PHOTON_PLAN)
    orders_node = d["nodes"].get("1", {})
    assert orders_node.get("outputCount") == 5
    output = orders_node.get("output", [])
    assert "user_id" in output
    assert "amount" in output


def test_details_partition_filters():
    """PartitionFilters should be parsed into a list."""
    d = parse_plan_details(PHOTON_PLAN)
    orders_node = d["nodes"].get("1", {})
    pf = orders_node.get("partitionFilters", [])
    assert len(pf) >= 1
    # Should contain the date filter
    assert any("order_date" in f for f in pf)


OSS_SPARK_PLAN = """\
== Physical Plan ==
AdaptiveSparkPlan isFinalPlan=false
+- HashAggregate(keys=[region#10], functions=[sum(amount#11)])
   +- Exchange hashpartitioning(region#10, 200)
      +- HashAggregate(keys=[region#10], functions=[partial_sum(amount#11)])
         +- FileScan parquet default.orders[region#10,amount#11]

(5) FileScan parquet default.orders
Output [2]: [region#10, amount#11]
PartitionFilters: []
DataFilters: []
ReadSchema: struct<region:string,amount:decimal(10,2)>
Location: InMemoryFileIndex [s3://warehouse/orders]
"""

def test_details_oss_spark_scan():
    """OSS Spark plan with FileScan should parse correctly."""
    d = parse_plan_details(OSS_SPARK_PLAN)
    assert len(d["scans"]) == 1
    scan = d["scans"][0]
    assert scan["table"] == "default.orders"
    assert scan["format"] == "parquet"
    assert scan["readColumnCount"] == 2
    # Empty partition filters still means hasPartitionFilter = False
    assert scan["hasPartitionFilter"] is False


def test_details_oss_spark_aqe_status():
    """isFinalPlan=false should be detected."""
    d = parse_plan_details(OSS_SPARK_PLAN)
    assert d["aqeFinalPlan"] is False


SMJ_PLAN = """\
== Physical Plan ==
AdaptiveSparkPlan isFinalPlan=true
+- SortMergeJoin [user_id#1L], [user_id#2L], Inner
   :- Sort [user_id#1L ASC]
   :  +- Exchange hashpartitioning(user_id#1L, 200)
   :     +- FileScan parquet default.orders[user_id#1L]
   +- Sort [user_id#2L ASC]
      +- Exchange hashpartitioning(user_id#2L, 200)
         +- FileScan parquet default.users[user_id#2L]

(3) SortMergeJoin
Left keys [1]: [user_id#1L]
Right keys [1]: [user_id#2L]
Join type: Inner
Join condition: None

(5) Exchange
Arguments: hashpartitioning(user_id#1L, 200)

(8) Exchange
Arguments: hashpartitioning(user_id#2L, 200)
"""

def test_details_smj_join():
    """SortMergeJoin should be parsed with keys and type."""
    d = parse_plan_details(SMJ_PLAN)
    assert len(d["joins"]) == 1
    join = d["joins"][0]
    assert join["strategy"] == "SortMergeJoin"
    assert join.get("joinType") == "Inner"
    left_keys = d["nodes"]["3"].get("leftKeys", [])
    assert "user_id" in left_keys


def test_details_smj_exchanges():
    """Exchange nodes should parse hashpartitioning arguments."""
    d = parse_plan_details(SMJ_PLAN)
    exchanges = d["exchanges"]
    assert len(exchanges) == 2
    for exch in exchanges:
        assert exch.get("exchangeType") == "shuffle"
        assert exch.get("numPartitions") == 200


def test_details_aqe_final_true():
    """isFinalPlan=true should be detected."""
    d = parse_plan_details(SMJ_PLAN)
    assert d["aqeFinalPlan"] is True


def test_details_empty_input():
    """Empty string should return empty nodes."""
    d = parse_plan_details("")
    assert d["nodes"] == {}
    assert d.get("scans", []) == []


# ── Formatted mode tests (actual Databricks Spark 4.1 Connect output) ──

FORMATTED_PLAN = """\
== Physical Plan ==
AdaptiveSparkPlan (10)
+- == Initial Plan ==
   ColumnarToRow (9)
   +- PhotonResultStage (8)
      +- PhotonProject (7)
         +- PhotonBroadcastHashJoin (6)
            :- PhotonScan parquet workspace.default.test_orders (1)
            +- PhotonShuffleExchangeSource (5)
               +- PhotonShuffleMapStage (4)
                  +- PhotonShuffleExchangeSink (3)
                     +- PhotonScan parquet workspace.default.test_customers (2)

(1) PhotonScan parquet workspace.default.test_orders
Output [3]: [order_id#26125L, customer_id#26126L, amount#26130]
DataFilters: [isnotnull(amount#26130), isnotnull(customer_id#26126L), (amount#26130 > 50.0)]
DictionaryFilters: [(amount#26130 > 50.0)]
Format: parquet
Location: PreparedDeltaFileIndex(1 paths)[s3://dbstorage-prod/data/orders]
OptionalDataFilters: [hashedrelationcontains(customer_id#26126L)]
PartitionFilters: []
ReadSchema: struct<order_id:bigint,customer_id:bigint,amount:double>
RequiredDataFilters: [isnotnull(amount#26130), isnotnull(customer_id#26126L), (amount#26130 > 50.0)]

(2) PhotonScan parquet workspace.default.test_customers
Output [2]: [customer_id#26135L, name#26136]
DataFilters: [isnotnull(customer_id#26135L)]
DictionaryFilters: []
Format: parquet
Location: PreparedDeltaFileIndex(1 paths)[s3://dbstorage-prod/data/customers]
OptionalDataFilters: []
PartitionFilters: []
ReadSchema: struct<customer_id:bigint,name:string>
RequiredDataFilters: [isnotnull(customer_id#26135L)]

(3) PhotonShuffleExchangeSink
Input [2]: [customer_id#26135L, name#26136]
Arguments: SinglePartition

(4) PhotonShuffleMapStage
Input [2]: [customer_id#26135L, name#26136]
Arguments: EXECUTOR_BROADCAST, [id=#25799]

(5) PhotonShuffleExchangeSource
Input [2]: [customer_id#26135L, name#26136]

(6) PhotonBroadcastHashJoin
Left keys [1]: [customer_id#26126L]
Right keys [1]: [customer_id#26135L]
Join type: Inner
Join condition: None

(7) PhotonProject
Input [5]: [order_id#26125L, customer_id#26126L, amount#26130, customer_id#26135L, name#26136]
Arguments: [order_id#26125L, name#26136, amount#26130]

(8) PhotonResultStage
Input [3]: [order_id#26125L, name#26136, amount#26130]

(9) ColumnarToRow
Input [3]: [order_id#26125L, name#26136, amount#26130]

(10) AdaptiveSparkPlan
Output [3]: [order_id#26125L, name#26136, amount#26130]
Arguments: isFinalPlan=false

== Photon Explanation ==
The query is fully supported by Photon.
== Optimizer Statistics (table names per statistics state) ==
  missing =\x20
  partial =\x20
  full    = test_customers, test_orders
"""


def test_formatted_operators_from_text():
    """operators_from_text should work with formatted mode tree."""
    ops = operators_from_text(FORMATTED_PLAN)
    assert "AdaptiveSparkPlan" in ops
    assert "PhotonBroadcastHashJoin" in ops
    assert "PhotonScan" in ops
    assert "ColumnarToRow" in ops


def test_formatted_tree_stops_at_details():
    """extract_physical_plan should not include detail section nodes."""
    tree = extract_physical_plan(FORMATTED_PLAN)
    # Tree should NOT contain detail section lines
    assert "(1) PhotonScan" not in tree
    assert "ReadSchema:" not in tree


def test_formatted_details_scans():
    """Formatted mode detail sections should parse scans."""
    d = parse_plan_details(FORMATTED_PLAN)
    assert len(d["scans"]) == 2
    orders = [s for s in d["scans"] if "orders" in s.get("table", "")][0]
    assert orders["format"] == "parquet"
    assert orders["table"] == "workspace.default.test_orders"
    assert orders.get("readColumnCount") == 3
    assert orders["hasDataFilter"] is True
    assert orders["hasPartitionFilter"] is False  # PartitionFilters: []


def test_formatted_details_read_schema():
    """ReadSchema should parse correctly from formatted detail sections."""
    d = parse_plan_details(FORMATTED_PLAN)
    orders_node = d["nodes"]["1"]
    schema = orders_node.get("readSchema", {})
    assert "order_id" in schema
    assert schema["order_id"] == "bigint"
    assert schema["amount"] == "double"


def test_formatted_details_join():
    """Join should be parsed from formatted detail section."""
    d = parse_plan_details(FORMATTED_PLAN)
    assert len(d["joins"]) == 1
    join = d["joins"][0]
    assert join["strategy"] == "BroadcastHashJoin"
    node = d["nodes"]["6"]
    assert node["joinType"] == "Inner"
    assert "customer_id" in node.get("leftKeys", [])
    assert "customer_id" in node.get("rightKeys", [])


def test_formatted_details_exchange():
    """Exchange should be parsed from formatted detail section."""
    d = parse_plan_details(FORMATTED_PLAN)
    exchanges = d["exchanges"]
    # PhotonShuffleExchangeSink is an exchange
    sink = [e for e in exchanges if "Sink" in e["operator"]]
    assert len(sink) == 1
    assert sink[0].get("exchangeType") == "broadcast"  # SinglePartition


def test_formatted_details_input_cols():
    """Non-scan operators should have inputCount from Input lines."""
    d = parse_plan_details(FORMATTED_PLAN)
    project = d["nodes"]["7"]  # PhotonProject
    assert project.get("inputCount") == 5
    assert "order_id" in project.get("input", [])


def test_formatted_details_photon():
    """Photon should be detected from formatted plan."""
    d = parse_plan_details(FORMATTED_PLAN)
    assert d["photonEnabled"] is True


def test_formatted_tables_from_text():
    """tables_from_text should find tables in formatted mode tree."""
    tables = tables_from_text(FORMATTED_PLAN)
    assert "workspace.default.test_orders" in tables
    assert "workspace.default.test_customers" in tables

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