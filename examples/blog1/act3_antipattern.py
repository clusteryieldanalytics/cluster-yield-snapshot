# Databricks notebook source
# MAGIC %pip install cluster-yield-snapshot==0.3.26 --force-reinstall --no-cache-dir

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from cluster_yield_snapshot import CYSnapshot
cy = CYSnapshot(spark).start()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType
from datetime import date, timedelta

spark.sql("CREATE DATABASE IF NOT EXISTS blog_demo")
spark.sql("USE blog_demo")

# COMMAND ----------

# ---- events: 800 GB, 30 columns, partitioned by event_date (365 days) ----
spark.sql("DROP TABLE IF EXISTS events")
spark.sql("""
CREATE TABLE events (
    event_id        BIGINT,
    event_date      DATE,
    event_type      STRING,
    user_id         BIGINT,
    session_id      STRING,
    page_url        STRING,
    referrer_url    STRING,
    device_type     STRING,
    browser         STRING,
    os              STRING,
    country         STRING,
    region          STRING,
    city            STRING,
    latitude        DOUBLE,
    longitude       DOUBLE,
    duration_sec    INT,
    scroll_depth    DOUBLE,
    click_count     INT,
    form_submits    INT,
    errors          INT,
    amount          DECIMAL(12,2),
    currency        STRING,
    product_id      BIGINT,
    category        STRING,
    search_query    STRING,
    ab_variant      STRING,
    campaign_id     STRING,
    utm_source      STRING,
    utm_medium      STRING,
    is_converted    BOOLEAN
)
USING DELTA
PARTITIONED BY (event_date)
""")

spark.range(100).select(
    F.col("id").alias("event_id"),
    F.date_add(F.lit("2024-01-01"), (F.col("id") % 30).cast("int")).alias("event_date"),
    F.lit("page_view").alias("event_type"),
    (F.col("id") % 500).alias("user_id"),
    F.lit("sess_xyz789").alias("session_id"),
    F.lit("/products/widget").alias("page_url"),
    F.lit("https://google.com").alias("referrer_url"),
    F.lit("desktop").alias("device_type"),
    F.lit("Chrome").alias("browser"),
    F.lit("macOS").alias("os"),
    F.lit("US").alias("country"),
    F.lit("IL").alias("region"),
    F.lit("Springfield").alias("city"),
    F.lit(39.78).alias("latitude"),
    F.lit(-89.65).alias("longitude"),
    (F.col("id") % 300 + 1).cast("int").alias("duration_sec"),
    (F.col("id") % 100 / 100.0).alias("scroll_depth"),
    (F.col("id") % 10).cast("int").alias("click_count"),
    F.lit(0).alias("form_submits"),
    F.lit(0).alias("errors"),
    (F.col("id") * 9.99).cast("decimal(12,2)").alias("amount"),
    F.lit("USD").alias("currency"),
    (F.col("id") % 50).alias("product_id"),
    F.lit("electronics").alias("category"),
    F.lit(None).cast("string").alias("search_query"),
    F.lit("control").alias("ab_variant"),
    F.lit(None).cast("string").alias("campaign_id"),
    F.lit("google").alias("utm_source"),
    F.lit("cpc").alias("utm_medium"),
    F.lit(False).alias("is_converted"),
).write.mode("overwrite").insertInto("events")

# COMMAND ----------

# ── Scale factor ──────────────────────────────────────────────
# Adjust this single number to control all table sizes proportionally.
SCALE = 100   # 0.01 = tiny (CI test), 0.1 = demo, 1.0 = production-like

NUM_EVENT_DAYS   = 365
NUM_USERS        = 5_000
NUM_PRODUCTS     = 5_000
EVENTS_PER_DAY   = int(30_000 * SCALE)    # ~10M rows at SCALE=1.0

print(f"Scale factor: {SCALE}")
print(f"  events:   {EVENTS_PER_DAY:,}/day × {NUM_EVENT_DAYS} days = ~{EVENTS_PER_DAY * NUM_EVENT_DAYS:,} rows")

# COMMAND ----------

import time

t0 = time.time()
total_rows = EVENTS_PER_DAY * NUM_EVENT_DAYS
print(f"Seeding events ({total_rows:,} rows) in monthly batches...")

# First batch overwrites, subsequent batches append
months = [
    ("2024-01-01", "2024-01-31"),
    ("2024-02-01", "2024-02-29"),
    ("2024-03-01", "2024-03-31"),
    ("2024-04-01", "2024-04-30"),
    ("2024-05-01", "2024-05-31"),
    ("2024-06-01", "2024-06-30"),
    ("2024-07-01", "2024-07-31"),
    ("2024-08-01", "2024-08-31"),
    ("2024-09-01", "2024-09-30"),
    ("2024-10-01", "2024-10-31"),
    ("2024-11-01", "2024-11-30"),
    ("2024-12-01", "2024-12-31"),
]

for i, (start, end) in enumerate(months):
    mode = "OVERWRITE" if i == 0 else "INTO"
    mt0 = time.time()
    
    spark.sql(f"""
    INSERT {mode} blog_demo.events
    SELECT
        monotonically_increasing_id()             AS event_id,
        dt                                        AS event_date,
        CASE WHEN rand() < 0.4 THEN 'page_view'
             WHEN rand() < 0.7 THEN 'click'
             WHEN rand() < 0.85 THEN 'purchase'
             WHEN rand() < 0.95 THEN 'signup'
             ELSE 'other'
        END                                       AS event_type,
        CAST(rand() * {NUM_USERS} AS BIGINT)      AS user_id,
        concat(
            hex(CAST(rand() * 4294967295 AS BIGINT)),
            hex(CAST(rand() * 4294967295 AS BIGINT))
        )                                         AS session_id,
        CASE
            WHEN rand() < 0.08 THEN concat('/checkout/order/', CAST(CAST(rand()*10000 AS INT) AS STRING))
            WHEN rand() < 0.16 THEN concat('/cart/view/', CAST(CAST(rand()*10000 AS INT) AS STRING))
            WHEN rand() < 0.22 THEN concat('/payment/confirm/', CAST(CAST(rand()*10000 AS INT) AS STRING))
            WHEN rand() < 0.27 THEN concat('/billing/invoice/', CAST(CAST(rand()*1000 AS INT) AS STRING))
            WHEN rand() < 0.30 THEN concat('/admin/users/', CAST(CAST(rand()*100 AS INT) AS STRING))
            WHEN rand() < 0.32 THEN '/internal/health'
            WHEN rand() < 0.33 THEN '/debug/spark-ui'
            ELSE concat('/page/', CAST(CAST(rand()*500 AS INT) AS STRING))
        END                                       AS page_url,
        CASE
            WHEN rand() < 0.05 THEN 'https://affiliate.partner-network.com/ref123'
            WHEN rand() < 0.08 THEN 'https://partner.example.com/landing'
            WHEN rand() < 0.10 THEN 'https://referral.program.io/invite'
            WHEN rand() < 0.12 THEN 'https://bot.scraperapi.com/crawl'
            WHEN rand() < 0.13 THEN 'https://spider.searchengine.net/index'
            WHEN rand() < 0.35 THEN 'https://google.com'
            WHEN rand() < 0.50 THEN 'https://facebook.com'
            ELSE NULL
        END                                       AS referrer_url,
        CASE WHEN rand() < 0.6 THEN 'desktop'
             WHEN rand() < 0.85 THEN 'mobile'
             ELSE 'tablet'
        END                                       AS device_type,
        CASE WHEN rand() < 0.5 THEN 'Chrome'
             WHEN rand() < 0.8 THEN 'Safari'
             ELSE 'Firefox'
        END                                       AS browser,
        CASE WHEN rand() < 0.4 THEN 'macOS'
             WHEN rand() < 0.7 THEN 'Windows'
             WHEN rand() < 0.9 THEN 'iOS'
             ELSE 'Android'
        END                                       AS os,
        CASE WHEN rand() < 0.6 THEN 'US'
             WHEN rand() < 0.8 THEN 'UK'
             WHEN rand() < 0.9 THEN 'DE'
             ELSE 'CA'
        END                                       AS country,
        concat('region_', CAST(CAST(rand()*50 AS INT) AS STRING))   AS region,
        concat('city_', CAST(CAST(rand()*200 AS INT) AS STRING))    AS city,
        CAST(30 + rand() * 20 AS DOUBLE)          AS latitude,
        CAST(-120 + rand() * 60 AS DOUBLE)        AS longitude,
        CAST(rand() * 600 AS INT)                 AS duration_sec,
        CAST(rand() AS DOUBLE)                    AS scroll_depth,
        CAST(rand() * 20 AS INT)                  AS click_count,
        CAST(rand() * 3 AS INT)                   AS form_submits,
        CAST(rand() * 2 AS INT)                   AS errors,
        CAST(rand() * 500 AS DECIMAL(12,2))       AS amount,
        'USD'                                     AS currency,
        CAST(rand() * {NUM_PRODUCTS} AS BIGINT)   AS product_id,
        CASE WHEN rand() < 0.3 THEN 'electronics'
             WHEN rand() < 0.5 THEN 'clothing'
             WHEN rand() < 0.7 THEN 'home'
             ELSE 'other'
        END                                       AS category,
        CASE WHEN rand() < 0.3 THEN concat('query_', CAST(CAST(rand()*100 AS INT) AS STRING))
             ELSE NULL
        END                                       AS search_query,
        CASE WHEN rand() < 0.5 THEN 'control' ELSE 'variant_a'
        END                                       AS ab_variant,
        CASE WHEN rand() < 0.2 THEN concat('camp_', CAST(CAST(rand()*20 AS INT) AS STRING))
             ELSE NULL
        END                                       AS campaign_id,
        CASE WHEN rand() < 0.4 THEN 'google'
             WHEN rand() < 0.7 THEN 'direct'
             ELSE 'social'
        END                                       AS utm_source,
        CASE WHEN rand() < 0.5 THEN 'cpc'
             WHEN rand() < 0.8 THEN 'organic'
             ELSE 'referral'
        END                                       AS utm_medium,
        rand() < 0.05                             AS is_converted
    FROM RANGE(0, {EVENTS_PER_DAY}) AS t(id)
    CROSS JOIN (
        SELECT explode(sequence(
            DATE '{start}',
            DATE '{end}',
            INTERVAL 1 DAY
        )) AS dt
    )
    """)
    
    elapsed_m = time.time() - mt0
    print(f"  Month {i+1}/12 ({start}): {elapsed_m:.0f}s")

elapsed = time.time() - t0
count = spark.sql("SELECT COUNT(*) FROM blog_demo.events").collect()[0][0]
print(f"\nDone: {count:,} rows ({elapsed:.0f}s)")

# COMMAND ----------

import time

current = spark.sql("SELECT COUNT(*) FROM blog_demo.events").collect()[0][0]
size_gb = spark.sql("DESCRIBE DETAIL blog_demo.events").collect()[0]["sizeInBytes"] / (1024**3)
print(f"Starting: {current:,} rows, {size_gb:.1f} GB")

target_gb = 100
iteration = 0

while size_gb < target_gb:
    iteration += 1
    t0 = time.time()
    
    spark.sql(f"""
    INSERT INTO blog_demo.events
    SELECT
        monotonically_increasing_id() AS event_id,
        event_date, event_type, user_id,
        concat(hex(CAST(rand() * 4294967295 AS BIGINT)),
               hex(CAST(rand() * 4294967295 AS BIGINT))) AS session_id,
        page_url, referrer_url, device_type, browser, os,
        country, region, city, latitude, longitude,
        duration_sec, scroll_depth, click_count, form_submits, errors,
        amount, currency, product_id, category, search_query,
        ab_variant, campaign_id, utm_source, utm_medium, is_converted
    FROM blog_demo.events
    """)
    
    current = spark.sql("SELECT COUNT(*) FROM blog_demo.events").collect()[0][0]
    size_gb = spark.sql("DESCRIBE DETAIL blog_demo.events").collect()[0]["sizeInBytes"] / (1024**3)
    elapsed = time.time() - t0
    print(f"  Iteration {iteration}: {current:,} rows, {size_gb:.1f} GB ({elapsed:.0f}s)")

print(f"\nDone: {current:,} rows, {size_gb:.1f} GB")

# COMMAND ----------

# The .count() + .write() without .cache() means Spark reads the full events table, 
# does the aggregation, counts — then does the entire thing again for the 
# write. The fix is two lines: add .cache() before the count, .unpersist() after the write.

events = spark.table("blog_demo.events")

result = (
    events
    .filter(F.col("amount") > 0)
    .groupBy("user_id", "category", "event_type")
    .agg(
        F.count("*").alias("event_count"),
        F.sum("amount").alias("total_amount"),
        F.avg("duration_sec").alias("avg_duration"),
        F.max("event_date").alias("last_event_date"),
    )
)

# Data quality check — how many users had events this period?
row_count = result.count()
print(f"Fiscal period rollup: {row_count:,} user-category groups")

# Write the rollup
result.write.mode("overwrite").saveAsTable("blog_demo.fiscal_events")

# COMMAND ----------

events = spark.table("blog_demo.events")

# "Give me high-value events" — looks correct, passes review
high_value = (
    events
    .filter(F.col("amount") > 100)
    .groupBy("category", "event_type")
    .agg(
        F.sum("amount").alias("total"),
        F.count("*").alias("cnt"),
    )
)
high_value.write.mode("overwrite").saveAsTable(
    "blog_demo.high_value_summary"
)

# COMMAND ----------

events = spark.table("blog_demo.events")

high_value_events = (
    events
    .filter(F.col("amount") > 100)
    .filter(F.col("is_converted") == True)
)

# Three outputs, no cache, no partition pruning
daily = (
    high_value_events
    .groupBy("event_date", "event_type")
    .agg(F.sum("amount").alias("total"), F.count("*").alias("cnt"))
)
daily.write.mode("overwrite").saveAsTable("blog_demo.converted_daily")

by_channel = (
    high_value_events
    .groupBy("utm_source", "utm_medium")
    .agg(
        F.sum("amount").alias("revenue"),
        F.countDistinct("user_id").alias("unique_users"),
    )
)
by_channel.write.mode("overwrite").saveAsTable("blog_demo.converted_channels")

quality = (
    high_value_events
    .agg(
        F.count("*").alias("total_rows"),
        F.countDistinct("user_id").alias("unique_users"),
        F.avg("amount").alias("avg_amount"),
    )
)
quality.write.mode("overwrite").saveAsTable("blog_demo.converted_quality")

# COMMAND ----------

cy.stop().save("/tmp/act3_antipattern_snapshot.json")
cy.display_html()

# COMMAND ----------

import base64
with open("/tmp/act3_antipattern_snapshot.json", "r") as f:
    content = f.read()
b64 = base64.b64encode(content.encode()).decode()
displayHTML(f'<a href="data:application/json;base64,{b64}" download="act3_antipattern_snapshot.json">📥 Download Snapshot</a>')