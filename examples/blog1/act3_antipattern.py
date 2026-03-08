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
spark.sql("DROP TABLE IF EXISTS orders")
spark.sql("""
CREATE TABLE orders (
    order_id            BIGINT,
    customer_id         BIGINT,
    order_date          DATE,
    order_status        STRING,
    total_amount        DECIMAL(12,2),
    tax_amount          DECIMAL(10,2),
    shipping_amount     DECIMAL(10,2),
    discount_amount     DECIMAL(10,2),
    net_amount          DECIMAL(12,2),
    currency            STRING,
    payment_method      STRING,
    shipping_method     STRING,
    channel             STRING,
    device_type         STRING,
    campaign_id         STRING,
    coupon_code         STRING,
    warehouse_id        INT,
    is_first_order      BOOLEAN,
    created_at          TIMESTAMP,
    updated_at          TIMESTAMP,
    shipped_at          TIMESTAMP,
    delivered_at        TIMESTAMP
)
USING DELTA
PARTITIONED BY (order_date)
""")

spark.range(100).select(
    F.col("id").alias("order_id"),
    (F.col("id") % 500).alias("customer_id"),
    F.date_add(F.lit("2024-01-01"), (F.col("id") % 30).cast("int")).alias("order_date"),
    F.lit("completed").alias("order_status"),
    F.lit(99.99).cast("decimal(12,2)").alias("total_amount"),
    F.lit(8.50).cast("decimal(10,2)").alias("tax_amount"),
    F.lit(5.00).cast("decimal(10,2)").alias("shipping_amount"),
    F.lit(0.00).cast("decimal(10,2)").alias("discount_amount"),
    F.lit(91.49).cast("decimal(12,2)").alias("net_amount"),
    F.lit("USD").alias("currency"),
    F.lit("credit_card").alias("payment_method"),
    F.lit("standard").alias("shipping_method"),
    F.lit("web").alias("channel"),
    F.lit("desktop").alias("device_type"),
    F.lit(None).cast("string").alias("campaign_id"),
    F.lit(None).cast("string").alias("coupon_code"),
    F.lit(1).alias("warehouse_id"),
    F.lit(True).alias("is_first_order"),
    F.current_timestamp().alias("created_at"),
    F.current_timestamp().alias("updated_at"),
    F.lit(None).cast("timestamp").alias("shipped_at"),
    F.lit(None).cast("timestamp").alias("delivered_at"),
).write.mode("overwrite").insertInto("orders")

# COMMAND ----------
spark.sql("DROP TABLE IF EXISTS users")
spark.sql("""
CREATE TABLE users (
    user_id             BIGINT,
    signup_date         DATE,
    email_domain        STRING,
    country             STRING,
    region              STRING,
    city                STRING,
    age_bucket          STRING,
    gender              STRING,
    lifetime_value      DECIMAL(12,2),
    order_count         INT,
    last_active_date    DATE,
    preferred_category  STRING,
    device_type         STRING,
    acquisition_source  STRING,
    is_active           BOOLEAN
)
USING DELTA
""")

spark.range(200).select(
    F.col("id").alias("user_id"),
    F.date_add(F.lit("2022-01-01"), (F.col("id") % 365).cast("int")).alias("signup_date"),
    F.lit("gmail.com").alias("email_domain"),
    F.lit("US").alias("country"),
    F.lit("IL").alias("region"),
    F.lit("Springfield").alias("city"),
    F.lit("25-34").alias("age_bucket"),
    F.lit("unknown").alias("gender"),
    (F.col("id") * 50.00).cast("decimal(12,2)").alias("lifetime_value"),
    (F.col("id") % 20).cast("int").alias("order_count"),
    F.date_add(F.lit("2024-06-01"), (F.col("id") % 30).cast("int")).alias("last_active_date"),
    F.lit("electronics").alias("preferred_category"),
    F.lit("mobile").alias("device_type"),
    F.lit("organic").alias("acquisition_source"),
    F.lit(True).alias("is_active"),
).write.mode("overwrite").insertInto("users")

# COMMAND ----------
spark.sql("DROP TABLE IF EXISTS product_features")
spark.sql("""
CREATE TABLE product_features (
    product_id          BIGINT,
    product_name        STRING,
    category            STRING,
    subcategory         STRING,
    brand               STRING,
    price               DECIMAL(10,2),
    avg_rating          DOUBLE,
    review_count        INT,
    embedding_v1        ARRAY<DOUBLE>,
    is_active           BOOLEAN,
    created_at          DATE,
    popularity_score    DOUBLE
)
USING DELTA
""")

spark.range(100).select(
    F.col("id").alias("product_id"),
    F.concat(F.lit("Product_"), F.col("id").cast("string")).alias("product_name"),
    F.lit("electronics").alias("category"),
    F.lit("gadgets").alias("subcategory"),
    F.lit("BrandX").alias("brand"),
    (F.col("id") * 9.99 + 10).cast("decimal(10,2)").alias("price"),
    (3.5 + F.col("id") % 15 / 10.0).alias("avg_rating"),
    (F.col("id") * 7).cast("int").alias("review_count"),
    F.array(F.lit(0.1), F.lit(0.2), F.lit(0.3)).alias("embedding_v1"),
    F.lit(True).alias("is_active"),
    F.current_date().alias("created_at"),
    (F.col("id") % 100 / 100.0).alias("popularity_score"),
).write.mode("overwrite").insertInto("product_features")

# COMMAND ----------

# ── Scale factor ──────────────────────────────────────────────
# Adjust this single number to control all table sizes proportionally.
SCALE = 10   # 0.01 = tiny (CI test), 0.1 = demo, 1.0 = production-like

NUM_EVENT_DAYS   = 365
NUM_ORDER_DAYS   = 365
EVENTS_PER_DAY   = int(30_000 * SCALE)    # ~10M rows at SCALE=1.0
ORDERS_PER_DAY   = int(15_000 * SCALE)    # ~5M rows at SCALE=1.0
NUM_USERS        = int(5_000)
NUM_PRODUCTS     = int(5_000)   

print(f"Scale factor: {SCALE}")
print(f"  events:   {EVENTS_PER_DAY:,}/day × {NUM_EVENT_DAYS} days = ~{EVENTS_PER_DAY * NUM_EVENT_DAYS:,} rows")
print(f"  orders:   {ORDERS_PER_DAY:,}/day × {NUM_ORDER_DAYS} days = ~{ORDERS_PER_DAY * NUM_ORDER_DAYS:,} rows")
print(f"  users:    {NUM_USERS:,} rows")
print(f"  products: {NUM_PRODUCTS:,} rows")

# COMMAND ----------

import time

t0 = time.time()
print(f"Seeding events ({EVENTS_PER_DAY * NUM_EVENT_DAYS:,} rows) ...")

spark.sql(f"""
INSERT OVERWRITE blog_demo.events
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
    uuid()                                    AS session_id,
    concat('/page/', CAST(CAST(rand()*500 AS INT) AS STRING))  AS page_url,
    CASE WHEN rand() < 0.3 THEN 'https://google.com'
         WHEN rand() < 0.5 THEN 'https://facebook.com'
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
    concat('region_', CAST(CAST(rand()*50 AS INT) AS STRING))  AS region,
    concat('city_', CAST(CAST(rand()*200 AS INT) AS STRING))   AS city,
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
        DATE '2024-01-01',
        DATE '2024-12-31',
        INTERVAL 1 DAY
    )) AS dt
)
""")

elapsed = time.time() - t0
count = spark.sql("SELECT COUNT(*) FROM blog_demo.events").collect()[0][0]
print(f"  Done: {count:,} rows ({elapsed:.0f}s)")

# COMMAND ----------

t0 = time.time()
print(f"Seeding orders ({ORDERS_PER_DAY * NUM_ORDER_DAYS:,} rows) ...")

spark.sql(f"""
INSERT OVERWRITE blog_demo.orders
SELECT
    monotonically_increasing_id()             AS order_id,
    CAST(rand() * {NUM_USERS} AS BIGINT)      AS customer_id,
    dt                                        AS order_date,
    CASE WHEN rand() < 0.7 THEN 'completed'
         WHEN rand() < 0.9 THEN 'shipped'
         ELSE 'pending'
    END                                       AS order_status,
    CAST(10 + rand() * 500 AS DECIMAL(12,2))  AS total_amount,
    CAST(rand() * 50 AS DECIMAL(10,2))        AS tax_amount,
    CAST(rand() * 20 AS DECIMAL(10,2))        AS shipping_amount,
    CAST(rand() * 30 AS DECIMAL(10,2))        AS discount_amount,
    CAST(10 + rand() * 450 AS DECIMAL(12,2))  AS net_amount,
    'USD'                                     AS currency,
    CASE WHEN rand() < 0.5 THEN 'credit_card'
         WHEN rand() < 0.8 THEN 'debit'
         ELSE 'paypal'
    END                                       AS payment_method,
    CASE WHEN rand() < 0.7 THEN 'standard'
         WHEN rand() < 0.9 THEN 'express'
         ELSE 'overnight'
    END                                       AS shipping_method,
    CASE WHEN rand() < 0.6 THEN 'web'
         WHEN rand() < 0.85 THEN 'mobile'
         ELSE 'app'
    END                                       AS channel,
    CASE WHEN rand() < 0.5 THEN 'desktop'
         ELSE 'mobile'
    END                                       AS device_type,
    CASE WHEN rand() < 0.3 THEN concat('camp_', CAST(CAST(rand()*20 AS INT) AS STRING))
         ELSE NULL
    END                                       AS campaign_id,
    CASE WHEN rand() < 0.2 THEN concat('COUP', CAST(CAST(rand()*100 AS INT) AS STRING))
         ELSE NULL
    END                                       AS coupon_code,
    CAST(rand() * 10 AS INT)                  AS warehouse_id,
    rand() < 0.3                              AS is_first_order,
    current_timestamp()                       AS created_at,
    current_timestamp()                       AS updated_at,
    CASE WHEN rand() < 0.6 THEN current_timestamp() ELSE NULL END AS shipped_at,
    CASE WHEN rand() < 0.4 THEN current_timestamp() ELSE NULL END AS delivered_at
FROM RANGE(0, {ORDERS_PER_DAY}) AS t(id)
CROSS JOIN (
    SELECT explode(sequence(
        DATE '2024-01-01',
        DATE '2024-12-31',
        INTERVAL 1 DAY
    )) AS dt
)
""")

elapsed = time.time() - t0
count = spark.sql("SELECT COUNT(*) FROM blog_demo.orders").collect()[0][0]
print(f"  Done: {count:,} rows ({elapsed:.0f}s)")

# COMMAND ----------

t0 = time.time()
print(f"Seeding users ({NUM_USERS:,} rows) ...")

spark.sql(f"""
INSERT OVERWRITE blog_demo.users
SELECT
    id                                        AS user_id,
    date_add(DATE '2020-01-01', CAST(rand() * 1500 AS INT))  AS signup_date,
    CASE WHEN rand() < 0.5 THEN 'gmail.com'
         WHEN rand() < 0.8 THEN 'outlook.com'
         ELSE 'yahoo.com'
    END                                       AS email_domain,
    CASE WHEN rand() < 0.6 THEN 'US'
         WHEN rand() < 0.8 THEN 'UK'
         WHEN rand() < 0.9 THEN 'DE'
         ELSE 'CA'
    END                                       AS country,
    concat('region_', CAST(CAST(rand()*50 AS INT) AS STRING))  AS region,
    concat('city_', CAST(CAST(rand()*200 AS INT) AS STRING))   AS city,
    CASE WHEN rand() < 0.25 THEN '18-24'
         WHEN rand() < 0.55 THEN '25-34'
         WHEN rand() < 0.80 THEN '35-44'
         ELSE '45+'
    END                                       AS age_bucket,
    CASE WHEN rand() < 0.45 THEN 'M'
         WHEN rand() < 0.9 THEN 'F'
         ELSE 'unknown'
    END                                       AS gender,
    CAST(rand() * 5000 AS DECIMAL(12,2))      AS lifetime_value,
    CAST(rand() * 50 AS INT)                  AS order_count,
    date_add(DATE '2024-06-01', CAST(rand() * 180 AS INT))  AS last_active_date,
    CASE WHEN rand() < 0.3 THEN 'electronics'
         WHEN rand() < 0.5 THEN 'clothing'
         WHEN rand() < 0.7 THEN 'home'
         ELSE 'other'
    END                                       AS preferred_category,
    CASE WHEN rand() < 0.6 THEN 'mobile'
         ELSE 'desktop'
    END                                       AS device_type,
    CASE WHEN rand() < 0.4 THEN 'organic'
         WHEN rand() < 0.7 THEN 'paid'
         ELSE 'referral'
    END                                       AS acquisition_source,
    rand() < 0.85                             AS is_active
FROM RANGE(0, {NUM_USERS}) AS t(id)
""")

elapsed = time.time() - t0
count = spark.sql("SELECT COUNT(*) FROM blog_demo.users").collect()[0][0]
print(f"  Done: {count:,} rows ({elapsed:.0f}s)")

# COMMAND ----------

t0 = time.time()
print(f"Seeding product_features ({NUM_PRODUCTS:,} rows) ...")

spark.sql(f"""
INSERT OVERWRITE blog_demo.product_features
SELECT
    id                                        AS product_id,
    concat('Product_', CAST(id AS STRING))    AS product_name,
    CASE WHEN rand() < 0.25 THEN 'electronics'
         WHEN rand() < 0.50 THEN 'clothing'
         WHEN rand() < 0.75 THEN 'home'
         ELSE 'other'
    END                                       AS category,
    concat('sub_', CAST(CAST(rand()*30 AS INT) AS STRING))  AS subcategory,
    concat('Brand_', CAST(CAST(rand()*50 AS INT) AS STRING)) AS brand,
    CAST(5 + rand() * 500 AS DECIMAL(10,2))   AS price,
    CAST(1.0 + rand() * 4.0 AS DOUBLE)        AS avg_rating,
    CAST(rand() * 2000 AS INT)                AS review_count,
    array(rand(), rand(), rand(), rand(), rand(),
          rand(), rand(), rand())             AS embedding_v1,
    rand() < 0.9                              AS is_active,
    date_add(DATE '2022-01-01', CAST(rand() * 1000 AS INT))  AS created_at,
    CAST(rand() AS DOUBLE)                    AS popularity_score
FROM RANGE(0, {NUM_PRODUCTS}) AS t(id)
""")

elapsed = time.time() - t0
count = spark.sql("SELECT COUNT(*) FROM blog_demo.product_features").collect()[0][0]
print(f"  Done: {count:,} rows ({elapsed:.0f}s)")

# COMMAND ----------

print("Running ANALYZE TABLE on all four tables...")
for table in ["events", "orders", "users", "product_features"]:
    fq = f"blog_demo.{table}"
    t0 = time.time()
    spark.sql(f"ANALYZE TABLE {fq} COMPUTE STATISTICS FOR ALL COLUMNS")
    elapsed = time.time() - t0

    desc = spark.sql(f"DESCRIBE DETAIL {fq}").collect()[0]
    size_gb = desc["sizeInBytes"] / (1024**3) if desc["sizeInBytes"] else 0
    num_files = desc["numFiles"] or 0
    row_count = spark.sql(f"SELECT COUNT(*) FROM {fq}").collect()[0][0]
    print(f"  {table:<20} {size_gb:>6.2f} GB   {num_files:>5} files   {row_count:>12,} rows   ({elapsed:.0f}s)")

print("\n✅ All tables seeded and stats computed.")

# COMMAND ----------

# The fiscal calendar logic: fiscal months start on the 4th Saturday
# of each calendar month.  A UDF is the natural way to express this.
@F.udf(BooleanType())
def is_in_fiscal_window(event_date):
    """Return True if event_date falls within the current fiscal period.

    Our fiscal month starts on the 4th Saturday of each calendar month.
    This is easier to express in Python than in Spark SQL builtins.
    """
    if event_date is None:
        return False
    from datetime import date, timedelta
    today = date.today()
    first_of_month = today.replace(day=1)
    days_until_saturday = (5 - first_of_month.weekday()) % 7
    first_saturday = first_of_month + timedelta(days=days_until_saturday)
    fourth_saturday = first_saturday + timedelta(weeks=3)
    prev_month = (first_of_month - timedelta(days=1)).replace(day=1)
    days_until_saturday_prev = (5 - prev_month.weekday()) % 7
    first_saturday_prev = prev_month + timedelta(days=days_until_saturday_prev)
    fourth_saturday_prev = first_saturday_prev + timedelta(weeks=3)
    return fourth_saturday_prev <= event_date <= fourth_saturday

# COMMAND ----------

events = spark.table("blog_demo.events")

# This looks correct.  The filter logic is right.  A code reviewer
# might approve it.  But Spark sees an opaque Python function where
# the PartitionFilters should be — and reads ALL 365 partitions.
#
# Plan shows:
#   PartitionFilters: []          ← empty, no pruning
#   BatchEvalPython               ← UDF, evaluated AFTER full scan

fiscal_events = (
    events
    .filter(is_in_fiscal_window(F.col("event_date"))) # UDF BAD!
    .filter(F.col("amount") > 0)
    .groupBy("user_id", "category", "event_type")
    .agg(
        F.count("*").alias("event_count"),
        F.sum("amount").alias("total_amount"),
        F.avg("duration_sec").alias("avg_duration"),
        F.max("event_date").alias("last_event_date"),
    )
)

fiscal_events.write.mode("overwrite").saveAsTable(
    "blog_demo.fiscal_events"
)

# COMMAND ----------

orders = spark.table("blog_demo.orders")

yesterday = date.today() - timedelta(days=1)

# The plan will show a full
# ShuffleExchange — every byte of the filtered orders moves across the
# network before being written.

daily_orders = (
    orders
    .filter(F.col("order_date") == F.lit(yesterday))
    .select(
        "order_id", "customer_id", "order_date", "order_status",
        "total_amount", "net_amount", "currency", "payment_method",
        "channel", "warehouse_id",
    )
)

(
    daily_orders
    .repartition(10) #shuffle!  BUT!  AQE to the rescue?!
    .write
    .mode("overwrite")
    .saveAsTable("blog_demo.order_export")
)

# COMMAND ----------

# The .count() + .write() without .cache() means Spark reads the full events table, 
# runs the UDF, does the aggregation, counts — then does the entire thing again for the 
# write. The fix is add .cache() before the count, .unpersist() after the write.

events = spark.table("blog_demo.events")

result = (
    events
    .filter(is_in_fiscal_window(F.col("event_date")))
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

cy.stop().save("/tmp/act3_antipattern_snapshot.json")
cy.display_html()

# COMMAND ----------

import base64
with open("/tmp/act3_antipattern_snapshot.json", "r") as f:
    content = f.read()
b64 = base64.b64encode(content.encode()).decode()
displayHTML(f'<a href="data:application/json;base64,{b64}" download="act3_antipattern_snapshot.json">📥 Download Snapshot</a>')
