import io, os, pandas as pd
from PIL import Image
import streamlit as st
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

st.set_page_config(layout="wide")
spark = SparkSession.builder.getOrCreate()
TABLE = os.getenv("ER_TABLE", "roboticscatalog.er_review.er_readings")

st.title("ER Review App")

df = spark.table(TABLE)
pdf = df.orderBy(F.col("er_name")).limit(1000).toPandas()
if pdf.empty:
    st.info("No rows in table"); st.stop()

row_idx = st.selectbox(
    "Pick a row",
    options=list(pdf.index),
    format_func=lambda i: f"{pdf.loc[i,'er_name']} | {pdf.loc[i,'photo_volume_path']}"
)
rec = pdf.loc[row_idx]

c1, c2 = st.columns([1,2])
with c1:
    st.write({
        "er_name": rec["er_name"],
        "er_value": rec["er_value"],
        "er_angle": rec["er_angle"],
        "er_confidence_score": rec["er_confidence_score"],
        "reviewer_score": None if pd.isna(rec["reviewer_score"]) else float(rec["reviewer_score"]),
        "reviewer_comment": "" if pd.isna(rec["reviewer_comment"]) else str(rec["reviewer_comment"]),
    })

with c2:
    p = rec["photo_volume_path"]
    try:
        with open(p, "rb") as f:
            img = Image.open(io.BytesIO(f.read()))
            st.image(img, caption=p, use_column_width=True)
    except Exception as e:
        st.error(f"Cannot open image: {e}")

st.divider()
score_default = float(rec["reviewer_score"]) if pd.notna(rec["reviewer_score"]) else 1.0
comment_default = "" if pd.isna(rec["reviewer_comment"]) else str(rec["reviewer_comment"])
new_score = st.slider("Reviewer score (0–1)", 0.0, 1.0, score_default, 0.01)
new_comment = st.text_area("Reviewer comment", value=comment_default, height=120)

if st.button("Save", type="primary"):
    esc_comment = new_comment.replace("'", "''")
    esc_name = str(rec["er_name"]).replace("'", "''")
    esc_path = str(rec["photo_volume_path"]).replace("'", "''")
    spark.sql(f"""
        UPDATE {TABLE}
        SET reviewer_score = {new_score},
            reviewer_comment = '{esc_comment}'
        WHERE er_name = '{esc_name}'
          AND photo_volume_path = '{esc_path}'
    """)
    st.success("Saved")