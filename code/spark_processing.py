from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
STUDENTS_PATH = str(OUTPUT_DIR / "students.csv")
INTERACTIONS_PATH = str(OUTPUT_DIR / "interactions.csv")


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("INT315_StudentInteractionAnalysis")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_data(spark):
    students_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(STUDENTS_PATH)
        .withColumn("student_id", F.col("student_id").cast("string"))
        .withColumn("network", F.col("network").cast("string"))
    )

    interactions_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(INTERACTIONS_PATH)
        .withColumn("network", F.col("network").cast("string"))
        .withColumn("source", F.col("source").cast("string"))
        .withColumn("target", F.col("target").cast("string"))
        .withColumn("weight", F.col("weight").cast("int"))
    )

    return students_df, interactions_df


def build_student_activity(students_df, interactions_df):
    incident_activity = (
        interactions_df
        .select(
            F.col("network"),
            F.col("source").alias("student_id"),
            F.col("weight")
        )
        .unionByName(
            interactions_df.select(
                F.col("network"),
                F.col("target").alias("student_id"),
                F.col("weight")
            )
        )
        .groupBy("network", "student_id")
        .agg(
            F.sum("weight").alias("total_weight"),
            F.count("*").alias("interaction_count")
        )
    )

    student_activity_df = (
        students_df
        .join(incident_activity, ["network", "student_id"], "left")
        .fillna({"total_weight": 0, "interaction_count": 0})
        .withColumn("total_weight", F.col("total_weight").cast("int"))
        .withColumn("interaction_count", F.col("interaction_count").cast("int"))
        .orderBy("network", "student_id")
    )

    return student_activity_df


def save_summary_files(student_activity_df, interactions_df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    network_summary_df = (
        interactions_df.groupBy("network")
        .agg(
            F.count("*").alias("interaction_records"),
            F.sum("weight").alias("total_weight")
        )
        .orderBy("network")
    )
    network_summary_df.coalesce(1).write.mode("overwrite").option("header", True).csv(
        str(OUTPUT_DIR / "network_summary")
    )

    most_active_df = (
        student_activity_df
        .orderBy(F.col("interaction_count").desc(), F.col("total_weight").desc(), F.col("student_id"))
        .limit(10)
    )
    most_active_df.toPandas().to_csv(OUTPUT_DIR / "most_active_students.csv", index=False)

    isolated_df = (
        student_activity_df.filter(F.col("interaction_count") == 0)
        .orderBy("network", "student_id")
    )
    isolated_df.toPandas().to_csv(OUTPUT_DIR / "isolated_students.csv", index=False)

    student_activity_df.toPandas().to_csv(OUTPUT_DIR / "student_activity.csv", index=False)

    network_summary_df.show(truncate=False)
    print("\nTop 10 active students:")
    most_active_df.show(truncate=False)
    print("\nIsolated students:")
    isolated_df.show(truncate=False)


def visualize_network(network_name, student_activity_df, interactions_df):
    network_students = (
        student_activity_df.filter(F.col("network") == network_name)
        .select("student_id", "label", "interaction_count")
        .toPandas()
    )

    network_interactions = (
        interactions_df.filter(F.col("network") == network_name)
        .select("source", "target", "weight")
        .toPandas()
    )

    graph = nx.DiGraph()

    for _, row in network_students.iterrows():
        node_id = str(row["student_id"])
        graph.add_node(node_id, label=node_id, degree=int(row["interaction_count"]))

    for _, row in network_interactions.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        weight = int(row["weight"])
        if not graph.has_edge(source, target):
            graph.add_edge(source, target, weight=weight)
        else:
            graph[source][target]["weight"] += weight

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, k=0.9, seed=42)
    node_sizes = [max(50, degree * 20) for _, degree in graph.nodes(data="degree")]
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=node_sizes,
        edge_color="gray",
        alpha=0.8,
        width=1.2,
        font_size=8,
    )

    plt.title(f"Interaction Network: {network_name}")
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"network_{network_name}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved network visualization: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spark = create_spark_session()

    try:
        students_df, interactions_df = load_data(spark)

        print("\n===== STUDENTS =====")
        students_df.show(5, truncate=False)
        print("\n===== INTERACTIONS =====")
        interactions_df.show(5, truncate=False)

        student_activity_df = build_student_activity(students_df, interactions_df)
        save_summary_files(student_activity_df, interactions_df)

        for network_name in ["ICS", "CD", "PS"]:
            visualize_network(network_name, student_activity_df, interactions_df)

        print("\nAnalysis complete. Files saved in:", OUTPUT_DIR)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
