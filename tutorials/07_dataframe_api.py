"""
Tutorial 07: Pandas-Compatible DataFrame API
============================================
This tutorial demonstrates the new high-performance, Pandas-compatible
DataFrame API introduced in CorePy v0.3.0. These operations are executed
in the Rust backend for memory safety and zero-copy performance.
"""

import os

import corepy as cp


def main():
    # 1. Loading data
    print("--- 1. Loading CSV Data ---")

    # Let's create a temporary CSV for demonstration
    csv_content = """id,category_id,price,qty
1,10,599.99,10
2,20,15.50,150
3,10,299.00,5
4,30,45.00,20
5,20,22.00,30
"""
    with open("temp_sales.csv", "w") as f:
        f.write(csv_content)

    # Read the CSV directly into a CorePy DataFrame
    print("Loading 'temp_sales.csv'...")
    sales_df = cp.read_csv("temp_sales.csv")
    print(sales_df)

    # 2. DataFrame GroupBy
    print("\n--- 2. GroupBy Aggregation ---")
    # We can group by category and compute the mean price for each
    grouped = sales_df.groupby("category_id")
    summary = grouped.agg({"price": "mean"})
    print("Average price per category_id:")
    print(summary)

    # 3. Merging (Joining) DataFrames
    print("\n--- 3. Relational Merging ---")

    # Creating a secondary dataframe for categories
    cat_df = cp.DataFrame()
    cat_df.add_int_column("category_id", [10, 20, 30, 40])
    cat_df.add_int_column("tax_rate", [15, 5, 8, 10])

    print("Category Tax Rates DataFrame:")
    print(cat_df)

    # Merge them together on the 'category_id' column
    # CorePy supports left, right, inner, and outer joins
    print("\nLeft join sales_df with cat_df:")
    merged_df = sales_df.merge(
        cat_df, left_on="category_id", right_on="category_id", how="left"
    )
    print(merged_df)

    # 4. Pivoting Data
    print("\n--- 4. Data Pivoting ---")
    # Group by id (index), Pivot on category_id (columns), Aggregate on price
    pivoted = sales_df.pivot(index="id", columns="category_id", values="price")
    print("Pivoted DataFrame:")
    print(pivoted)

    # Clean up
    if os.path.exists("temp_sales.csv"):
        os.remove("temp_sales.csv")


if __name__ == "__main__":
    main()
