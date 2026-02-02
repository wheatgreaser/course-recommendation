import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Grade_Student.csv", encoding="latin1")
df.columns = df.columns.str.strip()

df["GRADE NUMBER"] = pd.to_numeric(df["GRADE NUMBER"], errors="coerce")
df["academic_year"] = df["SEMETSER"].str.extract(r"(\d{4}-\d{4})")

yearly_summary = (
    df.groupby("academic_year")["GRADE NUMBER"]
      .agg(mean="mean", std="std", count="count")
      .sort_index()
)

print(yearly_summary)

yearly_summary["mean"].plot(marker="o")
plt.title("Average Grade per Academic Year")
plt.ylabel("Average Grade")
plt.grid(True)
plt.tight_layout()
plt.show()
