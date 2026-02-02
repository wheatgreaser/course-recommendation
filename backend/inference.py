# infer.py
import json
import torch
import numpy as np
from pathlib import Path

from robust_fm_recommender import (
    FactorizationMachine,
    build_mappings,
    load_user_input_courses,
    get_branch_filters,
    pick_proxy_user_for_input,
    filter_by_cdcs,
    write_courses_csv,
    load_grades
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path("models")

def main():
    FORCED_COURSES = [
    "PRACTICE SCHOOL I",
    "PRACTICE SCHOOL II"
    ]
    # ---------- Load metadata ----------
    with open(MODEL_DIR / "mappings.json") as f:
        meta = json.load(f)

    num_users = meta["num_users"]
    num_courses = meta["num_courses"]
    idx_to_name = {int(k): v for k, v in meta["idx_to_name"].items()}
    name_to_idx = meta["name_to_idx"]

    # ---------- Load grades and rebuild mappings ----------
    df = load_grades()
    df, _, _, _, _ = build_mappings(df)

    # ---------- Load user input ----------
    input_courses = load_user_input_courses()
    input_names = [c for c, _ in input_courses if c in name_to_idx]

    # ---------- Pick proxy user ----------
    proxy_user_idx = pick_proxy_user_for_input(input_names, df)

    # ---------- Load FM ----------
    fm = FactorizationMachine(
        num_features=num_users + num_courses,
        embedding_dim=32
    ).to(DEVICE)

    fm.load_state_dict(torch.load(MODEL_DIR / "robust_fm.pt", map_location=DEVICE))
    fm.eval()

    # ---------- Predict ----------
    all_items = torch.arange(num_courses, device=DEVICE)
    user_idx = torch.full_like(all_items, proxy_user_idx)
    fm_input = torch.stack([user_idx, all_items + num_users], dim=1)

    with torch.no_grad():
        preds = fm(fm_input).cpu().numpy()

    ranked = sorted(
        [
            (idx_to_name[i], float(preds[i]))
            for i in range(num_courses)
            if idx_to_name[i] not in input_names
        ],
        key=lambda x: x[1],
        reverse=True
    )

    # ---------- Apply branch filters ----------
    be, msc, columns = get_branch_filters()
    filtered = filter_by_cdcs([c for c, _ in ranked], be, msc, columns)

    # ---------- Write output ----------
    # ---------- output.csv ----------
    write_courses_csv(filtered, "output.csv", limit=5)

    # ---------- output_2.csv (Predicted GPA) ----------

    TOP_K = 5

    # Map course → predicted score
    name_to_pred = {
        idx_to_name[i]: float(preds[i])
        for i in range(num_courses)
    }

    # Base GPA courses = top-K recommendations
    gpa_courses = filtered[:TOP_K]

    # Add Practice School courses ONLY for GPA


    # Compute GPA
    pred_vals = [name_to_pred[c] for c in gpa_courses if c in name_to_pred]
    est_gpa = float(np.mean(pred_vals) * 10.0) if pred_vals else 0.0

    with open("output_2.csv", "w", newline="", encoding="utf-8") as f:
        import csv
        csv.writer(f).writerow([f"{est_gpa:.2f}"])


    # ---------- output_3.csv ----------
    write_courses_csv(filtered, "output_3.csv", limit=9)

    # ---------- output_4.csv ----------
    write_courses_csv(filtered, "output_4.csv", limit=9)


    print("Inference complete.")

if __name__ == "__main__":
    main()
