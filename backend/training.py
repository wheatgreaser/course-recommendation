# train.py
import torch
import json
import copy
from pathlib import Path

# import everything you already defined
from robust_fm_recommender import (
    FactorizationMachine,
    AdversarialAutoencoder,
    load_grades,
    build_mappings,
    build_user_item_matrix,
    build_fm_dataset_from_df,
    train_fm,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def main():
    df = load_grades()
    df, num_users, num_courses, idx_to_name, name_to_idx = build_mappings(df)

    user_item = build_user_item_matrix(df, num_users, num_courses)
    fm_dataset = build_fm_dataset_from_df(df, num_users)

    # ---------- Train FM ----------
    fm = FactorizationMachine(
        num_features=num_users + num_courses,
        embedding_dim=32
    ).to(DEVICE)

    train_fm(fm, fm_dataset, epochs=5)

    # ---------- Train AAE ----------
    LATENT_DIM = 20
    aae = AdversarialAutoencoder(
        input_dim=num_courses,
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    # (reuse your existing AAE training loop here verbatim)

    # ---------- Adversarial game ----------
    robust_fm = copy.deepcopy(fm)

    # (reuse your adversarial EMPSO loop verbatim)

    # ---------- Save everything ----------
    torch.save(robust_fm.state_dict(), MODEL_DIR / "robust_fm.pt")
    torch.save(aae.state_dict(), MODEL_DIR / "aae.pt")

    with open(MODEL_DIR / "mappings.json", "w") as f:
        json.dump({
            "num_users": num_users,
            "num_courses": num_courses,
            "idx_to_name": idx_to_name,
            "name_to_idx": name_to_idx
        }, f)

    print("Training complete. Models saved.")

if __name__ == "__main__":
    main()
