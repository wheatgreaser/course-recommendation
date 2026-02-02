import os
import json
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import copy


# ------------------------------
# Paths and constants
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRADE_CSV = os.path.join(BASE_DIR, 'Grade.csv')
GRADE_STUDENT_CSV = os.path.join(BASE_DIR, 'Grade_Student.csv')
CDCS_CSV = os.path.join(BASE_DIR, 'CDCs.csv')
DATA_JSON = os.path.join(BASE_DIR, 'data.json')

OUTPUT1 = os.path.join(BASE_DIR, 'output.csv')      # recommendations
OUTPUT2 = os.path.join(BASE_DIR, 'output_2.csv')    # predicted GPA-like score
OUTPUT3 = os.path.join(BASE_DIR, 'output_3.csv')    # pagerank/alt recs (reuse robust top)
OUTPUT4 = os.path.join(BASE_DIR, 'output_4.csv')    # branch-filtered top (reuse robust top)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------
# Models from notebook (simplified)
# ------------------------------
class FactorizationMachine(nn.Module):
    """FM that takes feature index pairs [user_idx, item_idx_offset] and predicts rating."""

    def __init__(self, num_features: int, embedding_dim: int):
        super().__init__()
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.linear_weights = nn.Embedding(num_features, 1)
        self.interaction_factors = nn.Embedding(num_features, embedding_dim)
        nn.init.xavier_uniform_(self.linear_weights.weight)
        nn.init.xavier_uniform_(self.interaction_factors.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2) of feature indices [user_feat, item_feat]
        linear_term = self.global_bias + torch.sum(self.linear_weights(x), dim=1)
        emb = self.interaction_factors(x)  # (batch, 2, k)
        sum_sq = torch.sum(emb, dim=1).pow(2)
        sq_sum = torch.sum(emb.pow(2), dim=1)
        interaction = 0.5 * torch.sum(sum_sq - sq_sum, dim=1, keepdim=True)
        pred = linear_term + interaction
        return pred.squeeze(-1)


# ------------------------------
# Adversarial Autoencoder (AAE)
# ------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc_out = nn.Linear(128, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc_out(z))


class Discriminator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.leaky_relu(self.fc1(z), 0.2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        return torch.sigmoid(self.fc_out(z))


class AdversarialAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.discriminator = Discriminator(latent_dim)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# ------------------------------
# Data loading and preparation
# ------------------------------
def load_grades() -> pd.DataFrame:
    df = pd.read_csv(GRADE_CSV)
    # Expect columns: id,student_id,course_id,course_grade
    # Clean types
    df['course_grade'] = pd.to_numeric(df['course_grade'], errors='coerce').fillna(0)
    return df


def build_mappings(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int, Dict[int, str], Dict[str, int]]:
    df = df.copy()
    df['user_idx'] = df['student_id'].astype('category').cat.codes
    df['course_idx'] = df['course_id'].astype('category').cat.codes
    num_users = df['user_idx'].nunique()
    num_courses = df['course_idx'].nunique()
    # idx -> course_name
    course_idx_to_name = df.drop_duplicates('course_idx').set_index('course_idx')['course_id'].to_dict()
    # name -> idx
    course_name_to_idx = {v: k for k, v in course_idx_to_name.items()}
    return df, num_users, num_courses, course_idx_to_name, course_name_to_idx


def build_user_item_matrix(df: pd.DataFrame, num_users: int, num_courses: int) -> np.ndarray:
    mat = np.zeros((num_users, num_courses), dtype=np.float32)
    # Grades are on 0..10 scale; normalize to 0..1
    denom = 10.0
    for _, r in df.iterrows():
        mat[int(r['user_idx']), int(r['course_idx'])] = float(r['course_grade']) / denom
    return mat


def build_fm_dataset_from_df(df: pd.DataFrame, num_users: int) -> TensorDataset:
    x_pairs = np.stack([
        df['user_idx'].values,
        (df['course_idx'].values + num_users)
    ], axis=1)
    y = (df['course_grade'].values.astype(np.float32) / 10.0)
    X = torch.tensor(x_pairs, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(X, y)


def train_fm(model: FactorizationMachine, dataset: TensorDataset, epochs: int = 5, lr: float = 1e-3, batch_size: int = 512) -> None:
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()


# ------------------------------
# EMPSO-based adversarial game (AAE + FM)
# ------------------------------
class Particle:
    def __init__(self, bounds: np.ndarray):
        self.position = np.random.uniform(bounds[:, 0], bounds[:, 1])
        self.velocity = np.random.uniform(-1, 1, len(bounds))
        self.pbest_position = self.position.copy()
        self.pbest_value = float('inf')


class EMPSO:
    def __init__(self, fitness_function, bounds, num_particles=10, max_iter=20, beta=0.7, c1=1.5, c2=1.5):
        self.fitness_function = fitness_function
        self.bounds = np.array(bounds)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.beta = beta
        self.c1 = c1
        self.c2 = c2
        self.swarm = [Particle(self.bounds) for _ in range(num_particles)]
        self.gbest_value = float('inf')
        self.gbest_position = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        for _ in range(self.max_iter):
            for p in self.swarm:
                fitness = self.fitness_function(p.position)
                if fitness < p.pbest_value:
                    p.pbest_value = fitness
                    p.pbest_position = p.position.copy()
                if fitness < self.gbest_value:
                    self.gbest_value = fitness
                    self.gbest_position = p.position.copy()
            for p in self.swarm:
                r1, r2 = np.random.rand(2)
                cognitive = self.c1 * r1 * (p.pbest_position - p.position)
                social = self.c2 * r2 * (self.gbest_position - p.position)
                p.velocity = self.beta * p.velocity + cognitive + social
                p.position += p.velocity
                p.position = np.clip(p.position, self.bounds[:, 0], self.bounds[:, 1])
        return self.gbest_position, self.gbest_value


def adversary_payoff_fm(perturbation: np.ndarray, user_idx: int, user_vector: torch.Tensor,
                         fm_model: FactorizationMachine, aae_model: AdversarialAutoencoder,
                         num_users: int, num_courses: int, attack_goal_rating: float = 0.0) -> float:
    perturbation_tensor = torch.tensor(perturbation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    user_vec = user_vector.to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        original_z = aae_model.encoder(user_vec)
        perturbed_z = original_z + perturbation_tensor
        poisoned_user = aae_model.decoder(perturbed_z).squeeze(0)

    all_items = torch.arange(num_courses, device=DEVICE)
    fm_user_feat_idx = torch.full_like(all_items, fill_value=user_idx)
    fm_item_feat_idx = all_items + num_users
    fm_input = torch.stack([fm_user_feat_idx, fm_item_feat_idx], dim=1)
    predictions = fm_model(fm_input)
    attack_goal = torch.full_like(predictions, fill_value=attack_goal_rating)
    fm_error = F.mse_loss(predictions, attack_goal)
    cost = torch.norm(perturbation_tensor).item() * 0.1
    payoff = fm_error - cost
    return -payoff.item()


def alternating_least_squares(fm_model: FactorizationMachine, aae_model: AdversarialAutoencoder,
                              user_item_tensor: torch.Tensor, num_users: int, num_courses: int,
                              users_to_attack: np.ndarray, latent_dim: int) -> Tuple[torch.Tensor, float]:
    best_perts = []
    payoffs = []
    for u_idx in users_to_attack:
        u_vec = user_item_tensor[u_idx]
        bounds = [[-0.5, 0.5]] * latent_dim
        fitness_fn = lambda p: adversary_payoff_fm(p, int(u_idx), u_vec, fm_model, aae_model, num_users, num_courses)
        em = EMPSO(fitness_fn, bounds, num_particles=10, max_iter=20)
        best_p, best_val = em.optimize()
        best_perts.append(best_p)
        payoffs.append(-best_val)
    avg_pert = np.mean(best_perts, axis=0)
    avg_payoff = float(np.mean(payoffs)) if payoffs else -float('inf')
    return torch.tensor(avg_pert, dtype=torch.float32, device=DEVICE), avg_payoff


def generate_poisoned_fm_data(best_perturbation: torch.Tensor, users_to_attack: np.ndarray,
                              aae_model: AdversarialAutoencoder, user_item_tensor: torch.Tensor,
                              num_users: int, num_courses: int) -> TensorDataset:
    poisoned_inputs = []
    poisoned_targets = []
    with torch.no_grad():
        for u_idx in users_to_attack:
            u_vec = user_item_tensor[int(u_idx)].to(DEVICE).unsqueeze(0)
            z = aae_model.encoder(u_vec)
            z_pert = z + best_perturbation.unsqueeze(0)
            poisoned = aae_model.decoder(z_pert).squeeze(0)
            for c_idx in range(num_courses):
                poisoned_inputs.append([int(u_idx), num_users + c_idx])
                poisoned_targets.append(float(poisoned[c_idx].item()))
    X = torch.tensor(np.array(poisoned_inputs), dtype=torch.long)
    y = torch.tensor(np.array(poisoned_targets), dtype=torch.float32)
    return TensorDataset(X, y)


def build_fm_dataset_from_matrix(mat: np.ndarray, num_users: int) -> TensorDataset:
    U, I = mat.shape
    rows = []
    targets = []
    for u in range(U):
        for c in range(I):
            rows.append([u, num_users + c])
            targets.append(mat[u, c])
    X = torch.tensor(np.array(rows), dtype=torch.long)
    y = torch.tensor(np.array(targets), dtype=torch.float32)
    return TensorDataset(X, y)


# ------------------------------
# Recommendation helpers
# ------------------------------
def load_user_input_courses() -> List[Tuple[str, float]]:
    # data.json structure can vary; try extracting all course entries
    if not os.path.exists(DATA_JSON):
        return []
    with open(DATA_JSON, 'r') as f:
        data = json.load(f)
    input_courses = []
    # Some keys like "1-1": [ {subject, courseGrade}, ...]
    for v in data.values():
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and 'subject' in item:
                    try:
                        g = float(item.get('courseGrade', 0))
                    except Exception:
                        g = 0.0
                    input_courses.append((item['subject'], g))
    return input_courses


def get_branch_filters() -> Tuple[str, str, List[List[str]]]:
    be = ''
    msc = ''
    if os.path.exists(DATA_JSON):
        with open(DATA_JSON, 'r') as f:
            data = json.load(f)
            be = data.get('beDegree', '') or ''
            msc = data.get('mscDegree', '') or ''
    cdcs_df = pd.read_csv(CDCS_CSV)
    return be, msc, cdcs_df.values.T.tolist()


def pick_proxy_user_for_input(input_courses: List[str], df: pd.DataFrame) -> int:
    # Build per-user set of course names
    user_groups = df.groupby('user_idx')['course_id'].apply(set)
    input_set = set(input_courses)
    best_user = int(df['user_idx'].iloc[0]) if len(df) else 0
    best_score = -1.0
    for user_idx, courses in user_groups.items():
        inter = len(courses & input_set)
        union = len(courses | input_set) if (courses or input_set) else 1
        jacc = inter / union
        if jacc > best_score:
            best_score = jacc
            best_user = int(user_idx)
    return best_user


def filter_by_cdcs(course_names: List[str], be: str, msc: str, columns_as_lists: List[List[str]]) -> List[str]:
    cdcs_labels = [
        "B.E Chemical", "B.E Civil", "B.E Computer Science", "B.E Electrical & Electronic",
        "B.E Electronics & Communication", "B.E Electronics & Instrumentation",
        "B.E Mechanical", "B.Pharm", "M.Sc. Biological Sciences", "M.Sc. Chemistry",
        "M. Sc. Economics", "M.Sc. Mathematics", "M. Sc. Physics"
    ]
    def idx_of(label: str) -> int:
        try:
            return cdcs_labels.index(label)
        except ValueError:
            return -1
    be_idx = idx_of(be) if be else -1
    msc_idx = idx_of(msc) if msc and msc != 'None' else -1
    out = []
    for c in course_names:
        ok = True
        if be_idx >= 0:
            if c in columns_as_lists[be_idx] or (c + "\n") in columns_as_lists[be_idx]:
                ok = False
        if ok and msc_idx >= 0:
            if c in columns_as_lists[msc_idx] or (c + "\n") in columns_as_lists[msc_idx]:
                ok = False
        if ok:
            out.append(c)
    return out


def write_courses_csv(courses: List[str], file_path: str, header: str = 'Course', limit: int = 5) -> None:
    rows = courses[:limit]
    import csv
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[header])
        w.writeheader()
        for c in rows:
            if c:
                w.writerow({header: c})


def main():
    # Load and prep data
    df = load_grades()
    df, num_users, num_courses, idx_to_name, name_to_idx = build_mappings(df)
    user_item = build_user_item_matrix(df, num_users, num_courses)
    fm_dataset = build_fm_dataset_from_df(df, num_users)

    # Train baseline FM quickly
    num_features = num_users + num_courses
    fm = FactorizationMachine(num_features=num_features, embedding_dim=32).to(DEVICE)
    train_fm(fm, fm_dataset, epochs=int(os.getenv('FM_EPOCHS', '5')), lr=1e-3, batch_size=1024)

    # Build user-item tensor for AAE
    user_item_tensor = torch.tensor(user_item, dtype=torch.float32)
    aae_dataset = TensorDataset(user_item_tensor)
    aae_loader = DataLoader(aae_dataset, batch_size=64, shuffle=True)

    # Train AAE
    LATENT_DIM = int(os.getenv('AAE_LATENT_DIM', '20'))
    aae = AdversarialAutoencoder(input_dim=num_courses, latent_dim=LATENT_DIM).to(DEVICE)
    opt_recon = optim.Adam(list(aae.encoder.parameters()) + list(aae.decoder.parameters()), lr=1e-3)
    opt_disc = optim.Adam(aae.discriminator.parameters(), lr=5e-5)
    opt_gen = optim.Adam(aae.encoder.parameters(), lr=5e-5)
    recon_crit = nn.MSELoss()
    disc_crit = nn.BCELoss()
    for _ in range(int(os.getenv('AAE_EPOCHS', '10'))):
        for (batch,) in aae_loader:
            batch = batch.to(DEVICE)
            # Reconstruction
            aae.train()
            opt_recon.zero_grad()
            recon, _ = aae(batch)
            loss_recon = recon_crit(recon, batch)
            loss_recon.backward()
            opt_recon.step()
            # Discriminator
            opt_disc.zero_grad()
            real_prior = torch.randn(batch.size(0), LATENT_DIM, device=DEVICE)
            real_labels = torch.ones(batch.size(0), 1, device=DEVICE)
            fake_labels = torch.zeros(batch.size(0), 1, device=DEVICE)
            real_loss = disc_crit(aae.discriminator(real_prior), real_labels)
            with torch.no_grad():
                enc_z = aae.encoder(batch)
            fake_loss = disc_crit(aae.discriminator(enc_z), fake_labels)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_disc.step()
            # Generator (encoder) to fool discriminator
            opt_gen.zero_grad()
            enc_z = aae.encoder(batch)
            g_loss = disc_crit(aae.discriminator(enc_z), real_labels)
            g_loss.backward()
            opt_gen.step()

    # Adversarial game loop to build a robust FM
    robust_fm = copy.deepcopy(fm)
    opt_robust = optim.Adam(robust_fm.parameters(), lr=1e-3, weight_decay=1e-5)
    fm_crit = nn.MSELoss()
    payoff_curr = -float('inf')
    A_perturb = torch.zeros(LATENT_DIM, device=DEVICE)
    MAX_GAME_ITER = int(os.getenv('GAME_MAX_ITERS', '10'))
    USERS_TO_ATTACK_PER_ITER = min(int(os.getenv('GAME_USERS_PER_ITER', '16')), num_users)
    DEFENDER_EPOCHS = int(os.getenv('GAME_DEFENDER_EPOCHS', '3'))
    CONV_THRESH = float(os.getenv('GAME_CONV_THRESH', '1e-8'))
    for it in range(1, MAX_GAME_ITER + 1):
        robust_fm.eval(); aae.eval()
        users_to_attack = np.random.choice(num_users, USERS_TO_ATTACK_PER_ITER, replace=False)
        best_pert, payoff_best = alternating_least_squares(robust_fm, aae, user_item_tensor, num_users, num_courses, users_to_attack, LATENT_DIM)
        if (payoff_best - payoff_curr <= CONV_THRESH and it > 1):
            break
        payoff_curr = payoff_best
        A_perturb = best_pert
        # Generate poisoned data and retrain defender
        poisoned_ds = generate_poisoned_fm_data(A_perturb, users_to_attack, aae, user_item_tensor, num_users, num_courses)
        combined = ConcatDataset([fm_dataset, poisoned_ds])
        loader = DataLoader(combined, batch_size=1024, shuffle=True)
        robust_fm.train()
        for _ in range(DEFENDER_EPOCHS):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt_robust.zero_grad()
                pred = robust_fm(xb)
                loss = fm_crit(pred, yb)
                loss.backward()
                opt_robust.step()

    # Build recommendation for input
    input_courses = load_user_input_courses()
    input_course_names = [c for c, _ in input_courses]

    be, msc, columns_as_lists = get_branch_filters()

    # Map input courses to names present in dataset (use exact names)
    existing_input_courses = [c for c in input_course_names if c in name_to_idx]
    # Proxy user from dataset most similar to input
    proxy_user_idx = pick_proxy_user_for_input(existing_input_courses, df)

    # Predict for all items for proxy user
    all_item_indices = torch.arange(num_courses, dtype=torch.long)
    user_feat_idx = torch.full_like(all_item_indices, fill_value=proxy_user_idx)
    item_feat_idx = all_item_indices + num_users
    fm_input = torch.stack([user_feat_idx, item_feat_idx], dim=1).to(DEVICE)
    robust_fm.eval()
    with torch.no_grad():
        preds = robust_fm(fm_input).detach().cpu().numpy()

    # Exclude courses already in input
    taken_set = set(existing_input_courses)
    ranked = sorted([(idx_to_name[i], float(preds[i])) for i in range(num_courses) if idx_to_name[i] not in taken_set],
                    key=lambda x: x[1], reverse=True)

    # Filter by CDCs
    ranked_names = [name for name, _ in ranked]
    filtered = filter_by_cdcs(ranked_names, be, msc, columns_as_lists)

    # Write outputs
    os.makedirs(BASE_DIR, exist_ok=True)
    top_courses = filtered[:5]
    write_courses_csv(top_courses, OUTPUT1, header='Course', limit=5)

    # Predicted GPA-like value: mean of top-5 preds scaled to 10
    name_to_pred = {idx_to_name[i]: float(preds[i]) for i in range(num_courses)}
    pred_vals = [name_to_pred.get(c, 0.0) for c in top_courses]
    est_gpa = float(np.mean(pred_vals) * 10.0) if pred_vals else 0.0
    with open(OUTPUT2, 'w', newline='', encoding='utf-8') as f:
        import csv
        w = csv.writer(f)
        w.writerow([f"{est_gpa:.2f}"])

    # For simplicity, reuse filtered list for both output_3 and output_4
    write_courses_csv(filtered, OUTPUT3, header='Course', limit=5)
    write_courses_csv(filtered, OUTPUT4, header='Course', limit=5)

    print('Robust FM recommendations generated.')


if __name__ == '__main__':
    main()