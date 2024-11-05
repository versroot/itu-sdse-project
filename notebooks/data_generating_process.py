import pandas as pd
import numpy as np
import string


def generate_data(n_rows=1_000, seed=42):
    np.random.seed(seed)

    df = pd.DataFrame.from_dict(
        {
            "lead_id": list(range(n_rows)),
            "lead_indicator": np.random.choice([0, 1, ""], size=n_rows, p=[0.5, 0.45, 0.05]),
            "date_part": np.random.choice([f"2024-01-{int(x)}" for x in range(1, 32)], size=n_rows),
            "is_active": np.random.choice([0, 1], size=n_rows),
            "marketing_consent": np.random.choice(["true", "false"], size=n_rows, p=[0.7, 0.3]),
            "first_booking": np.random.choice([f"2024-01-{int(x)}" for x in range(1, 32)], size=n_rows),
            "existing_customer": np.random.choice(["true", "false"], size=n_rows),
            "last_seen": np.random.choice([f"2024-01-{int(x)}" for x in range(1, 32)], size=n_rows),
            "source": np.random.choice(["signup", "fb", "organic", "li"], size=n_rows),
            "domain": np.random.choice([".com", ".cn", ".dk"], size=n_rows),
            "country": np.random.choice(["US", "CN", "DK"], size=n_rows),
            "visited_learn_more_before_booking": np.random.negative_binomial(1, 0.2, size=n_rows),
            "visited_faq": np.random.negative_binomial(1, 0.2, size=n_rows),
            "purchases": np.random.poisson(5, size=n_rows),
            "time_spent": np.random.normal(100, 10, size=n_rows),
            "customer_group": np.random.randint(1, 10, size=n_rows),
            "onboarding": np.random.choice([True, False], size=n_rows),
            "customer_code": [
                "".join(np.random.choice([x for x in string.ascii_uppercase], size=10)) 
                if idx%(n_rows//50)!=0 else "" for idx in range(n_rows)
            ]
        }
    )
    df["n_visits"] = (
        np.random.negative_binomial(1, 0.2, size=n_rows) 
        + np.random.randint(3, 15, n_rows) * df["lead_indicator"].replace({"": 0}).astype(int)
    )

    return df

