from dotenv import load_dotenv
import wandb
from random import randint
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
import os

import argparsing
from hsr_data import HsrData

ENV_PATH = ".env"

model_cls, config = argparsing.get_model_and_config()

source_data = HsrData(config.source_domain, config.label)
target_data = HsrData(config.target_domain, config.label)

source_na_samples = []
target_na_samples = []

if source_data.labels.isna().any():
    source_na_samples = source_data.labels[source_data.labels.isna()].index.to_list()
    warnings.warn(
        f"NA values in source data: {len(source_na_samples)}, Samples were removed."
    )
    source_data.feature_matrix = source_data.feature_matrix.drop(source_na_samples)
    source_data.labels = source_data.labels.drop(source_na_samples)

if target_data.labels.isna().any():
    target_na_samples = target_data.labels[target_data.labels.isna()].index.to_list()
    warnings.warn(
        f"NA values in target data: {len(target_na_samples)}, Samples were removed."
    )
    target_data.feature_matrix = target_data.feature_matrix.drop(target_na_samples)
    target_data.labels = target_data.labels.drop(target_na_samples)

seeds = [i for i in range(0, 100, 10)]
if config.sweep:
    seeds = [randint(1, 100)]

try:
    source_r_values = []
    source_r2_values = []
    target_r_values = []
    target_r2_values = []
    for seed in seeds:
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in cv.split(source_data.feature_matrix):

            Xs_train = source_data.feature_matrix.iloc[train_idx]
            ys_train = source_data.labels.iloc[train_idx]

            Xs_test = source_data.feature_matrix.iloc[test_idx]
            ys_test = source_data.labels.iloc[test_idx]

            scaler_X = StandardScaler()
            Xs_train_scaled = scaler_X.fit_transform(Xs_train)
            Xs_test_scaled = scaler_X.transform(Xs_test)

            scaler_y = StandardScaler()
            ys_train_scaled = scaler_y.fit_transform(
                ys_train.to_numpy().reshape(-1, 1)
            ).ravel()
            ys_test_scaled = scaler_y.transform(
                ys_test.to_numpy().reshape(-1, 1)
            ).ravel()

            target_scaler_X = StandardScaler()
            Xt_scaled = target_scaler_X.fit_transform(target_data.feature_matrix)
            target_scaler_y = StandardScaler()
            yt_scaled = target_scaler_y.fit_transform(
                target_data.labels.to_numpy().reshape(-1, 1)
            ).ravel()

            model = model_cls(config)
            model.seed = seed
            model.train(
                Xs_train_scaled,
                ys_train_scaled,
                Xt_scaled,
            )

            source_r2, source_r = model.validate(
                "src",
                Xs_test_scaled,
                ys_test_scaled,
            )
            target_r2, target_r = model.validate("tgt", Xt_scaled, yt_scaled)

            source_r_values.append(source_r)
            source_r2_values.append(source_r2)
            target_r_values.append(target_r)
            target_r2_values.append(target_r2)

    if config.use_csv:
        results_path = "results/results.csv"
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path, sep=";")
        else:
            results_df = pd.DataFrame(
                columns=[
                    "label",
                    "source",
                    "target",
                    "model",
                    "source_r2",
                    "source_r2_sd",
                    "source_r",
                    "source_r_sd",
                    "target_r2",
                    "target_r2_sd",
                    "target_r",
                    "target_r_sd",
                ]
            )

        new_row = {
            "label": config.label,
            "source": config.source_domain,
            "target": config.target_domain,
            "model": config.model,
            "source_r2": np.mean(source_r2_values),
            "source_r2_sd": np.std(source_r2_values),
            "source_r": np.mean(source_r_values),
            "source_r_sd": np.std(source_r_values),
            "target_r2": np.mean(target_r2_values),
            "target_r2_sd": np.std(target_r2_values),
            "target_r": np.mean(target_r_values),
            "target_r_sd": np.std(target_r_values),
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        results_df.to_csv("results/results.csv", sep=";", index=False)

    if config.use_wandb:
        load_dotenv(ENV_PATH)
        if config.sweep:
            run_name = f"{config.model}_sweep"
        else:
            run_name = f"{config.model}_{config.source_domain}_{config.target_domain}"

        run = wandb.init(
            project="hsr",
            entity="marleen-streicher",
            config=config,
            name=run_name,
            save_code=True,
            mode="online",
        )

        logs = {
            "source_r2": np.mean(source_r2_values),
            "source_r": np.mean(source_r_values),
            "target_r2": np.mean(target_r2_values),
            "target_r": np.mean(target_r_values),
        }

        wandb.log(logs)
        data = [
            ["source_r2", np.mean(source_r2_values), np.std(source_r2_values)],
            ["source_r", np.mean(source_r_values), np.std(source_r_values)],
            ["target_r2", np.mean(target_r2_values), np.std(target_r2_values)],
            ["target_r", np.mean(target_r_values), np.std(target_r_values)],
        ]
        log_table = wandb.Table(columns=["metric", "mean", "sd"], data=data)
        run.log({"Logs Table": log_table})
        run.finish()

except Exception as e:
    print(f"Script failed: {e}")
    raise
