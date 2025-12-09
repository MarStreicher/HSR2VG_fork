import pandas as pd
import matplotlib.pyplot as plt

results_df = pd.read_csv("results/results.csv", sep=";")

# Create a column for the source→target label
results_df["pair"] = results_df["source"] + "→" + results_df["target"]

# List of traits to plot
traits = results_df["label"].unique()

# Iterate over traits and plot one figure per trait
for trait in traits:
    df_trait = results_df[results_df["label"] == trait]

    # Sort for consistency
    df_trait = df_trait.sort_values(by=["pair", "model"])

    # Set up the figure
    plt.figure(figsize=(12, 6))

    # Get unique pairs and models
    pairs = df_trait["pair"].unique()
    models = df_trait["model"].unique()

    # Create grouped bar positions
    x = range(len(pairs))
    bar_width = 0.15
    offsets = [
        (i - len(models) / 2) * bar_width + bar_width / 2 for i in range(len(models))
    ]

    # Plot bars for each model
    for i, model in enumerate(models):
        df_model = df_trait[df_trait["model"] == model]
        plt.bar(
            [p + offsets[i] for p in x],
            df_model["target_r"],
            width=bar_width,
            yerr=df_model["target_r_sd"],
            capsize=3,
            label=model,
        )

    # Styling
    plt.xticks(x, pairs, rotation=45, ha="right")
    plt.ylabel("Target Pearson r")
    plt.xlabel("Source → Target Domain")
    plt.title(f"Model Performance (Trait: {trait})")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Show or save
    plt.show()
    plt.savefig(f"results_trait_{trait}.png", dpi=300, bbox_inches="tight")
