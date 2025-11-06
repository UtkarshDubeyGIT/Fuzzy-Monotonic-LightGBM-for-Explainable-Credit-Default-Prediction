from pathlib import Path
import json
import matplotlib.pyplot as plt


def main() -> None:
    ablation_path = Path("results/ablation_table.json")
    with ablation_path.open() as f:
        rows = json.load(f)

    # Preserve order from file
    variants = [r["variant"] for r in rows]
    prauc = [r["pr_auc"] for r in rows]

    plt.figure(figsize=(10, 6))
    # Set y-limits as requested before plotting bars
    plt.ylim(0.53, 0.56)
    plt.bar(variants, prauc)
    plt.ylabel("PR-AUC")
    plt.xlabel("Variant")
    plt.title("Ablation PR-AUC Comparison")
    plt.tight_layout()
    out = Path("results/ablation_pr_auc.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)

    # Print only the saved path
    print(str(out))


if __name__ == "__main__":
    main()
