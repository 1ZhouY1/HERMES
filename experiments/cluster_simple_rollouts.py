"""Cluster collected SimpleAgent rollouts with GMM and a DP Gaussian mixture."""

import argparse
import csv
import json
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_hex
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


EXPERIMENTS_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=("meta-world", "halfcheetah"))
    parser.add_argument("--input-dir", type=Path,
                        help="Default: experiments/datasets/simple_rollouts/<domain>")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: experiments/analysis/simple_rollout_clusters/<domain>")
    parser.add_argument("--tasks", type=int, nargs="+", help="Task IDs to include; default: all files")
    parser.add_argument("--scope", choices=("per-task", "pooled"), default="pooled")
    parser.add_argument("--features", choices=("state", "state-action", "action"),
                        default="state-action")
    parser.add_argument("--gmm-components", type=int, nargs="+", default=[4, 5, 8, 10])
    parser.add_argument("--dp-max-components", type=int, default=20,
                        help="Truncation limit for the Dirichlet-process mixture")
    parser.add_argument("--dp-concentration", type=float, default=1.0)
    parser.add_argument("--covariance-type", choices=("diag", "full"), default="diag")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--max-fit-samples", type=int, default=0,
                        help="Maximum rows used to fit each model (default: 0, all rows)")
    parser.add_argument("--plot-max-points", type=int, default=3_000,
                        help="Maximum points in each plot; all rows remain in the saved labels")
    parser.add_argument("--plot-dim", choices=("2d", "3d", "both"), default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.gmm_components or any(k < 1 for k in args.gmm_components):
        parser.error("--gmm-components must contain positive integers")
    if len(set(args.gmm_components)) != len(args.gmm_components):
        parser.error("--gmm-components contains duplicates")
    if args.dp_max_components < 1 or args.dp_concentration <= 0:
        parser.error("--dp-max-components and --dp-concentration must be positive")
    if args.max_iter < 1 or args.max_fit_samples < 0 or args.plot_max_points < 1:
        parser.error("--max-iter and --plot-max-points must be positive; --max-fit-samples must be non-negative")
    if args.tasks is not None and (len(set(args.tasks)) != len(args.tasks) or any(t < 0 for t in args.tasks)):
        parser.error("--tasks must contain distinct non-negative IDs")
    if args.plot_dim != "2d" and go is None:
        parser.error("Interactive 3D plots require Plotly. Install it with "
                     "python -m pip install 'plotly>=5.20,<8', or use --plot-dim 2d")
    return args


def load_rollout(path, domain, feature_mode):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata["domain"] != domain:
            raise ValueError(f"{path}: dataset domain {metadata['domain']} is not {domain}")
        states = np.asarray(data["states"])
        actions = np.asarray(data["actions"])
    if states.ndim != 2 or actions.ndim != 2 or len(states) != len(actions) or len(states) == 0:
        raise ValueError(f"{path}: states/actions must be nonempty aligned 2D arrays")
    if feature_mode == "state":
        features = states
    elif feature_mode == "action":
        features = actions
    else:
        features = np.concatenate((states, actions), axis=1)
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: clustering features contain NaN or infinity")
    return {
        "path": path,
        "task_id": int(metadata["task_id"]),
        "task_name": metadata["task_name"],
        "features": np.asarray(features, dtype=np.float64),
    }


def prepare_groups(args, input_dir):
    paths = sorted(input_dir.glob("task_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No rollout files matching task_*.npz in {input_dir}")
    datasets = [load_rollout(path, args.domain, args.features) for path in paths]
    if len({item["task_id"] for item in datasets}) != len(datasets):
        raise ValueError("Multiple rollout files have the same task ID")
    if args.tasks is not None:
        wanted = set(args.tasks)
        missing = wanted - {item["task_id"] for item in datasets}
        if missing:
            raise FileNotFoundError(f"No rollout files for task IDs {sorted(missing)}")
        datasets = [item for item in datasets if item["task_id"] in wanted]
    if args.scope == "per-task":
        return [(item["path"].stem, [item]) for item in datasets]
    dimensions = {item["features"].shape[1] for item in datasets}
    if len(dimensions) != 1:
        raise ValueError(f"Cannot pool datasets with different feature dimensions: {sorted(dimensions)}")
    return [("pooled", datasets)]


def save_plot(path, projection, labels, indices, variance, title, dimensions):
    unique, counts = np.unique(labels, return_counts=True)
    sampled_labels = labels[indices]
    colors = np.searchsorted(unique, sampled_labels)
    cmap = ListedColormap(category_colors(len(unique)))
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    if dimensions == 3:
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(projection[indices, 0], projection[indices, 1],
                             projection[indices, 2], c=colors, cmap=cmap,
                             vmin=-0.5, vmax=len(unique) - 0.5, s=6, alpha=0.7)
        ax.set_zlabel(f"PC3 ({variance[2]:.1%})")
        ax.view_init(elev=24, azim=-64)
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(projection[indices, 0], projection[indices, 1],
                             c=colors, cmap=cmap, vmin=-0.5,
                             vmax=len(unique) - 0.5, s=7, alpha=0.7, rasterized=True)
    ax.set_xlabel(f"PC1 ({variance[0]:.1%})")
    ax.set_ylabel(f"PC2 ({variance[1]:.1%})")
    ax.set_title(title)
    colorbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(len(unique)), shrink=0.8)
    colorbar.ax.set_yticklabels([f"{label} ({count})" for label, count in zip(unique, counts)])
    colorbar.set_label("Cluster (all samples)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def category_colors(count):
    palette = plt.get_cmap("tab20").colors
    order = [*range(0, 20, 2), *range(1, 20, 2)]
    return [to_hex(palette[order[index % len(order)]]) for index in range(count)]


def save_task_plot(path, projection, task_ids, task_names, indices, variance, title):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    unique_tasks = np.unique(task_ids)
    colors = category_colors(len(unique_tasks))
    for task_id, color in zip(unique_tasks, colors):
        selected = indices[task_ids[indices] == task_id]
        ax.scatter(projection[selected, 0], projection[selected, 1],
                   color=color, s=7, alpha=0.65, rasterized=True,
                   label=f"{task_id}: {task_names[int(task_id)]}")
    ax.set_xlabel(f"PC1 ({variance[0]:.1%})")
    ax.set_ylabel(f"PC2 ({variance[1]:.1%})")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, markerscale=2)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_interactive_3d(path, projection, labels, task_ids, task_names,
                        indices, variance, title, color_by):
    if go is None:
        raise RuntimeError("Interactive 3D plots require Plotly")
    categories = task_ids if color_by == "task" else labels
    unique = np.unique(categories)
    colors = category_colors(len(unique))
    fig = go.Figure()
    for category, color in zip(unique, colors):
        selected = indices[categories[indices] == category]
        hover = [f"Task: {int(task_ids[row])} ({escape(task_names[int(task_ids[row])])})"
                 f"<br>Cluster: {int(labels[row])}<br>Sample: {int(row)}"
                 for row in selected]
        label = (f"{int(category)}: {task_names[int(category)]}" if color_by == "task"
                 else f"Cluster {int(category)}")
        fig.add_trace(go.Scatter3d(
            x=projection[selected, 0], y=projection[selected, 1],
            z=projection[selected, 2], mode="markers", name=label,
            marker={"size": 3, "color": color, "opacity": 0.75},
            text=hover, hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(
        title=f"{title} | colored by {color_by}",
        scene={"xaxis_title": f"PC1 ({variance[0]:.1%})",
               "yaxis_title": f"PC2 ({variance[1]:.1%})",
               "zaxis_title": f"PC3 ({variance[2]:.1%})",
               "aspectmode": "data"},
        legend={"title": color_by.title()},
        margin={"l": 0, "r": 0, "b": 0, "t": 55},
        height=760)
    fig.write_html(path, include_plotlyjs=True,
                   config={"displaylogo": False, "responsive": True})


def save_task_composition(group_dir, method, datasets, labels, task_ids, cluster_labels):
    counts = np.array([[np.count_nonzero((task_ids == item["task_id"]) & (labels == cluster))
                        for cluster in cluster_labels] for item in datasets], dtype=np.int64)
    fractions = counts / counts.sum(axis=1, keepdims=True)
    header = ["task_id", "task_name", *[f"cluster_{int(cluster)}" for cluster in cluster_labels]]
    for kind, values in (("counts", counts), ("fractions", fractions)):
        with (group_dir / f"{method}_task_cluster_{kind}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for item, row in zip(datasets, values):
                writer.writerow([item["task_id"], item["task_name"], *row.tolist()])

    fig, ax = plt.subplots(figsize=(max(7, 3 + 0.5 * len(cluster_labels)),
                                    max(4, 1.5 + 0.45 * len(datasets))), constrained_layout=True)
    image = ax.imshow(fractions, aspect="auto", cmap="viridis", vmin=0,
                      vmax=max(0.1, float(fractions.max())))
    ax.set_xticks(np.arange(len(cluster_labels)), labels=[str(int(label)) for label in cluster_labels])
    ax.set_yticks(np.arange(len(datasets)),
                  labels=[f"{item['task_id']}: {item['task_name']}" for item in datasets])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Task")
    ax.set_title(f"{method} | fraction of each task assigned to each cluster")
    fig.colorbar(image, ax=ax, label="Fraction of task samples")
    fig.savefig(group_dir / f"{method}_task_cluster_heatmap.png", dpi=160)
    plt.close(fig)
    return {str(item["task_id"]): {str(int(cluster)): int(counts[row, col])
                                   for col, cluster in enumerate(cluster_labels)}
            for row, item in enumerate(datasets)}


def cluster_group(name, datasets, args, group_dir):
    features = np.concatenate([item["features"] for item in datasets], axis=0)
    task_names = {item["task_id"]: item["task_name"] for item in datasets}
    task_ids = np.concatenate([np.full(len(item["features"]), item["task_id"], dtype=np.int32)
                               for item in datasets])
    if min(features.shape) < 3:
        raise ValueError(f"{name}: at least 3 samples and 3 features are needed for 3D PCA plots")
    rng = np.random.default_rng(args.seed)
    fit_count = min(len(features), args.max_fit_samples or len(features))
    if fit_count < max(3, *args.gmm_components):
        raise ValueError(f"{name}: {fit_count} fit samples are insufficient for 3D PCA "
                         f"and GMM components {args.gmm_components}")
    fit_indices = np.sort(rng.choice(len(features), size=fit_count, replace=False))
    plot_count = min(len(features), args.plot_max_points)
    plot_indices = np.sort(rng.choice(len(features), size=plot_count, replace=False))
    scaler = StandardScaler().fit(features[fit_indices])
    scaled = scaler.transform(features)
    fit_data = scaled[fit_indices]
    pca = PCA(n_components=3, random_state=args.seed).fit(fit_data)
    projection = pca.transform(scaled).astype(np.float32)
    if not np.isfinite(projection).all():
        raise ValueError(f"{name}: PCA projection contains NaN or infinity")

    summary = {
        "group": name,
        "domain": args.domain,
        "scope": args.scope,
        "features": args.features,
        "feature_dim": int(features.shape[1]),
        "samples": int(len(features)),
        "fit_samples": int(fit_count),
        "plot_samples": int(plot_count),
        "seed": args.seed,
        "covariance_type": args.covariance_type,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "sources": [{"file": str(item["path"]), "task_id": item["task_id"],
                     "task_name": item["task_name"], "samples": len(item["features"])}
                    for item in datasets],
        "models": {},
    }
    models = {}
    for k in args.gmm_components:
        models[f"gmm_k{k}"] = GaussianMixture(
            n_components=k, covariance_type=args.covariance_type,
            reg_covar=1e-5, max_iter=args.max_iter, n_init=2, random_state=args.seed)
    models["dpmm"] = BayesianGaussianMixture(
        n_components=min(args.dp_max_components, fit_count),
        covariance_type=args.covariance_type, reg_covar=1e-5,
        max_iter=args.max_iter, n_init=1, random_state=args.seed,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=args.dp_concentration)

    for method, model in models.items():
        method_name = ("DPMM (Dirichlet-process BayesianGaussianMixture)"
                       if method == "dpmm" else method)
        print(f"[{name}] fitting {method_name} on {fit_count} samples", flush=True)
        model.fit(fit_data)
        labels = model.predict(scaled).astype(np.int32)
        unique, counts = np.unique(labels, return_counts=True)
        result = {
            "model_class": type(model).__name__,
            "cluster_counts": {str(int(label)): int(count) for label, count in zip(unique, counts)},
            "occupied_clusters": int(len(unique)),
            "converged": bool(model.converged_),
            "iterations": int(model.n_iter_),
            "weights": model.weights_.tolist(),
        }
        if method.startswith("gmm_"):
            result["aic_fit"] = float(model.aic(fit_data))
            result["bic_fit"] = float(model.bic(fit_data))
        else:
            result["max_components"] = int(model.n_components)
            result["concentration"] = args.dp_concentration
            result["weight_concentration_prior_type"] = model.weight_concentration_prior_type
            result["lower_bound"] = float(model.lower_bound_)
            result["inference"] = "truncated variational Dirichlet-process mixture"
        summary["models"][method] = result

        np.savez_compressed(group_dir / f"{method}.npz", labels=labels,
                            task_ids=task_ids, pca_3d=projection,
                            fit_indices=fit_indices, plot_indices=plot_indices)
        plot_dims = (2, 3) if args.plot_dim == "both" else (int(args.plot_dim[0]),)
        for dimensions in plot_dims:
            save_plot(group_dir / f"{method}_{dimensions}d.png", projection,
                      labels, plot_indices, pca.explained_variance_ratio_,
                      f"{name}\n{method} | {args.features}", dimensions)
        if 2 in plot_dims:
            save_task_plot(group_dir / f"{method}_2d_tasks.png", projection,
                           task_ids, task_names, plot_indices,
                           pca.explained_variance_ratio_,
                           f"{name}\n{method} | tasks")
        if 3 in plot_dims:
            for color_by in ("cluster", "task"):
                save_interactive_3d(
                    group_dir / f"{method}_3d_{color_by}.html", projection,
                    labels, task_ids, task_names, plot_indices,
                    pca.explained_variance_ratio_, f"{name} | {method} | {args.features}",
                    color_by)
        if args.scope == "pooled":
            result["task_cluster_counts"] = save_task_composition(
                group_dir, method, datasets, labels, task_ids, unique)
        print(f"[{name}] {method}: {len(unique)} occupied clusters", flush=True)

    with (group_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary


def main():
    args = parse_args()
    input_dir = args.input_dir or EXPERIMENTS_DIR / "datasets" / "simple_rollouts" / args.domain
    output_dir = args.output_dir or EXPERIMENTS_DIR / "analysis" / "simple_rollout_clusters" / args.domain
    groups = prepare_groups(args, input_dir)
    destinations = [(name, datasets, output_dir / args.features / args.scope / name)
                    for name, datasets in groups]
    if not args.overwrite:
        existing = [path for _, _, path in destinations if path.exists() and any(path.iterdir())]
        if existing:
            raise FileExistsError(f"Results already exist in {existing[0]}; use --overwrite to replace them")
    for index, (name, datasets, group_dir) in enumerate(destinations, start=1):
        group_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(destinations)}] clustering {name}", flush=True)
        cluster_group(name, datasets, args, group_dir)
        print(f"Saved results to {group_dir}", flush=True)


if __name__ == "__main__":
    main()
