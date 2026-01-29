import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List

sns.set(style="whitegrid")

# Helper to save & show
def _save_and_show(save: bool, save_path: str):
    plt.tight_layout()
    if save:
        plt.savefig(f"{save_path}.png", dpi=150)
    plt.show()

# ============================================================
# BASIC PLOTS
# ============================================================

def plot_scatter(df: pd.DataFrame, xcol: str, ycol: str,
                 color: Optional[str] = None,
                 sample: int = 20000,
                 title: str = "Scatter Plot",
                 save: bool = False,
                 save_path: str = "scatter_plot",
                 **kwargs):

    plt.figure(figsize=(8, 6))
    plot_df = df.sample(sample) if len(df) > sample else df

    if color:
        sns.scatterplot(
            data=plot_df, x=xcol, y=ycol, hue=color,
            palette=kwargs.get("palette", "tab10"),
            alpha=kwargs.get("alpha", 0.6), s=kwargs.get("s", 10)
        )
    else:
        plt.scatter(plot_df[xcol], plot_df[ycol], s=kwargs.get("s", 10), alpha=0.5)

    plt.title(title)
    _save_and_show(save, save_path)


def plot_line(df: pd.DataFrame, xcol: str, ycol: str,
              color: Optional[str] = None,
              title: str = "Line Plot",
              save: bool = False,
              save_path: str = "line_plot",
              **kwargs):

    plt.figure(figsize=(8, 6))
    if color:
        sns.lineplot(
            data=df, x=xcol, y=ycol, hue=color,
            palette=kwargs.get("palette", "tab10")
        )
    else:
        sns.lineplot(data=df, x=xcol, y=ycol)
    plt.title(title)
    _save_and_show(save, save_path)


def plot_bar(df: pd.DataFrame, xcol: str, ycol: Optional[str] = None,
             color: Optional[str] = None,
             title: str = "Bar Chart",
             save: bool = False,
             save_path: str = "bar_chart",
             **kwargs):

    plt.figure(figsize=(8, 6))
    if ycol:
        sns.barplot(
            data=df, x=xcol, y=ycol, hue=color,
            palette=kwargs.get("palette", "tab10")
        )
    else:
        sns.countplot(
            data=df, x=xcol, hue=color,
            palette=kwargs.get("palette", "tab10")
        )

    plt.title(title)
    _save_and_show(save, save_path)


def plot_histogram(df: pd.DataFrame, xcol: str,
                   bins: int = 30,
                   title: str = "Histogram",
                   save: bool = False,
                   save_path: str = "histogram",
                   category: str = None,
                   **kwargs):

    plt.figure(figsize=(8, 6))

    if category:
        sns.histplot(
            data=df,
            x=xcol,
            hue=category,
            bins=bins,
            palette=kwargs.get("palette", "tab10"),
            multiple=kwargs.get("multiple", "stack"),
            kde=kwargs.get("kde", False),
            **kwargs
        )
    else:
        sns.histplot(data=df, x=xcol, bins=bins, kde=True, **kwargs)

    plt.title(title)
    _save_and_show(save, save_path)


def plot_box(df: pd.DataFrame, ycol: str,
             xcol: Optional[str] = None,
             color: Optional[str] = None,
             title: str = "Box Plot",
             save: bool = False,
             save_path: str = "box_plot",
             **kwargs):

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x=xcol, y=ycol, hue=color,
        palette=kwargs.get("palette", "tab10")
    )
    plt.title(title)
    _save_and_show(save, save_path)


def plot_violin(df: pd.DataFrame, ycol: str,
                xcol: Optional[str] = None,
                color: Optional[str] = None,
                title: str = "Violin Plot",
                save: bool = False,
                save_path: str = "violin_plot",
                **kwargs):

    plt.figure(figsize=(8, 6))
    sns.violinplot(
        data=df,
        x=xcol, y=ycol, hue=color, split=kwargs.get("split", True),
        palette=kwargs.get("palette", "tab10")
    )
    plt.title(title)
    _save_and_show(save, save_path)

# ============================================================
# EXTRA PLOTS
# ============================================================

def plot_heatmap(df: pd.DataFrame, xcol: str, ycol: str,
                 gridsize: int = 50,
                 title: str = "Heatmap",
                 save: bool = False,
                 save_path: str = "heatmap",
                 **kwargs):

    plt.figure(figsize=(8, 6))
    plt.hexbin(df[xcol], df[ycol], gridsize=gridsize, cmap=kwargs.get("cmap", "viridis"))
    plt.colorbar(label='count')
    plt.title(title)
    _save_and_show(save, save_path)


def plot_correlation_matrix(df: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            title: str = "Correlation Matrix",
                            save: bool = False,
                            save_path: str = "correlation_matrix",
                            **kwargs):

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    corr = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=kwargs.get("cmap", "RdBu_r"))
    plt.title(title)
    _save_and_show(save, save_path)


def plot_density_contour(df: pd.DataFrame,
                         xcol: str, ycol: str,
                         title: str = "2D Density Contour",
                         save: bool = False,
                         save_path: str = "density_contour",
                         **kwargs):

    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df, x=xcol, y=ycol, fill=True, **kwargs)
    plt.title(title)
    _save_and_show(save, save_path)


def plot_ecdf(df: pd.DataFrame,
              xcol: str,
              title: str = "ECDF",
              save: bool = False,
              save_path: str = "ecdf_plot",
              **kwargs):

    plt.figure(figsize=(8, 6))
    sns.ecdfplot(data=df, x=xcol, **kwargs)
    plt.title(title)
    _save_and_show(save, save_path)


def plot_scatter_matrix(df: pd.DataFrame,
                        columns: Optional[List[str]] = None,
                        sample: int = 5000,
                        title: str = "Scatter Matrix",
                        save: bool = False,
                        save_path: str = "scatter_matrix",
                        **kwargs):

    plot_df = df.sample(sample) if len(df) > sample else df
    if columns is None:
        columns = plot_df.select_dtypes(include="number").columns.tolist()

    g = sns.pairplot(plot_df[columns], palette=kwargs.get("palette", "tab10"))
    plt.suptitle(title)

    if save:
        g.savefig(f"{save_path}.png")
    else:
        plt.show()

# ============================================================
# DISPATCHER
# ============================================================

def plot(df: pd.DataFrame,
         xcol: Optional[str] = None,
         ycol: Optional[str] = None,
         plot_type: str = "scatter",
         save: bool = False,
         save_path: str = "plot",
         **kwargs):

    plot_type = plot_type.lower()

    mapping = {
        "scatter": lambda: plot_scatter(df, xcol, ycol, save=save, save_path=save_path, **kwargs),
        "line":    lambda: plot_line(df, xcol, ycol, save=save, save_path=save_path, **kwargs),
        "bar":     lambda: plot_bar(df, xcol, ycol, save=save, save_path=save_path, **kwargs),
        "hist":    lambda: plot_histogram(df, xcol, save=save, save_path=save_path, **kwargs),
        "histogram": lambda: plot_histogram(df, xcol, save=save, save_path=save_path, **kwargs),
        "box":     lambda: plot_box(df, ycol, xcol, save=save, save_path=save_path, **kwargs),
        "violin":  lambda: plot_violin(df, ycol, xcol, save=save, save_path=save_path, **kwargs),
        "heatmap": lambda: plot_heatmap(df, xcol, ycol, save=save, save_path=save_path, **kwargs),
        "corr":    lambda: plot_correlation_matrix(df, save=save, save_path=save_path, **kwargs),
        "correlation": lambda: plot_correlation_matrix(df, save=save, save_path=save_path, **kwargs),
        "density": lambda: plot_density_contour(df, xcol, ycol, save=save, save_path=save_path, **kwargs),
        "ecdf":    lambda: plot_ecdf(df, xcol, save=save, save_path=save_path, **kwargs),
        "matrix":  lambda: plot_scatter_matrix(df, columns=kwargs.get("columns"),
                                               save=save, save_path=save_path, **kwargs),
        "scatter_matrix": lambda: plot_scatter_matrix(df, columns=kwargs.get("columns"),
                                                       save=save, save_path=save_path, **kwargs),
    }

    if plot_type not in mapping:
        raise ValueError(f"Unknown plot type: {plot_type}")

    return mapping[plot_type]()
