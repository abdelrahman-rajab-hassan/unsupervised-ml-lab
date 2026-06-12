import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
    mean_squared_error,
    classification_report,
    ConfusionMatrixDisplay,
    silhouette_score,
)


# ---------------------------------- functions related to regression tasks --------------------------

####################################################################################################


def regression_metrics(y_true, y_pred, label="", verbose=True, output_dict=False):
    # Get metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)

    if verbose == True:
        # Print Result with Label and Header
        header = "-" * 60
        print(header, f"Regression Metrics: {label}", header, sep="\n")
        print(f"- MAE = {mae:,.3f}")
        print(f"- MSE = {mse:,.3f}")
        print(f"- RMSE = {rmse:,.3f}")
        print(f"- R^2 = {r_squared:,.3f}")

    if output_dict == True:
        metrics = {
            "Label": label,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R^2": r_squared,
        }
        return metrics


######################################################################################


def evaluate_regression(
    reg, X_train, y_train, X_test, y_test, verbose=True, output_frame=False
):
    # Get predictions for training data
    y_train_pred = reg.predict(X_train)

    # Call the helper function to obtain regression metrics for training data
    results_train = regression_metrics(
        y_train,
        y_train_pred,
        verbose=verbose,
        output_dict=output_frame,
        label="Training Data",
    )
    print()

    # Get predictions for test data
    y_test_pred = reg.predict(X_test)

    # Call the helper function to obtain regression metrics for test data
    results_test = regression_metrics(
        y_test,
        y_test_pred,
        verbose=verbose,
        output_dict=output_frame,
        label="Test Data",
    )

    # Store results in a dataframe if output_frame is True
    if output_frame:
        results_df = pd.DataFrame([results_train, results_test])
        # Set the label as the index
        results_df = results_df.set_index("Label")
        # Set index.name to none to get a cleaner looking result
        results_df.index.name = None
        # Return the dataframe
        return results_df.round(3)


# --------------------------------- functions related to classifcation tasks --------------------------------


def classification_metrics(
    y_true,
    y_pred,
    label="",
    output_dict=False,
    figsize=(8, 4),
    normalize="true",
    cmap="Blues",
    colorbar=False,
):
    # Get the classification report
    report = classification_report(y_true, y_pred)
    ## Print header and report
    header = "-" * 70
    print(header, f" Classification Metrics: {label}", header, sep="\n")
    print(report)
    ## CONFUSION MATRICES SUBPLOT
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    # create a confusion matrix of raw counts
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        normalize=None,
        cmap="gist_gray",
        colorbar=colorbar,
        ax=axes[0],
    )
    axes[0].set_title("Raw Counts")
    # create a confusion matrix with the test data
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, normalize=normalize, cmap=cmap, colorbar=colorbar, ax=axes[1]
    )
    axes[1].set_title("Normalized Confusion Matrix")
    fig.tight_layout()
    plt.show()
    if output_dict == True:
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        return report_dict


def evaluate_classification(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    figsize=(6, 4),
    normalize="true",
    output_dict=False,
    cmap_train="Blues",
    cmap_test="Reds",
    colorbar=False,
):
    # Get predictions for training data
    y_train_pred = model.predict(X_train)
    # Call the helper function to obtain regression metrics for training data
    results_train = classification_metrics(
        y_train,
        y_train_pred,  # verbose = verbose,
        output_dict=True,
        figsize=figsize,
        colorbar=colorbar,
        cmap=cmap_train,
        label="Training Data",
    )
    print()
    # Get predictions for test data
    y_test_pred = model.predict(X_test)
    # Call the helper function to obtain regression metrics for test data
    results_test = classification_metrics(
        y_test,
        y_test_pred,  # verbose = verbose,
        output_dict=True,
        figsize=figsize,
        colorbar=colorbar,
        cmap=cmap_test,
        label="Test Data",
    )

    if output_dict == True:
        # Store results in a dataframe if ouput_frame is True
        results_dict = {"train": results_train, "test": results_test}
        return results_dict


# ---------------------------------- functions related to plotting --------------------------


###################################### Categorical vs Numerics & Categorical


# Previously defined function
def plot_categorical_vs_numeric(
    df,
    x_feature,
    y_target,
    figsize=(6, 4),
    fillna=True,
    placeholder="MISSING",
    order=None,
):

    # Copy only necessary columns to save memory
    temp_df = df[[x_feature, y_target]].copy()

    if fillna:
        temp_df[x_feature] = temp_df[x_feature].fillna(placeholder)
    else:
        temp_df = temp_df.dropna(subset=[x_feature])

    fig, ax = plt.subplots(figsize=figsize)

    # Barplot
    sns.barplot(
        data=temp_df,
        x=x_feature,
        y=y_target,
        ax=ax,
        order=order,
        alpha=0.6,
        linewidth=1,
        edgecolor="black",
        errorbar=None,
    )

    # Stripplot - Corrected 'y' parameter
    sns.stripplot(
        data=temp_df,
        x=x_feature,
        y=y_target,
        hue=x_feature,
        ax=ax,
        order=order,
        hue_order=order,
        legend=False,
        edgecolor="white",
        linewidth=0.5,
        size=3,
        zorder=0,
    )

    # Clean rotation handling
    ax.tick_params(axis="x", labelrotation=45)

    ax.set_title(f"{x_feature} vs. {y_target}")
    fig.tight_layout()

    return fig, ax


# Updating function
def plot_categorical_vs_categorical(
    df,
    x_feature,
    y_target,
    figsize=(6, 4),
    fillna=True,
    placeholder="MISSING",
    order=None,
    target_type="reg",
):
    # Make a copy of the dataframe and fillna
    temp_df = df.copy()
    # fillna with placeholder
    if fillna == True:
        temp_df[x_feature] = temp_df[x_feature].fillna(placeholder)
    # or drop nulls prevent unwanted 'nan' group in stripplot
    else:
        temp_df = temp_df.dropna(subset=[x_feature])
    # Create the figure and subplots
    fig, ax = plt.subplots(figsize=figsize)

    ax = sns.histplot(
        data=temp_df, hue=y_target, x=x_feature, stat="percent", multiple="fill"
    )
    # Rotate xlabels
    ax.set_xticks(ax.get_xticks())  # Added this to prevent a bug
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    # Add a title
    ax.set_title(f"{x_feature} vs. {y_target}")
    fig.tight_layout()
    return fig, ax


# make sure you pass the right argument for target_type
def plot_categorical_vs_target(
    df,
    x_feature,
    y_target,
    figsize=(6, 4),
    fillna=True,
    placeholder="MISSING",
    order=None,
    target_type="reg",
):
    # Make a copy of the dataframe and fillna
    temp_df = df.copy()
    # fillna with placeholder
    if fillna == True:
        temp_df[x_feature] = temp_df[x_feature].fillna(placeholder)
    # or drop nulls prevent unwanted 'nan' group in stripplot
    else:
        temp_df = temp_df.dropna(subset=[x_feature])
    # Create the figure and subplots
    fig, ax = plt.subplots(figsize=figsize)
    # REGRESSION-TARGET PLOT
    if target_type == "reg":  # Added if statement here
        # Barplot
        sns.barplot(
            data=temp_df,
            x=x_feature,
            y=y_target,
            ax=ax,
            order=order,
            alpha=0.6,
            linewidth=1,
            edgecolor="black",
            errorbar=None,
        )
        # Boxplot
        sns.stripplot(
            data=temp_df,
            x=x_feature,
            y=y_target,
            hue=x_feature,
            ax=ax,
            order=order,
            hue_order=order,
            legend=False,
            edgecolor="white",
            linewidth=0.5,
            size=3,
            zorder=0,
        )
    # CLASSIFICATION-TARGET PLOT # This is the new code for the classification task
    elif target_type == "class":
        ax = sns.histplot(
            data=temp_df, hue=y_target, x=x_feature, stat="percent", multiple="fill"
        )
    # Rotate xlabels
    ax.set_xticks(ax.get_xticks())  # Added this to prevent a bug
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    # Add a title
    ax.set_title(f"{x_feature} vs. {y_target}")
    fig.tight_layout()
    return fig, ax


######################################## Numeric vs Num & Cat ########################################################


# Previously defined function
def plot_numeric_vs_numeric(
    df, x_feature, y_target, figsize=(6, 4), **kwargs
):  # kwargs for sns.regplot
    # Calculate the correlation
    corr = df[[x_feature, y_target]].corr().round(2)
    r = corr.loc[x_feature, y_target]
    # Plot the data
    fig, ax = plt.subplots(figsize=figsize)
    scatter_kws = {"ec": "white", "linewidths": 1, "alpha": 0.8}
    sns.regplot(
        data=df, x=x_feature, y=y_target, ax=ax, scatter_kws=scatter_kws, **kwargs
    )  # Included the new argument within the sns.regplot function
    ## Add the title with the correlation
    ax.set_title(f"{x_feature} vs. {y_target} (r = {r})")
    # Make sure the plot is shown before the print statement
    plt.show()
    return fig, ax


def plot_numeric_vs_categorical(
    df,
    x,
    y,
    figsize=(6, 4),
    target_type="reg",
    estimator="mean",
    errorbar="ci",
    sorted=False,
    ascending=False,
):  # kwargs for sns.regplot

    nulls = df[[x, y]].isna().sum()
    if nulls.sum() > 0:
        print(f"- Excluding {nulls.sum()} NaN's")
        # print(nulls)
        temp_df = df.dropna(subset=[x, y])
    else:
        temp_df = df
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Sort the groups by median/mean
    if sorted == True:

        if estimator == "median":
            group_vals = temp_df.groupby(y)[x].median()
        elif estimator == "mean":
            group_vals = temp_df.groupby(y)[x].mean()

        ## Sort values
        group_vals = group_vals.sort_values(ascending=ascending)
        order = group_vals.index

    else:
        # Set order to None if not calcualted
        order = None

    # Left Subplot (barplot)
    sns.barplot(
        data=temp_df,
        x=y,
        y=x,
        order=order,
        estimator=estimator,
        errorbar=errorbar,
        ax=ax,
    )

    # Add title
    ax.set_title(f"{x} vs. {y}")

    # rotate xaxis labels
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Final Adjustments & return
    fig.tight_layout()
    return fig, ax


# Updating the function
def plot_numeric_vs_target(
    df,
    x,
    y,
    figsize=(6, 4),
    target_type="reg",
    estimator="mean",
    errorbar="ci",
    sorted=False,
    ascending=False,
    **kwargs,
):  # kwargs for sns.regplot

    nulls = df[[x, y]].isna().sum()
    if nulls.sum() > 0:
        print(f"- Excluding {nulls.sum()} NaN's")
        # print(nulls)
        temp_df = df.dropna(subset=[x, y])
    else:
        temp_df = df
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # REGRESSION-TARGET PLOT
    if "reg" in target_type:
        # Calculate the correlation
        corr = df[[x, y]].corr().round(2)
        r = corr.loc[x, y]
        # Plot the data
        scatter_kws = {"ec": "white", "lw": 1, "alpha": 0.8}
        sns.regplot(data=temp_df, x=x, y=y, ax=ax, scatter_kws=scatter_kws, **kwargs)
        # Included the new argument within the sns.regplot function
        ## Add the title with the correlation
        ax.set_title(f"{x} vs. {y} (r = {r})")

        # CLASSIFICATION-TARGET PLOT
    elif "class" in target_type:

        # Sort the groups by median/mean
        if sorted == True:

            if estimator == "median":
                group_vals = temp_df.groupby(y)[x].median()
            elif estimator == "mean":
                group_vals = temp_df.groupby(y)[x].mean()

            ## Sort values
            group_vals = group_vals.sort_values(ascending=ascending)
            order = group_vals.index

    else:
        # Set order to None if not calcualted
        order = None

    # Left Subplot (barplot)
    sns.barplot(
        data=temp_df,
        x=y,
        y=x,
        order=order,
        estimator=estimator,
        errorbar=errorbar,
        ax=ax,
        **kwargs,
    )

    # Add title
    ax.set_title(f"{x} vs. {y}")

    # rotate xaxis labels
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Final Adjustments & return
    fig.tight_layout()
    return fig, ax


# ---------------------------------- functions related to confusion matrix & errors types --------------------------


# ---------------------------------- funcions related to clustering tasks --------------------------


def plot_clusters(df, x, y, kmeans, scaler, color="Set2"):
    """
    Visualizes K-Means clustering results on a 2D scatter plot with centroids.

    Works with any number of clusters and any number of features — you just
    pick two columns to display. The centroids are inverse-transformed back
    to the original scale so the annotations are human-readable.

    Parameters
    ----------
    df      : DataFrame — must include a 'cluster' column from kmeans.labels_
    x       : str       — column name to plot on the x-axis
    y       : str       — column name to plot on the y-axis
    kmeans  : fitted KMeans object
    scaler  : fitted StandardScaler object used to scale the data before clustering

    Example
    -------
    plot_clusters(df=df, x='attendance_pct', y='avg_grade', kmeans=kmeans, scaler=scaler)
    """

    # Inverse transform centroids back to original scale (undo StandardScaler)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # Find the index of x and y inside the original feature columns
    # so we pull the correct centroid values regardless of column order
    cols = list(df.drop(columns="cluster").columns)
    x_idx = cols.index(x)
    y_idx = cols.index(y)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the data points, colored by cluster assignment
    ax.scatter(
        df[x],
        df[y],
        c=df["cluster"],
        cmap=color,
        s=80,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.8,
    )

    # Plot centroids as black X markers on top of the points
    ax.scatter(
        centers[:, x_idx], centers[:, y_idx], c="black", s=220, marker="X", zorder=5
    )

    # Annotate each centroid with its cluster number and coordinate values
    for i, center in enumerate(centers):
        cx, cy = center[x_idx], center[y_idx]
        ax.annotate(
            f"cluster {i}  ({cx:.1f}, {cy:.1f})",
            xy=(cx, cy),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=8.5,
            color="#333",
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ccc", alpha=0.9
            ),
        )

    ax.set_xlabel(x, fontsize=10)
    ax.set_ylabel(y, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------- funcions related to PCA tasks --------------------------


def plot_explained_variance(explained, threshold=0.9):
    """
    Plots the cumulative explained variance of PCA components.

    Parameters
    ----------
    explained : array-like or pd.Series
        The explained variance ratio from pca.explained_variance_ratio_
    threshold : float, optional (default=0.9)
        The variance threshold to mark on the plot (e.g. 0.9 = 90%)

    Returns
    -------
    n_components : int
        The number of components needed to reach the threshold

    Example
    -------
    plot_explained_variance(pca.explained_variance_ratio_)
    plot_explained_variance(pca.explained_variance_ratio_, threshold=0.95)
    """
    # convert to Series if numpy array
    explained = pd.Series(explained, name="Explained Variance Ratio")

    # find the number of components needed
    n_components = (explained.cumsum() >= threshold).argmax()

    # plot cumulative variance
    ax = explained.cumsum().plot(marker=".", figsize=(10, 6))

    # horizontal line at threshold
    ax.axhline(
        threshold, color="r", linestyle="--", label=f"{threshold*100:.0f}% variance"
    )

    # vertical line at intersection
    ax.axvline(
        n_components, color="g", linestyle="--", label=f"{n_components} components"
    )

    # dot at the intersection point
    ax.scatter(n_components, threshold, color="black", zorder=5, s=100)

    # annotate the x value
    ax.annotate(
        f"n = {n_components}",
        xy=(n_components, threshold),
        xytext=(n_components + 10, threshold - 0.05),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=12,
    )

    ax.set_title("PCA Explained Variance")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(
        f"Components needed to explain {threshold*100:.0f}% of variance: {n_components}"
    )
    return n_components


# ----------------------------------------- Function related to cluster --------------------------------------------


def plot_optimal_k(data, k_range=range(2, 11), random_state=42):
    """
    Plots the Elbow Method and Silhouette Score for a given dataset
    to help determine the optimal number of KMeans clusters.

    Parameters
    ----------
    data         : array-like or DataFrame — the scaled data to cluster
    k_range      : range — the range of K values to evaluate (default: 2–10)
    random_state : int   — random state for reproducibility (default: 42)
    """
    sil, inertia = [], []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        kmeans.fit(data)
        sil.append(silhouette_score(data, kmeans.labels_))
        inertia.append(kmeans.inertia_)

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow plot
    ax1.plot(k_range, inertia, marker="o")
    ax1.set_title("Elbow Method")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia (WCSS)")

    # Silhouette plot
    ax2.plot(k_range, sil, marker="o", color="orange")
    ax2.set_title("Silhouette Score")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")

    plt.tight_layout()
    plt.show()


# ----------------------------------------- Function related to data cleaning --------------------------------------------


def profile_column(df, column, show_value_counts=True):
    """
    Profiles a single DataFrame column by displaying two side-by-side summaries:

    - Summary table: data type, null count, null %, constant/quasi-constant status,
      cardinality level, and skewness (numeric columns only).
    - Value counts table: frequency of each unique value (optional).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the column.
    column : str
        The name of the column to profile.
    show_value_counts : bool, optional
        Whether to show the value counts table. Default is True.

    Example
    -------
    >>> profile_column(df, 'age')
    >>> profile_column(df, 'age', show_value_counts=False)
    """
    s = df[column]
    top_freq = s.value_counts(normalize=True).iloc[0]

    const_or_quasi = (
        "Constant"
        if s.nunique() == 1
        else "Quasi-constant" if top_freq >= 0.95 else "Neither"
    )

    n_unique = s.nunique()
    cardinality = (
        f"{n_unique} — Low"
        if n_unique <= 10
        else f"{n_unique} — Medium" if n_unique <= 50 else f"{n_unique} — High"
    )

    summary = pd.DataFrame(
        {
            "Value": [
                s.dtype,
                s.isna().sum(),
                f"{s.isna().mean():.1%}",
                const_or_quasi,
                cardinality,
                s.skew() if s.dtype in ["int64", "float64"] else "N/A",
            ]
        },
        index=[
            "Data Type",
            "Null Count",
            "Null %",
            "Constant/Quasi",
            "Cardinality",
            "Skewness",
        ],
    )

    from IPython.display import display, HTML

    if show_value_counts:
        value_counts = s.value_counts().to_frame()
        display(
            HTML(
                f"<div style='display:flex; gap:40px'>"
                f"<div><h4>Summary Table</h4>{summary.to_html().replace('<th>', '<th style=\"font-weight:bold\">')}</div>"
                f"<div><h4>Value Counts</h4>{value_counts.to_html()}</div>"
                f"</div>"
            )
        )
    else:
        display(
            HTML(
                f"<div><h4>Summary Table</h4>{summary.to_html().replace('<th>', '<th style=\"font-weight:bold\">')}</div>"
            )
        )

    # ----------------------------------------- Function related to univariate analysis --------------------------------------------


def plot_univariate_analysis(
    df, column, dtype="auto", figsize=(8, 4), show_pct=True, bins=20
):
    """
    Performs univariate analysis on a single column by plotting
    the appropriate chart based on the data type.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the column.
    column : str
        The name of the column to analyze.
    dtype : str, optional
        The data type of the column. Options: 'continuous', 'categorical', 'auto'.
        Default is 'auto', which infers the type from the column.
    figsize : tuple, optional
        The size of the figure. Default is (8, 4).
    show_pct : bool, optional
        Whether to show percentage labels on top of each bar. Default is True.

    Example
    -------
    >>> plot_univariate_analysis(df, 'age', dtype='continuous')
    >>> plot_univariate_analysis(df, 'gender', dtype='categorical')
    >>> plot_univariate_analysis(df, 'income')  # auto-detect
    """

    s = df[column]

    # auto-detect dtype if not specified
    if dtype == "auto":
        dtype = "continuous" if s.dtype in ["int64", "float64"] else "categorical"

    if dtype == "continuous":
        fig, (ax_hist, ax_box) = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [4, 1]}
        )

        # modern color palette
        hist_color = "#3498db"
        kde_color = "#2980b9"
        box_color = "#1abc9c"
        mean_color = "#e74c3c"
        median_color = "#f39c12"

        # histplot with KDE
        sns.histplot(s, kde=True, ax=ax_hist, color=hist_color, alpha=0.6, bins=bins)

        # fill KDE curve with soft transparent color
        kde_line = ax_hist.lines[0]
        ax_hist.fill_between(
            kde_line.get_xdata(), kde_line.get_ydata(), alpha=0.2, color=kde_color
        )

        # mean and median vertical lines
        ax_hist.axvline(
            s.mean(),
            color=mean_color,
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {s.mean():.2f}",
        )
        ax_hist.axvline(
            s.median(),
            color=median_color,
            linestyle="--",
            linewidth=1.5,
            label=f"Median: {s.median():.2f}",
        )
        ax_hist.legend(fontsize=10)

        # subtle background grid on histplot only
        ax_hist.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax_hist.set_axisbelow(True)

        ax_hist.set_title(f"Distribution of {column}", fontsize=16, fontweight="bold")
        ax_hist.set_ylabel("Count", fontsize=13)

        # boxplot with complementary color
        sns.boxplot(x=s, ax=ax_box, color=box_color)
        ax_box.set_xlabel(column, fontsize=13)

    elif dtype == "categorical":
        fig, ax = plt.subplots(figsize=figsize)

        order = s.value_counts().index
        counts = s.value_counts()

        # highlight dominant bar using a dict to ensure correct color mapping
        dominant = order[0]
        palette = {cat: "#2ecc71" if cat == dominant else "#3498db" for cat in order}

        sns.countplot(x=s, order=order, ax=ax, hue=s, palette=palette, legend=False)

        # add percentage labels on top of each bar
        if show_pct:
            total = len(s)
            for p in ax.patches:
                pct = f"{100 * p.get_height() / total:.1f}%"
                ax.annotate(
                    pct,
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # horizontal grid only
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        # font sizes
        ax.set_title(f"Frequency of {column}", fontsize=16, fontweight="bold")
        ax.set_xlabel(column, fontsize=13)
        ax.set_ylabel("Count", fontsize=13)
        ax.tick_params(axis="x", rotation=45, labelsize=9)

        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.tick_params(axis="y", labelsize=11)

    else:
        raise ValueError("dtype must be 'continuous', 'categorical', or 'auto'")

    plt.tight_layout()
    plt.show()




# ----------------------------------------- Function related to ploting analysis -------------------------------------------- 
import math

def plot_history(history, figsize=(12,10), marker='o'):
       
    # Get list of metrics from history
    metrics = [c for c in history.history if not c.startswith('val_')]
    
    # Calculate rows needed for 2 columns
    n_cols = 2
    n_rows = math.ceil(len(metrics) / n_cols)
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    
    # For each metric
    for i, metric_name in enumerate(metrics):
    
        ax = axes[i]
    
        metric_values = history.history[metric_name]
        epochs = history.epoch
    
        ax.plot(epochs, metric_values, label=metric_name, marker=marker)
    
        val_metric_name = f"val_{metric_name}"
        if val_metric_name in history.history:
            metric_values = history.history[val_metric_name]
            ax.plot(epochs, metric_values, label=val_metric_name, marker=marker)
    
        ax.legend()
        ax.set_title(metric_name)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    return fig, axes