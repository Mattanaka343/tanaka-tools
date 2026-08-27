""" Utilities for graphing during the EDA

This module contains functions that refer to graphs that a data scientist will 
commonly find themselves making. The module makes sure that they will be as readable
as possible and require relatively minimal preprocessing

"""
import warnings

import pandas               as pd
import numpy                as np
import matplotlib.pyplot    as plt
import seaborn              as sns

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import kstest, anderson, shapiro, linregress
from statsmodels.tsa.stattools import adfuller, acf
from statmodels.stats.diagnostic import acorr_ljungbox

from typing import Any, Literal





def corr_heatmap(data: pd.DataFrame, 
                 title:None|str, 
                 annot: bool = True, 
                 cmap: list[str]|str = 'coolwarm', 
                 show: bool = False, 
                 save_path:None|str = None) -> Figure:
    '''
    Generates a good looking correlation heatmap from a DataFrame

    Inputs:
        data -> pandas.DataFrame: The dataframe correlations are menat to be obtained from
        title -> string|None: The title of the plot or None for no title
        annot -> bool: If the plot should have annotations
        cmap -> string|list: If a string it should be a valid seabor colormap. If a list, the list should include the hexadecimal codes
                             for the colors to be included
        show -> bool: If the plot should be shown or just stored
        save_path -> string: The path here the image of the plot should be stored

    Output:
        fig -> Figure: Matplotlib figur containing the heatmap
    '''

    corr = data.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))

    if type(cmap) == list:
        cmap = LinearSegmentedColormap.from_list('my_cmap',cmap)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,         
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=annot,
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )

    

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=9
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=9
    )

    # Remove spines and axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    sns.despine(left=True, bottom=True)

    if title:
        plt.suptitle(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches = 'tight')

    if show:
        fig.show()

    return fig


def histogram(data: pd.Series, 
              fit_test: Literal['shapiro','ks','anderson'] = 'ks',
              dist:str = 'norm',
              show: bool = False,
              color: str = '#700202',
              save_path: None|str = None,
              ax: None|Axes = None,
              **kwargs
              ) -> Figure|Axes:
    """
    Creates a good looking histogram with goodness of fit test information on the corner

    Inputs:
        data -> pandas.Series: The series containing the data for the plot
        fit_test -> Literal['shapiro','ks','anderson']: One of three possible goodness of 
                                                        fit tests
        dist -> string: The name of the distribution that you are checking against for your data,
                        this distribution needs to be compatible with the test that was given.
        show -> bool: If the plot should be shown or not
        color -> string: The hexadecimal code of the color you want the plot to be
        save_path -> string|None: If present the path where the image of this plot will be saved to
        ax -> None|Axes: If an Axis it's the axis you wenat this plot to be added to.
        kwargs -> dict: Other arguments that may want to be passed onto histogram, like bins, kde,
                        alpha, etc.
    Outputs: 
        fig|ax -> Figure|Axes: The plot or the axis containing it. 
    """


    if fit_test == 'ks':
        result = kstest(data,dist)

    elif fit_test == 'shapiro':
        if dist != 'norm':
            warnings.warn('dist parameter was ignored as shapiro wilk test only tests for normality')
        result = shapiro(data)

    else:
        result = anderson(data,dist=dist)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    sns.histplot(
        data=data,
        stat="density",
        ax=ax,
        color=color,
        **kwargs
    )

    ax.set_title(f"{dist.capitalize()} Goodness-of-Fit")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")

    stats = [
        f"Test: {fit_test}",
        f"Distribution: {dist}",
        f"Statistic: {result.statistic:.4f}",
    ]

    if hasattr(result, "pvalue"):
        stats.append(f"P-value: {result.pvalue:.4g}")

    else:
        idx = np.where(result.significance_level == 5)[0][0]
        stats.append(f'Critical (5%): {result.critical_values[idx]:.4f}')

    stats.extend([
        f"Mean: {np.mean(data):.4f}",
        f"Std. Dev.: {np.std(data, ddof=1):.4f}",
    ])

    ax.text(
        0.98,
        0.98,
        "\n".join(stats),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            facecolor="white",
            edgecolor="0.5",
            alpha=0.9,
            boxstyle="round,pad=0.4",
        ),
    )

    if ax and save_path:
        fig.savefig(save_path)

    if show:
        fig.show()

    if ax:
        return ax

    else:
        return fig 


def time_series_plot(series: pd.Series,
                     seasonal_period: int | None = None,
                     alpha: float = 0.05,
                     figsize: tuple[int, int] = (14, 7),
                     c1: str = "#700202",
                     c2: str = "#027070",
                     show: bool = True,
                     save_dir: None|str = None
                     )->tuple[Figure,dict]:
    """
    Plot and diagnostically analyze a time series.

    Inputs:
        series -> pandas.Series: Time-series data. A DatetimeIndex is recommended.
        seasonal_period -> int|None: If present represents the expected period for 
                                     the seasonal observations. Else seasonality is automatically detected using ACF
        alpha -> float: 0.05 by default. I trefers to the significance level used in statistical tests 
        figsize -> tuple[int, int]:  (14, 7) by default, the size of the image that will be produced
        c1 -> string: The hex code of the color for the line of the main time-series
        c2 -> string: The hex code of the color for the trend line 
        show -> bool: True by default, if the graph should be shown
        save_dir -> str|None: if presenent it represents the directory where the image file version of the plot will 
                              be stored
    Outputs:
        fig -> Figure: The timeline plot.
        results -> dict: Numerical results and test conclusions.    
    """

    # ==========================================================
    # Prepare data
    # ==========================================================

    data = series.dropna()

    if len(data) < 20:
        raise ValueError(
            "At least 20 observations are required for time-series diagnostics."
        )

    x = np.arange(len(data))
    y = data.values

    # ==========================================================
    # Trend
    # ==========================================================

    slope, intercept, r_value, trend_pvalue, std_err = linregress(x, y)

    trend_line = intercept + slope * x

    if trend_pvalue < alpha:
        if slope > 0:
            trend_result = "Significant upward trend"
        else:
            trend_result = "Significant downward trend"
    else:
        trend_result = "No significant linear trend"

    # ==========================================================
    # Augmented Dickey-Fuller
    # ==========================================================

    adf = adfuller(y, autolag="AIC")

    adf_stat = adf[0]
    adf_pvalue = adf[1]

    stationary = adf_pvalue < alpha

    if stationary:
        adf_result = "Stationary"
    else:
        adf_result = "Non-stationary"

    # ==========================================================
    # Autocorrelation / seasonality
    # ==========================================================

    max_lag = min(len(data) // 2, 200)

    acf_values = acf(
        y,
        nlags=max_lag,
        fft=True
    )

    # Approximate 95% confidence interval
    acf_threshold = 1.96 / np.sqrt(len(data))

    if seasonal_period is not None:

        if seasonal_period >= len(acf_values):
            raise ValueError(
                "seasonal_period must be smaller than the number of observations."
            )

        seasonal_acf = acf_values[seasonal_period]

        seasonality_detected = (
            abs(seasonal_acf) > acf_threshold
        )

        detected_period = seasonal_period

    else:

        # Ignore lag 0 and very short lags
        candidate_lags = np.arange(2, len(acf_values))

        significant_lags = candidate_lags[
            np.abs(acf_values[candidate_lags]) > acf_threshold
        ]

        if len(significant_lags) > 0:

            detected_period = int(
                significant_lags[
                    np.argmax(
                        np.abs(acf_values[significant_lags])
                    )
                ]
            )

            seasonal_acf = acf_values[detected_period]

            seasonality_detected = True

        else:

            detected_period = None
            seasonal_acf = np.nan
            seasonality_detected = False

    # ==========================================================
    # Ljung-Box test
    # ==========================================================

    ljung_lag = min(10, len(data) // 5)

    ljung = acorr_ljungbox(
        data,
        lags=[ljung_lag],
        return_df=True
    )

    ljung_pvalue = ljung["lb_pvalue"].iloc[0]

    white_noise = ljung_pvalue >= alpha

    if white_noise:
        white_noise_result = "Consistent with white noise"
    else:
        white_noise_result = "Significant autocorrelation"

    # ==========================================================
    # Create figure
    # ==========================================================

    sns.set_theme(
        style="whitegrid",
        context="notebook"
    )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    # ==========================================================
    # Main time series
    # ==========================================================

    sns.lineplot(
        x=data.index,
        y=y,
        ax=ax,
        linewidth=1.6,
        label="Observed",
        color = c1
    )

    sns.lineplot(
        x=data.index,
        y=trend_line,
        ax=ax,
        linestyle="--",
        linewidth=2,
        label="Linear trend",
        color = c2
    )

    ax.set_title(
        f"Time Series: {series.name or 'Series'}",
        fontsize=18,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Time", fontsize=12)

    ax.set_ylabel(
        series.name or "Value",
        fontsize=12
    )

    # ==========================================================
    # Diagnostic information box
    # ==========================================================

    if detected_period is not None:

        seasonality_text = (
            f"Seasonality: detected "
            f"(period = {detected_period}, "
            f"ACF = {seasonal_acf:.3f})"
        )

    else:

        seasonality_text = "Seasonality: not detected"

    diagnostic_text = (
        f"TREND\n"
        f"{trend_result}\n"
        f"slope = {slope:.4g}, p = {trend_pvalue:.3g}\n\n"

        f"STATIONARITY\n"
        f"ADF: {adf_result}\n"
        f"p = {adf_pvalue:.3g}\n\n"

        f"SEASONALITY\n"
        f"{seasonality_text}\n\n"

        f"WHITE NOISE\n"
        f"{white_noise_result}\n"
        f"Ljung-Box p = {ljung_pvalue:.3g}"
    )

    ax.text(
        0.015,
        0.97,
        diagnostic_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=10.5,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9
        )
    )

    ax.legend(
        loc="upper right",
        frameon=True
    )

    sns.despine()

    plt.tight_layout()

    # ==========================================================
    # Results
    # ==========================================================

    results = {
        "trend": {
            "slope": slope,
            "r_squared": r_value**2,
            "p_value": trend_pvalue,
            "significant": trend_pvalue < alpha,
            "direction": (
                "increasing"
                if slope > 0
                else "decreasing"
            ),
            "conclusion": trend_result,
        },

        "adf": {
            "statistic": adf_stat,
            "p_value": adf_pvalue,
            "stationary": stationary,
            "conclusion": adf_result,
        },

        "seasonality": {
            "detected": seasonality_detected,
            "period": detected_period,
            "acf": seasonal_acf,
            "conclusion": seasonality_text,
        },

        "white_noise": {
            "ljung_box_lag": ljung_lag,
            "p_value": ljung_pvalue,
            "consistent_with_white_noise": white_noise,
            "conclusion": white_noise_result,
        },

        "figure": fig,
    }

    if show:
        plt.show()

    if save_dir:
        fig.savefig(save_dir)

    return fig, results
    
def aggregate_barplot(df: pd.DataFrame,
                      title: str,
                      category_col: str,
                      aggregate_col: str | None = None,
                      aggregate_value: Any=None,
                      figsize: tuple[int,int] =(8, 5),
                      color: str ='#700202',
                      show: bool = True,
                      save_dir: str|None = None,
                      **kwargs
                      ) -> Figure:
    """
    Plot counts by category, optionally restricted to a value
    in another column.

    Inputs:
        df -> pandas.DataFrame: The dataframe that contains the data to be analyzed
        title ->  str:  The title of the plot
        category_col -> string: Column whose categories will be shown on the x-axis.
        aggregate_col  -> str | None:  If present it is a secondary column to filter the main coulumn by
        aggregate_value -> Any: The value that is relevant in the aggregate column
        figsize -> tuple[int,int]: The size of the plot, takes (8, 5) by default
    Outputs:
        fig -> Figure: The matplotlib graph
        ax -> Axes: The matplotlib axes containing the graph
    """

    data = df.copy()

    # If an aggregate column and value are provided,
    # only keep rows matching that value
    if aggregate_col is not None:
        data = data[data[aggregate_col] == aggregate_value]

    # Count observations in each category
    counts = (
        data[category_col]
        .value_counts()
        .reset_index()
    )

    counts.columns = [category_col, "count"]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=counts,
        x=category_col,
        y="count",
        ax=ax,
        color=color,
        **kwargs
    )

    ax.set_xlabel(category_col)
    ax.set_ylabel("Count")

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    if show:
        plt.show()

    if save_dir:
        fig.savefig(save_dir)

    return fig, ax