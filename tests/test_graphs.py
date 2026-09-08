import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "ttools" / "graphs.py"
SPEC = importlib.util.spec_from_file_location("ttools_graphs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

histogram = MODULE.histogram
time_series_plot = MODULE.time_series_plot
stacked_percentage_barplot = MODULE.stacked_barplot

cleaners_path = Path(__file__).resolve().parents[1] / "src" / "ttools" / "cleaners.py"
cleaners_spec = importlib.util.spec_from_file_location("ttools_cleaners", cleaners_path)
cleaners_module = importlib.util.module_from_spec(cleaners_spec)
assert cleaners_spec is not None and cleaners_spec.loader is not None
cleaners_spec.loader.exec_module(cleaners_module)
clean_df = cleaners_module.clean_df


def test_histogram_accepts_custom_axis_labels():
    series = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 5])

    ax = histogram(
        series,
        fit_test="ks",
        show=False,
        x_label="Custom X",
        y_label="Custom Y",
        title="Custom Title",
    )

    assert ax.get_xlabel() == "Custom X"
    assert ax.get_ylabel() == "Custom Y"
    assert ax.get_title() == "Custom Title"


def test_time_series_plot_accepts_custom_axis_labels():
    series = pd.Series(
        [1.0, 1.5, 1.2, 1.8, 2.1, 2.5, 2.3, 2.9, 3.1, 3.5, 3.4, 3.8] * 3,
        index=pd.date_range("2020-01-01", periods=36, freq="D"),
        name="Revenue",
    )

    fig, _ = time_series_plot(
        series,
        seasonal_period=7,
        show=False,
        x_label="Date",
        y_label="Revenue USD",
        title="Revenue Trend",
    )

    ax = fig.axes[0]
    assert ax.get_xlabel() == "Date"
    assert ax.get_ylabel() == "Revenue USD"
    assert ax.get_title() == "Revenue Trend"


def test_time_series_plot_accepts_grouped_dataframe():
    grouped = pd.DataFrame(
        {
            "created_time": pd.date_range("2024-01-01", periods=30, freq="D").date,
            "count": [2, 5, 1, 4, 3, 6, 2, 7, 4, 5] * 3,
        }
    )

    fig, _ = time_series_plot(grouped, seasonal_period=7, show=False)

    ax = fig.axes[0]
    assert len(ax.lines[0].get_xdata()) == 30
    assert len(ax.lines[0].get_ydata()) == 30
    assert ax.get_ylabel() == "count"


def test_stacked_barplot_sorts_categories_by_total_count():
    df = pd.DataFrame(
        {
            "source": ["b", "a", "c", "b", "a", "a", "c", "b"],
            "status": [True, False, False, True, True, False, True, False],
        }
    )

    _, ax = stacked_percentage_barplot(
        df,
        title="Counts by Source",
        category_col="source",
        aggregate_col="status",
        colors=["#700202", "#027070"],
        figsize=(7, 4.5),
    )

    category_order = [tick.get_text() for tick in ax.get_xticklabels()]
    assert category_order == ["a", "b", "c"]


def test_clean_df_normalizes_selected_text_columns():
    df = pd.DataFrame(
        {
            "category": ["  foo   ", "BAR", "baz "],
            "value": [1, 2, 3],
        }
    )

    cleaned = clean_df(df, text_cols="category")

    assert cleaned["category"].tolist() == ["Foo", "Bar", "Baz"]
