# tanaka-tools (`ttools`)

Python package of data-analytics helpers for exploratory analysis and reporting: dataframe cleaning, reading SpreadsheetML XML files, joining dataframes through a JSON match list, and plotting (correlation heatmap, histogram with goodness-of-fit test, time-series diagnostics, stacked bar plot).

Distribution name `tanaka-tools`, import name `ttools`. Requires Python 3.11 or newer.

## Dependencies

numpy, pandas, scipy, matplotlib, seaborn, scikit-learn, statsmodels. Optional: `ollama` (only for `clean_df(model=...)`, which also needs a running local Ollama server).

## Install

```bash
pip install "git+ssh://git@github.com/Mattanaka343/tanaka-tools.git"
# or, from a local checkout
pip install -e .
```

## Usage

```python
import ttools as tt
```

Exports: `clean_df`, `corr_heatmap`, `histogram`, `json_join`, `read_excel_xml`, `time_series_plot`, `stacked_barplot`, `__version__`.

| Module | Contents |
|---|---|
| `ttools.cleaners` | `clean_df` |
| `ttools.graphs` | `corr_heatmap`, `histogram`, `time_series_plot`, `stacked_barplot` |
| `ttools.joiners` | `json_join` |
| `ttools.readers` | `read_excel_xml` |

### clean_df

```python
clean_df(data, reindex_col=None, bool_cols=None, model=None, true_pats=None,
         num_cols=None, text_cols=None, replace_comma_with='',
         drop_na=False, drop_dupes=False, drop_cols=None) -> pd.DataFrame
```

Returns a cleaned copy, applying in order: reindex, boolean conversion, numeric conversion, text normalisation, drop NA, drop duplicates, drop columns, column renaming.

- `reindex_col`: column moved to the index.
- `bool_cols` / `true_pats`: string columns converted to booleans using `str.contains(pattern, na=False)`. `true_pats` is one pattern for all columns or a list matched to `bool_cols` by position, and is required when `bool_cols` is given.
- `num_cols`: string columns converted to numeric. Values containing no digit become NaN; commas are replaced with `replace_comma_with` (`''` or `'.'`).
- `text_cols`: whitespace stripped and converted to sentence case (`"  foo  "` becomes `"Foo"`).
- Column names: accents removed, `. ? ¿ ¡ !` replaced by spaces, stripped, spaces to `_`, lowercased. If `model` is given, a local Ollama model proposes snake_case names from each column's name, dtype and sample values.

```python
df = tt.clean_df(raw, bool_cols=["Asistencia", "Contratado"], true_pats="SI",
                 text_cols="Ubicación Geográfica", drop_dupes=True)
```

### read_excel_xml

```python
read_excel_xml(path: str) -> pd.DataFrame
```

Reads SpreadsheetML 2003 XML (`urn:schemas-microsoft-com:office:spreadsheet`). The first row is the header; values are strings.

### json_join

```python
json_join(json_path, left, right, left_on, right_on,
          left_key="left_value", right_key="right_value",
          how="inner", suffixes=("_x", "_y")) -> pd.DataFrame
```

Joins `left` and `right` using value pairs from a JSON file holding a list of records, e.g. `[{"left_value": "ACME Inc", "right_value": "Acme"}]`. `left_on` values are matched to `left_key`, `right_on` values to `right_key`; `how` and `suffixes` go to pandas merge.

### corr_heatmap

```python
corr_heatmap(data, title=None, annot=True, cmap='coolwarm', show=False,
             save_path=None, x_label=None, y_label=None) -> Figure
```

Lower-triangle correlation heatmap of the numeric columns. `cmap` is a seaborn colormap name or a list of hex colours.

### histogram

```python
histogram(data: pd.Series, fit_test='ks', dist='norm', show=False,
          color='#700202', save_path=None, ax=None, x_label=None,
          y_label=None, title=None, **kwargs) -> Figure | Axes
```

Histogram with goodness-of-fit results shown on the plot. `fit_test` is `'ks'`, `'shapiro'` (normality only; `dist` ignored) or `'anderson'`. Extra keyword arguments are passed to the histogram call. Passing `ax` draws on that axes.

### time_series_plot

```python
time_series_plot(series, seasonal_period=None, alpha=0.05, figsize=(14, 7),
                 c1='#700202', c2='#027070', show=True, save_dir=None,
                 x_label=None, y_label=None, title=None) -> tuple[Figure, dict]
```

Plots a series with a trend line and runs diagnostics. Input is a Series with a datetime index, or a DataFrame with exactly one numeric column and one time column. At least 20 non-missing observations are required. `seasonal_period` is detected from the ACF if omitted. `save_dir` is a file path for the saved image. The dict has keys `trend`, `adf`, `seasonality`, `white_noise`, `figure`.

```python
fig, res = tt.time_series_plot(daily_counts, seasonal_period=7, show=False)
print(res["trend"]["conclusion"], res["adf"]["stationary"])
```

### stacked_barplot

```python
stacked_barplot(df, title, category_col, aggregate_col,
                colors=["#700202", "#027070"], figsize=(8, 5),
                ylabel="Count", xlabel=None, rotation=30) -> (fig, ax)
```

Stacked count bars per `category_col`, segmented by `aggregate_col`, ordered by descending total with totals labelled. Sets the seaborn theme (`whitegrid`, `talk`).

```python
fig, ax = tt.stacked_barplot(df, "Hires by source", "source", "hired")
```

## Testing

```bash
pip install pytest
pytest
```

Tests are in `tests/test_graphs.py`.

## Versioning

Version `0.1.0`, set in `pyproject.toml` and `ttools/__init__.py`.
