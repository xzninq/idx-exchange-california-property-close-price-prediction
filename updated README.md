## Dataset
- Source: CRMLS (California Regional Multiple Listing Service) sold-listing exports,
  files prefixed `CRMLSSold` (one file per month).
- Months used: {', '.join(str(m) for m in months_sorted)}
- Scope: `PropertyType = "Residential"` and `PropertySubType = "SingleFamilyResidence"` only,
  per the task spec.
- Rows after scoping and dropping missing target: {len(df):,}

## Train / test split
- Test set: most recent month ({test_month}).
- Training window: tuned by trying every window length from 1 up to
  {max_x} month(s) immediately before the test month and picking the one
  with the best Linear Regression test R². Chosen window: **{best_x} month(s)**
  ({', '.join(str(m) for m in train_months)}).
- See `window_tuning_results.csv` for the full sweep.

## Preprocessing
- Outlier trim: `ClosePrice` restricted to its 1st-99th percentile
  (${price_lo:,.0f} - ${price_hi:,.0f}), dropping {n_before_outlier_trim - len(df):,}
  of {n_before_outlier_trim:,} rows. Without this, a handful of multi-million/
  multi-hundred-million-dollar sales (and a few sub-$100 data-entry errors)
  dominate the error metrics and make every model -- especially the tree-based
  ones -- look far worse than they actually are on typical homes.
- Missing numeric values: median imputation.
- Missing categorical values: most-frequent imputation.
- Numeric features scaled with `StandardScaler`.
- Categorical features one-hot encoded (`handle_unknown="ignore"`).

## Feature sets
- **Old**: {numeric_features_old + categorical_features_old}
- **New**: {numeric_features_new + categorical_features_new}
  - `BedBathRatio` = bedrooms / bathrooms
  - `PropertyAge` = close year - year built
  - `SchoolDistrict`: {"spatially joined from the CA School District Areas 2024-25 boundaries" if school_join_method == "spatial_join" else f"taken from the MLS export's own '{school_join_method.split(':')[-1]}' field (spatial join to the CA school district boundary layer was not available in this run -- swap it back in wherever geopandas + internet access is available)"}

## Models tested
Linear Regression (baseline), Decision Tree, Random Forest, Gradient Boosting
(scikit-learn `GradientBoostingRegressor`; swap in XGBoost/LightGBM the same
way if those packages are available in your environment).

## Results (new feature set, test month = {test_month})
{metrics_summary.to_markdown(index=False)}

**Best model: {best_model_name}**

### Error by price band
{band_summary.to_markdown(index=False)}

## How to re-run
1. Upload at least 6 months of `CRMLSSold*.csv` files into Colab or in the same folder as
   `pipeline.py`
2. `pip install pandas numpy scikit-learn matplotlib geopandas shapely`
   (geopandas/shapely are optional -- the script falls back to the MLS
   export's own school-district field if they, or internet access, aren't
   available).
3. Run `python pipeline.py`. Plots and CSV summaries are written to `outputs/`.

## Files produced
- `outputs/metrics_summary.csv` -- R2/MAE/RMSE/MAPE/MdAPE per model.
- `outputs/price_band_summary.csv` -- error broken down by price quartile.
- `outputs/old_vs_new_features.csv` -- feature-engineering impact.
- `outputs/gb_hyperparameter_tuning.csv` -- Gradient Boosting tuning sweep.
- `outputs/window_tuning_results.csv` -- training-window length sweep.
- `outputs/*.png` -- EDA and comparison plots.
