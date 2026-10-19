import polars as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures"

#NOTE: this module is the whole pipeline for load data -> feature engineering -> training -> evaluation. It only runs because the dataset and training are toy examples and I would obviously create encapsulated pipeline steps down the line.


#NOTE: this combines loading the dataframe from file and some feature engineering on the assumption of using xgboost downstream. I went with integer encoding of the time steps, which cause the model to misbehave around the day cycle (ie 48->0 or 23:30->00:00). With more time I would build a pipeline around a purpose built library like PyTorch Forecast or similar. 
def load_df():
    df = pl.read_parquet("data/combined.parquet")
    df = df.with_columns(pl.col("interval_start").dt.ordinal_day().alias("day_of_year"))
    df = df.with_columns((pl.col("interval_start").dt.hour() * 2 + pl.col("interval_start").dt.minute() // 30 + 1).alias("time_of_day"))
    df = df.select(pl.col("regionid"),pl.col("forecast"),pl.col("day_of_year"),pl.col("time_of_day"))
    #TODO: create separate functionality to handle oversight of workflow steps
    print(df)
    return df

#NOTE: I wasted a bit of time researching/thinking of appropriate analysis methods, so in the end I just went with a method I had done before. Wouldn't stand up to scientific scrutiny. both the feature engineering and decision tree model lead to poor inference outcomes. Discussed further in write up.
#TODO: create ability to save/load model instead of rerunning training.
def regress(df: pl.DataFrame, target: str = "forecast", categorical: tuple[str, ...] = ("regionid",)):
    # Fit an XGBoost model predicting the target from the remaining columns
    features = df.drop(target)
    feature_types = ["c" if col in categorical else "q" for col in features.columns]
    X = features.to_numpy()
    y = df[target].to_numpy()
    #TODO: optimise splits and check assumptions around these arguments
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize XGBRegressor
    #TODO: optimise splits and check assumptions around these arguments
    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        enable_categorical=True,
        feature_types=feature_types,
    )

    # Fit the model to training data
    model.fit(X_train, y_train)

    # Make predictions on the held-out data
    predictions = model.predict(X_test)
    print(predictions[:5])
    print(f"test RMSE: {root_mean_squared_error(y_test, predictions):.2f}")
    return model


def save_figure(fig, name: str):
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def plot_day(model: XGBRegressor, day_of_year: int, region_ids: list[int] = [13]):
    # Feature columns must be in the same order the model was trained on: regionid, day_of_year, time_of_day
    time_of_day = list(range(1, 49))
    labels = [f"{(t - 1) // 2:02d}:{(t - 1) % 2 * 30:02d}" for t in time_of_day]

    fig, ax = plt.subplots(figsize=(10, 5))
    for region_id in region_ids:
        X = pl.DataFrame({
            "regionid": [region_id] * 48,
            "day_of_year": [day_of_year] * 48,
            "time_of_day": time_of_day,
        }).to_numpy()
        ax.plot(time_of_day, model.predict(X), label=f"region {region_id}")

    ax.set_xticks(time_of_day[::4], labels[::4])
    ax.set_xlabel("time of day (UTC)")
    ax.set_ylabel("predicted forecast (gCO2/kWh)")
    ax.set_title(f"Predicted carbon intensity, day {day_of_year}")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, f"day_{day_of_year:03d}.png")
    return fig


def plot_year(model: XGBRegressor, time_of_day: int, region_ids: list[int] = [13]):
    # Feature columns must be in the same order the model was trained on: regionid, day_of_year, time_of_day
    day_of_year = list(range(1, 367))
    label = f"{(time_of_day - 1) // 2:02d}:{(time_of_day - 1) % 2 * 30:02d}"

    fig, ax = plt.subplots(figsize=(10, 5))
    for region_id in region_ids:
        X = pl.DataFrame({
            "regionid": [region_id] * len(day_of_year),
            "day_of_year": day_of_year,
            "time_of_day": [time_of_day] * len(day_of_year),
        }).to_numpy()
        ax.plot(day_of_year, model.predict(X), label=f"region {region_id}")

    ax.set_xlabel("day of year")
    ax.set_ylabel("predicted forecast (gCO2/kWh)")
    ax.set_title(f"Predicted carbon intensity at {label} UTC")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, f"year_tod_{time_of_day:02d}.png")
    return fig


def run():
    model = regress(load_df())
    plot_day(model, day_of_year=2, region_ids=[1, 13])
    plot_year(model, time_of_day=40, region_ids=[1, 13])


if __name__ == "__main__":
    run()
