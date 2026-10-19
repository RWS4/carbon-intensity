#set page(paper: "a4", margin: 1.5cm)
#set text(size: 9.5pt)
#set par(justify: true)
#set heading(numbering: none)
#show heading.where(level: 1): set text(size: 12pt)
#show heading.where(level: 1): set block(above: 0.9em, below: 0.5em)

#align(center)[
  #text(size: 15pt, weight: "bold")[Regional Carbon Intensity: Prototype Model]
]

= Assumptions

- Half-hourly regional data from the Carbon Intensity API (16 Jan 2025 to 16 Oct 2026, regions 1–17) is complete and representative.
- The API's `forecast` value is a suitable target; actual intensity is not used.
- Carbon intensity can be explained by region, day of year and time of day alone, with no weather, demand or generation-mix inputs.
- Times are in UTC; daylight saving shifts are ignored.
- That linear models would be unsuitable to explore the day/yearly cycles of carbon intensity. I ended up going with decision trees, but they cause a number of artefacts discussed in findings.

= Methods

Data was downloaded asynchronously in 14-day windows per region, cleaned with polars, written to per-window parquet files and combined into one dataset. Features are `regionid` (categorical), `day_of_year` (1–366) and `time_of_day` (integer half-hour slot, 1–48). An XGBoost regressor (squared-error objective, default hyperparameters) was trained on a random 80/20 train/test split and evaluated by RMSE. Model predictions were then plotted for region 1 (North Scotland) and region 13 (London) across single days and across the year at fixed times.

= Findings

- The use of a random split was in hindsight inappropriate, and I should have used alternating days.
- Integer time encoding breaks continuity between 23:30 and 00:00, and the random split leaks neighbouring half-hours into the test set, so the reported RMSE is optimistic. This can be overcome with kernel methods, but it would be better in the long run to use a dedicated time series method.
- Within a day, both regions rise from mid-afternoon into an evening peak; 19:30 is higher than 11:30 across the year.
- London sees a higher magnitude in cycle from midday to midnight. This seems intuitively correct as the density of London would make it harder to integrate distributed renewables at scale required to overcome this.
- The yearly curves are very spiky rather than smoothly seasonal. With only one or two years of data, the model largely memorises the weather of specific days, e.g. London on day 180 is predicted higher than on day 2.
- The model predicts small negative values for North Scotland, which is physically impossible and indicates overfitting.

#v(0.4em)
#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 6pt,
    image("figures/day_002.png", width: 100%),
    image("figures/day_180.png", width: 100%),
    image("figures/year_tod_24.png", width: 100%),
    image("figures/year_tod_40.png", width: 100%),
  ),
  caption: [Predicted carbon intensity (gCO#sub[2]/kWh). Legend: region 1 = North Scotland (blue), region 13 = London (orange). Top: across the half-hours of day 2 and day 180. Bottom: across the year at 11:30 and 19:30 UTC.],
)
