# Vancouver Nightlife Analysis (2025)

An exploratory data analysis project exploring Vancouver's electronic music scene using event data scraped from Resident Advisor. The dataset covers 1,181 non-cancelled events from January to December 2025 across 199 venues.

---

## Table of Contents 

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Data Collection and Cleaning](#data-collection-and-cleaning)
- [Analysis Techniques](#analysis-techniques)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Project Overview

This project started from a simple question: what does the structure of Vancouver's nightlife scene actually look like when you put it in a spreadsheet? Resident Advisor publishes event listings publicly, but there is no clean dataset of Vancouver events available anywhere. This project builds that dataset from scratch and uses it to gain insights about timing, genre, venue performance, and what predicts whether an event gets traction.

The goal was not just to describe the scene but to apply real analytical methods (statistical testing, clustering, time series analysis, and machine learning) to a domain that does not usually get this treatment.

---

## Dataset

- **Source:** Resident Advisor event listings, scraped using an open-source scraping tool and cleaned manually
- **Scope:** 1,181 events (11 cancelled events removed), January 1 to December 31, 2025
- **Coverage:** 199 unique venues, 887 unique artists, 92,631 total interested counts
- **Key fields:** event name, date, start/end time, venue, artists, genres, number of guests interested, promoter/series

The "Number of guests attending" field reflects how many RA users marked themselves as interested in an event, not actual door counts. This is a proxy for demand, not a direct measure of attendance.

---

## How to Run

### Requirements

```
Python 3.9+
pandas
numpy
plotly
streamlit
scikit-learn
scipy
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the dashboard

```bash
streamlit run vancouver_nightlife_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Running the cleaning script

If you want to re-run the data cleaning from the raw scraped file:

```bash
python clean_data.py
```

This outputs `vancouver_events_cleaned.csv`.

### Running advanced analysis

```bash
python advanced_eda.py
```

---

## Project Structure

```
├── vancouver_nightlife_dashboard.py   # Main Streamlit dashboard
├── clean_data.py                      # Data cleaning pipeline
├── advanced_eda.py                    # Standalone analysis scripts
├── vancouver_events_cleaned.csv       # Cleaned dataset
└── README.md
```

---

## Data Collection and Cleaning

The raw data came from a GitHub scraping tool pointed at Resident Advisor's Vancouver listings. The scraper handles pagination and pulls structured event data, but the output required significant cleaning before it was usable.

**Venue normalization.** Venue names had inconsistent casing, spacing, and abbreviations. For example, "Industrial 236" appeared as several variants. These were collapsed into a single canonical name per venue using fuzzy matching and manual review.

**Promoter deduplication.** Promoter names like "VANTEK", "Vantek Presents", and "VANTEK PRESENTS" were all referring to the same entity. These were standardized using substring matching rules rather than automated deduplication, because automated tools were collapsing genuinely different promoters with similar names.

**Genre parsing.** Resident Advisor stores genres as a comma-separated string (e.g., "Techno, House"). The first genre was extracted as the primary genre, which is used for most analysis. The full string is preserved in the `Genres` column.

**Time features.** Duration was calculated from start and end times. Events ending after midnight were handled carefully to avoid negative durations. Start hour, time of day category, day of week, month, season, and week of year were all derived from the datetime fields.

**Cancelled events.** 11 events (0.9%) were flagged as cancelled on RA and removed from all analysis.

**Missing attendance data.** 10.2% of events had no interested count, either because the event was too new or because RA did not record it. These were excluded from attendance-related analysis rather than imputed, since imputing a zero would misrepresent low-traction events as no-data events.

---

## Analysis Techniques

### Exploratory Data Analysis

Standard distributional analysis across event count, attendance, genre, venue, day, and month. The main challenge here was distinguishing signal from noise in a dataset where most events are small and a few are very large. Mean and median are both reported throughout because the mean is consistently inflated by outliers.

### Statistical Testing

A two-sample t-test was used to compare weekend and weekday attendance. The test confirmed a statistically significant difference (t = 3.56, p = 0.0006). Effect size was also examined because statistical significance alone does not tell you whether the difference is large enough to matter practically.

A Pearson correlation matrix was computed across numeric features (number of artists, duration, number of genres, attendance). Correlation was used to identify relationships worth investigating further rather than as a standalone finding.

### Clustering (K-Means)

K-means clustering was applied to events with attendance data, using start hour, duration, number of artists, and attendance as features. Features were standardized before clustering since they operate on different scales. The number of clusters was selected using silhouette scores across k = 2 to 6, with k = 4 producing the most interpretable result. Clusters were then characterized by their mean values to give them descriptive labels.

One limitation of k-means is that it assumes spherical clusters and is sensitive to outliers, both of which are potential issues in this dataset given the skewed attendance distribution.

### Time Series Analysis

Events were aggregated by week to examine volume and interest trends over the year. A lag-1 autocorrelation of 0.48 in weekly event count indicates that activity clusters on the calendar rather than being randomly distributed. This has practical implications: the market is not equally competitive in every week of the year.

Genre share was compared between the first and second halves of the year to identify which genres were growing or declining in representation. This is a simple but direct way to detect drift in a single-year dataset where longer-term trend analysis is not possible.

Venue consistency was measured by counting how many months each venue appeared in. Only 12 venues were active across 10 or more months. 129 venues appeared exactly once, representing pop-ups, one-off warehouse events, and outdoor locations that never recur.

### Machine Learning (Random Forest)

A Random Forest regressor was trained to predict event attendance using features available before the event (day of week, start hour, number of artists, genre, venue, season). The model was evaluated using train/test split with the test set drawn from the later portion of the year to avoid data leakage from temporal structure. Feature importance analysis identified number of artists as the strongest predictor at 38% importance, followed by venue.

A 62% R² on the test set means the model explains roughly 62% of the variance in attendance. This is reasonable given that many attendance drivers (artist reputation, ticket price, marketing budget) are not in the dataset.

---

## Key Findings

**Weekend vs weekday attendance.** Weekend events average 93 interested compared to 46 on weekdays, a statistically confirmed 2x gap. The median tells a more useful story though: weekend median is 23 vs 7 on weekdays. Most weekend events are still small. The gap is real but the ceiling is not as high as the mean suggests.

**Lineup size has a threshold effect.** Solo events average about 45 interested. Events with 2 artists average 77, and events with 3 artists average 127. The jump from 2 to 3 is larger than from 1 to 2. Three artists appears to be roughly where a booking starts reading as a proper lineup to potential attendees rather than a single act.

**More events in a month does not mean more interest per event.** August has the most events (130) but only averages 82 interested. December and January have the fewest events but the highest average interest (114 and 109). When more promoters enter the market in summer, average attendance drops. The total audience does not grow proportionally with supply.

**Saturday is saturated.** 46% of all events fall on Saturday. The average Saturday event is not significantly better attended than a strong Friday event, and Thursday has only 6% of events despite capturing much of the same going-out demand.

**Trance is underrepresented relative to its performance.** Techno dominates at 37% of events and earns it with the highest average interest per event (~137). But Trance has only 17 events and averages 117 interested, the second-highest mean of any genre with a meaningful sample size. The supply of Trance events does not match the apparent demand.

**Industrial 236 is the strongest consistent venue.** It averages 335 interested per event across 86 events. Gorg-O-Mish runs more events (105) but at a lower average. Outdoor and warehouse venues perform well when they run but appear infrequently.

**Genre share is shifting.** Experimental went from 11 events in H1 to 41 in H2, a 3x increase in the second half of the year. Drum and Bass declined slightly. The scene is not static across the calendar year.

---

## Limitations

The attendance data is RA interest counts, not door counts. These are correlated with actual attendance but not the same thing. Events with strong local followings who don't use RA will appear artificially small.

This is one year of data from one city. Findings about seasonality, genre trends, and venue performance are specific to Vancouver in 2025 and should not be generalized.

The dataset only includes events listed on Resident Advisor. Events promoted entirely through Instagram, private mailing lists, or other channels are not captured. The dataset likely skews toward established promoters with RA presence.

K-means clustering is sensitive to initial conditions and the choice of k. The four-cluster solution was chosen because it produced interpretable results, not because it is provably optimal.

---

## Future Work

- Compare Vancouver's scene structure to other mid-sized cities on RA (Montreal, Toronto, Berlin) to assess whether the patterns here are local or universal
- Track individual artist trajectories over time to identify emerging acts before they break through
- Incorporate ticket pricing data to separate demand signals from capacity constraints
- Apply survival analysis to model venue longevity. Which venue characteristics predict whether a space stays active?
- Build a simple recommender system that suggests events based on a user's genre and timing preferences

---

## License

Student project for educational purposes. Event data sourced from Resident Advisor.
