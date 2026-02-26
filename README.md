# 2025-Vancouver-Nightlife-Dashboard

**Vancouver Nightlife Data Analysis** <br>
A data analysis dashboard exploring Vancouver's electronic music scene using event data from Resident Advisor.

**About This Project** <br>
This project analyzes 1,192 nightlife events in Vancouver from 2025 to identify trends, patterns, and insights about the local music scene.
**Data Source:** Resident Advisor (scraped event data)
**Location:** Vancouver, British Columbia, Canada
**Time Period:** January 1 - December 31 2025

**What's Included**
- Data Cleaning Script (clean_data.py) 
- Interactive Dashboard (vancouver_nightlife_dashboard.py) - Streamlit web app for exploring the data
- Advanced Analysis (advanced_eda.py)
- Cleaned Dataset (vancouver_events_cleaned.csv) 

**Key Findings**

Summary Statistics
- 1,192 total events analyzed
- 199 unique venues
- 887 unique artists
- 92,631 total events marked interested

Main Insights
- Weekend events get 2x more interested-s than weekday events (93 vs 46 average)
- Techno is the dominant genre (37% of all events)
- August is the busiest month (131 events)
- Saturday is the most popular day (544 events)
- Most events start between 10-11pm

Statistical Findings
- Weekend vs weekday attendance difference is statistically significant (p = 0.0006)
- Built a Random Forest model that predicts event RSVPs with 62% accuracy
- Number of artists is the strongest predictor of attendance (38% feature importance)



Pre-Processessing Steps: 
- Fix inconsistent venue names
- Standardize promoter names (e.g., "VANTEK" and "Vantek Presents")
- Extract primary genre from multi-genre fields
- Calculate event duration and timing features
- Remove cancelled events
- Handle missing values

Feature engineering:
- Day of week, month, season
- Start hour, time of day
- Event duration categories
- Attendance categories
- Promoter/series extraction

**Dashboard Features**

  Filters:
  - Date range selector
  - Genre filter
  - Venue filter
  - Day of week filter

  Visualizations:
  - Time series of events and RSVPs
  - Genre and venue breakdowns
  - Event timing patterns
  - Promoter rankings
  - Correlation analysis

  Data Explorer:
  - Searchable event table
  - Sortable columns
  - CSV download

Analysis Methods
- Statistical Testing:
- T-tests for group comparisons
- Chi-square tests for independence
- Correlation analysis

Machine Learning:
- K-means clustering for market segmentation
- Random Forest for RSVP prediction
- Feature importance analysis

Time Series:
- Weekly aggregation
- Seasonality detection
- Trend analysis

**Technical Stack**
Python 3.9+
Pandas - Data manipulation
NumPy - Numerical operations
Matplotlib/Seaborn - Statistical plotting
Plotly - Interactive charts
Streamlit - Web dashboard
Scikit-learn - Machine learning
SciPy - Statistical tests

**Future Improvements**
- Add sentiment analysis from event descriptions
- Include ticket pricing data
- Compare Vancouver to other cities
- Track artist touring patterns
- Build recommendation system

**Data Notes**
"Number of guests attending" refers to RSVPs/interested on Resident Advisor, not actual attendance
Some venues listed as "TBA" (to be announced)
11 events were cancelled (0.9%)
89.8% of events have RSVP data

**Contact**
Questions about this project? Feel free to reach out!

**License**
This is a student project for educational purposes. Data sourced from Resident Advisor.
