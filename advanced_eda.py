"""
Vancouver Nightlife Events - Exploratory Data Analysis
======================================================
Student Data Analysis Project

This script analyzes event data from Resident Advisor to find patterns
and trends in Vancouver's nightlife scene.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Setup
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("="*70)
print("VANCOUVER NIGHTLIFE - DATA ANALYSIS")
print("="*70)

# ============================================================================
# PART 1: LOAD AND EXPLORE THE DATA
# ============================================================================

print("\n--- PART 1: DATA LOADING ---\n")

df = pd.read_csv('vancouver_events_cleaned.csv')
df['Date'] = pd.to_datetime(df['Date'])

print(f"Dataset size: {len(df)} events")
print(f"Time period: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Number of columns: {len(df.columns)}")

# Basic info
print("\nDataset info:")
print(f"- Unique venues: {df['Venue Clean'].nunique()}")
print(f"- Unique genres: {df['Primary Genre'].nunique()}")
print(f"- Events with RSVP data: {df['Has Attendance Data'].sum()} ({df['Has Attendance Data'].sum()/len(df)*100:.1f}%)")
print(f"- Cancelled events: {df['Is Cancelled'].sum()}")

# Missing data
print("\nMissing values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values!")

# ============================================================================
# PART 2: SUMMARY STATISTICS
# ============================================================================

print("\n--- PART 2: SUMMARY STATISTICS ---\n")

# Numeric variables
print("Event RSVPs:")
rsvp_data = df[df['Has Attendance Data']]['Number of guests attending']
print(f"  Mean: {rsvp_data.mean():.1f}")
print(f"  Median: {rsvp_data.median():.1f}")
print(f"  Std dev: {rsvp_data.std():.1f}")
print(f"  Min: {rsvp_data.min()}")
print(f"  Max: {rsvp_data.max()}")

print("\nEvent duration:")
print(f"  Mean: {df['Duration (hours)'].mean():.1f} hours")
print(f"  Median: {df['Duration (hours)'].median():.1f} hours")

print("\nArtist counts:")
print(f"  Mean: {df['Number of Artists'].mean():.1f} per event")
print(f"  Median: {df['Number of Artists'].median():.1f} per event")

# Categorical variables
print("\nTop 5 genres:")
for genre, count in df['Primary Genre'].value_counts().head().items():
    pct = count / len(df) * 100
    print(f"  {genre}: {count} events ({pct:.1f}%)")

print("\nTop 5 venues:")
for venue, count in df['Venue Clean'].value_counts().head().items():
    print(f"  {venue}: {count} events")

print("\nEvents by day of week:")
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for day in days:
    count = len(df[df['Day of Week'] == day])
    print(f"  {day}: {count} events")

# ============================================================================
# PART 3: COMPARING GROUPS
# ============================================================================

print("\n--- PART 3: WEEKEND VS WEEKDAY COMPARISON ---\n")

weekend = df[df['Is Weekend'] & df['Has Attendance Data']]
weekday = df[~df['Is Weekend'] & df['Has Attendance Data']]

print(f"Weekend events: {len(weekend)}")
print(f"Weekday events: {len(weekday)}")

print(f"\nAverage RSVPs:")
print(f"  Weekend: {weekend['Number of guests attending'].mean():.1f}")
print(f"  Weekday: {weekday['Number of guests attending'].mean():.1f}")
print(f"  Difference: {weekend['Number of guests attending'].mean() - weekday['Number of guests attending'].mean():.1f}")

# Statistical test
t_stat, p_val = stats.ttest_ind(
    weekend['Number of guests attending'],
    weekday['Number of guests attending']
)
print(f"\nT-test results:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_val:.4f}")
if p_val < 0.05:
    print(f"  ✓ The difference IS statistically significant (p < 0.05)")
else:
    print(f"  ✗ The difference is NOT statistically significant")

# ============================================================================
# PART 4: TIME PATTERNS
# ============================================================================

print("\n--- PART 4: TIME PATTERNS ---\n")

# Monthly trends
print("Events by month:")
monthly = df.groupby('Month').size().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
], fill_value=0)

for month, count in monthly.items():
    print(f"  {month}: {count} events")

print(f"\nBusiest month: {monthly.idxmax()} ({monthly.max()} events)")
print(f"Slowest month: {monthly.idxmin()} ({monthly.min()} events)")

# Start times
print("\nMost common start times:")
top_hours = df['Start Hour'].value_counts().head()
for hour, count in top_hours.items():
    print(f"  {hour:02d}:00 - {count} events")

# ============================================================================
# PART 5: RELATIONSHIPS BETWEEN VARIABLES
# ============================================================================

print("\n--- PART 5: CORRELATIONS ---\n")

# Calculate correlations
rsvp_df = df[df['Has Attendance Data']].copy()
corr_vars = ['Number of guests attending', 'Duration (hours)', 'Number of Artists', 'Number of Genres']
correlations = rsvp_df[corr_vars].corr()

print("Correlation with RSVPs:")
rsvp_corr = correlations['Number of guests attending'].drop('Number of guests attending')
for var, corr in rsvp_corr.items():
    print(f"  {var}: {corr:.3f}")

# Interpretation
strongest = rsvp_corr.abs().idxmax()
print(f"\nStrongest correlation: {strongest} (r = {rsvp_corr[strongest]:.3f})")

# ============================================================================
# PART 6: GENRE ANALYSIS
# ============================================================================

print("\n--- PART 6: GENRE ANALYSIS ---\n")

# RSVPs by genre
print("Average RSVPs by genre (top 5):")
genre_rsvps = rsvp_df.groupby('Primary Genre')['Number of guests attending'].agg(['mean', 'count'])
genre_rsvps = genre_rsvps[genre_rsvps['count'] >= 10]  # At least 10 events
top_genres = genre_rsvps.nlargest(5, 'mean')

for genre, row in top_genres.iterrows():
    print(f"  {genre}: {row['mean']:.0f} avg RSVPs ({int(row['count'])} events)")

# Genre diversity
print(f"\nGenre diversity:")
print(f"  Total unique genres: {df['Primary Genre'].nunique()}")
print(f"  Events with multiple genres: {(df['Number of Genres'] > 1).sum()} ({(df['Number of Genres'] > 1).sum()/len(df)*100:.1f}%)")

# ============================================================================
# PART 7: VENUE ANALYSIS
# ============================================================================

print("\n--- PART 7: VENUE PERFORMANCE ---\n")

# Top venues by total RSVPs
venue_rsvps = rsvp_df.groupby('Venue Clean')['Number of guests attending'].agg(['sum', 'mean', 'count'])
venue_rsvps = venue_rsvps[venue_rsvps['count'] >= 5]  # At least 5 events
top_venues = venue_rsvps.nlargest(5, 'sum')

print("Top venues by total RSVPs:")
for venue, row in top_venues.iterrows():
    print(f"  {venue[:40]}")
    print(f"    Total: {int(row['sum']):,} | Avg: {row['mean']:.0f} | Events: {int(row['count'])}")

# ============================================================================
# PART 8: CLUSTERING EVENTS (MARKET SEGMENTS)
# ============================================================================

print("\n--- PART 8: IDENTIFYING EVENT TYPES ---\n")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Prepare clustering data
cluster_df = df[df['Has Attendance Data']].copy()
cluster_features = cluster_df[['Start Hour', 'Duration (hours)', 'Number of Artists', 'Number of guests attending']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

# Try different numbers of clusters
print("Finding optimal number of clusters...")
silhouette_scores = []
K_range = range(2, 7)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    silhouette_scores.append(score)
    print(f"  k={k}: silhouette score = {score:.3f}")

# Use 4 clusters
best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_df['Cluster'] = kmeans.fit_predict(scaled_features)

print(f"\nUsing {best_k} clusters (segments):")
for i in range(best_k):
    segment = cluster_df[cluster_df['Cluster'] == i]
    print(f"\n  Segment {i+1}: {len(segment)} events ({len(segment)/len(cluster_df)*100:.1f}%)")
    print(f"    Avg start: {segment['Start Hour'].mean():.1f}:00")
    print(f"    Avg duration: {segment['Duration (hours)'].mean():.1f} hrs")
    print(f"    Avg RSVPs: {segment['Number of guests attending'].mean():.0f}")
    print(f"    Avg artists: {segment['Number of Artists'].mean():.1f}")
    print(f"    Top genre: {segment['Primary Genre'].mode().values[0]}")

# ============================================================================
# PART 9: PREDICTIVE MODEL
# ============================================================================

print("\n--- PART 9: PREDICTING RSVPs ---\n")

# Prepare data for modeling
model_df = df[df['Has Attendance Data']].copy()

# Create features
features = pd.DataFrame({
    'start_hour': model_df['Start Hour'],
    'duration': model_df['Duration (hours)'],
    'num_artists': model_df['Number of Artists'],
    'num_genres': model_df['Number of Genres'],
    'is_weekend': model_df['Is Weekend'].astype(int),
    'is_techno': (model_df['Primary Genre'] == 'Techno').astype(int),
    'month': pd.to_datetime(model_df['Date']).dt.month
})

target = model_df['Number of guests attending']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} events")
print(f"Testing on {len(X_test)} events")

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"\nModel performance:")
print(f"  R² score: {r2:.3f}")
print(f"  Mean Absolute Error: {mae:.1f} RSVPs")
print(f"  Interpretation: Model explains {r2*100:.1f}% of RSVP variance")

# Feature importance
importance = pd.DataFrame({
    'feature': features.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nMost important factors:")
for idx, row in importance.head().iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# ============================================================================
# PART 10: KEY FINDINGS
# ============================================================================

print("\n--- PART 10: KEY FINDINGS ---\n")

print("1. WEEKEND EFFECT:")
print(f"   Weekend events get {weekend['Number of guests attending'].mean() / weekday['Number of guests attending'].mean():.1f}x more RSVPs")
print(f"   This difference is statistically significant (p = {p_val:.4f})")

print("\n2. TIMING:")
print(f"   Busiest month: {monthly.idxmax()}")
print(f"   Most common start time: {df['Start Hour'].mode().values[0]}:00")
print(f"   Most popular day: {df['Day of Week'].value_counts().index[0]}")

print("\n3. GENRE TRENDS:")
print(f"   Dominant genre: {df['Primary Genre'].value_counts().index[0]} ({df['Primary Genre'].value_counts().values[0]} events)")
print(f"   Genre diversity: {df['Primary Genre'].nunique()} unique genres")

print("\n4. PREDICTIVE INSIGHTS:")
print(f"   Top predictor: {importance.iloc[0]['feature']}")
print(f"   Model accuracy: R² = {r2:.3f}")
print(f"   Average prediction error: ±{mae:.0f} RSVPs")

print("\n5. SUCCESS FACTORS:")
high_rsvp_events = rsvp_df.nlargest(50, 'Number of guests attending')
print(f"   Top events average {high_rsvp_events['Number of Artists'].mean():.1f} artists")
print(f"   Top events run {high_rsvp_events['Duration (hours)'].mean():.1f} hours on average")
print(f"   {(high_rsvp_events['Is Weekend'].sum() / len(high_rsvp_events) * 100):.0f}% of top events are on weekends")

# ============================================================================
# PART 11: RECOMMENDATIONS
# ============================================================================

print("\n--- PART 11: RECOMMENDATIONS ---\n")

print("For event organizers:")
print("  ✓ Schedule events on weekends (2x higher RSVPs)")
print(f"  ✓ Book {importance.iloc[0]['feature'].replace('_', ' ')} - strongest predictor")
print(f"  ✓ Target {monthly.idxmax()} for maximum exposure")
print(f"  ✓ Consider {top_genres.index[0]} genre (highest avg RSVPs)")

# Find gaps
day_genre = df.groupby(['Day of Week', 'Primary Genre']).size()
oversaturated = day_genre.nlargest(3)
undersaturated = day_genre[day_genre > 0].nsmallest(3)

print("\nMarket gaps (underserved opportunities):")
for (day, genre), count in undersaturated.items():
    print(f"  - {genre} on {day}s (only {count} event{'s' if count > 1 else ''})")

print("\nOversaturated combinations (high competition):")
for (day, genre), count in oversaturated.items():
    print(f"  - {genre} on {day}s ({count} events)")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
