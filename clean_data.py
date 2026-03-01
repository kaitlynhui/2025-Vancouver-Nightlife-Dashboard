import pandas as pd
import numpy as np
from datetime import datetime
import re

print("=" * 80)
print("VANCOUVER NIGHTLIFE DATA CLEANING")
print("=" * 80)

# Load the data
print("\n1. Loading data...")
df = pd.read_csv('/mnt/user-data/uploads/2025__events.csv')
print(f"   ✓ Loaded {len(df)} events")
print(f"   ✓ Columns: {list(df.columns)}")

# Create a copy for cleaning
df_clean = df.copy()

print("\n2. Inspecting data quality...")
print(f"   • Shape: {df_clean.shape}")
print(f"   • Missing values:\n{df_clean.isnull().sum()}")
print(f"   • Data types:\n{df_clean.dtypes}")

print("\n3. Converting date/time columns...")
# Convert date columns to datetime
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
df_clean['Start Time'] = pd.to_datetime(df_clean['Start Time'])
df_clean['End Time'] = pd.to_datetime(df_clean['End Time'])

# Extract useful time features
df_clean['Day of Week'] = df_clean['Date'].dt.day_name()
df_clean['Month'] = df_clean['Date'].dt.month_name()
df_clean['Month Number'] = df_clean['Date'].dt.month
df_clean['Week of Year'] = df_clean['Date'].dt.isocalendar().week
df_clean['Is Weekend'] = df_clean['Day of Week'].isin(['Friday', 'Saturday', 'Sunday'])

# Calculate event duration in hours
df_clean['Duration (hours)'] = (df_clean['End Time'] - df_clean['Start Time']).dt.total_seconds() / 3600

# Start hour (for understanding event start times)
df_clean['Start Hour'] = df_clean['Start Time'].dt.hour

print("   ✓ Converted dates and extracted time features")

print("\n4. Cleaning venue names...")
# Clean venue names - remove "TBA - " prefix
df_clean['Venue Clean'] = df_clean['Venue'].str.replace(r'^TBA\s*-\s*', '', regex=True).str.strip()
df_clean['Is TBA Venue'] = df_clean['Venue'].str.contains('TBA', case=False, na=False)

print("   ✓ Cleaned venue names")

print("\n5. Processing genres...")
# Handle multiple genres - create primary genre column
def extract_primary_genre(genre_str):
    """Extract the first genre from comma-separated list"""
    if pd.isna(genre_str) or genre_str == '':
        return 'Unknown'
    genres = str(genre_str).split(',')
    return genres[0].strip()

df_clean['Primary Genre'] = df_clean['Genres'].apply(extract_primary_genre)

# Count number of genres per event
df_clean['Number of Genres'] = df_clean['Genres'].apply(
    lambda x: len(str(x).split(',')) if pd.notna(x) and x != '' else 0
)

print("   ✓ Processed genre information")

print("\n6. Processing artists...")
# Count number of artists
df_clean['Number of Artists'] = df_clean['Artists'].apply(
    lambda x: len(str(x).split(',')) if pd.notna(x) and x != '' else 0
)

# Check if event has artists listed
df_clean['Has Artists'] = df_clean['Number of Artists'] > 0

print("   ✓ Processed artist information")

print("\n7. Handling cancelled events...")
# Identify cancelled events
df_clean['Is Cancelled'] = df_clean['Event name'].str.contains('CANCELLED', case=False, na=False)

# Clean event names (remove [CANCELLED] prefix)
df_clean['Event Name Clean'] = df_clean['Event name'].str.replace(r'\[CANCELLED\]\s*', '', regex=True, case=False).str.strip()

print(f"   ✓ Found {df_clean['Is Cancelled'].sum()} cancelled events")

print("\n8. Processing attendance data...")
# Handle missing/zero attendance
df_clean['Has Attendance Data'] = df_clean['Number of guests attending'] > 0

# Categorize attendance
def categorize_attendance(num):
    if num == 0:
        return 'No data'
    elif num < 50:
        return 'Small (0-50)'
    elif num < 150:
        return 'Medium (50-150)'
    elif num < 300:
        return 'Large (150-300)'
    else:
        return 'Very Large (300+)'

df_clean['Attendance Category'] = df_clean['Number of guests attending'].apply(categorize_attendance)

print("   ✓ Categorized attendance data")

print("\n9. Identifying event series/promoters...")
# Extract promoter/series from event name
def extract_promoter(event_name):
    """Extract promoter/series name from event title"""
    # Common patterns: "PROMOTER:", "PROMOTER presents", "PROMOTER -"
    patterns = [
        r'^([^:]+):',  # Everything before first colon
        r'^([^-]+)\s*-',  # Everything before first dash
        r'^(\w+)\s+presents',  # Promoter presents
    ]
    
    for pattern in patterns:
        match = re.search(pattern, str(event_name), re.IGNORECASE)
        if match:
            promoter = match.group(1).strip()
            # Filter out very short or generic matches
            if len(promoter) > 3 and not promoter.isdigit():
                return promoter
    return 'Other'

df_clean['Promoter/Series'] = df_clean['Event Name Clean'].apply(extract_promoter)

print("   ✓ Extracted promoter/series information")

print("\n10. Creating additional useful features...")
# Season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df_clean['Season'] = df_clean['Month Number'].apply(get_season)

# Time of day category
def categorize_start_time(hour):
    if 6 <= hour < 12:
        return 'Morning (6am-12pm)'
    elif 12 <= hour < 17:
        return 'Afternoon (12pm-5pm)'
    elif 17 <= hour < 22:
        return 'Evening (5pm-10pm)'
    else:
        return 'Late Night (10pm-6am)'

df_clean['Time of Day'] = df_clean['Start Hour'].apply(categorize_start_time)

# Event type based on duration
def categorize_duration(hours):
    if hours < 3:
        return 'Short (< 3 hrs)'
    elif hours < 6:
        return 'Medium (3-6 hrs)'
    else:
        return 'Long (6+ hrs)'

df_clean['Duration Category'] = df_clean['Duration (hours)'].apply(categorize_duration)

print("   ✓ Created additional features")

print("\n11. Data quality summary...")
print(f"\n   Data Quality Report:")
print(f"   {'='*60}")
print(f"   Total events: {len(df_clean)}")
print(f"   Date range: {df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}")
print(f"   Cancelled events: {df_clean['Is Cancelled'].sum()}")
print(f"   Events with attendance data: {df_clean['Has Attendance Data'].sum()}")
print(f"   Events with artists listed: {df_clean['Has Artists'].sum()}")
print(f"   Events with genre data: {(df_clean['Primary Genre'] != 'Unknown').sum()}")
print(f"   Unique venues: {df_clean['Venue Clean'].nunique()}")
print(f"   Unique genres: {df_clean['Primary Genre'].nunique()}")
print(f"   {'='*60}")

print("\n12. Saving cleaned data...")
# Select final columns for export
columns_to_export = [
    'Event Name Clean', 'Date', 'Start Time', 'End Time',
    'Day of Week', 'Month', 'Month Number', 'Season', 'Week of Year',
    'Is Weekend', 'Start Hour', 'Time of Day',
    'Duration (hours)', 'Duration Category',
    'Artists', 'Number of Artists', 'Has Artists',
    'Venue', 'Venue Clean', 'Is TBA Venue',
    'Number of guests attending', 'Attendance Category', 'Has Attendance Data',
    'Genres', 'Primary Genre', 'Number of Genres',
    'Genre IDs', 'Event URL',
    'Promoter/Series', 'Is Cancelled'
]

df_export = df_clean[columns_to_export].copy()

# Save to CSV
output_path = '/home/claude/vancouver_events_cleaned.csv'
df_export.to_csv(output_path, index=False)
print(f"   ✓ Saved cleaned data to: {output_path}")

# Save summary statistics
summary_path = '/home/claude/data_cleaning_summary.txt'
with open(summary_path, 'w') as f:
    f.write("VANCOUVER NIGHTLIFE DATA CLEANING SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Original dataset: {len(df)} events\n")
    f.write(f"Cleaned dataset: {len(df_export)} events\n")
    f.write(f"Date range: {df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}\n\n")
    
    f.write("COLUMNS ADDED:\n")
    new_cols = [col for col in df_export.columns if col not in df.columns]
    for col in new_cols:
        f.write(f"  • {col}\n")
    
    f.write(f"\nDATA QUALITY:\n")
    f.write(f"  • Cancelled events: {df_clean['Is Cancelled'].sum()}\n")
    f.write(f"  • Events with attendance: {df_clean['Has Attendance Data'].sum()}\n")
    f.write(f"  • Events with artists: {df_clean['Has Artists'].sum()}\n")
    f.write(f"  • Events with genres: {(df_clean['Primary Genre'] != 'Unknown').sum()}\n")
    
    f.write(f"\nTOP INSIGHTS:\n")
    f.write(f"  • Most common genre: {df_clean['Primary Genre'].value_counts().index[0]}\n")
    f.write(f"  • Most popular venue: {df_clean[df_clean['Has Attendance Data']].groupby('Venue Clean')['Number of guests attending'].sum().idxmax()}\n")
    f.write(f"  • Busiest month: {df_clean['Month'].value_counts().index[0]}\n")
    f.write(f"  • Most common day: {df_clean['Day of Week'].value_counts().index[0]}\n")

print(f"   ✓ Saved summary to: {summary_path}")

print("\n" + "="*80)
print("CLEANING COMPLETE!")
print("="*80)
print("\nNext steps:")
print("  1. Review the cleaned CSV file")
print("  2. Check the summary statistics")
print("  3. Load into the Streamlit dashboard for EDA")
print("\n" + "="*80)
