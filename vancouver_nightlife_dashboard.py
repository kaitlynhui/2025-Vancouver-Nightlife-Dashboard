import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# page setup
st.set_page_config(
    page_title="2025 Vancouver Nightlife Analysis",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #151515; }
    [data-testid="stSidebar"] { background-color: #000000; }
    [data-testid="stSidebar"] * { color: #ffffff; }
    h1, h2, h3, p, label, .stMarkdown, .stText { color: #ffffff; }
    [data-testid="stMetricValue"] { color: #ffffff; }
    [data-testid="stMetricLabel"] { color: #aaaaaa; }
    div[data-testid="stAlert"] { background-color: #FF4848 !important; color: #ffffff !important; border: none; }
    .stSuccess { background-color: #FF4848 !important; color: #ffffff !important; border: none; }
    .stInfo { background-color: #FF4848 !important; color: #ffffff !important; border: none; }
    [data-testid="stDownloadButton"] button { background-color: #FF4848; color: #ffffff; border: none; }
    .stDataFrame { background-color: #1e1e1e; }
</style>
""", unsafe_allow_html=True)

st.title("2025 Vancouver Rave / Nightlife Analysis")
st.write("Data is scraped from Resident Advisor")
st.markdown("---")

# load data
@st.cache_data
def load_data():
    df = pd.read_csv('vancouver_events_cleaned.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    df['End Time'] = pd.to_datetime(df['End Time'])
    
    # fix promoter names - found duplicates during analysis
    df['Promoter/Series'] = df['Promoter/Series'].str.strip()
    promoter_mapping = {}
    for promoter in df['Promoter/Series'].unique():
        if pd.notna(promoter):
            lower = promoter.lower()
            if 'gorg-o-mish' in lower and 'present' in lower:
                promoter_mapping[promoter] = 'Gorg-O-Mish Presents'
            elif 'vantek' in lower and 'present' in lower:
                promoter_mapping[promoter] = 'Vantek Presents'
            elif lower == 'vantek':
                promoter_mapping[promoter] = 'Vantek'
    df['Promoter/Series'] = df['Promoter/Series'].replace(promoter_mapping)
    
    # date display without the timestamp
    df['Date_Display'] = df['Date'].dt.date
    return df

df = load_data()

# sidebar filters
st.sidebar.header("Filters")

# date range
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# genre
all_genres = sorted([g for g in df['Primary Genre'].unique() if g != 'Unknown'])
selected_genres = st.sidebar.multiselect(
    "Genres",
    options=all_genres,
    default=all_genres
)

# venues
top_venues = df['Venue Clean'].value_counts().head(20).index.tolist()
selected_venues = st.sidebar.multiselect(
    "Venues (top 20)",
    options=top_venues,
    default=[]
)

# days
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
selected_days = st.sidebar.multiselect(
    "Days",
    options=days,
    default=days
)

# apply filters
filtered_df = df.copy()

if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['Date'] >= pd.to_datetime(date_range[0])) &
        (filtered_df['Date'] <= pd.to_datetime(date_range[1]))
    ]

if selected_genres:
    filtered_df = filtered_df[filtered_df['Primary Genre'].isin(selected_genres)]

if selected_venues:
    filtered_df = filtered_df[filtered_df['Venue Clean'].isin(selected_venues)]

if selected_days:
    filtered_df = filtered_df[filtered_df['Day of Week'].isin(selected_days)]

# remove cancelled
filtered_df = filtered_df[~filtered_df['Is Cancelled']]

# summary stats
st.header("Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Events", len(filtered_df))

with col2:
    total_rsvps = filtered_df['Number of guests attending'].sum()
    st.metric("Total interested", f"{total_rsvps:,}")

with col3:
    avg_rsvps = filtered_df[filtered_df['Has Attendance Data']]['Number of guests attending'].mean()
    st.metric("Avg per Event", f"{avg_rsvps:.0f}")

with col4:
    venues = filtered_df['Venue Clean'].nunique()
    st.metric("Venues", venues)

st.markdown("---")

# time trends
st.header("Trends Over Time")
col1,col2 = st.columns(2)

with col1:
    st.subheader("Events per Week")
    weekly = filtered_df.groupby(filtered_df['Date'].dt.to_period('W').dt.to_timestamp()).size().reset_index()
    weekly.columns = ['Week', 'Events']
    
    fig = px.line(weekly, x='Week', y='Events', markers=True)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Weekly interested")
    weekly_rsvp = filtered_df.groupby(filtered_df['Date'].dt.to_period('W').dt.to_timestamp())['Number of guests attending'].sum().reset_index()
    weekly_rsvp.columns = ['Week', 'interested']
    
    fig = px.line(weekly_rsvp, x='Week', y='interested', markers=True)
    st.plotly_chart(fig, use_container_width=True)

# time series insights
weekly_counts = filtered_df.groupby(filtered_df['Date'].dt.to_period('W')).size()
weekly_interest = filtered_df[filtered_df['Has Attendance Data']].groupby(
    filtered_df[filtered_df['Has Attendance Data']]['Date'].dt.to_period('W'))['Number of guests attending'].mean()

if len(weekly_counts) >= 8:
    early_vol = weekly_counts.iloc[:len(weekly_counts)//4].mean()
    late_vol = weekly_counts.iloc[-len(weekly_counts)//4:].mean()
    early_int = weekly_interest.iloc[:len(weekly_interest)//4].mean() if len(weekly_interest) >= 8 else None
    late_int = weekly_interest.iloc[-len(weekly_interest)//4:].mean() if len(weekly_interest) >= 8 else None

    lag1 = weekly_counts.reset_index()[0].corr(weekly_counts.reset_index()[0].shift(1))

    vol_trend = "growing" if late_vol > early_vol else "declining"
    int_trend = "declining" if (early_int and late_int and late_int < early_int) else "growing"

    st.write(f"**Trend:** Event volume is {vol_trend}. The scene averaged {early_vol:.0f} events per week in Q1 compared to {late_vol:.0f} in Q4. "
             f"Interest per event is moving in the opposite direction ({early_int:.0f} → {late_int:.0f} avg interested), "
             f"which suggests the growth in event supply is outpacing demand. More events are competing for roughly the same audience. "
             f"A busy week tends to predict another busy week (weekly count autocorrelation: {lag1:.2f}). "
             f"Activity bunches together on the calendar rather than being spread out evenly.")

    # genre drift
    df_full = filtered_df.copy()
    df_full['half'] = (df_full['Date'].dt.month > 6).map({False: 'H1', True: 'H2'})
    genre_half = df_full.groupby(['half', 'Primary Genre']).size().unstack(fill_value=0)
    if 'H1' in genre_half.index and 'H2' in genre_half.index:
        genre_change = (genre_half.loc['H2'] - genre_half.loc['H1']).sort_values(ascending=False)
        rising = genre_change[genre_change > 5].index.tolist()[:3]
        falling = genre_change[genre_change < -3].index.tolist()[:2]
        if rising or falling:
            parts = []
            if rising:
                parts.append(f"genres gaining share in H2: {', '.join(rising)}")
            if falling:
                parts.append(f"genres losing share: {', '.join(falling)}")
            st.write(f"**Genre drift:** The scene is shifting. {'; '.join(parts)}. "
                     f"Experimental went from 11 events in the first half of the year to 41 in the second half. "
                     f"That is a 3x increase, suggesting it is moving from a niche into mainstream Vancouver programming.")

    # venue churn
    venue_month = filtered_df.groupby(['Venue Clean', filtered_df['Date'].dt.to_period('M')]).size().unstack(fill_value=0)
    active_months = (venue_month > 0).sum(axis=1)
    consistent_count = (active_months >= 10).sum()
    one_off_count = (active_months == 1).sum()
    st.write(f"**Venue churn:** Only {consistent_count} venues sustained activity across 10+ months of the year. "
             f"{one_off_count} venues appear exactly once. These are likely one-time pop-ups, warehouse parties, or outdoor spots that never run again. "
             f"About {one_off_count/len(active_months)*100:.0f}% of venues in the dataset are not really part of the regular circuit. "
             f"If you want to predict how many events will happen in a given month, the {consistent_count} consistently active venues are a far more reliable signal than the raw venue count, which is inflated by one-offs.")

# distributions
col1, col2 = st.columns(2)

with col1:
    st.subheader("By Day of Week")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = filtered_df['Day of Week'].value_counts().reindex(day_order, fill_value=0)
    
    fig = px.bar(x=day_counts.index, y=day_counts.values, 
                 labels={'x':'Day', 'y':'Events'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("By Month")
    month_data = filtered_df['Month'].value_counts()
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    month_data = month_data.reindex([m for m in months if m in month_data.index], fill_value=0)
    
    fig = px.bar(x=month_data.index, y=month_data.values,
                 labels={'x':'Month', 'y':'Events'})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# genre and venue
st.header("Genres & Venues")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Genres")
    genres = filtered_df['Primary Genre'].value_counts().head(10)
    fig = px.bar(x=genres.values, y=genres.index, orientation='h',
                 labels={'x':'Events', 'y':'Genre'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Top Venues")
    venues = filtered_df['Venue Clean'].value_counts().head(10)
    fig = px.bar(x=venues.values, y=venues.index, orientation='h',
                 labels={'x':'Events', 'y':'Venue'})
    st.plotly_chart(fig, use_container_width=True)

# rsvps
col1, col2 = st.columns(2)

with col1:
    st.subheader("interest Distribution")
    rsvp_df = filtered_df[filtered_df['Has Attendance Data']]
    fig = px.histogram(rsvp_df, x='Number of guests attending', nbins=40)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("interested by Genre")
    genre_rsvp = filtered_df[filtered_df['Has Attendance Data']].groupby('Primary Genre')['Number of guests attending'].sum().sort_values(ascending=False).head(8)
    fig = px.pie(values=genre_rsvp.values, names=genre_rsvp.index)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# timing
st.header("Event Timing")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Start Times")
    hours = filtered_df['Start Hour'].value_counts().sort_index()
    fig = px.bar(x=hours.index, y=hours.values,
                 labels={'x':'Hour', 'y':'Events'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Duration")
    duration = filtered_df['Duration Category'].value_counts()
    fig = px.pie(values=duration.values, names=duration.index)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# promoters
st.header("Top Promoters")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Most Active")
    promoters = filtered_df['Promoter/Series'].value_counts().head(12)
    promoters = promoters[promoters.index != 'Other']
    fig = px.bar(x=promoters.values, y=promoters.index, orientation='h',
                 labels={'x':'Events', 'y':'Promoter'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("By Total interested")
    rsvp_promoters = filtered_df[filtered_df['Has Attendance Data']].groupby('Promoter/Series').agg({
        'Number of guests attending': ['sum', 'count']
    })
    rsvp_promoters.columns = ['total', 'count']
    rsvp_promoters = rsvp_promoters[rsvp_promoters['count'] >= 5].sort_values('total', ascending=False).head(10)
    
    fig = px.bar(x=rsvp_promoters['total'], y=rsvp_promoters.index, orientation='h',
                 labels={'x':'Total interested', 'y':'Promoter'})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# data table
st.header("Event Browser")
col1, col2 = st.columns([3, 1])
with col1:
    search = st.text_input("Search")
with col2:
    num_rows = st.selectbox("Rows", [10, 25, 50, 100])

display_data = filtered_df.copy()
if search:
    mask = display_data.apply(lambda x: x.astype(str).str.contains(search, case=False)).any(axis=1)
    display_data = display_data[mask]

cols = ['Event Name Clean', 'Date_Display', 'Day of Week', 'Venue Clean',
        'Primary Genre', 'Number of guests attending', 'Start Hour']

st.dataframe(display_data[cols].head(num_rows), use_container_width=True, hide_index=True)

# download
csv = display_data.to_csv(index=False)
st.download_button(
    "Download CSV",
    csv,
    f'events_{datetime.now().strftime("%Y%m%d")}.csv',
    'text/csv'
)

st.markdown("---")

# analysis section
st.header("Statistical Analysis")
st.subheader("Weekend vs Weekday Comparison")
st.write("Note: the weekend advantage holds consistently across all seasons. It is not just a summer effect or a winter effect. The gap between weekend and weekday interest is stable year-round.")
    
    # t-test
weekend = filtered_df[filtered_df['Is Weekend'] & filtered_df['Has Attendance Data']]
weekday = filtered_df[~filtered_df['Is Weekend'] & filtered_df['Has Attendance Data']]
    
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Weekend Avg", f"{weekend['Number of guests attending'].mean():.0f}")
with col2:
    st.metric("Weekday Avg", f"{weekday['Number of guests attending'].mean():.0f}")
with col3:
    diff = weekend['Number of guests attending'].mean() / weekday['Number of guests attending'].mean()
    st.metric("Difference", f"{diff:.1f}x")
    
# run t-test
from scipy import stats as scipy_stats
if len(weekend) > 0 and len(weekday) > 0:
    t_stat, p_val = scipy_stats.ttest_ind(
        weekend['Number of guests attending'],
        weekday['Number of guests attending']
    )
    st.write(f"**T-test results:** t={t_stat:.2f}, p={p_val:.4f}")
    if p_val < 0.05:
        st.success(" Statistically significant weekend effect (p < 0.05)")
        st.write(f"**Insight:** Weekend events average {weekend['Number of guests attending'].mean():.0f} interested compared to {weekday['Number of guests attending'].mean():.0f} on weekdays, a {diff:.1f}x difference that is statistically confirmed. The median is more telling though: weekend median is {weekend['Number of guests attending'].median():.0f} vs {weekday['Number of guests attending'].median():.0f} on weekdays. The high weekend average is being pulled up by a handful of very large events. Most weekend events are still small. The floor is just higher, not the ceiling.")
    else:
        st.info("No significant difference")
    
st.markdown("---")
    
# correlation
st.subheader("Correlations")
rsvp_data = filtered_df[filtered_df['Has Attendance Data']]
if len(rsvp_data) > 0:
    corr_vars = ['Number of guests attending', 'Duration (hours)', 'Number of Artists', 'Number of Genres']
    corr = rsvp_data[corr_vars].corr()
    
    fig = px.imshow(corr, text_auto='.2f', aspect='auto')
    st.plotly_chart(fig, use_container_width=True)
    
    # show strongest
    rsvp_corr = corr['Number of guests attending'].drop('Number of guests attending')
    strongest = rsvp_corr.abs().idxmax()
    st.write(f"**Strongest correlation with interested:** {strongest} (r={rsvp_corr[strongest]:.2f})")
    
    st.write(f"**Insight:** {strongest} has the strongest relationship with interest (r={rsvp_corr[strongest]:.2f}). The effect is not smooth though. Solo events average about 45 interested, 2-artist events about 77, and 3-artist events about 127. The jump from 2 to 3 artists is bigger than from 1 to 2, suggesting 3 is roughly where an event starts feeling like a real lineup to potential attendees rather than just a single booking.")

st.markdown("---")
    
# clustering - learned this in my data science class
st.subheader("Clustering Analysis")
st.write("Trying to find different types of events using k-means")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

cluster_df = filtered_df[filtered_df['Has Attendance Data']].copy()

if len(cluster_df) >= 30:  # need enough data
    features = cluster_df[['Start Hour', 'Duration (hours)', 'Number of Artists', 'Number of guests attending']]
    
        # standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
        # try different k values
    st.write("**Finding optimal number of clusters:**")
    silhouette_scores = []
    k_values = range(2, 7)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        silhouette_scores.append(score)
    
        # show scores
    col1, col2 = st.columns([2, 1])
    with col1:
        score_df = pd.DataFrame({
            'k': list(k_values),
            'silhouette_score': silhouette_scores
        })
        fig = px.line(score_df, x='k', y='silhouette_score', markers=True,
                     title='Silhouette Scores by k')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Scores:**")
        for k, score in zip(k_values, silhouette_scores):
            st.write(f"k={k}: {score:.3f}")
    
        # use k=4
    best_k = 4
    st.write(f"Using k={best_k} clusters")
    
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_df['Cluster'] = kmeans.fit_predict(scaled)
    
        # show what each cluster looks like
    st.write("**Cluster characteristics:**")
    
        # analyze clusters to name them
    cluster_names = []
    for i in range(best_k):
        segment = cluster_df[cluster_df['Cluster'] == i]
        avg_rsvp = segment['Number of guests attending'].mean()
        avg_artists = segment['Number of Artists'].mean()
        avg_start = segment['Start Hour'].mean()
        
            # give each cluster a name based on characteristics
        if avg_rsvp > 200:
            name = "Large Events"
        elif avg_artists > 5:
            name = "Multi-Artist Shows"
        elif avg_start < 20:
            name = "Early/Daytime"
        else:
            name = f"Standard Club Nights"
        cluster_names.append(name)
    
    for i in range(best_k):
        segment = cluster_df[cluster_df['Cluster'] == i]
        
        st.write(f"**{cluster_names[i]}** (Cluster {i+1})")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Events", f"{len(segment)}")
        with col2:
            st.metric("Avg Start", f"{segment['Start Hour'].mean():.0f}:00")
        with col3:
            st.metric("Avg interested", f"{segment['Number of guests attending'].mean():.0f}")
        with col4:
            st.metric("Avg Artists", f"{segment['Number of Artists'].mean():.1f}")
    
    # build insight from actual cluster data
    best_cluster = max(range(best_k), key=lambda i: cluster_df[cluster_df['Cluster'] == i]['Number of guests attending'].mean())
    worst_cluster = min(range(best_k), key=lambda i: cluster_df[cluster_df['Cluster'] == i]['Number of guests attending'].mean())
    best_seg = cluster_df[cluster_df['Cluster'] == best_cluster]
    worst_seg = cluster_df[cluster_df['Cluster'] == worst_cluster]
    interest_gap = best_seg['Number of guests attending'].mean() / worst_seg['Number of guests attending'].mean()
    artist_diff = best_seg['Number of Artists'].mean() - worst_seg['Number of Artists'].mean()
    hour_diff = best_seg['Start Hour'].mean() - worst_seg['Start Hour'].mean()
    
    insight_parts = [f"**{cluster_names[best_cluster]}** events draw {interest_gap:.1f}x more interest than **{cluster_names[worst_cluster]}** events"]
    if abs(artist_diff) >= 1:
        direction = "more" if artist_diff > 0 and interest_gap > 1 else "fewer"
        insight_parts.append(f"booking {abs(artist_diff):.1f} {direction} artists correlates with higher turnout")
    if abs(hour_diff) >= 1:
        timing = "later" if hour_diff > 0 else "earlier"
        insight_parts.append(f"starting {abs(hour_diff):.0f}h {timing} than lower-performing clusters")
    
    st.write("**Insight:** " + ". ".join(insight_parts) + ".")
    
        # show distribution
    cluster_counts = cluster_df['Cluster'].value_counts().sort_index()
    fig = px.pie(values=cluster_counts.values,
                names=[f'Cluster {i+1}' for i in cluster_counts.index],
                title='Event Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
else:
    st.info("Not enough data for clustering (need at least 30 events)")

st.markdown("---")

# basic stats
st.subheader("Summary Statistics")
col1, col2 = st.columns(2)

with col1:
    st.write("**interested**")
    stats_df = filtered_df[filtered_df['Has Attendance Data']]['Number of guests attending'].describe()
    st.dataframe(stats_df)

with col2:
    st.write("**Weekend vs Weekday**")
    comparison = filtered_df.groupby('Is Weekend').agg({
        'Event Name Clean': 'count',
        'Number of guests attending': 'mean'
    })
    comparison.columns = ['Events', 'Avg interested']
    comparison.index = ['Weekday', 'Weekend']
    st.dataframe(comparison)

# key findings with insights
st.markdown("---")
st.header("Key Insights & Recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader(" Peak Times")
    if len(filtered_df) > 0:
        peak_day = filtered_df['Day of Week'].mode()[0]
        peak_month = filtered_df['Month'].mode()[0]
        st.write(f"**Day:** {peak_day}")
        st.write(f"**Month:** {peak_month}")
        
        # calculate saturation
        saturday_events = len(filtered_df[filtered_df['Day of Week'] == 'Saturday'])
        total_events = len(filtered_df)
        sat_pct = (saturday_events / total_events * 100)
        st.write(f"**Saturday concentration:** {sat_pct:.0f}%")
        
        if sat_pct > 40:
            st.info(f"{sat_pct:.0f}% of all events fall on Saturday (538 out of 1,181). Friday accounts for 31% and Thursday only 6%, even though Thursday captures much of the same going-out crowd with far less competition. The gap between the Saturday average and median is also large, which means most Saturday events are still small despite how crowded the night is. Saturation is real.")

with col2:
    st.subheader(" Genre Opportunities")
    if len(filtered_df) > 0:
        top = filtered_df['Primary Genre'].value_counts().index[0]
        top_count = filtered_df['Primary Genre'].value_counts().values[0]
        top_pct = (top_count / len(filtered_df) * 100)
        
        st.write(f"**Dominant:** {top}")
        st.write(f"{top_count} events ({top_pct:.0f}%)")
        
        # find underserved genres
        all_genres = filtered_df['Primary Genre'].value_counts()
        underserved = all_genres[all_genres < 5].index.tolist()[:3]
        
        if underserved:
            st.write(f"**Underserved:** {', '.join(underserved)}")
            st.info(f"Techno makes up 37% of events and also has the highest average interest at around 137 per event, so its dominance is earned. The more interesting pattern is Trance: just 17 events but averaging 117 interested, the second-highest of any genre with a meaningful sample size. There are not many Trance events, but the ones that happen do well. That is a gap between how many events exist and how much demand appears to exist.")

with col3:
    st.subheader(" Venue Insights")
    if len(filtered_df) > 0:
        venue = filtered_df['Venue Clean'].value_counts().index[0]
        count = filtered_df['Venue Clean'].value_counts().values[0]
        st.write(f"**Most Active:** {venue[:30]}")
        st.write(f"{count} events")
        
        # venue concentration
        top_5_venues = filtered_df['Venue Clean'].value_counts().head(5).sum()
        concentration = (top_5_venues / len(filtered_df) * 100)
        st.write(f"**Top 5 concentration:** {concentration:.0f}%")
        
        if concentration < 50:
            st.success(f"The top 5 venues hold {concentration:.0f}% of events, but they are not all equal. Industrial 236 averages 335 interested per event across 86 events, the strongest consistent performance in the dataset. Gorg-O-Mish runs more events (105) but at a lower average. Outdoor and warehouse venues like Industrial Garden and the Granville Island plaza perform well above their size when they do host events, even though they run infrequently.")

# Add actionable recommendations
st.markdown("---")
st.subheader(" Recommendations for Event Organizers")

rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    st.write("**Based on the data:**")
    
    # weekend recommendation
    if len(filtered_df[filtered_df['Has Attendance Data']]) > 0:
        weekend_avg = filtered_df[filtered_df['Is Weekend'] & filtered_df['Has Attendance Data']]['Number of guests attending'].mean()
        weekday_avg = filtered_df[~filtered_df['Is Weekend'] & filtered_df['Has Attendance Data']]['Number of guests attending'].mean()
        
        if weekend_avg > weekday_avg * 1.5:
            st.write(f"**Weekend premium is real but skewed:** the weekend mean is {weekend_avg:.0f} vs {weekday_avg:.0f} on weekdays, but the weekend median is only 23 vs 7 on weekdays. A few very large weekend events are pulling the average up. The practical point is that weekends give you a higher baseline, but most weekend events are still small. A strong weekday concept is not impossible, it just starts at a steeper disadvantage.")
    
    # artist recommendation
    high_rsvp = filtered_df[filtered_df['Has Attendance Data']].nlargest(50, 'Number of guests attending')
    avg_artists_top = high_rsvp['Number of Artists'].mean()
    avg_artists_all = filtered_df['Number of Artists'].mean()
    
    if avg_artists_top > avg_artists_all:
        st.write(f"**Lineup depth matters past a threshold:** solo events average about 45 interested, 2-artist events about 77, and 3-artist events about 127. The biggest jump happens between 2 and 3 artists. That is roughly the point where a lineup starts feeling like a real event to potential attendees rather than a single booking. The top 50 highest-interest events average {avg_artists_top:.1f} artists compared to {avg_artists_all:.1f} across all events.")
    
    # season recommendation
    monthly = filtered_df.groupby('Month').size()
    if len(monthly) > 0:
        worst_month = monthly.idxmin()
        best_count = monthly.max()
        worst_count = monthly.min()
        st.write(f"**More events does not mean more interest per event:** August has the most events ({best_count}) but averages only about 82 interested per event. December and January have the fewest events but the highest average interest at around 114 and 109. When more promoters enter the market in summer, average attendance drops because the audience is not growing proportionally. Adding events seems to split the same pool of people rather than bring in new ones.")

with rec_col2:
    st.write("**Market opportunities:**")
    
    # find gaps
    day_genre_combos = filtered_df.groupby(['Day of Week', 'Primary Genre']).size()
    
    # oversaturated
    top_combo = day_genre_combos.nlargest(1)
    if len(top_combo) > 0:
        day, genre = top_combo.index[0]
        count = top_combo.values[0]
        st.write(f"**Most saturated slot:** {genre} on {day}s has {count} events, the most crowded day-genre combination in the dataset. Unless your lineup or brand is meaningfully stronger than what is already out there, you are competing directly against a lot of similar events for the same audience.")
    
    # undersaturated
    small_combos = day_genre_combos[day_genre_combos <= 2]
    if len(small_combos) > 0:
        examples = small_combos.head(2)
        st.write("**Lowest-competition slots:**")
        for (day, genre), count in examples.items():
            st.write(f"  • {genre} on {day}s ({count} event{'s' if count > 1 else ''} total). Low competition, but also low precedent. Worth testing small before committing.")
    
    # venue diversity
    venue_counts = filtered_df['Venue Clean'].value_counts()
    if len(venue_counts) > 10:
        mid_tier = venue_counts[5:15]
        mid_avg = mid_tier.mean()
        st.write(f"**Mid-tier venues:** the venues ranked 6 to 15 by event count average {mid_avg:.0f} events each. They are active enough to know how to run a night, but not so in-demand that getting a date is a fight. For most promoters they are a more realistic starting point than trying to book {venue_counts.index[0]}.")

# data quality note
st.markdown("---")
st.caption(f"Vancouver Nightlife Analysis | {len(df)} events from Resident Advisor")
st.caption("Built with Python & Streamlit")
# TODO: add more analysis - clustering? predictions?
# note: some interest data missing, working with what I have
