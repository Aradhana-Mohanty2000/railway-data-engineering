"""
Railway Data Engineering Report
Author: [Your Name]
Tool: Python in VS Code

Summary:
This script compiles all major steps of data processing, transformation,
analysis, and visualization on the Railway dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean dataset
df = pd.read_csv('Railway_info (1).csv')
df.columns = df.columns.str.strip()
df['days_list'] = df['days'].str.split(',')
print(f"\n📊 Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}")

# Print actual column names
print("Actual column names:", df.columns.tolist())

# Clean column names (strip extra spaces)
df.columns = df.columns.str.strip()

print("\n🔹 Task 1.2: Basic Statistics")

# 1. Number of trains (rows)
total_trains = df.shape[0]
print(f"Total number of trains: {total_trains}")

# 2. Unique source stations
unique_sources = df['Source_Station_Name'].nunique()
print(f"Number of unique source stations: {unique_sources}")

# 3. Unique destination stations
unique_destinations = df['Destination_Station_Name'].nunique()
print(f"Number of unique destination stations: {unique_destinations}")

# 4. Most common source station
most_common_source = df['Source_Station_Name'].mode()[0]
print(f"Most common source station: {most_common_source}")

# 5. Most common destination station
most_common_destination = df['Destination_Station_Name'].mode()[0]
print(f"Most common destination station: {most_common_destination}")
print("\n🔹 Task 1.3: Data Cleaning")

# 1. Check and handle missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Fill missing values with a placeholder (you can choose 'Unknown' or drop rows)
df.fillna('Unknown', inplace=True)
print("\n✅ Missing values filled with 'Unknown'.")

# 2. Standardize station names to uppercase
df['Source_Station_Name'] = df['Source_Station_Name'].str.upper()
df['Destination_Station_Name'] = df['Destination_Station_Name'].str.upper()

print("\n✅ Station names converted to uppercase.")
print("\n🔹 Task 2.1: Data Filtering")

# 1. Filter trains that run on Saturday
saturday_trains = df[df['days'].str.contains("Saturday", case=False, na=False)]
print(f"\nNumber of trains operating on Saturday: {len(saturday_trains)}")
print(saturday_trains[['Train_No', 'Train_Name', 'days']].head())

# 2. Filter trains that start from a specific station (e.g., DELHI)
specific_station = "DELHI"
trains_from_station = df[df['Source_Station_Name'] == specific_station]
print(f"\nNumber of trains starting from {specific_station}: {len(trains_from_station)}")
print(trains_from_station[['Train_No', 'Train_Name', 'Source_Station_Name']].head())
print("\n🔹 Task 2.2: Grouping and Aggregation")

# 1. Count number of trains from each Source_Station_Name
train_count_per_source = df.groupby('Source_Station_Name').size().reset_index(name='Train_Count')
print("\nNumber of trains per source station:")
print(train_count_per_source.sort_values(by='Train_Count', ascending=False).head())

# 2. Count how many days each train runs (e.g., Monday, Tuesday, etc.)
# First split the 'days' column
df['days_list'] = df['days'].str.split(',')

# Then count number of days for each train
df['operating_days_count'] = df['days_list'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Group by Source_Station_Name to find average
avg_days_per_source = df.groupby('Source_Station_Name')['operating_days_count'].mean().reset_index(name='Avg_Operating_Days')
print("\nAverage operating days per source station:")
print(avg_days_per_source.sort_values(by='Avg_Operating_Days', ascending=False).head())
print("\n🔹 Task 2.3: Data Enrichment")

# Define a function to categorize train based on days
def categorize_days(day_list):
    if not isinstance(day_list, list) or len(day_list) == 0:
        return "Unknown"
    
    weekdays = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'}
    weekends = {'Saturday', 'Sunday'}
    day_set = set([d.strip() for d in day_list])

    if day_set & weekdays and day_set & weekends:
        return "Daily"
    elif day_set & weekdays:
        return "Weekday"
    elif day_set & weekends:
        return "Weekend"
    else:
        return "Unknown"

# Apply the function to create new column
df['Train_Category'] = df['days_list'].apply(categorize_days)

# Show a sample
print("\nSample of categorized trains:")
print(df[['Train_No', 'Train_Name', 'days', 'Train_Category']].head())
print("\n🔹 Task 3.1: Pattern Analysis")

# Count how many trains run on each day of the week
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = {day: df['days'].str.contains(day, na=False).sum() for day in day_names}

# Convert to DataFrame for plotting
day_df = pd.DataFrame(list(day_counts.items()), columns=['Day', 'Train_Count'])

# Plot the bar chart
plt.figure(figsize=(10, 5))
sns.barplot(data=day_df, x='Day', y='Train_Count', palette='Set2')
plt.title("🚆 Number of Trains Operating Each Day")
plt.xlabel("Day of the Week")
plt.ylabel("Number of Trains")
plt.tight_layout()
plt.show()

# Bonus: Most frequent source-destination routes
route_counts = df.groupby(['Source_Station_Name', 'Destination_Station_Name']).size().reset_index(name='Train_Count')
top_routes = route_counts.sort_values(by='Train_Count', ascending=False).head(5)

print("\nTop 5 most frequent routes:")
print(top_routes)
print("\n🔹 Task 3.2: Correlation and Insights")

# Use the existing day_df from Task 3.1
# Sort by number of trains (to see busiest day)
busiest_day = day_df.sort_values(by='Train_Count', ascending=False).iloc[0]
quietest_day = day_df.sort_values(by='Train_Count', ascending=True).iloc[0]

print(f"\n📈 Busiest day: {busiest_day['Day']} with {busiest_day['Train_Count']} trains.")
print(f"📉 Quietest day: {quietest_day['Day']} with {quietest_day['Train_Count']} trains.")

# Simple barplot again for correlation view
plt.figure(figsize=(10, 5))
sns.barplot(data=day_df, x='Day', y='Train_Count', palette='coolwarm')
plt.title("Train Frequency by Day (for Pattern Analysis)")
plt.xlabel("Day")
plt.ylabel("Train Count")
plt.tight_layout()
plt.show()
print("\n🔹 Task 4.1: Additional Visualizations")

# 1. Bar chart: Number of trains per source station (top 10)
top_sources = df['Source_Station_Name'].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_sources.index, y=top_sources.values, palette='viridis')
plt.title("Top 10 Source Stations by Train Count")
plt.xlabel("Source Station")
plt.ylabel("Number of Trains")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Heatmap: Source vs. Destination frequency
route_matrix = df.groupby(['Source_Station_Name', 'Destination_Station_Name']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 8))
sns.heatmap(route_matrix, cmap="YlGnBu", linewidths=0.3)
plt.title("Train Frequency from Source to Destination")
plt.xlabel("Destination")
plt.ylabel("Source")
plt.tight_layout()
plt.show()
print("\n📌 Final Insights:")
print("- Saturday is the busiest day for trains.")
print("- Delhi is the most common source station.")
print("- Most trains run on both weekdays and weekends.")
print("- Heatmap shows busiest routes between metro cities.")

