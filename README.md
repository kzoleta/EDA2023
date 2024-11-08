# Python - Exploratory Data Analysis on Spotify 2023 Dataset

## Overview

This repository provides a structured Exploratory Data Analysis (EDA) of Spotify's "Most Streamed Songs of 2023," based on the Kaggle dataset available [here](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023) . This EDA provides a comprehensive look into Spotify’s trending songs for 2023, with insights that may offer valuable context for future music popularity analyses.

## 🍄Files

| File Name                                                                                      | Description                                                                                               |              
| :---:                                                                                          | :---:                                                                             
|🌸 [EDA_Spotify2023.ipnyb](https://github.com/kzoleta/EDA2023/blob/main/EDA_Spotify2023.ipynb)  | This file contains the code for the Exploratory Data Analysis of Spotify's "Most Streamed Songs of 2023". |                                        
| 🌸 [README.md](https://github.com/kzoleta/EDA2023/blob/main/README.md)                         | This file caters the overview of the repository                                                          |


## 🍄Code
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from the specified file path, using ISO-8859-1 encoding to handle special characters.
df = pd.read_csv(r'C:\Users\FRIMAKK\Downloads\archive\spotify-2023.csv', encoding='ISO-8859-1')
```

```python
# Display basic information about the dataset, including its shape, data types, and the number of missing values in each column.
print(f"Dataset shape: {df.shape}")
print(f"Data types:\n{df.dtypes}")
print(f"Null values:\n{df.isnull().sum()}")
```

```python
# Convert the 'streams' column to numeric values, setting any errors (non-numeric values) to NaN, and display the updated data type.
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
print(f"Updated data type of 'streams' column: {df['streams'].dtypes}")
```

```python
# Calculate and display basic statistics (mean, median, standard deviation) for the 'streams' column.
mean_streams = df['streams'].mean()
median_streams = df['streams'].median()
std_streams = df['streams'].std()
print(f"Mean: {mean_streams}")
print(f"Median: {median_streams}")
print(f"Standard Deviation: {std_streams}")
```

```python
# Plot the distribution of release years and artist count side by side for comparison using histograms.
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(df['released_year'], kde=False, ax=axes[0]).set(title='Release Year Distribution')
sns.histplot(df['artist_count'], kde=False, ax=axes[1]).set(title='Artist Count Distribution')
plt.tight_layout()
plt.show()
```

```python
# Plot a scatter plot showing the relationship between the release year and artist count, with reduced opacity for better visibility.
plt.figure(figsize=(8, 4))
sns.scatterplot(data=df, x='released_year', y='artist_count', alpha=0.4)
plt.title('Released Year vs Artist Count')
plt.show()
```

```python
# Identify and plot the top 5 tracks by the highest stream count using a horizontal bar plot.
top_tracks = df.sort_values(by='streams', ascending=False).head()
plt.figure(figsize=(9, 6))
sns.barplot(x='streams', y='track_name', data=top_tracks, hue='streams', palette='pastel', legend=False)
plt.title('Top 5 Most Streamed Tracks')
plt.xticks(rotation=45)
plt.show()
```

```python
# Display the top 5 artists with the most tracks in the dataset by counting the frequency of their appearances in the 'artist(s)_name' column.
top_artists = df['artist(s)_name'].value_counts().head().index.tolist()
print(f"Top 5 Artists: {top_artists}")
```

```python
# Filter the dataset for the top 5 artists and plot the number of tracks for each artist using a count plot.
top_artists_data = df[df['artist(s)_name'].isin(top_artists)]
sns.countplot(x='artist(s)_name', data=top_artists_data, hue='artist(s)_name', palette='muted', legend=False)
plt.title('Top 5 Artists by Track Count')
plt.xlabel('Artist')
plt.ylabel('Number of Tracks')
plt.show()
```

```python
# Calculate and display the year with the highest number of tracks, as well as the count of tracks released in that year.
release_year_counts = df['released_year'].value_counts().reset_index()
release_year_counts.columns = ['Year', 'Track Count']
max_year = release_year_counts.loc[release_year_counts['Track Count'].idxmax(), 'Year']
max_tracks = release_year_counts['Track Count'].max()
print(f"Year with most tracks: {max_year} ({max_tracks} tracks)")
```

```python
# Plot the number of tracks released each year using a bar plot, highlighting the track count for each year.
plt.figure(figsize=(14, 9))
sns.barplot(data=release_year_counts, x='Year', y='Track Count', hue='Year', palette='pastel', legend=False)
plt.title('Tracks Released Per Year')
plt.xlabel('Year')
plt.ylabel('Track Count')
plt.xticks(rotation=45)
plt.show()
```

```python
# Calculate and display the month with the highest number of releases, then plot the number of releases per month in a bar plot.
month_counts = df['released_month'].value_counts().reset_index()
month_counts.columns = ['Month', 'Track Count']
month_counts['Month'] = month_counts['Month'].map({
    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
})
max_month = month_counts.loc[month_counts['Track Count'].idxmax(), 'Month']
max_month_tracks = month_counts['Track Count'].max()
print(f"Month with highest releases: {max_month} ({max_month_tracks} tracks)")
```

```python
# Plot the number of tracks released each month using a bar plot, with months displayed in order.
sns.barplot(data=month_counts, x='Month', y='Track Count', hue='Month', palette='pastel', legend=False)
plt.title('Tracks Released Per Month')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.show()
```

```python
# Calculate and display the correlation matrix between streams and other features such as danceability, energy, and valence.
correlation_matrix = df[['streams', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".1g")
plt.title('Correlation Matrix')
plt.show()
```

```python
# Calculate and print the correlation between specific features, such as danceability vs. energy and valence vs. acousticness.
dance_energy_corr = df[['danceability_%', 'energy_%']].corr()
valence_acoustic_corr = df[['valence_%', 'acousticness_%']].corr()
print(f"Danceability vs. Energy:\n{dance_energy_corr}")
print(f"Valence vs. Acousticness:\n{valence_acoustic_corr}")
```

```python
# Analyze the number of tracks featured in different playlists and plot the total count using a bar plot.
playlist_counts = df[['in_spotify_playlists', 'in_deezer_playlists', 'in_apple_playlists']].apply(pd.to_numeric, errors='coerce').sum()
sns.barplot(x=playlist_counts.index, y=playlist_counts.values, hue=playlist_counts.index, palette='pastel', legend=False)
plt.title('Tracks in Playlists')
plt.ylabel('Track Count')
plt.yscale('log')
plt.show()
```

```python
# Display information about the top 5 tracks that appear in multiple playlists across platforms.
top_playlist_tracks = df.loc[[55, 179, 86, 620, 41], ['track_name', 'artist(s)_name', 'in_spotify_playlists', 'in_deezer_playlists', 'in_apple_playlists']]
print(top_playlist_tracks)
```

```python
# Calculate and display the average number of streams for each mode and key combination, sorted by streams.
average_streams = df.groupby(['mode', 'key'])['streams'].mean().reset_index()
sorted_streams = average_streams.sort_values(by='streams', ascending=False)
print(f"Average streams by mode and key:\n{sorted_streams}")
```

```python
# Plot the average streams for each key, broken down by mode, using a bar plot.
sns.barplot(data=sorted_streams, x='key', y='streams', hue='mode', palette='muted')
plt.title('Average Streams by Mode and Key')
plt.xlabel('Key')
plt.ylabel('Average Streams')
plt.show()
```

```python
# Aggregate the number of appearances of artists across various platforms and plot the top 5 artists with the most appearances.
playlist_columns = ['in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts']
df[playlist_columns] = df[playlist_columns].apply(pd.to_numeric, errors='coerce')
artist_appearances = df.groupby('artist(s)_name')[playlist_columns].sum()
artist_appearances['Total_appearances'] = artist_appearances.sum(axis=1)
sorted_appearances = artist_appearances.sort_values(by='Total_appearances', ascending=False)
```

```python
# Plot the top 5 artists by the total number of appearances in playlists using a bar plot.
sns.barplot(data=sorted_appearances.head(), x='artist(s)_name', y='Total_appearances', hue='artist(s)_name', palette='pastel', legend=False)
plt.title('Top 5 Artists by Appearances in Playlists')
plt.ylabel('Total Appearances')
plt.show()
```

## 🍄Addressed Guide Questions:

### 🦢Overview of Dataset
1. The dataset has 953 rows and 24 columns
___
2.
![image](https://github.com/user-attachments/assets/cb8b7d81-3802-4375-8ce5-216ed7f2453b)
___
The missing values:

![image](https://github.com/user-attachments/assets/25130fd7-2606-403b-a504-69317a9ea170)

___

### 🦢Basic Descriptive Statistics
3.
![image](https://github.com/user-attachments/assets/6d36ae3e-05a5-4fbd-a558-d7b0110f4db3)
___
4.

![image](https://github.com/user-attachments/assets/9e3f2602-3283-403c-88f1-c6c1122df217)

___

![image](https://github.com/user-attachments/assets/4604ec51-fdd6-4ac6-b617-a3079380c530)

___

### 🦢Top Performers
5.
![image](https://github.com/user-attachments/assets/9e2c4976-ff93-47f6-9251-55ce4269266a)
___
6.

![image](https://github.com/user-attachments/assets/cd0a2333-b5f8-4609-9585-3a8f80c875ef)

___

### 🦢Temporal Trends
7.
![image](https://github.com/user-attachments/assets/8cac03c1-df0e-46da-b766-7f71e826e7e3)
___
8.
![image](https://github.com/user-attachments/assets/0303bc28-54a6-4a55-add2-676b56d60274)

___

### 🦢Genre and Music Characteristics
9.
![image](https://github.com/user-attachments/assets/d43f54ff-54e6-42de-97c1-9e73f540ff0b)
___
10. ans [TEXT]

### 🦢Platform Popularity
11.
![image](https://github.com/user-attachments/assets/1d9ad641-00b6-4bad-b97c-541eb90d26f9)
___

### 🦢Advanced Analysis
13.
![image](https://github.com/user-attachments/assets/4c805e35-432b-47be-8f9d-37133d8e8d25)
___

14.
![image](https://github.com/user-attachments/assets/1f6bb8a3-5447-4f17-af89-a11628231ac3)
___

## 🍄Main Features
:sparkles: feature_1

:sparkles: feature 2

:sparkles: feature_3

## 🍄Author
Made by: [Karizza Dea R. Zoleta](https://github.com/kzoleta). If you have any queries or comments, feel free to reach out! :heart:


#### ❗NOTE: Make sure to tap [here](https://github.com/kzoleta/EDA2023/commits/main/README.md) to see the process of making this repository.





    
