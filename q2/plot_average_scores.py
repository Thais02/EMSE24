from pathlib import Path
from collections import namedtuple
import multiprocessing as mp
from itertools import combinations

import pandas as pd  # pip install pandas
import plotly.express as px  # pip install plotly
from fastdtw import fastdtw  # pip install fastdtw
from tqdm import tqdm


JSON_ROOT = Path('../DATAR/release_related/all_reviews')
IMAGE_ROOT = Path('plots')


def preprocess_plotting(df):
    """
    Takes a review dataframe and return a preprocessed copy for plotting
    """
    df_scores_time = df[['score', 'at']].copy()  # Create a copy of the DataFrame to avoid view issues
    # Adds a column to keep counts during grouping
    df_scores_time['count'] = 1
    # Add year and month to group on
    df_scores_time['year'] = pd.to_datetime(df_scores_time['at']).dt.year
    df_scores_time['month'] = pd.to_datetime(df_scores_time['at']).dt.month
    # Group by year -> month -> score
    df_scores_time = df_scores_time.groupby(by=['year', 'month', 'score']).sum(numeric_only=True)
    df_grouped_scores_time = pd.DataFrame(df_scores_time['count'].index.to_list(), columns=['year', 'month', 'score'])
    # Keep only the values
    df_grouped_scores_time['count'] = df_scores_time['count'].values
    # Add a datetime column to the values
    df_grouped_scores_time['day'] = 1
    df_grouped_scores_time['date'] = pd.to_datetime(df_grouped_scores_time[['day', 'month', 'year']])
    return df_grouped_scores_time


def calculate_average(df: pd.DataFrame):
    """
    Takes a plotting dataframe and adds the average per month
    """
    # Initialize variables
    dict_average = {}
    Old_row_tuple = namedtuple('old_row', ['month', 'year'])
    old_row = Old_row_tuple(0, 0)
    total_score = 0
    total_reviews = 0

    for index, row in df.iterrows():
        # Is true if the new row is of the same month
        if (row.month, row.year) == (old_row.month, old_row.year):
            total_score += row.score * row['count']
            total_reviews += row['count']
            dict_average[(row.month, row.year)] = (total_score, total_reviews, total_score / total_reviews, row['date'])
            old_row = row
        # If the new row is the next month, create a new total score and total review
        else:
            total_score = row.score * row['count']
            total_reviews = row['count']
            dict_average[(row.month, row.year)] = (total_score, total_reviews, total_score / total_reviews, row['date'])
            old_row = row

    # Turns the dictionary into a dataframe
    df_average = pd.DataFrame(dict_average).T
    df_average.columns = ['total score', 'total amount of reviews', 'average score', 'date']

    return df_average


def generate_similarities(json_paths: list[Path]):
    distances = {}
    scores = {}
    for json_path in tqdm(json_paths):
        scores[json_path.stem] = calculate_average(preprocess_plotting(pd.read_json(json_path)))['average score']
    for path1, path2 in tqdm(list(combinations(json_paths, 2))):
        try:
            distance, _ = fastdtw(scores[path1.stem], scores[path2.stem], dist=2)
            distances[(path1.stem, path2.stem)] = distance
        except Exception as e:
            print(path1.stem, path2.stem, e)
    print(distances)


def generate_plots(json_path: Path):
    app_name = json_path.stem
    try:
        df = pd.read_json(json_path)
        df_grouped = preprocess_plotting(df)
        df_average = calculate_average(df_grouped)

        plot_hist = px.histogram(df_grouped, x='date', y='count', barmode='group', color='score')
        plot_hist.write_image(IMAGE_ROOT / f"{app_name}__hist.png")
        # plot_hist.write_html(IMAGE_ROOT / f"{app_name}__hist.html")  # DO NOT if you want to keep your RAM :)

        plot_scatter = px.scatter(df_average, x='date', y='average score', trendline='ols')
        plot_scatter.write_image(IMAGE_ROOT / f"{app_name}__scat.png")
        # plot_scatter.write_html(IMAGE_ROOT / f"{app_name}__scat.html")  # DO NOT if you want to keep your RAM :)
    except Exception as e:
        print(app_name, e)


def main():
    IMAGE_ROOT.mkdir(exist_ok=True)
    json_paths = list(JSON_ROOT.glob('*.json'))
    generate_similarities(json_paths)
    return
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(generate_plots, json_paths)


if __name__ == "__main__":
    main()
