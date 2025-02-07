from pathlib import Path
import multiprocessing as mp

import pandas as pd  # pip install pandas

JSON_ROOT = Path('../DATAR/release_related/all_jsons')
REVIEWS_ROOT = Path('../DATAR/release_related/all_reviews')
IMAGE_ROOT = Path('plots')
DELTADAYS = 90

json_paths = list(JSON_ROOT.glob('*.json'))


def _get_best_release(json_path: Path):
    review_path = REVIEWS_ROOT / f"{json_path.stem}.json"
    if not review_path.exists():
        print('FAILED: no reviews\t\t\t\t', json_path.stem)
        return "no reviews"

    df_reviews = pd.read_json(review_path)[['reviewId', 'score', 'reviewCreatedVersion', 'at']]
    df_reviews['at'] = pd.to_datetime(df_reviews['at'], utc=True)
    df_reviews_v = df_reviews.groupby('reviewCreatedVersion').count().rename(columns={'reviewId': 'reviews'})
    df_reviews_v['vmajor'] = df_reviews_v.index.astype(str).str.replace('v', '').str.replace('-', '.').str.split('.').str[0]
    df_reviews_v = df_reviews_v[['reviews', 'vmajor']]
    df_reviews_v['vmajor'] = pd.to_numeric(df_reviews_v['vmajor'], errors='coerce')

    df_roll = df_reviews_v.rolling(2).mean().dropna()
    try:
        df_roll['vshift'] = df_roll['vmajor'] % 1 != 0
    except:
        print('FAILED: no valid major release\t', json_path.stem)
        return "no valid major release"

    df_roll_vshift = df_roll[df_roll['vshift']]
    if df_roll_vshift.empty:
        print('FAILED: not enough releases\t\t', json_path.stem)
        return "not enough releases"

    df_json = pd.read_json(json_path)
    if df_json.empty:
        print('FAILED: empty json\t\t\t\t', json_path.stem)
        return "empty json"

    # best_version = df_roll_vshift.idxmax()['reviews']
    best_versions = df_roll_vshift.sort_values(by='reviews', ascending=False).index.to_list()
    for version in best_versions:
        if version in df_json['google_play_tag'].to_list():
            best_version = version
            break
    else:
        print('FAILED: no valid major release\t', json_path.stem)
        return "no valid major release"

    df_roll_idx = df_roll.reset_index()
    i = df_roll_idx.index[df_roll_idx['reviewCreatedVersion'] == best_version]
    prev_version = df_roll_idx.iloc[i - 1]['reviewCreatedVersion'].to_list()[0]

    df_json = df_json[df_json['google_play_tag'] == best_version]
    df_json['end_date'] = pd.to_datetime(df_json['end_date'], utc=True)
    center_datetime = df_json['end_date'].min()

    issues_closed = len(df_json['closed_issues'].to_list()[0]) if df_json['closed_issues'].to_list() else 0

    df_reviews_filtered = df_reviews[(df_reviews['reviewCreatedVersion'] == best_version) | (df_reviews['reviewCreatedVersion'] == prev_version)]
    df_reviews_filtered_c = df_reviews_filtered.copy()
    df_reviews_filtered_c['deltadays'] = (df_reviews_filtered_c['at'] - center_datetime).dt.days
    df_reviews_filtered_c = df_reviews_filtered_c[(df_reviews_filtered_c['deltadays'] >= -DELTADAYS) & (df_reviews_filtered_c['deltadays'] <= DELTADAYS)]
    df_reviews_filtered_c = df_reviews_filtered_c[((df_reviews_filtered_c['deltadays'] < 0) & (df_reviews_filtered_c['reviewCreatedVersion'] == prev_version))
                                                  | ((df_reviews_filtered_c['deltadays'] >= 0) & (df_reviews_filtered_c['reviewCreatedVersion'] == best_version))]
    filtered_reviews = df_reviews_filtered_c[['score', 'reviewCreatedVersion', 'deltadays']].to_dict(orient='index')

    if not filtered_reviews:
        print('FAILED: no github data\t\t\t', json_path.stem)
        return "no github data"
    elif (df_reviews_filtered_c['deltadays'].min() > 0) or (df_reviews_filtered_c['deltadays'].max() <= 0):
        print(f'FAILED: ({df_reviews_filtered_c["deltadays"].min()}/{df_reviews_filtered_c["deltadays"].max()}) deltadays\t\t', json_path.stem)
        return "no negative or positive reviews (deltadays)"

    print('SUCCESS\t\t\t\t\t\t\t', json_path.stem)
    return {
        'name': json_path.stem,
        'version': best_version,
        'num_reviews': df_roll.max()['reviews'],
        'issues_closed': issues_closed,
        'center_datetime': center_datetime,
        'filtered_reviews': filtered_reviews
    }


def get_best_releases(as_df=False, return_errors=False):
    results = {}
    errors = {}
    with mp.Pool(mp.cpu_count()) as pool:
        for res in pool.imap_unordered(_get_best_release, json_paths):
            if isinstance(res, dict):
                results[res['name']] = res
            else:
                errors[res] = errors.get(res, 0) + 1
    errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)
    if as_df:
        results = pd.DataFrame.from_dict(results, orient='index').sort_values('num_reviews', ascending=False)
    if return_errors:
        return results, errors
    else:
        return results


if __name__ == "__main__":
    print(JSON_ROOT.resolve(), Path.cwd(), sep='\n')
    dfr, errors = get_best_releases(as_df=True, return_errors=True)
    dfr.to_csv('../supplementary_data/test_data.csv')
    print("\nTotal apps succeeded: ", len(dfr))
    print(errors)
    print(dfr.head(50))
