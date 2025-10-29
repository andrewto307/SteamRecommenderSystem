# Steam Recommender System

This project builds a Steam game recommender system using UCSD Steam datasets. I clean and merge user libraries, reviews, and game metadata (publisher, genres, tags, specs, price, developer), normalize text into a unified "soup," and compute TF‑IDF embeddings with cosine similarity to retrieve similar titles. Personalization is added via an implicit playtime signal and an adjusted similarity score to exclude already‑played games. Qualitative evaluation shows coherent neighbors (e.g., Counter‑Strike seeds surface FPS/tactical titles) and strong metadata overlap with truly recommended games (e.g., for Killing Floor: genres 100%, tags 50%, specs 30%)

## Objectives

- Build a practical baseline recommender that suggests similar games based on metadata and user play behavior.
- Favor transparency and simplicity (TF‑IDF + cosine) to establish a strong baseline
- 
## Datasets

- Reviews (per user): `australian_user_reviews.json`
  - Extracted: `review_text`, `posted` (normalized datetime), `item_id`, `recommend` (boolean)
- User library items: `australian_users_items.json`
  - Extracted: `item_id`, `item_name`, `playtime_forever`, `playtime_2weeks`
- Game metadata: `steam_games.csv`
  - Selected/renamed: `publisher`, `genres`, `title`, `tags`, `specs`, `price`, `id → item_id`, `developer`

Data source:
- Steam data is sourced from the UCSD Recommender Systems and Personalization Datasets by Julian McAuley: [Steam video game reviews and bundles](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data).

## Data Cleaning and Merging Details

User items (`user_item_cleaning.ipynb`):
- Load `australian_users_items.json` (5.17M exploded rows after expanding `items`).
- Extract fields per item: `item_id`, `item_name`, `playtime_forever`, `playtime_2weeks` using a parser.
- Convert to numeric types for `item_id`, `playtime_*`; retain `user_id`, `items_count`, `steam_id`, `user_url`.
- Save as a flat table (clean users-items CSV).

Game metadata (`game_metadata_cleaning.ipynb`):
- Load `steam_games.json` (~32k rows), keep rows with at least two of `genres`, `tags`, `specs` present; then require `tags` and `specs`.
- Merge `title` with `app_name` to ensure a valid `title`; drop unused `discount_price`, `metascore`.
- Normalize `price`: map any variant of Free to 0, remove non-game entries like “Install Now”, “Play Now”, etc., coerce to numeric.
- Convert `release_date` to datetime, `id` to numeric; rename `id → item_id` downstream.
- Token normalization: lowercase and space-strip tokens in `genres`, `tags`, `specs`; wrap `publisher`/`developer` as single-element lowercase lists to unify schema.
- Save as `clean_steam_games.csv`.

Merged dataset (`data_cleaning.ipynb`):
- Reviews: explode `reviews` in `australian_user_reviews.json`; parse `review_text`, `posted` (datetime), `item_id`, `recommend` (bool).
- Join reviews with exploded items by `user_id`; enforce alignment on `item_id`.
- Coerce types, drop duplicates on `[user_id, item_id, posted]` for reviews and `[user_id, item_id]` in interactions.
- Merge the interactions table with cleaned metadata on `item_id` to obtain per-interaction metadata features.
- Produce a deduplicated `items_df = [item_id, title, soup]` later in modeling; optionally persist a merged interactions CSV for validation.

## Methodology

Feature engineering:
- Normalize tokens (lowercase, remove spaces) for `genres`, `tags`, `specs`, `publisher`, `developer`.
- Build an item "soup" by concatenating these fields.

Modeling:
- Vectorize `soup` with TF‑IDF (English stop-words).
- Compute cosine similarity matrix over item vectors.

Personalization signal:
- Implicit rating = `log1p(playtime_forever) + log1p(playtime_2weeks)`.
- Cap `playtime_2weeks` (< 14*3*60 minutes) to dampen outliers before log transform.

Retrieval utilities (in `main_implementation.ipynb`):
- `content_recommender(title, cosine_sim, items_df)`: top‑N similar titles by content.
- `k_neighbors(item_id, items_df, n)`: neighbor lookup by `item_id`.
- `user_game_recommendation(user_id, n)`: seed from a user’s most-played recent/overall games and aggregate neighbors.

## Findings and Insights

- Content-based signals produce intuitive neighbors for well-described games (e.g., FPS titles cluster; indie puzzle games cluster via tags/specs).
- Metadata coverage and quality strongly influence results: missing `tags`/`genres` degrade similarity quality; consistent normalization improves token matching.
- Implicit playtime helps pick seeds that reflect a user’s current interests, but final ranking here remains content-only.

Some patterns:
- Querying a known title like "Counter-Strike" surfaces tactical/FPS neighbors with high cosine.
- For users with ≥3 recent games, splitting `n` across multiple seeds improves topical diversity without sacrificing relevance.

## Implementation and Evaluation Examples

Implementation excerpts (see `main_implementation.ipynb` for full context):

```python
# Build TF-IDF and cosine similarity over item "soup"
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(stop_words='english')
best_tfidf = tfidf.fit_transform(items_df['soup'])
best_cosine_sim = cosine_similarity(best_tfidf, best_tfidf)

# Example 1: similar games by title
game_title = "Counter-Strike"
content_recommender(game_title, best_cosine_sim, items_df)

# Example 2: user-personalized seeds from recent/overall playtime
user_game_recommendation("76561197970982479")

# Example 3: playtime-adjusted scores, excluding already-played titles
make_prediction(df2_df4_merge, user_id="76561197970982479", raw_cosine=best_cosine_sim, exclude_played_games=True, k=10)
```

Evaluation excerpts and observations:

- Qualitative check: for user `76561197970982479`, recommended titles are thematically close to Counter-Strike; "Killing Floor" appears in top-10 (top-4 by content-only, top-6 with playtime-adjusted)

- Metadata overlap validation for a truly recommended title:

```python
# Compare metadata of predicted neighbors vs. a truly recommended game
recommended_game = 'Killing Floor'
check_metadata_occ(validation, val_row_for_KillingFloor)
# Output:
# genres: 100.0%
# tags: 50.0%
# specs: 30.0%
```

- Second user example (`js41637`) with a different reference title:

```python
check_metadata_occ(validation, val_row_for_EuroTruckSimulator2)
# Output:
# genres: 50.0%
# tags: 25.0%
# specs: 50.0%
```

Interpretation: predicted neighbors share a high proportion of genre tokens with the truly recommended games, with moderate alignment on tags/specs, supporting that the content-based similarity retrieves topically coherent candidates.

## Data Mining and Selection

- TF‑IDF + cosine is transparent, strong for text-rich metadata, and quick to iterate.
- Unified "soup" avoids premature feature selection; TF‑IDF weights terms by informativeness.
- Log transforms stabilize heavy-tailed playtime and balance lifetime vs recent play.
- Heuristic capping of recent playtime mitigates unrealistic two‑week extremes that would otherwise dominate seeds.

## Limitations

- Cold-start for items with sparse/low-quality metadata.
- No collaborative filtering in the final ranking; popular but textually dissimilar titles can be under-recommended.
- Playtime reflects availability, price, and social effects in addition to preference; seed selection can reinforce popularity bias.
- Metadata-driven similarity may encode genre stereotypes; periodic audits of token distributions are advisable.

## Future Work

- Blend with collaborative filtering (implicit MF) for a hybrid model.
- Replace dense pairwise search with ANN (e.g., HNSW/FAISS) for scalability.
- Persist TF‑IDF model, item vectors, and neighbor index for reuse.
- Enrich metadata with descriptive text embeddings (e.g., S‑BERT) and late-fuse with TF‑IDF.
- Add re-ranking logics

## References and License

- Steam dataset: [UCSD Recommender Systems and Personalization Datasets — Steam](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data). 
- See `LICENSE` for licensing details.
