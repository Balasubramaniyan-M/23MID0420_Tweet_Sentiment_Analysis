# Dataset Card: Twitter US Airline Sentiment

## Overview

| Property | Value |
|---|---|
| **Dataset Name** | Twitter US Airline Sentiment |
| **Source** | CrowdFlower (Figure Eight) / Kaggle |
| **URL** | https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment |
| **Access Date** | 2026-08-24 |
| **License** | CC BY-NC-SA 4.0 (CrowdFlower open data) |
| **Expected Rows** | ~14,640 |
| **Expected Columns** | 15 |
| **Language** | English (Twitter/social media) |
| **Domain** | US airline customer tweets |
| **Collection Period** | February 2015 |

## Sentiment Labels

| Label | Meaning |
|---|---|
| `negative` | Tweet expresses negative sentiment toward the airline |
| `neutral` | Tweet expresses neither clearly positive nor negative sentiment |
| `positive` | Tweet expresses positive sentiment toward the airline |

## Expected Class Distribution

The dataset is **class-imbalanced**, with negative tweets being the majority class.

| Class | Approximate Count | Approximate Percentage |
|---|---|---|
| negative | ~9,178 | ~63% |
| neutral | ~3,099 | ~21% |
| positive | ~2,363 | ~16% |

> **Note**: Exact values will be confirmed upon loading the actual dataset.

## Schema (Expected Columns)

| Column | Type | Role | Notes |
|---|---|---|---|
| `tweet_id` | int | Identifier | Unique tweet identifier. **Excluded from modeling.** |
| `airline_sentiment` | str | **Target** | Ground-truth sentiment label |
| `airline_sentiment_confidence` | float | **Leakage risk** | Annotator confidence. **Excluded from modeling.** |
| `negativereason` | str | **Leakage risk** | Reason for negative label. **Excluded from modeling.** |
| `negativereason_confidence` | float | **Leakage risk** | Confidence of reason. **Excluded from modeling.** |
| `airline` | str | Entity | Airline name (6 US airlines) |
| `airline_sentiment_gold` | str | **Leakage risk** | Gold-standard label. **Excluded from modeling.** |
| `name` | str | **Privacy** | Twitter username. **Excluded from modeling.** |
| `negativereason_gold` | str | **Leakage risk** | Gold reason. **Excluded from modeling.** |
| `retweet_count` | int | **Privacy/Metadata** | Number of retweets. **Excluded from modeling.** |
| `text` | str | **Input** | Raw tweet text (primary input feature) |
| `tweet_coord` | str | **Privacy** | Tweet coordinates. **Excluded from modeling.** |
| `tweet_created` | str | Metadata | Timestamp of tweet |
| `tweet_location` | str | **Privacy** | User location. **Excluded from modeling.** |
| `user_timezone` | str | **Privacy** | User timezone. **Excluded from modeling.** |

## Data Quality Notes

### Missing Values
- Several columns contain missing values (e.g., `negativereason`, `tweet_coord`, `tweet_location`, `user_timezone`)
- The `text` and `airline_sentiment` columns should have no missing values
- Exact missing counts are documented in the notebook upon loading

### Duplicate Analysis
- Duplicate tweet IDs and duplicate text are checked and documented
- Near-duplicate tweets may exist due to retweets or common phrases
- Policy: duplicates are retained but documented for transparency

### Annotation Method
- Crowdsourced annotation via CrowdFlower (now Figure Eight/Appen)
- Multiple annotators per tweet with confidence scoring
- Annotation noise is expected, especially for neutral vs. mildly negative/positive tweets

### Sampling Limitations
- Tweets collected during February 2015 only
- Only US airlines covered (6 airlines)
- Twitter-specific language, abbreviations, and conventions
- Not representative of all customers, all airlines, or all time periods
- Users who tweet are not representative of the general customer population

### Privacy Considerations
- Username (`name`), coordinates (`tweet_coord`), and location (`tweet_location`) are excluded from modeling
- Tweet examples in reports are anonymized (handles redacted)
- No individual user profiling or tracking is performed
- Data is used only for academic sentiment classification research

### Leakage Risks
- `airline_sentiment_confidence` directly reveals annotation certainty about the target
- `negativereason` and `negativereason_confidence` reveal post-labeling metadata
- `airline_sentiment_gold` and `negativereason_gold` are alternate label sources
- All leakage-risk columns are excluded from the feature set

## Download Instructions

1. Visit https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
2. Download `Tweets.csv`
3. Place the file in this `data/` directory
4. The notebook will automatically detect and load it

## Citation

CrowdFlower. "Twitter US Airline Sentiment." Kaggle, 2015.
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

---

*This dataset card was created as part of MDI3003 Lab 05 by Balasubramaniyan M (23MID0420).*
