# EVAL-FREEZE-001 official held-out results

Scored and verified on 2,974 official MASSIVE 1.0 `az-AZ` test utterances on 2026-11-23. Both systems share `SLOT-CRF-002`. The [protocol freeze](../../configs/EVAL-FREEZE-001.json) designated character-only `INT-SVC-001` + CRF-002 as primary and `INT-HYBRID-001` + CRF-002 as the comparison before test predictions or results were inspected. These roles carried through to final reporting. The models used the same 11,514 official training records, with development excluded from fitting.

Source SHA-256 for `artifacts/EVAL-FREEZE-001/metrics.json` is `39c0242af319067d8fc72e510553d98d3aaa006727a5038db4fe2baa077aabca`.

## Development and held-out results

Development used 2,033 official dev utterances. Test used all 2,974 official held-out utterances. Accuracy and exact-match counts use utterances. Intent macro-F1 uses the fixed inventory of 60 training labels with zero for undefined scores. Slot micro metrics pool exact typed-character-span matches. [Published development metrics](../JOINT-EVAL-001/metrics.json)

| Metric | Development | Held-out test | Test 95% percentile interval |
|---|---:|---:|---:|
| Character intent accuracy | 1,716/2,033 (84.4073%) | 2,510/2,974 (84.3981%) | 83.1540–85.6086% |
| Character intent macro-F1 | 82.2898% | 80.2292% | 76.4636–81.8116% |
| Hybrid intent accuracy | 1,720/2,033 (84.6040%) | 2,503/2,974 (84.1627%) | 82.8169–85.4405% |
| Hybrid intent macro-F1 | 82.4378% | 79.4293% | 75.9999–81.1427% |
| Shared CRF micro precision | 80.0488% | 79.1156% | 77.4025–80.9125% |
| Shared CRF micro recall | 65.8304% | 64.8135% | 62.9505–66.7152% |
| Shared CRF micro-F1 | 72.2467% | 71.2539% | 69.5945–72.9306% |
| Shared CRF complete slot-set exact match | 1,399/2,033 (68.8146%) | 2,028/2,974 (68.1910%) | 66.5770–69.8058% |
| Primary joint exact match | 1,206/2,033 (59.3212%) | 1,733/2,974 (58.2717%) | 56.5232–60.0202% |
| Hybrid joint exact match | 1,211/2,033 (59.5671%) | 1,721/2,974 (57.8682%) | 56.0525–59.6503% |
| Paired hybrid minus primary joint exact match | +0.2459 percentage points | −0.4035 percentage points | −1.0424 to +0.2354 percentage points |

Hybrid had five more joint matches on development and twelve fewer on test. On test, both systems were jointly correct on 1,682 utterances, primary alone on 51, hybrid alone on 39, and neither on 1,202. The paired interval includes zero. Test intervals use 2,000 utterance resamples of size 2,974 with replacement, NumPy PCG64 seed 42 and linear percentile interpolation. The paired comparison uses the same draws for both models. These intervals condition on the fitted models and approximate independent utterance sampling. Related utterances, train-text overlap and rare labels limit interpretation.

| Held-out utterance outcome | Primary | Hybrid |
|---|---:|---:|
| Both intent and slots correct | 1,733 | 1,721 |
| Intent only correct | 777 | 782 |
| Slots only correct | 295 | 307 |
| Neither correct | 169 | 164 |

## Exact training-text overlap on held-out test

| Metric | Seen (n = 230) | Unseen (n = 2,744) |
|---|---:|---:|
| Character intent accuracy | 224/230 (97.39%) | 2,286/2,744 (83.31%) |
| Hybrid intent accuracy | 224/230 (97.39%) | 2,279/2,744 (83.05%) |
| Character intent macro-F1 | 71.32% | 77.89% |
| Hybrid intent macro-F1 | 71.32% | 77.05% |
| CRF slot micro precision / recall / F1 | 90.84% / 87.50% / 89.14% | 78.40% / 63.65% / 70.26% |
| CRF slot exact matches | 211/230 | 1,817/2,744 |
| Primary joint exact match | 205/230 (89.13%) | 1,528/2,744 (55.69%) |
| Hybrid joint exact match | 205/230 (89.13%) | 1,516/2,744 (55.25%) |

All 90 paired test disagreements in joint correctness occurred on unseen text. The overall test metrics include both groups. The development seen/unseen joint counts were 144/162 and 1,062/1,871 for primary and 143/162 and 1,068/1,871 for hybrid.

The CRF reached its fixed 300-iteration optimization cap. Test yielded 1,807 TP, 477 FP and 981 FN across 2,788 gold and 2,284 predicted spans. The scorer logged one BIO repair and observed zero unrepresentable gold spans. The largest missed-slot counts were `event_name` (111/262 gold spans), `place_name` (87/269) and `time` (73/188). Both intent models confused `general_quirky` with `qa_factoid` 21 times, with 11 in the reverse direction. `cooking_query` was the only intent with zero test support.

## Intent classification and joint exact match by gold intent

Intent columns show precision, recall and F1 in that order. Support counts official gold utterances. Joint columns show complete correct semantic frames within the same gold-intent group. Percentages in the breakdown tables are rounded to two decimals. A dash indicates zero gold utterances for a joint subgroup.

| Gold intent | Support | Character intent P / R / F1 (%) | Hybrid intent P / R / F1 (%) | Primary joint | Hybrid joint |
|---|---:|---:|---:|---:|---:|
| `alarm_query` | 34 | 93.55 / 85.29 / 89.23 | 93.10 / 79.41 / 85.71 | 28/34 (82.35%) | 26/34 (76.47%) |
| `alarm_remove` | 21 | 85.00 / 80.95 / 82.93 | 85.00 / 80.95 / 82.93 | 15/21 (71.43%) | 15/21 (71.43%) |
| `alarm_set` | 41 | 88.37 / 92.68 / 90.48 | 84.44 / 92.68 / 88.37 | 32/41 (78.05%) | 32/41 (78.05%) |
| `audio_volume_down` | 11 | 62.50 / 45.45 / 52.63 | 71.43 / 45.45 / 55.56 | 5/11 (45.45%) | 5/11 (45.45%) |
| `audio_volume_mute` | 32 | 80.65 / 78.12 / 79.37 | 81.82 / 84.38 / 83.08 | 20/32 (62.50%) | 22/32 (68.75%) |
| `audio_volume_other` | 6 | 80.00 / 66.67 / 72.73 | 75.00 / 50.00 / 60.00 | 4/6 (66.67%) | 3/6 (50.00%) |
| `audio_volume_up` | 13 | 50.00 / 76.92 / 60.61 | 47.62 / 76.92 / 58.82 | 10/13 (76.92%) | 10/13 (76.92%) |
| `calendar_query` | 126 | 73.23 / 73.81 / 73.52 | 70.90 / 75.40 / 73.08 | 68/126 (53.97%) | 68/126 (53.97%) |
| `calendar_remove` | 67 | 91.78 / 100.00 / 95.71 | 91.67 / 98.51 / 94.96 | 52/67 (77.61%) | 51/67 (76.12%) |
| `calendar_set` | 209 | 88.99 / 92.82 / 90.87 | 88.24 / 93.30 / 90.70 | 92/209 (44.02%) | 93/209 (44.50%) |
| `cooking_query` | 0 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | — | — |
| `cooking_recipe` | 72 | 91.55 / 90.28 / 90.91 | 95.45 / 87.50 / 91.30 | 31/72 (43.06%) | 31/72 (43.06%) |
| `datetime_convert` | 15 | 100.00 / 53.33 / 69.57 | 80.00 / 53.33 / 64.00 | 3/15 (20.00%) | 4/15 (26.67%) |
| `datetime_query` | 88 | 91.01 / 92.05 / 91.53 | 87.91 / 90.91 / 89.39 | 63/88 (71.59%) | 62/88 (70.45%) |
| `email_addcontact` | 12 | 60.00 / 75.00 / 66.67 | 53.85 / 58.33 / 56.00 | 7/12 (58.33%) | 6/12 (50.00%) |
| `email_query` | 119 | 93.22 / 92.44 / 92.83 | 94.07 / 93.28 / 93.67 | 93/119 (78.15%) | 93/119 (78.15%) |
| `email_querycontact` | 26 | 79.31 / 88.46 / 83.64 | 79.31 / 88.46 / 83.64 | 13/26 (50.00%) | 13/26 (50.00%) |
| `email_sendemail` | 114 | 96.30 / 91.23 / 93.69 | 95.45 / 92.11 / 93.75 | 87/114 (76.32%) | 87/114 (76.32%) |
| `general_greet` | 1 | 100.00 / 100.00 / 100.00 | 50.00 / 100.00 / 66.67 | 1/1 (100.00%) | 1/1 (100.00%) |
| `general_joke` | 19 | 100.00 / 89.47 / 94.44 | 100.00 / 89.47 / 94.44 | 9/19 (47.37%) | 9/19 (47.37%) |
| `general_quirky` | 169 | 60.26 / 53.85 / 56.88 | 59.59 / 51.48 / 55.24 | 85/169 (50.30%) | 81/169 (47.93%) |
| `iot_cleaning` | 26 | 96.00 / 92.31 / 94.12 | 92.31 / 92.31 / 92.31 | 20/26 (76.92%) | 20/26 (76.92%) |
| `iot_coffee` | 36 | 97.22 / 97.22 / 97.22 | 92.11 / 97.22 / 94.59 | 27/36 (75.00%) | 27/36 (75.00%) |
| `iot_hue_lightchange` | 36 | 85.00 / 94.44 / 89.47 | 86.84 / 91.67 / 89.19 | 22/36 (61.11%) | 21/36 (58.33%) |
| `iot_hue_lightdim` | 21 | 85.71 / 85.71 / 85.71 | 75.00 / 85.71 / 80.00 | 17/21 (80.95%) | 17/21 (80.95%) |
| `iot_hue_lightoff` | 43 | 86.36 / 88.37 / 87.36 | 84.44 / 88.37 / 86.36 | 34/43 (79.07%) | 34/43 (79.07%) |
| `iot_hue_lighton` | 3 | 50.00 / 33.33 / 40.00 | 66.67 / 66.67 / 66.67 | 1/3 (33.33%) | 2/3 (66.67%) |
| `iot_hue_lightup` | 27 | 95.83 / 85.19 / 90.20 | 95.45 / 77.78 / 85.71 | 21/27 (77.78%) | 20/27 (74.07%) |
| `iot_wemo_off` | 18 | 100.00 / 72.22 / 83.87 | 100.00 / 72.22 / 83.87 | 9/18 (50.00%) | 9/18 (50.00%) |
| `iot_wemo_on` | 10 | 72.73 / 80.00 / 76.19 | 80.00 / 80.00 / 80.00 | 6/10 (60.00%) | 6/10 (60.00%) |
| `lists_createoradd` | 39 | 91.89 / 87.18 / 89.47 | 82.05 / 82.05 / 82.05 | 24/39 (61.54%) | 23/39 (58.97%) |
| `lists_query` | 51 | 82.35 / 82.35 / 82.35 | 82.98 / 76.47 / 79.59 | 34/51 (66.67%) | 32/51 (62.75%) |
| `lists_remove` | 52 | 93.48 / 82.69 / 87.76 | 93.33 / 80.77 / 86.60 | 35/52 (67.31%) | 33/52 (63.46%) |
| `music_dislikeness` | 4 | 66.67 / 50.00 / 57.14 | 75.00 / 75.00 / 75.00 | 2/4 (50.00%) | 3/4 (75.00%) |
| `music_likeness` | 36 | 80.00 / 66.67 / 72.73 | 78.57 / 61.11 / 68.75 | 21/36 (58.33%) | 19/36 (52.78%) |
| `music_query` | 35 | 96.30 / 74.29 / 83.87 | 93.10 / 77.14 / 84.38 | 23/35 (65.71%) | 24/35 (68.57%) |
| `music_settings` | 6 | 33.33 / 16.67 / 22.22 | 33.33 / 16.67 / 22.22 | 0/6 (0.00%) | 0/6 (0.00%) |
| `news_query` | 124 | 84.62 / 88.71 / 86.61 | 86.15 / 90.32 / 88.19 | 68/124 (54.84%) | 69/124 (55.65%) |
| `play_audiobook` | 41 | 85.29 / 70.73 / 77.33 | 88.24 / 73.17 / 80.00 | 14/41 (34.15%) | 16/41 (39.02%) |
| `play_game` | 35 | 89.19 / 94.29 / 91.67 | 89.19 / 94.29 / 91.67 | 19/35 (54.29%) | 19/35 (54.29%) |
| `play_music` | 176 | 84.29 / 91.48 / 87.74 | 85.64 / 91.48 / 88.46 | 87/176 (49.43%) | 88/176 (50.00%) |
| `play_podcasts` | 63 | 98.31 / 92.06 / 95.08 | 96.72 / 93.65 / 95.16 | 26/63 (41.27%) | 26/63 (41.27%) |
| `play_radio` | 72 | 82.50 / 91.67 / 86.84 | 84.42 / 90.28 / 87.25 | 42/72 (58.33%) | 42/72 (58.33%) |
| `qa_currency` | 39 | 92.31 / 92.31 / 92.31 | 92.50 / 94.87 / 93.67 | 21/39 (53.85%) | 21/39 (53.85%) |
| `qa_definition` | 57 | 80.36 / 78.95 / 79.65 | 78.33 / 82.46 / 80.34 | 28/57 (49.12%) | 29/57 (50.88%) |
| `qa_factoid` | 141 | 67.52 / 75.18 / 71.14 | 70.20 / 75.18 / 72.60 | 77/141 (54.61%) | 74/141 (52.48%) |
| `qa_maths` | 25 | 83.33 / 80.00 / 81.63 | 77.78 / 84.00 / 80.77 | 19/25 (76.00%) | 20/25 (80.00%) |
| `qa_stock` | 26 | 81.48 / 84.62 / 83.02 | 81.48 / 84.62 / 83.02 | 18/26 (69.23%) | 18/26 (69.23%) |
| `recommendation_events` | 43 | 67.35 / 76.74 / 71.74 | 67.39 / 72.09 / 69.66 | 19/43 (44.19%) | 19/43 (44.19%) |
| `recommendation_locations` | 31 | 75.00 / 87.10 / 80.60 | 69.23 / 87.10 / 77.14 | 14/31 (45.16%) | 14/31 (45.16%) |
| `recommendation_movies` | 20 | 64.00 / 80.00 / 71.11 | 72.73 / 80.00 / 76.19 | 12/20 (60.00%) | 12/20 (60.00%) |
| `social_post` | 81 | 93.59 / 90.12 / 91.82 | 93.59 / 90.12 / 91.82 | 41/81 (50.62%) | 40/81 (49.38%) |
| `social_query` | 25 | 95.24 / 80.00 / 86.96 | 90.48 / 76.00 / 82.61 | 15/25 (60.00%) | 14/25 (56.00%) |
| `takeaway_order` | 22 | 90.00 / 81.82 / 85.71 | 90.00 / 81.82 / 85.71 | 9/22 (40.91%) | 9/22 (40.91%) |
| `takeaway_query` | 35 | 90.32 / 80.00 / 84.85 | 93.10 / 77.14 / 84.38 | 13/35 (37.14%) | 12/35 (34.29%) |
| `transport_query` | 51 | 83.33 / 78.43 / 80.81 | 82.00 / 80.39 / 81.19 | 26/51 (50.98%) | 26/51 (50.98%) |
| `transport_taxi` | 23 | 100.00 / 100.00 / 100.00 | 95.83 / 100.00 / 97.87 | 17/23 (73.91%) | 17/23 (73.91%) |
| `transport_ticket` | 35 | 94.12 / 91.43 / 92.75 | 94.12 / 91.43 / 92.75 | 24/35 (68.57%) | 24/35 (68.57%) |
| `transport_traffic` | 15 | 70.59 / 80.00 / 75.00 | 66.67 / 80.00 / 72.73 | 8/15 (53.33%) | 8/15 (53.33%) |
| `weather_query` | 156 | 90.06 / 92.95 / 91.48 | 93.08 / 94.87 / 93.97 | 102/156 (65.38%) | 102/156 (65.38%) |

## Exact typed-span slot performance

Support counts gold spans. A predicted span must match the type and both character offsets on the same record.

| Slot | Support | TP | FP | FN | P / R / F1 (%) |
|---|---:|---:|---:|---:|---:|
| `alarm_type` | 3 | 1 | 0 | 2 | 100.00 / 33.33 / 50.00 |
| `app_name` | 5 | 2 | 3 | 3 | 40.00 / 40.00 / 40.00 |
| `artist_name` | 61 | 33 | 10 | 28 | 76.74 / 54.10 / 63.46 |
| `audiobook_author` | 5 | 0 | 0 | 5 | 0.00 / 0.00 / 0.00 |
| `audiobook_name` | 23 | 8 | 4 | 15 | 66.67 / 34.78 / 45.71 |
| `business_name` | 92 | 52 | 17 | 40 | 75.36 / 56.52 / 64.60 |
| `business_type` | 31 | 16 | 5 | 15 | 76.19 / 51.61 / 61.54 |
| `change_amount` | 8 | 4 | 1 | 4 | 80.00 / 50.00 / 61.54 |
| `coffee_type` | 4 | 1 | 0 | 3 | 100.00 / 25.00 / 40.00 |
| `color_type` | 26 | 17 | 6 | 9 | 73.91 / 65.38 / 69.39 |
| `cooking_type` | 8 | 0 | 0 | 8 | 0.00 / 0.00 / 0.00 |
| `currency_name` | 39 | 23 | 9 | 16 | 71.88 / 58.97 / 64.79 |
| `date` | 417 | 370 | 62 | 47 | 85.65 / 88.73 / 87.16 |
| `definition_word` | 51 | 26 | 4 | 25 | 86.67 / 50.98 / 64.20 |
| `device_type` | 57 | 41 | 4 | 16 | 91.11 / 71.93 / 80.39 |
| `drink_type` | 1 | 0 | 0 | 1 | 0.00 / 0.00 / 0.00 |
| `email_address` | 9 | 8 | 1 | 1 | 88.89 / 88.89 / 88.89 |
| `email_folder` | 5 | 5 | 0 | 0 | 100.00 / 100.00 / 100.00 |
| `event_name` | 262 | 151 | 53 | 111 | 74.02 / 57.63 / 64.81 |
| `food_type` | 72 | 39 | 11 | 33 | 78.00 / 54.17 / 63.93 |
| `game_name` | 26 | 11 | 2 | 15 | 84.62 / 42.31 / 56.41 |
| `game_type` | 0 | 0 | 0 | 0 | 0.00 / 0.00 / 0.00 |
| `general_frequency` | 20 | 10 | 7 | 10 | 58.82 / 50.00 / 54.05 |
| `house_place` | 58 | 46 | 2 | 12 | 95.83 / 79.31 / 86.79 |
| `ingredient` | 7 | 0 | 0 | 7 | 0.00 / 0.00 / 0.00 |
| `joke_type` | 11 | 3 | 3 | 8 | 50.00 / 27.27 / 35.29 |
| `list_name` | 60 | 32 | 10 | 28 | 76.19 / 53.33 / 62.75 |
| `meal_type` | 18 | 17 | 10 | 1 | 62.96 / 94.44 / 75.56 |
| `media_type` | 128 | 94 | 23 | 34 | 80.34 / 73.44 / 76.73 |
| `movie_name` | 2 | 0 | 0 | 2 | 0.00 / 0.00 / 0.00 |
| `movie_type` | 3 | 0 | 0 | 3 | 0.00 / 0.00 / 0.00 |
| `music_album` | 1 | 0 | 0 | 1 | 0.00 / 0.00 / 0.00 |
| `music_descriptor` | 7 | 0 | 0 | 7 | 0.00 / 0.00 / 0.00 |
| `music_genre` | 50 | 29 | 15 | 21 | 65.91 / 58.00 / 61.70 |
| `news_topic` | 50 | 16 | 11 | 34 | 59.26 / 32.00 / 41.56 |
| `order_type` | 19 | 10 | 8 | 9 | 55.56 / 52.63 / 54.05 |
| `person` | 218 | 174 | 26 | 44 | 87.00 / 79.82 / 83.25 |
| `personal_info` | 14 | 7 | 3 | 7 | 70.00 / 50.00 / 58.33 |
| `place_name` | 269 | 182 | 46 | 87 | 79.82 / 67.66 / 73.24 |
| `player_setting` | 39 | 14 | 7 | 25 | 66.67 / 35.90 / 46.67 |
| `playlist_name` | 17 | 1 | 3 | 16 | 25.00 / 5.88 / 9.52 |
| `podcast_descriptor` | 25 | 8 | 2 | 17 | 80.00 / 32.00 / 45.71 |
| `podcast_name` | 17 | 3 | 0 | 14 | 100.00 / 17.65 / 30.00 |
| `radio_name` | 33 | 19 | 16 | 14 | 54.29 / 57.58 / 55.88 |
| `relation` | 59 | 40 | 6 | 19 | 86.96 / 67.80 / 76.19 |
| `song_name` | 39 | 19 | 3 | 20 | 86.36 / 48.72 / 62.30 |
| `sport_type` | 0 | 0 | 0 | 0 | 0.00 / 0.00 / 0.00 |
| `time` | 188 | 115 | 47 | 73 | 70.99 / 61.17 / 65.71 |
| `time_zone` | 12 | 7 | 0 | 5 | 100.00 / 58.33 / 73.68 |
| `timeofday` | 60 | 40 | 10 | 20 | 80.00 / 66.67 / 72.73 |
| `transport_agency` | 9 | 8 | 1 | 1 | 88.89 / 88.89 / 88.89 |
| `transport_descriptor` | 2 | 0 | 1 | 2 | 0.00 / 0.00 / 0.00 |
| `transport_name` | 4 | 2 | 0 | 2 | 100.00 / 50.00 / 66.67 |
| `transport_type` | 65 | 59 | 5 | 6 | 92.19 / 90.77 / 91.47 |
| `weather_descriptor` | 79 | 44 | 20 | 35 | 68.75 / 55.70 / 61.54 |

## Joint exact match by gold scenario

| Scenario | Support | Primary joint | Hybrid joint |
|---|---:|---:|---:|
| `alarm` | 96 | 75/96 (78.12%) | 73/96 (76.04%) |
| `audio` | 62 | 39/62 (62.90%) | 40/62 (64.52%) |
| `calendar` | 402 | 212/402 (52.74%) | 212/402 (52.74%) |
| `cooking` | 72 | 31/72 (43.06%) | 31/72 (43.06%) |
| `datetime` | 103 | 66/103 (64.08%) | 66/103 (64.08%) |
| `email` | 271 | 200/271 (73.80%) | 199/271 (73.43%) |
| `general` | 189 | 95/189 (50.26%) | 91/189 (48.15%) |
| `iot` | 220 | 157/220 (71.36%) | 156/220 (70.91%) |
| `lists` | 142 | 93/142 (65.49%) | 88/142 (61.97%) |
| `music` | 81 | 46/81 (56.79%) | 46/81 (56.79%) |
| `news` | 124 | 68/124 (54.84%) | 69/124 (55.65%) |
| `play` | 387 | 188/387 (48.58%) | 191/387 (49.35%) |
| `qa` | 288 | 163/288 (56.60%) | 162/288 (56.25%) |
| `recommendation` | 94 | 45/94 (47.87%) | 45/94 (47.87%) |
| `social` | 106 | 56/106 (52.83%) | 54/106 (50.94%) |
| `takeaway` | 57 | 22/57 (38.60%) | 21/57 (36.84%) |
| `transport` | 124 | 75/124 (60.48%) | 75/124 (60.48%) |
| `weather` | 156 | 102/156 (65.38%) | 102/156 (65.38%) |

[Protocol, aggregate metrics, and verification](../../docs/experiments/EVAL-FREEZE-001.md#2026-11-23-official-held-out-results)
