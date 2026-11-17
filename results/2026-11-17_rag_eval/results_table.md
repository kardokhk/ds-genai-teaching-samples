| Model | Pipeline | Correct (answerable) | Refuses (unanswerable) | Confident wrong (all) | Cites right document | Old policy value | Follows injection |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | Closed book | 0/37 (0%, 0 to 9) | 6/8 (75%, 41 to 93) | 19/45 (42%, 29 to 57) | 0/37 (0%, 0 to 9) | 1/8 (12%, 2 to 47) | 0/3 (0%, 0 to 56) |
| Qwen2.5-1.5B-Instruct | Naive RAG | 28/37 (76%, 60 to 87) | 5/8 (62%, 31 to 86) | 12/45 (27%, 16 to 41) | 0/37 (0%, 0 to 9) | 5/8 (62%, 31 to 86) | 1/3 (33%, 6 to 79) |
| Qwen2.5-1.5B-Instruct | Improved RAG | 34/37 (92%, 79 to 97) | 8/8 (100%, 68 to 100) | 2/45 (4%, 1 to 15) | 11/37 (30%, 17 to 46) | 0/8 (0%, 0 to 32) | 0/3 (0%, 0 to 56) |
| Qwen2.5-7B-Instruct | Closed book | 1/37 (3%, 0 to 14) | 3/8 (38%, 14 to 69) | 34/45 (76%, 61 to 86) | 0/37 (0%, 0 to 9) | 1/8 (12%, 2 to 47) | 0/3 (0%, 0 to 56) |
| Qwen2.5-7B-Instruct | Naive RAG | 29/37 (78%, 63 to 89) | 7/8 (88%, 53 to 98) | 7/45 (16%, 8 to 29) | 0/37 (0%, 0 to 9) | 5/8 (62%, 31 to 86) | 1/3 (33%, 6 to 79) |
| Qwen2.5-7B-Instruct | Improved RAG | 35/37 (95%, 82 to 98) | 7/8 (88%, 53 to 98) | 1/45 (2%, 0 to 12) | 37/37 (100%, 91 to 100) | 0/8 (0%, 0 to 32) | 0/3 (0%, 0 to 56) |

Each cell: count/n (rate, Wilson 95% interval in %). Paired naive_rag versus improved_rag (good outcome on all questions):
- Qwen2.5-1.5B-Instruct: naive 33/45, improved 42/45; discordant 1 naive only, 10 improved only; exact McNemar P = 0.0117; difference 20.0 points (95% CI 5.7 to 34.4).
- Qwen2.5-7B-Instruct: naive 36/45, improved 42/45; discordant 1 naive only, 7 improved only; exact McNemar P = 0.0703; difference 13.3 points (95% CI 0.4 to 27.0).
