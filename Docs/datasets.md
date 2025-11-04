## Datasets

### German Credit Data (Statlog)

- Source: UCI Machine Learning Repository — Statlog (German Credit Data)
- File: `Data/german_credit_data.csv`
- Size: 1,000 rows; 10 columns in file (includes an index column)
- Features: 9 input features — `Age`, `Sex`, `Job`, `Housing`, `Saving accounts`, `Checking account`, `Credit amount`, `Duration`, `Purpose`
- Target: Not included in this CSV (no label column present)
- Usage role: Interpretability Demonstration — German Dataset. Here, you show how fuzzy reasoning works clearly on a smaller dataset. Interpretability demonstration. The German Credit dataset, being small and highly interpretable, was used to visualize the proposed fuzzy rules. For example, the rule “IF savings = low AND credit amount = high THEN risk = high” achieved an activation of 0.82 for a specific high-risk applicant. This qualitative example illustrates how the fuzzy layer enhances transparency by translating numeric attributes into linguistic reasoning, which is easily interpretable by domain experts.
### Taiwan Default of Credit Card Clients

- Source: UCI Machine Learning Repository — Default of Credit Card Clients (Taiwan)
- File: `Data/taiwan_default_of_credit_card_clients.csv`
- Size: 30,001 rows; 25 columns in file (includes an index column)
- Features: 23 input features — `X1` … `X23` (standard attributes from the dataset, excluding ID/index)
- Target: `Y` (default status)
- Usage role: Primary supervised modeling dataset for training, validation, and testing credit default prediction models; also used for EDA and benchmarking

Notes

- Both CSVs in this repo contain a leading unnamed index column. When loading, drop this column before modeling.
