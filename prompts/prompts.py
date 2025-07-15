COT_COLUMN_ENRICHMENT = [{
    "role": "user",
    "content": """
You are given a column name and the table it belongs to. Your task is to generate three alternative names for this column using a structured reasoning process. Explain your approach at each step.

**Instructions:**

1. **Understand the domain:**

   - Use the table name and description to infer the domain or subject area.

2. **Generate six candidate names:**

   - Use domain-specific terminology to propose six alternative names for the column.
   - Do **not** reuse any exact words that appear in the original column or table name.
   - Avoid abbreviations or acronyms.

3. **Reflect and select and rank the most relevant names:**

   - Review the six candidates and select three that best match the inferred domain.
   - Rank top three from best to worst.
   - Ensure they reflect technical or contextual relevance.

4. **Output the final result in JSON format:**

```json
{
  "name_1": "name_1",
  "name_2": "name_2",
  "name_3": "name_3"
}
"""}, 
{ "role": "assistant", "content": "Certainly! Please provide the table and column name." },

{ "role": "user", "content": """ 
Column Name: c_id
Table Name: purchase_history
Table Description: This table contains the purchase history of customers.
""" }, 
{ "role": "assistant", "content": """ Steps:

1. **Understand the domain:**

The domain is marketing or e-commerce, focusing on customer transactions.

2. **Generate six candidate names:**

I am avoiding any words that appear in the original column or table name or their descriptions such as customer, so I will propose:
 
 consumer_identifier, user_profile_identifier, shopper_key, buyer_token, transaction_participant_id, participant_id

3. **Reflect and select and rank the most relevant names:**

user_profile_identifier, participant_id and transaction_participant_id are too long or abstract.

I will select these three ranked names: consumer_identifier, shopper_key, buyer_token

4. **Output the final result in JSON format:**

        ```json
        { 
            "name_1": "consumer_identifier",
            "name_2": "shopper_key", 
            "name_3": "buyer_token" 
        }
        ```
""",
    }]

COT_COLUMN_ENRICHMENT_SYNONYMS = [
    {
        "role": "user",
        "content": """
You are given a column name and the table it belongs to. Your task is to generate semantically three enriched names for this column using a structured reasoning process. Explain your approach for each step.

**Instructions:**

1. **Understand the domain:**
   - Use the table and column names along with their descriptions to infer the domain or subject area.

2. **Generate enriched names:**
   - Propose six alternative names that are semantically richer and more descriptive and synonymous.
   - Avoid using any abbreviations or acronyms.

3. **Reflect and select and rank final names:**
   - Review your proposed names and select the three most relevant ones based on domain-specific relevance.
   - Rank top three from best to worst.


4. **Output the final result in JSON format:**

```json
{
  "name_1": "name_1",
  "name_2": "name_2",
  "name_3": "name_3"
}
""", }, { "role": "assistant", "content": "Certainly! Please provide the table and column name along with their descriptions.", }, 
    
{ "role": "user", "content": """ 

Column Name: c_id
Column Description: This column contains the unique identifier for each customer.
Table Name: purchase_history
Table Description: This table contains the purchase history of customers. """, }, 

{ "role": "assistant", "content": """ Steps:

1. **Understand the domain:**

The domain is marketing or e-commerce. The focus is on customer interactions and transactions.

2. **Generate enriched names:**
 
customer_identifier, consumer_identifier, client_identifier, account_holder_identifier, transaction_participant_id, shopper_profile_key

3. **Reflect and select and rank final names:**

"account_holder_identifier" and "transaction_participant_id" are more generic or overly formal. The three most relevant names within the marketing/e-commerce context are: customer_identifier, consumer_identifier, client_identifier


4. **Output the final result in JSON format:**
```json
{
  "name_1": "customer_identifier",
  "name_2": "consumer_identifier",
  "name_3": "client_identifier"
}

""" } ]


TABLE_SELECTION = [{
    "role": "user",
    "content": """
        You are provided with a table along with a list of candidate table names and descriptions.
        Your task is to carefully evaluate and select tables that are relevant to the given table.
        You need to select at least 3 tables.
        Return only the names of the selected tables in JSON format as follows:
        {
            "selected_table_names": ["table_name_1", "table_name_2", ...]
        }
    """
}]



RANKING_PROMPT = [
    {"role": "system", "content": """
You are tasked with thoroughly evaluating and scoring column candidates for a given query column based on their relevance, granularity, and specificity, while also considering data types. Your goal is to rank candidates and select the top 5 most suitable ones.
"""}, 
    {"role": "user", "content": """
**Instructions:**

1. **Analyze Query and Each Candidate:**
   - Conduct a detailed analysis of the query column and each candidate column using their metadata.

2. **Understand Data Granularity:**
   - Evaluate whether each table stores broad or specific data based on the table name and description, identifying the level of detail and scope each table addresses.

3. **Align Table Granularity:**
   - Compare the granularity of candidate tables to the query table, focusing initially on the table names. Prioritize candidates whose table granularity aligns closely with the query table's granularity.

4. **Strict Scoring Criteria:**
   - **Domain-Specific Table Names:** Deprioritize candidates with generic, non-domain-specific table names (e.g., MAN, SUPPLIER).
   - **Analytical Comparison:** Compare the table names and descriptions between the query table and candidate tables.
   - **Close Match Considerations:**
      - In closely matched scenarios, prefer tables that provide more granular data for detailed insights.
      - Prefer columns with compatible data types to the query column for effective data comparison and analysis.

5. **Candidate Ranking:**
   - Rank all candidates based on evaluations considering table granularity alignment and data types.
   - Select the top 5 candidates from the ranked list to ensure suitability.

6. **Explanations:**
   - Provide detailed explanations for each step and sub-step in the process.
   - Clarify your reasoning for the ranking order of columns, especially for the top 5 selections.

7. **Output Formatting:**
   - Format the ranking output as JSON, listing only the indices of the top 5 ranked columns under the key "selected_columns."
   - Present explanations separately, outside of the JSON format.
"""}, 
    {"role": "assistant", "content": """
Great! I can help you with that. Please provide the query column information and the candidate columns you would like to evaluate.
"""}
]

