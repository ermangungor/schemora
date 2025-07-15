COT_COLUMN_ENRICHMENT_THREE = [{
        "role": "user",
        "content": """
You are given a column name and the table it belongs to. Your task is to generate three names for this column using a structured reasoning process. Explain your approach for each step.

**Instructions:**

1. **Understand the domain:**

   - Use the table name and description to infer the domain or subject area.

2. **Generate three candidate names:**

   - Use domain-specific terminology to propose three alternative names for the column.
   - Do **not** reuse any words that appear in the original column or table name.
   - Avoid abbreviations or acronyms.

3. **Reflect and select the most relevant names:**

   - Review the five candidates and select three that best match the inferred domain.
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

2. **Generate five candidate names:**
 
 consumer_identifier, shopper_profile_key, client_reference_number, buyer_token, transaction_participant_id

3. **Reflect and select the most relevant names:**

shopper_profile_key and transaction_participant_id are too long or abstract.

I will select: consumer_identifier, client_reference_number, buyer_token

4. **Output the final result in JSON format:**

        ```json
        { 
            "name_1": "shopper_idenfier",
            "name_2": "client_reference_number", 
            "name_3": "buyer_token" 
        }
        ```
        """,
    }]

COT_COLUMN_ENRICHMENT_ONE = [
    {
        "role": "user",
        "content": """
You are given a column name and the table it belongs to. Your task is to generate and select one name for this column using a structured reasoning process. Explain your approach for each step.

**Instructions:**

1. **Understand the domain:**

   - Use the table name and description to infer the domain or subject area.

2. **Generate three candidate names:**

   - Use domain-specific terminology to propose three alternative names for the column.
   - Do **not** reuse any words that appear in the original column or table name.
   - Avoid abbreviations or acronyms.

3. **Reflect and select the most relevant name:**

   - Review the three candidates and select the one that best matches the inferred domain.
   - Ensure it reflects technical or contextual relevance.

4. **Output the final result in JSON format:**

```json
{
  "name_1": "name_1"
}
"""
    }, 
    { "role": "assistant", "content": "Certainly! Please provide the table and column name." },

    { "role": "user", "content": """ 
Column Name: c_id
Table Name: purchase_history
Table Description: This table contains the purchase history of customers.
""" }, 
    { 
        "role": "assistant", "content": """ Steps:

1. **Understand the domain:**

The domain is marketing or e-commerce, focusing on customer transactions.

2. **Generate three candidate names:**

consumer_identifier, client_reference_number, buyer_token

3. **Reflect and select the most relevant name:**

While all names are relevant, "consumer_identifier" most clearly aligns with the focus on identifying individual customers.

I will select: consumer_identifier

4. **Output the final result in JSON format:**

```json
{ 
    "name_1": "consumer_identifier"
}
```""",
    }]

COT_COLUMN_ENRICHMENT_TWO = [
    {
        "role": "user",
        "content": """
You are given a column name and the table it belongs to. Your task is to generate and select two names for this column using a structured reasoning process. Explain your approach for each step.

**Instructions:**

1. **Understand the domain:**

   - Use the table name and description to infer the domain or subject area.

2. **Generate four candidate names:**

   - Use domain-specific terminology to propose four alternative names for the column.
   - Do **not** reuse any words that appear in the original column or table name.
   - Avoid abbreviations or acronyms.

3. **Reflect and select the most relevant names:**

   - Review the four candidates and select two that best match the inferred domain.
   - Ensure they reflect technical or contextual relevance.

4. **Output the final result in JSON format:**

```json
{
  "name_1": "name_1",
  "name_2": "name_2"
}
"""
    }, 
    { "role": "assistant", "content": "Certainly! Please provide the table and column name." },

    { "role": "user", "content": """ 
Column Name: c_id
Table Name: purchase_history
Table Description: This table contains the purchase history of customers.
""" }, 
    { 
        "role": "assistant", "content": """ Steps:

1. **Understand the domain:**

The domain is marketing or e-commerce, focusing on customer transactions.

2. **Generate four candidate names:**

consumer_identifier, client_reference_number, buyer_token, transaction_member_id

3. **Reflect and select the most relevant names:**

After reviewing the options, "consumer_identifier" and "client_reference_number" are the most relevant and clear.

I will select: consumer_identifier, client_reference_number

4. **Output the final result in JSON format:**

```json
{ 
    "name_1": "consumer_identifier",
    "name_2": "client_reference_number"
}
```""",
    }]

COT_COLUMN_ENRICHMENT_SYNONYMS_THREE = [
    {
        "role": "user",
        "content": """
You are given a column name and the table it belongs to. Your task is to generate semantically three enriched names for this column using a structured reasoning process. Explain your approach for each step.

**Instructions:**

1. **Understand the domain:**
   - Use the table and column names along with their descriptions to infer the domain or subject area.

2. **Generate enriched names:**
   - Propose five alternative names that are semantically richer and more descriptive.
   - Avoid using any abbreviations or acronyms.

3. **Reflect and select final names:**
   - Review your proposed names and select the three most relevant ones based on domain-specific relevance.

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
 
customer_identifier, consumer_identifier, client_identifier, account_holder_identifier, transaction_participant_id

3. **Reflect and select final names:**

"account_holder_identifier" and "transaction_participant_id" are more generic or overly formal. The three most relevant names within the marketing/e-commerce context are: customer_identifier, consumer_identifier, client_identifier


4. **Output the final result in JSON format:**
```json
{
  "name_1": "customer_identifier",
  "name_2": "consumer_identifier",
  "name_3": "client_identifier"
}

""", }, ]

COT_COLUMN_ENRICHMENT_SYNONYMS_ONE = [
    {
        "role": "user",
        "content": """
You are given a column name and the table it belongs to. Your task is to generate and select one enriched name for this column using a structured reasoning process. Explain your approach for each step.

**Instructions:**

1. **Understand the domain:**
   - Use the table and column names along with their descriptions to infer the domain or subject area.

2. **Generate three enriched names:**
   - Propose three alternative names that are semantically richer and more descriptive.
   - Avoid using any abbreviations or acronyms.

3. **Reflect and select the final name:**
   - Review your proposed names and select the one most relevant based on domain-specific relevance.

4. **Output the final result in JSON format:**

```json
{
  "name_1": "name_1"
}
"""
    }, 
    { "role": "assistant", "content": "Certainly! Please provide the table and column name along with their descriptions." }, 
    
    { "role": "user", "content": """ 

Column Name: c_id
Column Description: This column contains the unique identifier for each customer.
Table Name: purchase_history
Table Description: This table contains the purchase history of customers. 
""", }, 

    { "role": "assistant", "content": """ Steps:

1. **Understand the domain:**

The domain is marketing or e-commerce. The focus is on customer interactions and transactions.

2. **Generate three enriched names:**

customer_identifier, consumer_identifier, client_identifier

3. **Reflect and select the final name:**

Among the options, "customer_identifier" most effectively captures the essence of uniquely identifying individual customers.

I will select: customer_identifier

4. **Output the final result in JSON format:**

```json
{
  "name_1": "customer_identifier"
}
```""",
    }]
COT_COLUMN_ENRICHMENT_SYNONYMS_TWO = [
    {
        "role": "user",
        "content": """
You are given a column name and the table it belongs to. Your task is to generate and select two enriched names for this column using a structured reasoning process. Explain your approach for each step.

**Instructions:**

1. **Understand the domain:**
   - Use the table and column names along with their descriptions to infer the domain or subject area.

2. **Generate enriched names:**
   - Propose four alternative names that are semantically richer and more descriptive.
   - Avoid using any abbreviations or acronyms.

3. **Reflect and select final names:**
   - Review your proposed names and select the two most relevant ones based on domain-specific relevance.

4. **Output the final result in JSON format:**

```json
{
  "name_1": "name_1",
  "name_2": "name_2"
}
"""
    }, 
    { "role": "assistant", "content": "Certainly! Please provide the table and column name along with their descriptions." },

    { "role": "user", "content": """ 

Column Name: c_id
Column Description: This column contains the unique identifier for each customer.
Table Name: purchase_history
Table Description: This table contains the purchase history of customers. 
""" }, 

    { "role": "assistant", "content": """ Steps:

1. **Understand the domain:**

The domain is marketing or e-commerce. The focus is on customer interactions and transactions.

2. **Generate enriched names:**

customer_identifier, consumer_identifier, client_identifier, buyer_identification_number

3. **Reflect and select final names:**

The two most relevant names within the marketing/e-commerce context are: customer_identifier and consumer_identifier, as they best communicate the concept of unique customer identification.

I will select: customer_identifier, consumer_identifier

4. **Output the final result in JSON format:**

```json
{
  "name_1": "customer_identifier",
  "name_2": "consumer_identifier"
}
```""",
    }]


TABLE_SELECTION = [{
    "role": "user",
    "content": """
        You are provided with a table along with a list of candidate table names and descriptions.
        Your task is to carefully evaluate and select tables that are relevant to the given table.
        
        
        Return only the names of the selected tables in JSON format as follows:
        
        {
            "selected_table_names": ["table_name_1", "table_name_2", ...]
        }
    """
}]
