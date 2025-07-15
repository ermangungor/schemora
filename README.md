# Schemora 🚀

Welcome to **Schemora**! 🎉  
This repository helps you efficiently manage and match schemas. Below, you’ll find instructions to set up your environment and get started.

---

## 🛠️ Environment Setup

Follow these steps to prepare your development environment:

1. **Clone the repository**
    ```bash
    git clone https://github.com/your-username/schemora.git
    cd schemora
    ```

2. **Create and activate the conda environment**
    ```bash
    conda create -n schemora python=3.10.10
    conda activate schemora
    pip install -r requirements.txt
    ```

    ```bash
    export PYTHONPATH=.:$PYTHONPATH
    ```

3. **Set up OpenAI credentials**
   Create a `.env` file and save your OpenAI API key there.
   

---

## 🚀 How to Run

### 1. Data

- The raw data used in our paper is provided at the following link:

  > *(https://drive.google.com/drive/folders/1w7mx55ExgEdm-4cIrSckFLRqbOMzr9xG?usp=drive_link)*

- Download and save it under the `schemora/` directory.

- If you want to run the pipeline on new data, ensure it follows the structure below. You can use the provided data as an example:

    ```
    raw_data/
    └── benchmark-name/
        ├── source.[csv|parquet]
        ├── target.[csv|parquet]
        └── mapping
    ```

---

### 2. Prepare your config

- We provide two template config files:  
  - `config_omop_template.toml`
  - `config_synt_template.toml`

- Use these as references to create your own configuration file.

---

### 3. Run Schemora

- Update the config file name in the `Makefile`, then run:

    ```bash
    make all
    ```

- This will execute the entire Schemora pipeline.

---

### 4. Batch Mode

- To perform experiments in batch mode:

  1. Run the config generator script to create multiple configs:
      ```bash
      python scripts/parameter_search_config_generator.py
      ```
      *(Modify this script as needed for your experiments.)*

  2. Run Schemora in batch mode:
      ```bash
      bash scripts/batch_script.sh
      ```

---

### 5. Evaluation

- By default, `make all` performs evaluation at the end.

- Alternatively, for batch evaluations, run:
    ```bash
    python scripts/batch_evaluation.py
    ```

## FAQ

1. **Can I access the experiments shared in the paper?**

   Yes. All experiments and their results are available at the following link:  
   [Google Drive Folder](https://drive.google.com/drive/folders/1w7mx55ExgEdm-4cIrSckFLRqbOMzr9xG?usp=drive_link)

