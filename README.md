## Gen-Z Slang Translator with FLAN-T5

This project builds a Gen-Z Slang Translator using the FLAN-T5 transformer model. It fine-tunes the model to translate informal Gen-Z expressions into formal English, useful for cross-generational communication in professional or academic settings.

---

## Project Overview

- Translates Gen-Z slang (e.g., "rizz", "sus", "cap") into proper English.
- Based on a curated dataset from Hugging Face and manual refinement.
- Fine-tuned with `FLAN-T5` using LoRA for efficiency.
- Evaluation metrics include BLEU, METEOR, ROUGE-L, BERTScore, and Cosine Similarity.
- Demo provided with an interactive **Streamlit web app**.

---

## 1. Installation & Dependencies

```bash
pip install -r requirements.txt
```

Key libraries:
- `transformers`
- `datasets`
- `peft`
- `evaluate`
- `scikit-learn`
- `streamlit`

---

## 2. Dataset Curation

### Sources:
- [`genz_slang_dataset`](https://huggingface.co/datasets/MLBtrio/genz-slang-dataset)
- Extended with context using ChatGPT to form: `genZ_slangs(1.7k_rows)_edited.csv`

### Combined Dataset:
A merged dataset (`genz_finetune_combined_dataset.csv`) is created and stored in `/dataset`, containing:
- `input`: slang + context
- `target`: formal English sentence

---

## 3. Data Preprocessing

Steps include:
- Removing empty entries
- Stripping whitespace
- Tokenizing inputs using Hugging Face's `AutoTokenizer`
- Formatting the dataset for FLAN-T5's text-to-text schema

---

## 4. Evaluation Metrics

The model is evaluated using multiple semantic and lexical metrics:

- **BLEU-1 & BLEU-2**
- **ROUGE-L**
- **METEOR**
- **BERTScore**
- **Cosine Similarity**

All evaluations are run using the `evaluate` library and custom scoring functions.

---

## 5. Model Fine-tuning

- **Base model**: `google/flan-t5-base`
- **Parameter-efficient tuning** via `LoRA`
- Training Arguments:
  - `learning_rate=2e-5`
  - `batch_size=8`
  - `weight_decay=0.01`
  - `epochs=10`

Implemented with `Seq2SeqTrainer` from `transformers`.

---

## 6. Model Inference

Example predictions are generated using `model.generate()` on test entries, with decoding via `tokenizer.batch_decode()`.

---

## 7. Streamlit Web App

A live demo is built using **Streamlit**, featuring:
- 🔍 Slang input to formal English translation
- 📂 Dataset preview tab
- 📈 Metrics & evaluation visualizations

### Run it locally:

```bash
streamlit run app.py
```

---

## 📁 Folder Structure

```
├── dataset/
│   └── genz_finetune_combined_dataset.csv
├── models/
│   └── flan-t5-finetuned/
├── app.py
├── GenzNLP.ipynb
├── requirements.txt
└── README.md
```

---

## Authors

- Joel – Evaluation Metrics, Results & Model Inference 
- Donovan – Data Augmentation, Preprocessing & Streamlit frontend  
- Iggy – Modified Model Pipeline & Hyperparameter Tuning 
- Vernis – Dataset Sourcing, Cleaning & Augmentation  
- Sneha – Dataset Sourcing, Cleaning & Augmentation  


---

## 📜 License

This project is for Nanyang Technological Univeristy's academic purposes (EE6405 module) and not intended for commercial use. Attribution required if reused.
