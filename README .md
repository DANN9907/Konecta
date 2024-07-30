
# Konecta Knowledge Tests

This repository contains a series of knowledge tests designed for employment evaluation at Konecta. The tests aim to assess various skills and competencies required for the Data Scientist position, focusing on machine learning (ML) and generative artificial intelligence (GenAI) concepts.

## Table of Contents

- [Overview](#overview)
- [Contents](#contents)
- [Usage](#usage)
- [Submission Guidelines](#submission-guidelines)
- [Contact Information](#contact-information)

## Overview

I am participing in the Konecta Knowledge Tests. These tests cover a range of topics including data cleaning, exploratory data analysis, predictive modeling, and the development of a chatbot using generative AI.

## Contents

### Part 1: Traditional ML Skills

**Objective:** Analyze customer data to predict and mitigate customer churn.

- **Datasets:**
  - `train.csv`: Training dataset. Churn is the binary target to predict.
  - `inference.csv`: Inference dataset for predicting churn.
  - `inference_target.csv`: The inference dataset with the new column "Churn" predictions.

**Tasks:**
1. Data exploration and cleaning.
2. Building and evaluating predictive models.
3. Selecting the best model and justifying the choice.
4. Discussing potential improvements.
5. Considering object-oriented programming (OOP) for the solution.
6. Deploying the model to production.

**Results**

- `Class_ML_skills.py`: Script with a class and method for data exploration, data cleaning, create models and test the results
- `Class_ML_skills.py`: Script where the method of "Class_ML_skills.py" is used and show all the result.
- `inference_target.csv`: The inference dataset with the new column "Churn" predictions.
- `Questions_konecta.pdf`: This file contain the answer to the question in the technical test and the justification of the chosen model

### Part 2: Generative AI Skills

**Objective:** Develop a chatbot to answer questions about a product catalog for children.

- **Shared File:**
  - `Bruno_child_offers.pdf`: Catalog to be used as the knowledge base for the chatbot.

**Tasks:**
1. Create a chatbot to answer questions related to the catalog.
2. Ensure the chatbot reduces hallucination and answers only relevant questions.
3. Deploy the chatbot to production.

**Results**

- `Part_2_Generative_ai_skills.ipynb`: Notebook with the implementation of the chatbot using technologies like, langchain, openai, RAG.
- `Questions_konecta.pdf`: This file contain the answer to the question in the technical test and the justification of the chosen model

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/DANN9907/Konecta.git
   ```
2. Navigate to the repository directory:
   ```bash
   cd Konecta
   ```
3. Follow the instructions provided in each test directory to complete the tests.

Thank you for your interest in my, Konecta.
