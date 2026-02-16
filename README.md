# IL-17A Inhibitor Response Predictor 🏥

## Overview
This repository contains a machine learning-based web application designed to predict the treatment outcome (specifically the risk of non-response, defined as "Event 0") for psoriasis patients receiving IL-17A inhibitors. 

The application utilizes a trained **Random Forest** classifier and integrates **SHAP (SHapley Additive exPlanations)** to provide transparent, individualized interpretations for clinical decision-making.

## Key Features
* **Clinical Prediction**: Calculates the exact probability of a patient being a "Non-Responder" based on 7 accessible clinical and hematological parameters.
* **Interpretable AI**: Automatically generates SHAP Force Plots from the perspective of Event 0.
  * 🔴 **Red bars**: Risk factors pushing the prediction towards treatment failure.
  * 🔵 **Blue bars**: Protective factors pushing towards treatment success.
* **User-Friendly UI**: Built with Streamlit, requiring no coding experience for clinicians to use.

## Clinical Features Required
1. **BMI** (Body Mass Index)
2. **Biologics History** (0 = No, 1 = Yes)
3. **Baseline PASI**
4. **Hemoglobin** (g/L)
5. **ALP** (Alkaline Phosphatase, U/L)
6. **IBil** (Indirect Bilirubin, μmol/L)
7. **SII** (Systemic Immune-Inflammation Index)

## Installation & Local Usage
To run this application locally on your machine:

1. Clone this repository:
   ```bash
   git clone [https://github.com/your-username/psoriasis-response-predictor.git](https://github.com/your-username/psoriasis-response-predictor.git)
   cd psoriasis-response-predictor
