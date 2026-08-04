# Model Training & Inference Architecture

This directory houses the machine learning ecosystem, dataset processing pipelines, and deep neural network training logs. The primary codebase for data generation, YOLOv8 training, and PatchCore inference tuning is contained within:

📓 **`RV_ProjectV1_Gem.ipynb`**

---

## 🛠️ Environment Setup & Running the Notebook

Because the notebook contains `!pip install` commands designed for cloud environments like Google Colab, you **do not** need to execute those installation cells when running the notebook locally.

Instead, we have provided an automated script to cleanly set up your local environment and launch Jupyter Notebook.

### Setup Instructions:

1. **Run the Setup Script:**
   Execute the following batch script to automatically generate a virtual environment (`.venv`), install all requirements, and register the IPython kernel:
   
   ▶️ **[`run_model_setup.bat`](run_model_setup.bat)**

2. **Open the Notebook:**
   Once the script finishes, the Jupyter Notebook interface will open in your web browser. Click on `RV_ProjectV1_Gem.ipynb` to open it.

3. **Select the Correct Kernel:**
   In the top right corner of the Jupyter interface, ensure that your kernel is set to **`Python (RV Model Env)`**. If it is not, click the kernel name and select it from the dropdown.

You are now ready to run the training pipelines, evaluate the heatmaps, or export weights without worrying about dependency conflicts!
