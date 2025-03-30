# Project Setup and Usage Guide

## 📌 Setting Up the Virtual Environment
To ensure a smooth execution of the model, create a virtual environment and install dependencies using the provided `requirements.txt` file.

### **1. Create and Activate the Virtual Environment**
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 📌 Download and Place the Model Weights
The model's weights can be found at the provided link: https://drive.google.com/drive/folders/1I-fich1bqEIj3Y4wwa2gAO_KTp3M54eT?usp=sharing. After downloading, move the weights file to:
```
utils/modelscheckpoints/
```

Ensure the model weights are correctly placed before running the test script.

---

## 📌 Running the Model
Once the setup is complete, you can test the model by running the `test.py` script located in the `utils/` directory.

### **Basic Usage**
```bash
python utils/test.py --testingdata /path/to/testdata
```

### **Get Accuracy Instead of Predictions**
If you want the script to return accuracy metrics instead of just predictions, specify `Accuracy` in the argument:
```bash
python utils/test.py --testingdata /path/to/testdata --Predictions Accuracy
```

If the `--Predicitions` argument is not provided, the script will return only the model's predictions.

---

## 📌 Quick Testing via Google Colab
A simple way to test the model is using the `colab_demo.ipynb` notebook provides sample code for testing.

### **Steps:**
1. Open `colab_demo.ipynb`.
2. Load only the test data.
3. Follow the sample code in the notebook to test the model after downloading the weight file.

---

## ⚠️ **Important Note**
Your test data folder must be structured as:
- `test_folder/class/images`
- **or** `test_folder/images`

If you use the second structure (`test_folder/images`), you need to:
1. **Comment out** how `MGDataset` is loading your data in **line 132** of `test.py`.
2. **Uncomment** line **134** in `test.py`.

---

## 🎯 Summary
✅ **Create & activate virtual environment**
✅ **Install dependencies**
✅ **Download & place model weights in `utils/modelscheckpoints/`**
✅ **Run `test.py` with test data**
✅ **Use Colab for quick testing**

Now you're all set to run the model! 🚀

