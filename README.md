

# Project Setup Guide

This guide provides step-by-step instructions for setting up the project, including installing Ollama, pulling the `deepseek-r1:1.5b` model, setting up a Python virtual environment, and running the Python script.

---

## 1. Install Ollama

### **For Windows**
1. Visit the official [Ollama download page](https://ollama.com/download).
2. Download the Windows installer.
3. Run the installer and follow the on-screen instructions.

### **For macOS**
1. Install Ollama via Homebrew:
   ```bash
   brew install ollama
   ```
   Alternatively, download the macOS installer from the [Ollama download page](https://ollama.com/download).

### **For Linux**
1. Download the Linux binary from the [Ollama GitHub repository](https://github.com/ollama/ollama).
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

---

## 2. Pull the `deepseek-r1:1.5b` Model Using Ollama
Run the following command to pull the model:
```bash
ollama pull deepseek-r1:1.5b
```

---

## 3. Create a Python Virtual Environment
1. Navigate to your project directory:
   ```bash
   cd /path/to/your/project
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

---

## 4. Activate the Virtual Environment
### **Windows**
```bash
venv\Scripts\activate
```

### **macOS/Linux**
```bash
source venv/bin/activate
```

---

## 5. Install Dependencies
Ensure you have a `requirements.txt` file in your project directory. Then, install the dependencies:
```bash
pip install -r requirements.txt
```

---

## 6. Run the Python Script
Execute the script using the following command:
```bash
streamlit run app.py
```

---

## Notes
- Ensure you have Python 3.7 or higher installed.
- For any issues with Ollama, refer to the [Ollama Documentation](https://ollama.com).

