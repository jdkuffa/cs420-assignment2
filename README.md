# CS420 Assignment 2: Fine-Tuning CodeT5 for Predicting if Statements

# **1. Introduction**  

# **2. Getting Started**  

## **2.1 Preparations**  

1. Clone the repository to your workspace:  
```shell
git clone https://github.com/jdkuffa/cs420-assignment1.git
```

2. Navigate into the repository:

```
cd cs420-assignment1
```

3. Set up a virtual environment and activate it:

### For macOS/Linux:

```
python -m venv ./venv/
```
```
source venv/bin/activate
```

### For Windows:

1. Install ```virtualenv```:
```
pip install virtualenv
```

2. Create a virtual environment:
```
python -m virtualenv venv
```

3. Activate the environment
```
venv\Scripts\activate
```

The name of your virtual environment should now appear within parentheses just before your commands.

To deactivate the virtual environment, use the command:

```
deactivate
```

## **2.2 Install Packages**

Install the required dependencies:

```
pip install -r requirements.txt
```

## **2.3 Run Program**

(1) Run ```main.py```

This program fine-tunes the pre-trained CodeT5-small Transformer model from Hugging Face to automatically recommend suitable if statements in Python functions. After preparing the dataset by masking the if conditions and tokenize the input using Hugging Face's pre-trained AutoTokenizer, the model is trained and fine-tuned on this data and evaluated using multiple evaluation metrics, including exact match, BLEU & CodeBLEU.

```
python main.py
```

## 3. Report

The assignment report is available in the file "Assignment_Report.pdf."
