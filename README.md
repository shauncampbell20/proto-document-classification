# proto-document-classification
Document classification using embedded prototypes

## Setup
1. Clone the repo:
```
git clone https://github.com/shauncampbell20/proto-document-classification.git
```

2. Set up environment:
```
cd <path to proto-document-classification>
python -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
```

3. Install Tesseract (Windows install link on [UB-Mannheim GitHub](https://github.com/UB-Mannheim/tesseract/wiki))
- Add Tesseract location to config.py ```TESSERACT = <path to tesseract>```

## Building a Model
1. Create a new folder in the /models directory to store the model.
2. Create a sub-folder called "examples"
3. For each desired label, create a new folder to contain example documents
4. Add sample documents to each label folder (10-20 recommended)
5. The directory should look like this:

```
proto-document-classification/
├── .gitignore
├── README.md
├── requirements.txt
├── .venv/
├── src/
│    ├── __init__.py
│    ├── classify.py
│    ├── config.py
│    ├── extract_text.py
├── models/
│    ├── my-model/
│          ├── examples/
│                ├── label1/
│                      ├── <files>
│                ├── label2/
│                      ├── <files>
│                ├── label3/
│                      ├── <files>
│                ├── ...
```
6. Add the location of the model to config.py ```MODEL = <path to my-model>```
7. Run command ```python src/classify.py --build

## Classifying New Documents

To classify a single file, run the following command:
```
python src/classify.py --file <path to file>
```

To classify a directory of files, run the following command. It will generate a comma-delimited text file containing the file path, label, and score.
```
python src/classify.py --directory <path to directory> --output <path to output file>
```
