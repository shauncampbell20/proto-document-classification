import warnings
import os
from sentence_transformers import SentenceTransformer, util
import numpy as np
from transformers import AutoTokenizer, utils
from tqdm import tqdm
import re, unicodedata
import torch
import sys
import argparse
from extract_text import OCR, get_text
from config import MODEL

warnings.filterwarnings("ignore")
utils.logging.set_verbosity_error()

REGEXES = {
    "page": re.compile(r'page\s+\d+\s+of\s+\d+', flags=re.I),
    "numbers": re.compile(r'\b\d+\b'),
    "hyphen": re.compile(r'-\s+'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "money": re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?'),
    "phone": re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
}

def walk(root):
    ''' yield all full file paths in a directory '''
    for file in os.listdir(root):
        if os.path.isdir(os.path.join(root, file)):
            yield from walk(os.path.join(root, file))
        else:
            yield os.path.join(root, file)

def clean_text(text):
    ''' clean text '''
    text = unicodedata.normalize("NFKC", text).lower()  # normalize unicode
    text = re.sub(r'\s+', ' ', text).strip()
    for pat in REGEXES.values():
        text = pat.sub('',text)
    return text
    
def chunk_text(text, max_tokens=400, overlap=50):
    ''' split text into chunks '''
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = tokens[i:i+max_tokens]
        chunks.append(tokenizer.decode(chunk))
    return chunks
    
def embed_document(raw, batch_size=16):
    ''' clean text, chunk, embed each chunk, mean-pool'''
    text = clean_text(raw)
    chunks = chunk_text(text)
    
    with torch.no_grad():
        emb = model.encode(
            chunks, 
            batch_size=batch_size, 
            convert_to_tensor=True, 
            normalize_embeddings=True, 
            show_progress_bar=False
        )
    return emb.mean(dim=0)


def classify_document(file=None, text=None):
    ''' embed document and return most similar prototype '''
    if not text:
        if file:
            text = get_text(file)
            if text == '':
                text = OCR(file)
        else:
            raise TypeError('file or text must be provided')
    emb = embed_document(text)
    scores = util.cos_sim(emb, protos).cpu().numpy()
    am = np.argmax(scores)
    pred_label = labels[am]
        
    return pred_label, scores[0][am]

def create_prototypes():       
    ''' create prototypes from examples'''
    # examples directory
    wd = os.path.join(MODEL, 'examples')
    if not os.path.exists(wd):
        raise Exception(wd+' not found')

    # Extract text from examples
    print('Extracting texts...')
    examples = {}
    for label in os.listdir(wd):
        print(label)
        if label not in examples.keys():
            examples[label] = []
        for file in tqdm(os.listdir(os.path.join(wd, label))):
            text = get_text(os.path.join(wd, label, file))
            if text == '':
                text = OCR(os.path.join(wd, label, file))
            examples[label].append(text)

    # create prototype embeddings
    print('Embedding texts...')
    class_embeddings = {}
    for label, texts in examples.items():
        print(label)
        embs = []
        for text in tqdm(texts):
            embs.append(embed_document(text))
        class_embeddings[label] = sum(embs)/len(texts)
    
    # save labels and prototypes
    labels = list(class_embeddings.keys())
    protos = torch.stack(list(class_embeddings.values()))
    torch.save(labels, os.path.join(MODEL,'labels.pt'))
    torch.save(protos, os.path.join(MODEL,'protos.pt'))

    print(MODEL,'built successfully!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--info', action='store_true', help='Display information')
    parser.add_argument('-b','--build', action='store_true', help='Build prototypes from examples')
    parser.add_argument('-f','--file', help='Classify a single file')
    parser.add_argument('-d','--directory',help='Classify and rename files in a directory.')
    parser.add_argument('-o','--output',help='Output file for directory classification.')
    args = parser.parse_args()

    # Model and tokenizer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Build prototypes
    if args.build:
        create_prototypes()
        sys.exit()

    else:
        try:
            labels = torch.load(os.path.join(MODEL, 'labels.pt'))
            protos = torch.load(os.path.join(MODEL, 'protos.pt'))
        except:
            raise Exception('Unable to load labels and protos from '+MODEL)

    # display info
    if args.info:
        print('Model:',MODEL)
        print('Labels:')
        for lab in labels:
            print('  - '+lab)
        sys.exit()

    # single file
    elif args.file:
        lab, score = classify_document(args.file)
        print('File: ',args.file)
        print('Predicted label: ',lab)
        print('Score: ',score)

    # directory of files
    elif args.directory:
        if not args.output:
            raise Exception('Please provide output file using --output argument')
        if not os.path.exists(args.directory):
            raise Exception(args.directory+' not found.')
        try:
            output = args.output
            output = os.path.splitext(output)[0]+'.txt'
            with open(output,'w') as f:
                f.write('')
        except:
            raise Exception('Unable to write to '+output)
        
        files = [f for f in walk(args.directory)]
        for file in tqdm(files):
            lab, score = classify_document(file)
            output_str = file+','+lab+','+str(score)+'\n'
            with open(output,'a') as f:
                f.write(output_str)
