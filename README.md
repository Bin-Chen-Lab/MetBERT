# MetBERT
# Step by Step instruction for replicating results

## Requirements:

* Python3
* Pytorch
* Transformers
* Scikit-learn
* Captum
* Numpy
* Pandas

## Project Structure

├── LICENSE
├── README.md 
├── data
│   ├── annotations   
│   ├── notevents      
│   ├── processed     
|
├── src              
│   ├── preprocess.py       
│   ├── model.py        
│   ├── train_val.py      
│   └── run.py            
|
├── models           
│
├── notebooks
│
├── references    
│
├── results           




## Data Source
* Get access to MIMIC III by going over to https://physionet.org/content/mimiciii-demo/1.4/
* Once granted access, download NOTEEVENTS.csv.gz which has all the unstructred patient notes in MIMIC III
* To get annotated phenotypes for the notes go over to https://github.com/sebastianGehrmann/phenotyping
* They have annotations in data/annotations.csv, where they have admission ID, subject ID, and chart time as columns
* Load both csv files and merge on HADM_ID. As we are only focusing on Discharge summary, filter 'CATEGORY' for 'Discharge Summary', rename the csv as final.csv 


## Pre-Processing
* Run preprocess.py and you will get 2/3 csv files depending on if you are planning to train-test or train-validation-test
* You can customize code in order to get desired outcome for pre-processing.

## Fine-tuning
* run run.py file 

## Examples
A simple example:

```bash
python3 run.py --epochs 4
```

A more advanced example:

```bash
python3 run.py --epochs 4 \
    --model_type 'pubmedbert' \
    --train_batch 64 \
    --device 'gpu' \
    --scheduler 'yes' \
    --aproach 'mixed'
```

* AAFter successfully fine-tuning the model, you will have model metrics as png file as well as examples missclassifed by the mdoel for further investigation as a csv file

## Vizualization
* run captumviz.py once you have fine-tuned the model and placed in model/ folder
* Following args are needed inside the python script.
1) pre-trained model link from huggingface (should be same as the model_type in run.py)
2) finetuned model file path
3) Text you want to visualize
4) Actual label for the text
