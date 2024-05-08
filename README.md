# Title


## Abstract <a name="abstract"></a>
***

<br>

![workflow](https://github.com/B1607/SirtuinNAD/blob/b49ebc651c953b041f5e2ef0dfd2778277020526/other/Figure_Sirtuin.png)
## Dataset <a name="Dataset"></a>

| Dataset            | Protein Sequence | ATP Interacting Residues | Non-Interacting Residues |
|--------------------|------------------|--------------------------|--------------------------|
| General NAD        | 195              | 4772                     | 61256                    |
| Sirtuin NAD        | 7                | 186                      | 2728                     |
| Sirtuin like NAD   | 45               | 1162                     | 18687                    |
| Total              | 247              | 6120                     | 82671                    |


## Quick start <a name="quickstart"></a>

### Step 1: Generate Data Features

Navigate to the data folder and utilize the FASTA file to produce additional data features, saving them in the dataset folder.

Example usage:
```bash
python get_ProtTrans.py -in "Your FASTA file folder" -out "The destination folder of your output"
```
Alternative example:
```bash
python get_ProtTrans -in ./Fasta/Train -out ./Train
```

### Step 2: Generate Dataset Using Data Features

Transition to the dataset folder and utilize the data features to produce a dataset.

Example usage:
```bash
python get_dataset.py -in "Your data feature Folder" -label "Your data label Folder" -w "Window Size" -ds "Datatype of your feature" -out "The destination folder of your output"

```
Alternative example:
```bash
python get_dataset.py -in ../Data/Train -label ../Data/Fasta/Train/label -w 7 -ds ".prottrans" -out ./Train
```

### Step 3: Execute Prediction

Navigate to the code folder to execute the prediction.

Command-line usage:
```bash
python main.py -ds Sirtuin7 -n_dep 7 -n_fil 256 -ws 2 4 -vm independent
```
Alternatively, utilize the Jupyter notebook:
```bash
main.ipynb
```

