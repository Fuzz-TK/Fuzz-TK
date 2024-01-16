# Fuzz-TK
## [Fuzz-TK]A tool that employs fuzzy set relus using features based on semantics, spectra, and mutation to achieve effective spectrum-based fault localization and semantic-based program repair.
**Requirements:**
1. **Java 1.7**
2. **Python 3.6**
3. **Defects4J 1.2.0**
4. **SVN >= 1.8**
5. **Git >= 1.9**
6. **Perl >= 5.0.10**
7. **PyTorch-1.5.1**
***
### Datatest:
If you wish to use publicly available datasets, you can access *[data_for_transfer.zip](https://mega.nz/file/u0wQzRga#Q2BHCuRD2aW_61vshVbcxj-ObYh2cyGhqOAmAXNn-T0)*.Unzip it and put the extracted datasets in the corresponding path,Put `dataset_fl.pkl` into `./fault_localization/`,Put `dataset_pr.pkl` into `./program_repair/`,Put `src_code.pkl` into `./fault_localization/binary_classification/d4j_data/` .
***
## Our Works
Fuzz-TK is an automated software debugging method that aims to relieve developers of the significant burden of fixing software failures.
### Learing Transfer Knowledge

For generated SLforge or realistic models, place them in the `courpus_seed` folder. At the command line, type: ```ModelProcessing```. The results will be displayed in the `directed_graph.txt` file.

### Fault Localization Task

Run `python train.py <fix_template>` to train the corresponding binary classifier, and run `python data_preprocess_for_d4j.py <fix_template>` to preprocess each suspicious statement in the Defects4J dataset. Then execute `python predict_for_d4j.py` to obtain the final 11-dimensional semantic features.

### Program Repair Task
Run `python pipeline.py` and `python train_github.py` to train the multi-classifier, then execute `python train_d4j.py` to fine-tune the parameters of the model. Place `model_save_d4j.m` and `w2v_32.m` into the `dnn_model` directory, then run `python parse_version.py`. Finally, run `python get_sus_file.py`.

### Specifically, if you want to use our model quickly, some steps need to be done.


Firstly, navigate to the path `./fault_localization/ranking_task/run_model/` and execute the command `python run_group.py`.

Next, proceed to the directory `./program_repair/automatic_fix/shell_71_versions/` and run the `python generate_shell.py` command to create a shell file that includes 71 sequential commands for the 71 bugs in Defects4J.

Finally, execute `./run_repair.sh` to acquire the repair results. All repair logs are located in `./program_repair/dnn_model/dnn_tbar_log/`, and all generated plausible patches can be found in `./program_repair/automatic_fix/OUTPUT/`.
#### Introduction to each code file
`data.m`: Preserving transferred knowledge

`fault_localization.m`: Obtaining the fault location information

`program_repair.m`: The fault is repaired by migrating knowledge combined with Tbar
***


### datasetfl: 
datasetfl consists of a large number of historical bug fixes from popular open-source Java projects on GitHub. Specifically, if a commit message contains keywords such as "error", "bug", "fix", "issue", "incorrect", "fault", "defect", "type", and others, the commit is considered to be related to a bug. There are a total of 1,010,628 bug-related commits identified. The dataset then identifies bug types from bug fix commits. For all 15 repair templates defined in TBar, if the code element \textit{d} with a bug is fixed after applying repair template  \textit{gu}, \textit{d} is considered to have a bug of the corresponding type of \textit{gu}. Once the repair template is determined, the code elements before the commit are labeled with their respective error types. After collecting bug fix commits with error type labels, the dataset assigns each commit to one or more groups based on their error types. To collect negative samples, a novel approach is proposed by gathering all statements containing necessary syntax components and their corresponding methods as negative sample candidates to construct a balanced dataset. Finally, the 11 constructed datasets include 392,567 positive samples and an equal number of negative samples (a total of 785,134) for \textit{datasetfl}.

### datasetpr: 
datasetpr is a dataset that contains 408,091 samples with 11 different categories. The dataset is used for program fixing, where the goal is to select the correct repair template to fix a given erroneous statement. This task can be seen as a multi-class classification problem. Each sample in the dataset represents a version of the code (prior to the submission) and is classified into one of the 11 predefined error types. Since error fixes can be labeled with one or multiple error types, the dataset assigns them to deeper error types in the Abstract Syntax Tree (AST) hierarchy. This allows for the use of smaller change regions in the repair templates, which is preferred by actual developers.

### The following tables show the data sizes for datasetfl and datasetpr:
## Statistics of Datasetfl and Datasetpr

| Fix Templates                  | Datasetfl | Datasetpr |
|-------------------------------|-----------|-----------|
| Insert Null Pointer Checker   | 11,320    | 7,600     |
| Insert Missed Statement       | 94,240    | 54,445    |
| Mutate Conditional Expression | 62,334    | 30,530    |
| Mutate Data Type              | 6,586     | 5,178     |
| Mutate Literal Expression     | 71,072    | 55,044    |
| Mutate Method Invocation Expr.| 397,508   | 166,943   |
| Mutate Operators              | 3,346     | 2,326     |
| Mutate Return Statement       | 25,022    | 20,680    |
| Mutate Variable               | 59,836    | 35,156    |
| Move Statement                | 14,626    | 7,812     |
| Remove Buggy Statement        | 39,244    | 22,377    |
| Total                         | 785,134   | 408,091   |
