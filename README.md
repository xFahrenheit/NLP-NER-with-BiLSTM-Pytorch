## Named entity entity recognition (NER) using GloVE embeddings for training a Bidirectional LSTM and LSTM-CNN model
Embedding → BLSTM → Linear → ELU → classifier

Task 1:

To run task 1:
1. Save the blstm1.pt model in the same folder as test1_run.py and make sure the test data and dev data are in a folder called data which is also located in the same folder as test1_run.py .
2. Make sure that the unzipped glove.6B.100d file is also in the same folder as the test1_run.py file
3. Move to the folder directory where the test1_run.py file is and type 'python3 test1_run.py' in the cmd prompt to generate the dev1.out and test1.out files


Task 2:

To run task 2:
1. Save the blstm2.pt model in the same folder as test2_run.py and make sure the test data and dev data are in a folder called data which is also located in the same folder as test2_run.py .
2. Make sure that the unzipped glove.6B.100d file is also in the same folder as the test2_run.py file
3. Move to the folder directory where the test2_run.py file is and type 'python3 test2_run.py' in the cmd prompt to generate the dev2.out and test2.out files.

More detailed documentation of the code is included in task1.ipynb and task2.ipynb files which are also included in the folder. 

