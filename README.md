# Maskul_NN

## Scripts Description
  - **Normalizer**: This function normalizes the "raw" input data using a _MinMax_ approach. The idea is to make each feature of the input to belong the interval **[0,1]**; each feature maximum value is marked as _1_, while each feature minimum value is marked as _0_. 
  
  - **do_windows**: This function is required to create the windows according to the approach one wants to use. It is possible to create either "static" windows (e.g. discarding transitions) or dynamic windows, overlapping windows (_**sliding windows**_).
  
  - **do_folds_inter_subjects**: This function creates the folds to feed the NN by mixing up the whole set of subjects.
  
  - **do_folds_intra_subject**: This function creates the folds to feeed the NN on the _**same patient**_.
  
  
## NN Description
### Hyperparameters

 - **Learning Rate**: The net has to find, 
step-by-step, an optimum weight in the _set of weight_. 
The optima could be either _global_ or _local_, and the 
net must be able to find them. To do so, the net has to 
"change" of a certain amount in each searching step, 
and this quantity is called _**learning rate**_. It has 
been set to a value of **0.1** by default.

 - **Batch Size**: This value defines the number of 
_samples_ which will be propagated through the network 
per time. For instance, if the _training set_ has 512 
samples, and the _batch size_ is set to 32, then the 
network will be trained 32 samples per time, until the 
end of the 512 total samples. This value has been set 
to **32**.

 - **Number of Folds**: A neural network must be 
trained many times by "shuffling" the data with which 
has been fed. This is done in order to increase the 
variability of data, as well as to see "better" how the 
network works. This approach is **_mandatory_**, and a 
value of **10 folds** is suggested.

 - **Validation Split**: It is the percentage of the 
training set allocated for creating the _dev set_ 
(_development set_). We decided to split the initial 
training set with a 80/20 ration, thus its value has 
been set to **0.2**.

 - **Test Validation Split**: It is the percentage of 
the training set allocated for creating the _test val 
set_ (_test validation set_). Actually, this value is 
not used if we are using three files from the folding 
process (Train, Test Learned, and Test Unlearned). See 
the **Folding Procedure** for a better explanation.

 - **Samples per Window**: It represents the number of 
samples composing each window (i.e. each row of the 
input files) for which a label has been assigned 
according to the _**do_windows**_ criteria. It has been 
set to **20** by default, and it is dependent on how 
the windowing procedure has been chosen. 

 - **Exclude Features**: If _true_, it allows to 
exclude the some of the features, selecting them into 
the variable ``feature select''
