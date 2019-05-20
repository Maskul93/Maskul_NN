# Maskul_NN

## Functions Description
  - **Normalizer**: This function normalizes the "raw" input data using a _MinMax_ approach. The idea is to make each feature of the input to belong the interval **[0,1]**; each feature maximum value is marked as _1_, while each feature minimum value is marked as _0_. 
  
  - **do_windows**: This function is required to create the windows according to the approach one wants to use. It is possible to create either "static" windows (e.g. discarding transitions) or dynamic windows, overlapping windows (_**sliding windows**_).
  
  - **do_folds_inter_subjects**: This function creates the folds to feed the NN by mixing up the whole set of subjects.
  
  - **do_folds_intra_subject**: This function creates the folds to feeed the NN on the _**same patient**_.
  
  
