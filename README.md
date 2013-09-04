fresco
======

A tool for choosing optimal variable-size features in classification problems.

Dependencies:

- python (tested with 2.7.3)
- numpy (tested with 1.7.1)
- scipy (tested with 0.12.0)
- scikit-learn (tested with 0.14.1)

To test your fresco installation, run the following command:

    python script.py -h

You should see the script's help text displayed.

Next, try running fresco over the test dataset (Crawford et al., 2009) that is
included in the repository:

    python fresco.py --group_map_files test_data/otu_maps.txt --mapping_file test_data/study_451_mapping_file.txt --prediction_field TREATMENT --start_level 3 --n_procs 4 --model lr --prediction_testing_output test_data/prediction_testing_output.txt --feature_vector_output test_data/feature_vector_output.txt

This command assumes that you are at the root level of the fresco directory.
You can change the number of parallel processes that are launched by changing
```--n_procs``` in the above command.

References
----------

The test dataset that is included with fresco is derived from the following
study:

Crawford, P. A., Crowley, J. R., Sambandam, N., Muegge, B. D., Costello, E. K.,
Hamady, M., et al. (2009). Regulation of myocardial ketone body metabolism by
the gut microbiota during nutrient deprivation. Proc Natl Acad Sci U S A,
106(27), 11276-11281.
