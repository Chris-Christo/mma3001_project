# Reproduction evidence, 24 October 2026

The student reported installing the recorded dependencies into a fresh Python environment in a separate GitHub clone and running all six notebooks without errors. These records support that account but are not a complete execution log.

The supplied test report contains 113 cases with no failures, errors or skips. The counts are 21 evaluation tests, 21 feature and linear regression tests, 25 SVR tests and 46 final evaluation tests.

The nine rows in comparison_metrics.csv were compared with the earlier results stored in the uploaded project archive. Example counts, MAE and RMSE match exactly at the recorded numeric precision. The selected SVR settings remain C=10, epsilon=0.1 and gamma=0.01. The split counts remain 14,772, 3,388 and 2,260. The preparation settings match the supplied evaluation protocol.

The source fingerprints differ from the original protocol after the documented comment and docstring revision. Both protocols should be retained as records of their respective executions. No fingerprint was edited to make an old record appear to describe new source files.

This is reproduction of a previously completed experiment. The test results were already known; this is not a new independent test and must not be used to justify retuning.

The records do not establish reproduction on different hardware, independent accuracy of the sensors, or agreement of every individual prediction. A fresh timing report and the exact reproduction Git commit were not supplied. The report uses the earlier archived runtime measurements and labels them accordingly.

Keep this folder beside the original results, not in place of them. The original outputs and source files are not modified by adding this evidence.
