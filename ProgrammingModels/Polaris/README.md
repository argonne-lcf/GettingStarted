# ProgrammingModels - Polaris

The examples in this directory demonstrate compiling and running examples for programming models supported on Polaris.

Users can access Polaris resource by first sshing into Polaris and entering their passcode.
```
ssh username@polaris.alcf.anl.gov
```
Users can build their applications on the Polaris login nodes. If a GPU is required to be present, then one can build and test applications on one of the Polaris compute nodes. This can conveniently be done by submitting a short interactive job.
```
qsub -I -l select=1,walltime=1:00:00 -q debug -l filesystems=home:grand:eagle -A <PROJECT>
```
[//]: # (Additional info on submitting jobs on Polaris is available here:)
