# SUrface-wave Grader with ARtificial intelligence (SUGAR)

Developed by Chengping Chai (chaic@ornl.gov) in collaboration with Jonas Kintner, K. Michael Cleveland, Jingyi Luo, Monica Maceira, Charles J. Ammon

This package (SUrface-wave Grader with ARtificial intelligence or SUGAR) automatically assigns a quality score to surface-wave seismograms (SAC format) using a trained artificial neural network model (included). Specifically, the python script 01_apply_ann.py calculates probability scores for a list of SAC files. You may consider seismograms with probability scores larger than 0.5 as acceptable data. Note no scores will be given to seismograms that do not pass an initial check (e.g., insufficient number of data points). See the following paper for more details. 

If you use this package for your research, please consider cite:

* Chai, C., Kintner, J., Cleveland, K. M., Luo, J., Maceira, M., & Ammon, C. J. (2021). Automatic Waveform Quality Control for Surface Waves Using Machine Learning. https://doi.org/10.1002/essoar.10507941.3
* Chai, C., Luo, J., and Maceira, M. ccp137/SUGAR. Computer Software. https://github.com/ccp137/SUGAR. 06 Jan. 2022. Web. https://doi.org/10.11578/dc.20220106.9


## Install Required Packages

This package was developed for Python 3.7. The following python packages are required.

* pandas (1.3.2)
* numpy (1.21.1)
* scipy (1.7.0)
* h5py (2.10.0)
* keras (2.4.3)
* sklearn (0.24.2)
* obspy (1.2.2)

Installation these packages through Anaconda (or Miniconda) is highly recommended. You can find instructions on how to install Anaconda at https://docs.continuum.io/anaconda/install.

Once you have Anaconda installed, the required packages can be installed using these commands in the terminal. You may need to create an Anaconda environment to avoid conflicts with existing Python packages (see https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). If you get an error message that says a package can not be found, you can install the package using a similar command or search online for instructions.

```bash
conda install numpy=1.21.1
conda install scipy=1.7.0
conda install pandas=1.3.2
conda install keras=2.4.3
conda install scikit-learn=0.24.2
conda config --add channels conda-forge
conda install obspy=1.2.2
```

## SAC File Requirement

The following SAC headers are required. The time (nzyear etc.) in SAC headers has to be the origin time of the corresponding earthquake.

* stla
* stlo
* evla
* evlo
* nzyear
* nzjday
* nzhour
* nzmin
* nzsec
* nzmsec
* mag
* evdp

## Apply SUGAR

The trained model can be found as trained_model/ann_202110131500.hdf5. To apply the model to new data, you will need to provide a plain text file with a list of filenames in it. These filenames will be the waveforms (in SAC format) that you want to grade. See filelist.txt for an example. After installing all the required packages, you can use the machine leanring model with the following command. 

```bash
python 01_apply_ann.py
```

More details on the useage of 01_apply_ann.py.

```
usage: 01_apply_ann.py [-h] [--filelist_name FILELIST_NAME]
                       [--data_dir DATA_DIR] [--model_name MODEL_NAME]
                       [--output_dir OUTPUT_DIR] [--batch_size BATCH_SIZE]

Predict quality probability using a pre-trained model

optional arguments:
  -h, --help            show this help message and exit
  --filelist_name FILELIST_NAME
                        Path to a file contains filenames of SAC files that
                        need to be graded (default: filelist.txt)
  --data_dir DATA_DIR   Folder where the surface-wave seismograms are stored
                        (default: ./test_data/).
  --model_name MODEL_NAME
                        Machine learning model to be used (default:
                        ./trained_model/ann_202110131500.hdf5)
  --output_dir OUTPUT_DIR
                        Folder to save output (default: ./)
  --batch_size BATCH_SIZE
                        Batch size.
```

The output are two text files named as "predicted_prob.txt" and "files_not_graded.txt". The first file (predicted_prob.txt) contains two columns. The first column is the filename of the seismogram. The second column is the probability score (between 0 and 1) on whether a waveform should be accpeted. You may consider seismograms with probability scores larger than 0.5 as acceptable data. Waveforms that are not long enough in time will not have probability scores. The second file (files_not_graded.txt) stores the filenames that were not graded by the package. Note the output will be appended if the output files are already exists in the working directory.

## Change SAC Header

To change the SAC header "iqual", you can use the following command.

```bash
python 02_change_sac_header.py
```

--threshold can be used to set the threshold (>= 0.5) for accepting waveforms. If 0.5 was used, two labels (0 for rejected and 2 for accepted) are assigned to all the waveforms. If a value larger than 0.5 was used, three labels (0 for rejected; 1 for unclear; 2 for accepted) will be assigned. 

More details on the usage of 02_change_sac_header.py can be found by running "python 02_change_sac_header.py -h".

```
usage: 02_change_sac_header.py [-h] [--threshold THRESHOLD]
                               [--data_dir DATA_DIR] [--pred_file PRED_FILE]

Change SAC header iqual

optional arguments:
  -h, --help            show this help message and exit
  --threshold THRESHOLD
                        Probability threshold to compute quality grades
                        (default: 0.5)
  --data_dir DATA_DIR   Data directory (default: ./test_data/)
  --pred_file PRED_FILE
                        Path to the file that contains the predicted
                        probabilities (default: ./predicted_prob.txt)
```

## Contact

If you have questions or noticed anything unusual in the results, please contact me at chaic@ornl.gov.


## Acknowledgement

* Kipton Barros
* Singanallur Venkatakrishnan
* Derek Rose
* Chanel Deane

## Copyright

Copyright (C) 2022 Chengping Chai, Jingyi Luo, and Monica Maceira

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.
