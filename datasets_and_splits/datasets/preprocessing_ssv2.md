# Preprocessing Something Something V2 Dataset

## Dataset Background

The Something Something V2 Dataset was created with the help of more than 5,400 unique crowd actors. Developers have successfully created classification models based on the training set, achieving scores of up to 91 percent on the test set.

### Dataset Details:
- Video data is provided as a large TGZ archive, split into parts of 1 GB maximum.
- Total download size is 19.4 GB.
- The archive contains webm-files, using the VP9 codec, with a height of 240px.
- Files are numbered from 1 to 220847.
- The dataset includes object annotations in addition to video labels for the training and validation sets.
- Total annotations: 318,572 involving 30,408 unique objects.

## Dataset Instructions

### Step 1: Download and Extract the Dataset
1. Download all 20 zip files from the dataset source:  [Qualcomm AI Datasets - Something Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something).
2. Open a terminal on an Ubuntu system.
3. Run the following command to unzip all files:
   ```bash
   unzip 20bn-something-something-v2-\??.zip
   ```
4. Concatenate the files and extract the TGZ archive:
   ```bash
   cat 20bn-something-something-v2-?? | tar -xvzf -
   ```
5. Rename the extracted folder:
   ```bash
   mv 20bn-something-something-v2 ssv2_dataset
   ```

### Step 2: Install FFmpeg
1. Install FFmpeg by running the following command:
   ```bash
   sudo apt-get install ffmpeg
   ```

### Step 3: Extract Frames and Split Dataset
1. Run the `extract_ssv2.py` script to get the train-test split and extract frames for each video:
   ```bash
   python extract_ssv2.py
   ```
2. It will create a folder called `ssv2_256x256q5`, where 256 is the height and the width output.

### Step 4: Zip the Processed Dataset
1. Zip the folder that has been split into train and test sets and contains the extracted frames:
   ```bash
   zip -r ssv2_256x256q5.zip ssv2_256x256q5
   ```

Following these steps will prepare the Something Something V2 Dataset for use in your machine learning projects.
