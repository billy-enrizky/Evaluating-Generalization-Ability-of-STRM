# Preprocessing HMDB51

## Introduction
HMDB51 is a large collection of realistic videos from various sources, including movies and web videos. Introduced by Hildegard Kuehne et al. in "HMDB: A large video database for human motion recognition," the dataset is composed of 6,766 video clips from 51 action categories (such as “jump”, “kiss”, and “laugh”), with each category containing at least 101 clips. The original evaluation scheme uses three different training/testing splits. In each split, each action class has 70 clips for training and 30 clips for testing. The average accuracy over these three splits is used to measure the final performance.

## Steps to Preprocess HMDB51

1. **Download the dataset:**
   Download the dataset from [here](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads). The file will be named `hmdb51_org.rar`.

2. **Install `unrar`:**
   ```bash
   sudo apt-get install unrar
   ```

3. **Extract the dataset:**
   ```bash
   mkdir hmdb51_dataset
   unrar x hmdb51_org.rar hmdb51_dataset
   ```
   The folder `hmdb51_dataset` will contain .rar files for each action category in HMDB51.

4. **Unrar all action files:**
   ```bash
   cd hmdb51_dataset
   for rar in **.rar; do unrar x $rar; done
   ```

5. **Remove the .rar files:**
   ```bash
   for rar in **.rar; rm $rar; done
   ```

6. **Run the `extract_hmdb.py` script:**
   This script will create the train-test split and extract frames for each video.
   ```bash
   python extract_hmdb.py
   ```
   It will create a folder called `hmdb_256x256q5`, where 256 is the height and width of the output frames.

7. **Zip the preprocessed folder:**
   ```bash
   zip -r hmdb_256x256q5.zip hmdb_256x256q5
   ```

By following these steps, you will have a preprocessed version of the HMDB51 dataset, ready for training and testing machine learning models.