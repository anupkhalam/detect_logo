# Objective
The objective is to Identify logo of Allianz in an image.

# Architecture
FasterRCNN

# Installation
1.  Create a virtual environment
2. Activate the virtual environment
3. Copy the code
4. Change directory to be inside logo_detection

`cd logo_detection`
5.  Install required packages

`pip3 --no-cache-dir install -r requirements.txt`

# Training
1. Trained model is available. You may skip to inferencing to save time.
2. Run the following to generate training data

`python3 data_generator.py`
3. Run the following to start the training

`python3 train.py`

Note: There shall be atleast 7GB CUDA free memory. If not, reduce the batch-size accordingly.

# Inference
Run the following steps for inferencing.
1. Clear all CUDA memory
2. Run `uvicorn main:app --reload`
3. Open Postman or any API Platform
4. Make a POST request to http://127.0.0.1:8000/process/ using following JSON format

`{
    "source_path": "xx/xx/xx/yy.jpg")
}`
where "xx/xx/xx/yy.jpg" is the absolute file path

Note:  
- Error handling is at its minimum. Hence provide a legitimate file path.
- This method works only within the local system

# Result
The results will be saved in the folder of source image. One resulting file contains the bounding box and the other contains cropped logo.  
The code has been tested in an Ubuntu 20.04.4 LTS machine with GeForce RTX 2070.
