# Age and Gender Detection

This project uses OpenCV and pre-trained deep learning models to detect faces, predict their age and gender, and display the results in real-time using a webcam.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Install the required Python packages:
    ```sh
    pip install opencv-python numpy
    ```

2. Download the pre-trained models:

    - Age detection model:
        - `age.prototxt`: [Download Link](https://github.com/spmallick/learnopencv/blob/master/AgeGender/AgeGender/age_deploy.prototxt)
        - `age.caffemodel`: [Download Link](https://github.com/spmallick/learnopencv/blob/master/AgeGender/AgeGender/age_net.caffemodel)

3. Place the downloaded model files in the same directory as your script.

## Usage

Run the script:
    ```sh
    python main.py
    ```

## Code Explanation

The script performs the following steps:

1. Initializes the webcam.
2. Loads the pre-trained age and gender detection models.
3. Uses OpenCV's Haar Cascade classifier to detect faces in the video feed.
4. For each detected face, predicts the age and gender.
5. Draws bounding boxes around the faces and displays the predicted age and gender.
6. Logs the predictions with timestamps to a file named `predictions.log`.

## Example

When you run the script, it will open a window showing the webcam feed with bounding boxes around detected faces and the predicted age and gender displayed above each box.

## License

This project is licensed under the MIT License.