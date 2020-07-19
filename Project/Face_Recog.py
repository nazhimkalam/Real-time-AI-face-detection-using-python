import cv2
from random import randrange

print("code running...")

# Load some pre trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect a face in a picture
# img = cv2.imread('rdj.jpg')

# Capturing video from web cam
webCam = cv2.VideoCapture(0)        # 0 means the default web cam else you can add a video link to it, eg:- 'video.mp4 location of the video in your computer

# Iterate over the video frames forever until we stop
while True:
    # read the current frame
    successful_frame_read, frame = webCam.read()

    # converting to greyscale (black and white) (we use grey scale so that the data set checks for brightness on the image/frame to assume that it is a face)
    greyScaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(greyScaled_img)  # result in this form [x, y, width, height]

    # Draw Rectangles around the face
    for (x, y, w, h) in face_coordinates:  # this loops through each face in the view
        # cv2.rectangle(img, (x,y ), (x+width, y+height), (colour of the square in BGR format), width of the stroke
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5)

    # Viewing the image
    cv2.imshow("Face_Detector",frame)   # this takes the window name in with the img
    key = cv2.waitKey(1)                                # keeps displaying the frames

    # Stops if the "Q" key is pressed
    if key==81 or key==113:             # ASCII value
        break


# release or empty the Video capture object
webCam.release()
