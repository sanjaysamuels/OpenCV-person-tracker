'''
FIT4009 Advanced Topics in Intelligent Systems
Assignment 4: Intrusion detection system
By: Sanjay Sekar Samuel

To run the file type: python intrusion_detection_system.py --video 1_2_intruder_1.mp4
or "python intrusion_detection_system.py --video 1_2_intruder_2.mp4"
Link for 2nd video: https://1drv.ms/v/s!AnFXaSD_he3hiOlsP_IFVhEyXj-S1A
'''



'''
Here we import all the necessary files needed for the detection of video
'''
from imutils.video import VideoStream
import argparse
import imutils
import cv2

'''
To construct the user friendly argument parser
'''
Argument_parser = argparse.ArgumentParser()
Argument_parser.add_argument("-v", "--video")
Argument_parser.add_argument("-a", "--min-area", type=int, default=1500)   # To set the minimum area to be detected
args = vars(Argument_parser.parse_args())


Video_Stream = cv2.VideoCapture(args["video"])

# The first frame will be used as the reference frame for the rest of the frames
firstVideo_Frame = None

# The while loop is used to loop over all the Video_Frames in the video
'''
The loop grabs all the video frames and detects if there are any intruders or not
'''
while True:
	Video_Frame = Video_Stream.read()
	Video_Frame = Video_Frame if args.get("video", None) is None else Video_Frame[1]
	Alert_text = "No intruder detected"


	'''
	To show when there is no more video frames, that means the video is ended and the code breaks
	'''
	if Video_Frame is None:
		break

	'''
	Here we apply some preprocessing techniques of resizing, grayscale and bluring it for the video
	code from: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding
	'''
	Video_Frame = imutils.resize(Video_Frame, width=500)
	preprocessing = cv2.threshold(Video_Frame,127,255,cv2.THRESH_BINARY)
	preprocessing = cv2.cvtColor(Video_Frame, cv2.COLOR_BGR2GRAY)
	preprocessing = cv2.GaussianBlur(preprocessing, (21, 21), 0)

	'''
	Initialize the first frame if it is None
	'''
	if firstVideo_Frame is None:
		firstVideo_Frame = preprocessing
		continue

	'''
	Here we compute the difference between the first frame and the current frame (Background subsrtraction)
	'''
	Video_Frame_D = cv2.absdiff(firstVideo_Frame, preprocessing)  # (background - current frame)
	Threshold_motion = cv2.threshold(Video_Frame_D, 40, 255, cv2.THRESH_BINARY)[1]   # Set the threshold value if less than 50 ignore and keep it black

	'''
	Here we dilate the threshold image to fill in holes in the video, then find contours on the threshold old image
	'''
	Threshold_motion = cv2.dilate(Threshold_motion, None, iterations=4)
	contours = cv2.findContours(Threshold_motion.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if imutils.is_cv2() else contours[1]

	'''
	Here we loop over the contours of the video
	'''
	for i in contours:
		# Min area is the minimum area for the box, if the box is less than the min area, it ignores it
		if cv2.contourArea(i) < args["min_area"]:
			continue

		'''
		This part of the code will draw a rectangle box on the person detected in the room
		After that it updates the room to be occupired
		'''
		(x, y, w, h) = cv2.boundingRect(i)
		cv2.rectangle(Video_Frame, (x, y), (x + w, y + h), (500,500,500),3)
		Alert_text = "ALERT: INTRUDER DETECTED"

	'''
	Insert the text on the video frame for the user
	code taken from: https://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html
	'''
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(Video_Frame, "Intruder detector feed: {}".format(Alert_text), (20, 20), font, 0.5, (0, 0, 0), 2)

	'''
	Here we will display the video feed to the user showing the detection of intruder
	'''
	cv2.imshow("Intruder detection Feed", Video_Frame)
	key_press = cv2.waitKey(1) & 0xFF

	'''To set the exit key 'q', and break from the loop'''
	if key_press == ord("q"):
		break

'''
Clean and exit the video
'''
Video_Stream.stop() if args.get("video", None) is None else Video_Stream.release()
cv2.destroyAllWindows()
