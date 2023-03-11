import time as t
import av
import numpy as np
from PIL import Image
from current import process, sudoku
from models import model_wrapper
from preprocessing import preprocess
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer,WebRtcMode


@st.cache_resource
def transform(frame: av.VideoFrame) -> av.VideoFrame:
    img1 = frame.to_ndarray(format="bgr24")
    frame_rate = 30
    my_model = model_wrapper.model_wrapper(None, False, None, "model.hdf5")

    prev = 0

    seen = dict()

    while True:
        time_elapsed = t.time() - prev

        #success, img = img1

        if time_elapsed > 1. / frame_rate:
            prev = t.time()

            img_result = img1.copy()
            img_corners = img1.copy()

            processed_img = preprocess.preprocess(img1)
            corners = process.find_contours(processed_img, img_corners)

            if corners:
                warped, matrix = process.warp_image(corners, img1)
                warped_processed = preprocess.preprocess(warped)

                vertical_lines, horizontal_lines = process.get_grid_lines(warped_processed)
                mask = process.create_grid_mask(vertical_lines, horizontal_lines)
                numbers = cv2.bitwise_and(warped_processed, mask)

                squares = process.split_into_squares(numbers)
                squares_processed = process.clean_squares(squares)

                squares_guesses = process.recognize_digits(squares_processed, my_model)

                # if it is impossible, continue
                if squares_guesses in seen and seen[squares_guesses] is False:
                    continue

                # if we already solved this puzzle, just fetch the solution
                if squares_guesses in seen:
                    process.draw_digits_on_warped(warped, seen[squares_guesses][0], squares_processed)
                    img_result = process.unwarp_image(warped, img_result, corners, seen[squares_guesses][1])

                else:
                    solved_puzzle, time = sudoku.solve_wrapper(squares_guesses)
                    if solved_puzzle is not None:
                        process.draw_digits_on_warped(warped, solved_puzzle, squares_processed)
                        img_result = process.unwarp_image(warped, img_result, corners, time)
                        seen[squares_guesses] = [solved_puzzle, time]

                    else:
                        seen[squares_guesses] = False
        return av.VideoFrame.from_ndarray(img_result,format='bgr24')

def load_image(image_file):
    img = Image.open(image_file) 
    return img      

def ImageProcess(uploaded_file):
        img1 = uploaded_file
        frame_rate = 30
        my_model = model_wrapper.model_wrapper(None, False, None, "model.hdf5")
        prev = 0
        seen = dict()
        while True:
            time_elapsed = t.time() - prev
            #success, img = img1
            if time_elapsed > 1. / frame_rate:
                prev = t.time()

                img_result = img1.copy()
                img_corners = img1.copy()

                processed_img = preprocess.preprocess(img1)
                corners = process.find_contours(processed_img, img_corners)

                if corners:
                    warped, matrix = process.warp_image(corners, img1)
                    warped_processed = preprocess.preprocess(warped)

                    vertical_lines, horizontal_lines = process.get_grid_lines(warped_processed)
                    mask = process.create_grid_mask(vertical_lines, horizontal_lines)
                    numbers = cv2.bitwise_and(warped_processed, mask)

                    squares = process.split_into_squares(numbers)
                    squares_processed = process.clean_squares(squares)

                    squares_guesses = process.recognize_digits(squares_processed, my_model)

                    # if it is impossible, continue
                    if squares_guesses in seen and seen[squares_guesses] is False:
                        continue

                    # if we already solved this puzzle, just fetch the solution
                    if squares_guesses in seen:
                        process.draw_digits_on_warped(warped, seen[squares_guesses][0], squares_processed)
                        img_result = process.unwarp_image(warped, img_result, corners, seen[squares_guesses][1])

                    else:
                        solved_puzzle, time = sudoku.solve_wrapper(squares_guesses)
                        if solved_puzzle is not None:
                            process.draw_digits_on_warped(warped, solved_puzzle, squares_processed)
                            img_result = process.unwarp_image(warped, img_result, corners, time)
                            seen[squares_guesses] = [solved_puzzle, time]

                        else:
                            seen[squares_guesses] = False
                return img_result

def main():
    # Face Analysis Application #
    st.title("Real Time Suduko solver Application")
    activiteis = ["Home", "Webcam Suduko Detection","Image Suduko Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Deepak Gond    
            Email : gonddeepak786@gmail.com  """)
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Suduko Solver application using OpenCV, Custom OCR model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.
                 1. Real time Suduko Detection using web cam feed.
                 2. Real time Suduko Solver On the image itself.
                 """)
    elif choice == "Webcam Suduko Detection":
        st.header("Webcam Live Feed")
        st.markdown("Click on start to use webcam and detect your Suduko")
        webrtc_streamer(key="example",video_frame_callback = transform,media_stream_constraints={"video": True, "audio": False},
        async_processing=True, mode=WebRtcMode.SENDRECV)

    elif choice == "Image Suduko Detection":
        st.header("Image Upload Detection")
        st.markdown("Click on here and Select the image and detect your Suduko")
        uploaded_file = st.file_uploader("Choose a file",type=["jpg","png","jpeg"])
        if uploaded_file is not None:
            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8),1)
            img1 = load_image(uploaded_file)
            st.write(f'<p style="font-size:24px; color:red;text-align:center;"><b>{"Original Image"}</b></p>', unsafe_allow_html=True)
            st.image(img1,use_column_width=True,output_format="auto")
            if st.button("Process Image"):                
                processed_image1 = ImageProcess(img)
                st.write(f'<p style="font-size:24px; color:red;text-align:center;"><b>{"Processed Image"}</b></p>', unsafe_allow_html=True)
                st.image(processed_image1, use_column_width=True)
    
    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Suduko Solver application using OpenCV, Custom Trained OCR model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)
        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Deepak Gond using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose.  </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""
        st.markdown(html_temp4, unsafe_allow_html=True)
    else:
        pass


if __name__ == "__main__":
    main()