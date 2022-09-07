import cv2 
import streamlit as st
import face_recognition
import pandas as pd
import numpy as np
import time
import os
from PIL import Image
from streamlit_option_menu import option_menu
from datetime import datetime

from csv import DictWriter

st.set_page_config(
    page_title="MOD-AI: Video Analytics", 
    page_icon=Image.open("assets/logo.png")
)

DEFAULT_URL = "rtsp://admin:Things22@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"#"rtsp://"+IPAddr+":554/h264"
COLOR_DARK = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLOR_KNOW = (91, 166, 38)#(255, 0, 255)
COLOR_UNKNOW = (15, 0, 207)#(91, 166, 38)
PATH_DATA = 'data/DB.csv'
COLS_INFO = ['name', 'description', 'linetoken']
COLS_ENCODE = [f'v{i}' for i in range(128)]
DISPLAY_LOCATION='MOD'
COLS_DETECT_INFO = ['name', 'confidence', 'image', 'location', 'year', 'month', 'date', 'time', 'timestamp']

with st.sidebar:
    st.sidebar.image('assets/logo.png')
    st.sidebar.markdown("<center>MOD-AI: Video Analytics</center>", unsafe_allow_html=True) 
    chooseMenu = option_menu("Main Menu", ["Faces Recognition", "Faces Knowns Detected","Faces UnKnows Detected", "Faces Configuration"],
                         icons=['camera fill', 'kanban', 'book', 'gear'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    ) 
@st.cache(allow_output_mutation=True)
def init_data(data_path=PATH_DATA):
    if os.path.isfile(data_path):
        return pd.read_csv(data_path)
    else:
        return pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)

def init_detected_data(data_path):
    if os.path.isfile(data_path):
        return pd.read_csv(data_path)
    else:
        return pd.DataFrame(columns=COLS_DETECT_INFO)

def byte_to_array(image_in_byte):
    return cv2.imdecode(
        np.frombuffer(image_in_byte.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

def load_known_data():
    DB = pd.read_csv(PATH_DATA)
    return (
        DB['name'].values, 
        DB[COLS_ENCODE].values
        )
def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))

def main():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    st.title("MOD-AI: Video Analytics") 
    
    if chooseMenu == "Faces Knowns Detected" :  
        st.markdown('Face Knows Detected')
        TODAY_CODE = datetime.now().strftime('%Y-%m-%d')
        PATH_DETECT_DATA = 'detected/detected_data/'+str(TODAY_CODE)+'-knowns.csv'
        DB = init_detected_data(PATH_DETECT_DATA) 
        dataframe = DB[COLS_DETECT_INFO]
        st.dataframe(
            dataframe.sort_values("timestamp").iloc[:500].set_index('name')
        )
    if chooseMenu == "Faces UnKnows Detected" :  
        st.markdown('Face UnKnows Detected')
        TODAY_CODE = datetime.now().strftime('%Y-%m-%d')
        PATH_DETECT_DATA = 'detected/detected_data/'+str(TODAY_CODE)+'-unknowns.csv'
        DB = init_detected_data(PATH_DETECT_DATA) 
        dataframe = DB[COLS_DETECT_INFO]
        st.dataframe(
            dataframe.sort_values("timestamp").iloc[:500].set_index('name')
        )

    if chooseMenu == "Faces Configuration" :
        image_byte = st.file_uploader(
            label="Select a picture contains faces:", type=['jpg', 'png']
        )
        # detect faces in the loaded image
        max_faces = 0 
        rois = []  # region of interests (arrays of face areas)
        if image_byte is not None:
            image_array = byte_to_array(image_byte)
            face_locations = face_recognition.face_locations(image_array)
            for idx, (top, right, bottom, left) in enumerate(face_locations):
                # save face region of interest to list
                rois.append(image_array[top:bottom, left:right].copy())

                # Draw a box around the face and lable it
                cv2.rectangle(image_array, (left, top),
                            (right, bottom), COLOR_DARK, 2)
                cv2.rectangle(
                    image_array, (left, bottom + 35),
                    (right, bottom), COLOR_DARK, cv2.FILLED
                )
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(
                    image_array, f"#{idx}", (left + 5, bottom + 25),
                    font, .55, COLOR_WHITE, 1
                )

            st.image(BGR_to_RGB(image_array), width=720)
            max_faces = len(face_locations)
        
        if max_faces > 0: 
            face_idx = st.selectbox("Select face#", range(max_faces))
            roi = rois[face_idx]
            st.image(BGR_to_RGB(roi), width=min(roi.shape[0], 300))

            DB = init_data()
            face_encodings = DB[COLS_ENCODE].values
            dataframe = DB[COLS_INFO]

            face_to_compare = face_recognition.face_encodings(roi)[0]
            dataframe['distance'] = face_recognition.face_distance(
                face_encodings, face_to_compare
            )
            dataframe['similarity'] = dataframe.distance.apply(
                lambda distance: f"{face_distance_to_conf(distance):0.2%}"
            )
            st.dataframe(
                dataframe.sort_values("distance").iloc[:5].set_index('name')
            )

            # add roi to known database
            if st.checkbox('Add it to knonwn faces'):
                line_token = ''
                face_name = st.text_input('Name:', '')
                face_des = st.text_input('Desciption:', '')
                if st.checkbox('Notify to LINE'):
                    line_token = st.text_input('LINE Token:', '')
                if st.button('Add New Faces'):
                    encoding = face_to_compare.tolist()
                    if face_name and face_des:
                        DB.loc[len(DB)] = [face_name, face_des, line_token] + encoding
                        DB.to_csv(PATH_DATA, index=False)  
                        face_name=""
                        face_des=""
                        line_token=""

    if chooseMenu == "Faces Recognition":  

        fshow01, fshow02 = st.columns(2)
        with fshow01:
            FRAME_WINDOW = st.image([])
        with fshow02:
            st.markdown("<center>**Results**</center>", unsafe_allow_html=True)
            fshow02_text = st.markdown("<h1><center>UNKNOWN</center></h1>", unsafe_allow_html=True)
            fshow03_text = st.markdown("<h2 style='margin:0; padding:0'><center>00.00%</center></h2>", unsafe_allow_html=True)


        kpi0, kpi1, kpi2, kpi3 = st.columns(4)
        with kpi0:
            st.markdown("**Detected Face**")
            DETECTED_WINDOW = st.image([]) 
        with kpi1:
            st.markdown("**FrameRate**")
            kpi1_text = st.markdown("0")
        with kpi2:
            st.markdown("**Detected Faces**")
            kpi2_text = st.markdown("0")
        with kpi3:
            st.markdown("**Image Width**")
            kpi3_text = st.markdown(f" 320 Pixels", unsafe_allow_html=True)
        
        st.subheader('Faces Recgonition')
        detection_confidence = st.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.75)
        process_resolution = st.slider('Processing Resolution', min_value = 100,max_value = 1024,value = 320) 
        rtsp_url = st.text_input('Input RTSP URL', value=DEFAULT_URL)
            
        if st.button("Start Procssing"): 
            vid = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            known_face_names, known_face_encodings = load_known_data() 
            while(True):  
                prevTime = time.time()
                ret, frame = vid.read() 
                if ret:
                    TODAY_CODE = datetime.now().strftime('%Y-%m-%d')
                    TODAY_Day = datetime.now().strftime('%d')
                    TODAY_Month = datetime.now().strftime('%m')
                    TODAY_Year = datetime.now().strftime('%Y')
                    width=process_resolution
                    (h, w) = frame.shape[:2] 
                    r = width / float(w)
                    dim = (width, int(h * r))
                    # face detection
                    small_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA) 
                    rgb_frame = small_frame[:, :, ::-1]
                    
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    face_count=0
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # Draw a box around the face
                        face_count+=1
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        name = known_face_names[best_match_index]
                        similarity = face_distance_to_conf(face_distances[best_match_index], 0.5)
                        #cv2.rectangle(small_frame, (left, top), (right, bottom), COLOR_DARK, 2)
                        box_color=COLOR_UNKNOW
                        if similarity > detection_confidence:
                            box_color=COLOR_KNOW
                        cv2.rectangle(small_frame, (left, top), (right, bottom), box_color, 1)
                        # Top Left  x,y
                        cv2.line(small_frame, (left, top), (left+10, top), box_color, 3)
                        cv2.line(small_frame, (left, top), (left, top+10), box_color, 3) 
                        # Top Right  x1,y
                        cv2.line(small_frame, (right, top), (right-10, top), box_color, 3)
                        cv2.line(small_frame, (right, top), (right, top+10), box_color, 3)
                        # Bottom Left  x,y1
                        cv2.line(small_frame, (left, bottom), (left+10, bottom), box_color, 3)
                        cv2.line(small_frame, (left, bottom), (left, bottom-10), box_color, 3)
                        # Bottom Right  x1,y1
                        cv2.line(small_frame, (right, bottom), (right-10, bottom), box_color, 3)
                        cv2.line(small_frame, (right, bottom), (right, bottom-10), box_color, 3)
                        cv2.rectangle( small_frame, (left-2, bottom + 10), (right+2, bottom-3), box_color, cv2.FILLED)
                        if similarity > detection_confidence:
                            cv2.putText(small_frame, f"{name}", (left + 3, bottom + 6), cv2.FONT_HERSHEY_PLAIN, .50, COLOR_WHITE, 1)
                        else:
                            name='UNKNOW'
                            cv2.putText(small_frame, f"UNKNOW", (left + 3, bottom + 6), cv2.FONT_HERSHEY_PLAIN, .50, COLOR_WHITE, 1) 

                        DISPLAY_IMAGE = cv2.cvtColor(small_frame[top:bottom, left:right], cv2.COLOR_BGR2RGB)
                        STORAGE_FOLDER = "detected/detected_faces/"+str(TODAY_CODE)
                        if not os.path.exists(STORAGE_FOLDER): 
                            os.mkdir(STORAGE_FOLDER)
                            os.mkdir(STORAGE_FOLDER+'/knowns')
                            os.mkdir(STORAGE_FOLDER+'/unknows')
                        if name=='UNKNOW': 
                            FILE_NAME = STORAGE_FOLDER+"/unknows/"+str(time.time())+".jpg"
                            cv2.imwrite(FILE_NAME, DISPLAY_IMAGE[:, :, ::-1])
                            PATH_DETECT_DATA = 'detected/detected_data/'+str(TODAY_CODE)+'-unknowns.csv'
                            fshow02_text.write(f"<h1><center style='color:red'> {name} </center></h1>", unsafe_allow_html=True)
                            fshow03_text.write(f"<h2 style='margin:0; padding:0'><center> {similarity:.2%} </center></h2>", unsafe_allow_html=True)
                        else:
                            FILE_NAME = STORAGE_FOLDER+"/knowns/"+str(time.time())+".jpg"
                            cv2.imwrite(FILE_NAME, DISPLAY_IMAGE[:, :, ::-1])
                            PATH_DETECT_DATA = 'detected/detected_data/'+str(TODAY_CODE)+'-knowns.csv'
                            fshow02_text.write(f"<h1><center style='color:green'> {name} </center></h1>", unsafe_allow_html=True)
                            fshow03_text.write(f"<h2 style='margin:0; padding:0'><center> {similarity:.2%} </center></h2>", unsafe_allow_html=True)

                        DETECTED_WINDOW.image(DISPLAY_IMAGE) 
                        dict={'name':str(name), 'confidence': str(similarity), 'image':str(FILE_NAME), 'location':DISPLAY_LOCATION, 'year':str(TODAY_Year), 'month':str(TODAY_Month), 'date':str(TODAY_Day), 'time': str(datetime.now().strftime('%H:%M:%S')), 'timestamp':str(time.time())}
                        
                        if not os.path.isfile(PATH_DETECT_DATA): 
                            with open(PATH_DETECT_DATA, 'a', encoding='UTF8', newline='') as f_object:
                                dictwriter_object = DictWriter(f_object, fieldnames=COLS_DETECT_INFO)
                                dictwriter_object.writeheader()
                                dictwriter_object.writerow(dict)
                                f_object.close()                        
                        else: 
                            with open(PATH_DETECT_DATA, 'a', encoding='UTF8', newline='') as f_object:
                                dictwriter_object = DictWriter(f_object, fieldnames=COLS_DETECT_INFO)
                                #dictwriter_object.writeheader()
                                dictwriter_object.writerow(dict)
                                f_object.close()
                    
                    fps = 1 / (time.time() - prevTime)
                    kpi1_text.write(f" {int(fps)} fps ", unsafe_allow_html=True)
                    kpi2_text.write(f" {face_count} ", unsafe_allow_html=True)
                    kpi3_text.write(f" {process_resolution} Pixels", unsafe_allow_html=True)
                    #st.markdown('---') 
                    FRAME_WINDOW.image(small_frame[:, :, ::-1]) 
                else:
                    vid = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    continue
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
