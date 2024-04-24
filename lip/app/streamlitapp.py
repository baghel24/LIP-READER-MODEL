# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import tensorflow as tf 
from utils import load_data, num_to_char, char_to_num
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(
    page_title='LipNet Full Stack App',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Set the theme
st.markdown("""
    <style>
        .stApp {
            background-color: #011627;
            color: white;
        }
        [data-testid=stSidebar] {
        background-color: #C6D8FF;
    }
    [data-testid=stSidebar]{
            font-size: 50px;
            text-align: center;
        }
         .block-container {
                    padding-top: 2rem;
                    padding-bottom: 3rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
    </style>
""", unsafe_allow_html=True)

# Setup the sidebar
with st.sidebar: 
    # st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.sidebar.image("/Users/visha/Downloads/main image.png", use_column_width=True)
    st.title('LipReader')
    # Generating a list of options or videos 
    options = os.listdir(os.path.join('..', 'data', 's1'))
    selected_video = st.selectbox('Choose video', options)
    # st.info('This application is originally developed from the LipNet deep learning model.')

st.title(' :blue[_AI LIP READING MODEL_] :sunglasses:') 

# # Generating a list of options or videos 
# options = os.listdir(os.path.join('..', 'data', 's1'))
# selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 
    # Rendering the video 
    with col1: 
        st.info(' :blue[The video below displays the converted video in mp4 format]')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)

    with col2: 
        st.info(' :blue[This is all the machine learning model sees when making a prediction]')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400) 

        st.info(' :blue[This is the output of the machine learning model as tokens]')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info(' :blue[Decode the raw tokens into words]')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
