import streamlit as st
import pandas as pd
from io import StringIO
from manual_submission import get_preds
import matplotlib.pyplot as plt
import plotly.express as px

def visualize_probabilities(probabilities):
    labels = ['Direct Acceptance', 'Direct Rejection', 'Scientific Interview']
    percentages = [round(prob * 100, 2) for prob in probabilities]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, percentages, color=['red', 'green', 'blue'])
    ax.set_xlabel('Probability (%)')
    ax.set_ylabel('Classes')
    ax.set_title('Class Probabilities')

    for index, value in enumerate(percentages):
        ax.text(value, index, f'{value}%', ha='left', va='center')

    fig.patch.set_alpha(0.0)  # Set background color to be transparent
    st.pyplot(fig)


def set_page_layout():
    # Set page title and description
    # Set page title and description
    st.set_page_config(page_title="IE 492 Project", page_icon=":chart_with_upwards_trend:", layout="wide")

    # Centered header for the main project title
    st.markdown("""
           <h1 style="text-align: center;">Design of Classifiers for Graduate Admission</h1>
           <h2 style="text-align: center;">IE 492 Project</h2>
           <h3 style="text-align: center;">Ömer Coşkun</h3>
           <h3 style="text-align: center;">Önder Yılmaz</h3>
           <h3 style="text-align: center;">Emre Çağan Kanlı</h3>
           <h3 style="text-align: center;">İ. Kuban Altınel - Supervisor</h3>
       """, unsafe_allow_html=True)

    st.markdown("---")
    # Centered smaller header for the application process
    st.markdown("""
           <h2 style="text-align: center;">Application Process</h2>
       """, unsafe_allow_html=True)

    # Add a separator
    st.markdown("---")

    # Grid layout for text and image
    col1, col2 = st.columns([1, 1])

    with col1:
        text_objects = [
            {
                "text": [
                    "In this demo, you can test the classifiers we designed to predict the outcomes of MSc applications to Boğaziçi University Industrial Engineering programs."
                ],
                "sublists": [
                    [
                        "For both 1st and 2nd Stages, model can be tested by providing:",
                        "i. an excel file containing the records of students in the right format,",
                        "ii. manual input"
                    ]
                ],
                "font_size": "20px",
                "margin_top": "20px"
            },
            {
                "text": [],
                "sublists": [
                    [
                        "You can try different the models:",
                        "i. Neural Network (recommended)",
                        "ii. Decision Tree",
                        "iii. Support Vector Machine",
                        "iv. K-Nearest Neighbors"
                    ]
                ],
                "font_size": "20px",
                "margin_top": "20px"
            },
            {
                "text": [
                    "You can change the inputs to see how they affect the predictions."
                ],
                "sublists": [

                ],
                "font_size": "20px",
                "margin_top": "20px"
            }
        ]
        for text_obj in text_objects:
            for paragraph in text_obj["text"]:
                st.markdown(f'<li style="font-size: {text_obj["font_size"]}; margin-top: {text_obj["margin_top"]}">{paragraph}</li>', unsafe_allow_html=True)
            for sublist in text_obj["sublists"]:
                for index, item in enumerate(sublist):
                    if index == 0:
                        st.markdown(f'<li style="font-size: {text_obj["font_size"]}; margin-top: {text_obj["margin_top"]};">{item}</li>', unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div style="font-size: 17px; margin-top: {text_obj["margin_top"]}; padding-left: 80px;"><em>{item}</em></div>',
                            unsafe_allow_html=True)


    # Right column for image
    with col2:
        st.image("process.png", use_column_width=True, caption="Application Scheme")
    # Custom CSS for linear gradient background
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(5deg, #4186c6 0%, #094f94 20%, #094f94 80%, #4186c6 100%);
                height: 100vh;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("---")
    # Centered smaller header for the application process
    st.markdown("""
              <h2 style="text-align: center;">Testing The Model</h2>
          """, unsafe_allow_html=True)
    st.markdown("---")

def center_columns():
    # Centering the columns horizontally
    st.markdown("""
        <style>
            .center {
                display: flex;
                justify-content: center;
                width: 100%;
            }
            .slide-in {
                animation: slide-in 0.5s ease-out;
            }
            @keyframes slide-in {
                from {
                    opacity: 0;
                    transform: translateY(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            button[kind="primary"] {
                background-color: #4186c6;
                color: black;
                border-color: #4186c6;
            }
            button[kind="secondary"] {
                background-color: #dedede;
                color: black;
                border-color: #4186c6;
            }
             button[kind="primary"]:hover {
                background-color: #4186c6;
                color: black;
                border-color: #4186c6;
                }
            button[kind="secondary"]:hover {
                background-color: #4186c6;
                color: black;
                 border-color: #4186c6;
                }
            button[kind="primary"]:active,
                button[kind="secondary"]:active {
                    background-color: #4186c6 !important;
                    color: black !important;
                    border-color: #4186c6 !important;
                }
}

                 
                
        </style>
    """, unsafe_allow_html=True)
    # Create a container to center the columns
    st.markdown('<div class="center">', unsafe_allow_html=True)

def create_columns():
    # Define the columns
    grid = st.columns([4.5,1,1,4])

    with grid[1]:
        if st.session_state.selected_stage == 1:
            btn_color = "primary"
        else:
            btn_color = "secondary"
        if st.button("1st Stage", type=btn_color, key="1"):
            with st.empty():
                if st.session_state.selected_stage == 1:
                    st.session_state.selected_stage = None
                    st.session_state.show_file_upload = False
                else:
                    st.session_state.selected_stage = 1
                    st.session_state.show_file_upload = True
                st.session_state.preds = None
                st.session_state.preds2 = None
                st.rerun()

    # Second column
    with grid[2]:
        if st.session_state.selected_stage == 2:
            btn_color = "primary"
        else:
            btn_color = "secondary"
        if st.button("2nd Stage", type=btn_color, key="2"):
            with st.empty():
                if st.session_state.selected_stage == 2:
                    st.session_state.show_file_upload = False
                    st.session_state.selected_stage = None
                else:
                    st.session_state.selected_stage = 2
                    st.session_state.show_file_upload = True
                st.session_state.preds = None
                st.session_state.preds2 = None
                st.rerun()



def file_upload_section():
    # File uploader section
    st.header("File Upload")
    st.write("Please upload the file you want to process.")

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)
        # To read file as string:
        string_data = stringio.read()
        st.write(string_data)
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

def main():
    set_page_layout()
    center_columns()
    create_columns()

    if st.session_state.selected_stage:
        stage_text = "1st" if st.session_state.selected_stage == 1 else "2nd"
        st.markdown(f"""
            <h2 style="text-align: center;">Testing {stage_text} Stage</h2>
        """, unsafe_allow_html=True)


    if st.session_state.show_file_upload:
        # Two-column layout for input fields and file uploader
        upload_grid = st.columns([3, 1, 4])

        # Left column for input fields
        with upload_grid[0]:
            with st.container(border=True):

                # Title
                st.markdown("""
                              <h1 style="text-align: center;">Application Details</h1>
                          """, unsafe_allow_html=True)

                # Input fields
                with st.container():
                    st.markdown("<h3 style='font-size: 18px;'>Student Information</h2>", unsafe_allow_html=True)
                    is_boun = st.checkbox("Boğaziçi University Student")
                    is_ie = st.checkbox("Industrial Engineering Student")

                with st.container():
                    st.markdown("<h3 style='font-size: 18px;'>Application Information</h2>", unsafe_allow_html=True)
                    if st.session_state.selected_stage == 2:
                        interview_score = st.slider("Interview Score", 50, 100, 50, 1)
                    term = st.radio("Term", ["Fall", "Spring"])
                    gpa = st.slider("GPA", 0.0, 4.0, 3.0, 0.01)
                    ales = st.slider("ALES Score", 0, 100, 80)
                    uni_score = st.slider("University Score", 0, 560, 400)
                    n_of_applicants = st.number_input("Number of Applicants", min_value=1, step=1, value=60)

                # Button to submit inputs
                submit_button = st.button("Get Predictions")

            with st.container(border=True):
                if st.session_state.preds is not None and st.session_state.preds.any():
                    predictions = st.session_state.preds
                    # visualize_probabilities(predictions)
                    # Title
                    st.markdown("""
                                                                     <h1 style="text-align: center;">Results</h1>
                                                                 """, unsafe_allow_html=True)
                    # Convert probabilities to percentage strings for displaying when hovered
                    percentage_strings = [f"{round(float(pred) * 100, 2)}%" for pred in predictions]

                    st.write(
                        f"<h3 style='font-size: 24px; font-weight: bold;'>Direct Acceptance: {round(float(predictions[0]) * 100, 2)}%</h3>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f'<progress max="100" value="{round(float(predictions[0]) * 100, 2)}" '
                        f'style="width: 100%; height: 40px; background-color: red;" '
                        f'title="{percentage_strings[0]}">{percentage_strings[0]}</progress>',
                        unsafe_allow_html=True
                    )

                    st.write(
                        f"<h3 style='font-size: 24px; color: red;'>Direct Rejection: {round(float(predictions[1]) * 100, 2)}%</h3>",
                        unsafe_allow_html=True)

                    st.markdown(
                        f'<progress max="100" value="{round(float(predictions[1]) * 100, 2)}" '
                        f'style="width: 100%; height: 40px; background-color: red;" '
                        f'title="{percentage_strings[1]}">{percentage_strings[1]}</progress>',
                        unsafe_allow_html=True
                    )

                    st.write(f"<h3 style='font-size: 24px; color: yellow;'>Scientific Interview: {round(float(predictions[2]) * 100, 2)}%</h3>", unsafe_allow_html=True)

                    st.markdown(
                        f'<progress max="100" value="{round(float(predictions[2]) * 100, 2)}" '
                        f'style="width: 100%; height: 40px; background-color: red;" '
                        f'title="{percentage_strings[2]}">{percentage_strings[2]}</progress>',
                        unsafe_allow_html=True
                    )
                if st.session_state.preds2 is not None and st.session_state.preds2.any():
                    predictions = st.session_state.preds2
                    # visualize_probabilities(predictions)
                    # Title
                    st.markdown("""
                                                                     <h1 style="text-align: center;">Results</h1>
                                                                 """, unsafe_allow_html=True)
                    # Convert probabilities to percentage strings for displaying when hovered
                    percentage_strings = [f"{round(float(pred) * 100, 2)}%" for pred in predictions]

                    st.write(
                        f"<h3 style='font-size: 24px; font-weight: bold;'>Acceptance: {round(float(predictions[0]) * 100, 2)}%</h3>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f'<progress max="100" value="{round(float(predictions[0]) * 100, 2)}" '
                        f'style="width: 100%; height: 40px; background-color: red;" '
                        f'title="{percentage_strings[0]}">{percentage_strings[0]}</progress>',
                        unsafe_allow_html=True
                    )

                    st.write(
                        f"<h3 style='font-size: 24px; color: red;'>Rejection: {round(float(predictions[1]) * 100, 2)}%</h3>",
                        unsafe_allow_html=True)

                    st.markdown(
                        f'<progress max="100" value="{round(float(predictions[1]) * 100, 2)}" '
                        f'style="width: 100%; height: 40px; background-color: red;" '
                        f'title="{percentage_strings[1]}">{percentage_strings[1]}</progress>',
                        unsafe_allow_html=True
                    )

                # Display inputs after submission
                if submit_button:
                    if st.session_state.selected_stage == 1:
                        predictions = get_preds(is_boun, is_ie, term, gpa, ales, uni_score, n_of_applicants)
                        st.session_state.preds = predictions
                        st.session_state.preds2 = None
                    else:
                        predictions = get_preds(is_boun, is_ie, term, gpa, ales, uni_score, n_of_applicants, interview_score)
                        st.session_state.preds2 = predictions
                        st.session_state.preds = None
                    st.rerun()




        # Right column for file uploader
        with upload_grid[2]:
            with st.container():
                st.markdown('<div class="slide-in">', unsafe_allow_html=True)
                file_upload_section()
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    if "preds" not in st.session_state:
        st.session_state.preds = None
    if "preds2" not in st.session_state:
        st.session_state.preds2 = None
    if "show_file_upload" not in st.session_state:
        st.session_state.show_file_upload = False
    if "selected_stage" not in st.session_state:
        st.session_state.selected_stage = None
    main()
