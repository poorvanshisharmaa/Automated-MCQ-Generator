import streamlit as st
import pandas as pd
from openai import RateLimitError, APIError

from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging


class MCQGeneratorApp:

    def __init__(self):
        st.set_page_config(
            page_title="MCQ Generator",
            layout="centered"
        )
        
        # ---------- CUSTOM CSS (MATCHING YOUR DESIGN) ----------
        st.markdown("""
        <style>
            /* ---------- APP BACKGROUND ---------- */
            .stApp {
                background-color: #ECEDE8;
            }

            /* ---------- REMOVE STREAMLIT TOP GAP ---------- */
            section.main > div {
                padding-top: 0rem;
            }

            /* ---------- FULL-WIDTH HEADER (SAFE) ---------- */
            .header {
                background-color: #628141;
                padding: 36px 60px;
                color: white;

                width: 100%;
                margin: 0;
                border-radius: 0;
            }

            .header h1 {
                margin: 0;
                font-size: 32px;
                font-weight: 700;
            }

            .header p {
                margin-top: 6px;
                font-size: 16px;
                opacity: 0.9;
            }

            /* ---------- CARD ---------- */
            .card {
                background-color: #FFFFFF;
                padding: 30px;
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.06);
                margin-bottom: 28px;
            }

            .section-title {
                font-size: 20px;
                font-weight: 700;
                color: #1B211A;
                margin-bottom: 18px;
            }

            /* ---------- INPUTS ---------- */
            label {
                color: #1B211A !important;
                font-weight: 600;
            }

            .stTextInput input,
            .stNumberInput input,
            .stSelectbox div[data-baseweb="select"] {
                background-color: #EBD5AB !important;
                border-radius: 10px !important;
                border: none !important;
                padding: 12px !important;
                font-size: 15px;
            }

            /* ---------- FILE UPLOADER ---------- */
            .stFileUploader {
                background-color: #F4F5F2;
                border-radius: 14px;
                padding: 24px;
                border: 1px dashed #CFCFCF;
            }

            /* ---------- CENTER BUTTON ---------- */
            .stButton {
                display: flex;
                justify-content: center;
            }

            .stButton > button {
                background-color: #8BAE66;
                color: #1B211A;
                border-radius: 14px;
                height: 52px;
                font-size: 17px;
                font-weight: 700;
                width: 280px;
                border: none;
                margin: 20px 0;
            }

            .stButton > button:hover {
                background-color: #628141;
                color: white;
            }
        </style>
        """, unsafe_allow_html=True)

        # ---------- HEADER ----------
        st.markdown("""
        <div class="header">
            <h1>📘 MCQ Generator</h1>
            <p>Create quizzes from your documents</p>
        </div>
        """, unsafe_allow_html=True)

    def run(self):
        try:
            # ---------- SETTINGS CARD ----------
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Upload File",
                type=["pdf", "txt"]
            )

            subject = st.text_input(
                "Subject",
                placeholder="e.g. Machine Learning"
            )

            tone = st.selectbox(
                "Difficulty",
                ["Simple", "Moderate", "Advanced"]
            )

            number = st.number_input(
                "Number of Questions",
                min_value=1,
                max_value=20,
                value=5
            )

            st.markdown("</div>", unsafe_allow_html=True)

            # ---------- GENERATE BUTTON ----------
            if st.button("Generate MCQs"):

                if uploaded_file is None:
                    st.warning("Please upload a file")
                    return

                with st.spinner("Reading document..."):
                    text = read_file(uploaded_file)

                response_json = """
                {
                    "1": {
                        "mcq": "",
                        "options": {
                            "A": "",
                            "B": "",
                            "C": "",
                            "D": ""
                        },
                        "correct": ""
                    }
                }
                """

                with st.spinner("Generating MCQs..."):
                    response = generate_evaluate_chain(
                        {
                            "text": text,
                            "number": number,
                            "subject": subject,
                            "tone": tone,
                            "response_json": response_json
                        }
                    )

                quiz = response.get("quiz")
                review = response.get("review")

                # ---------- OUTPUT ----------
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Generated MCQs</div>', unsafe_allow_html=True)
                st.code(quiz, language="json")
                st.markdown("</div>", unsafe_allow_html=True)

                table_data = get_table_data(quiz)
                if table_data:
                    df = pd.DataFrame(table_data)
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">MCQs Table</div>', unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">AI Review</div>', unsafe_allow_html=True)
                st.success(review)
                st.markdown("</div>", unsafe_allow_html=True)

        except RateLimitError:
            st.error("API quota exceeded. Please check your OpenAI usage.")
        except APIError:
            st.error("OpenAI API error. Please try again later.")
        except Exception as e:
            logging.error("Error occurred", exc_info=True)
            st.error("Something went wrong. Please check logs.")


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    app = MCQGeneratorApp()
    app.run()
