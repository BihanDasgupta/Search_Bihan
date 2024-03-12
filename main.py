# Import Libraries
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import openai

# Get OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load CSV file (corpus of customized information)
load_dotenv()

loader = CSVLoader(file_path="Bihan-Corpus.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
# Set documents to variable database
database = FAISS.from_documents(documents, embeddings)


def retrieve_info(query):
    """ Function to retrieve relevant information for given query using database """
    similar_response = database.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array


# Call to OpenAI LLM, gpt-3.5-turbo-1106 (change model as needed)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

# Prompt template feeding question (user's query) and relavant data (corpus)
temp = """
You are to answer any question the user inputs. You can answer general questions, but also questions that are specific to Bihan Dasgupta. You have extensive knowledge about Bihan from the data fed to you below. When referring to Bihan, please use pronouns she/her/hers.
Question: {question}

Data about Bihan: {relevant_data}
"""

# Apply prompt template to PromptTemplate function
prompt = PromptTemplate(
    input_variables=["question", "relevant_data"],
    template=temp
)

chain = LLMChain(llm=llm, prompt=prompt)


def generate_response(question):
    """ Function to generate response from relevant data """
    relevant_data = retrieve_info(question)
    response = chain.run(question=question, relevant_data=relevant_data)
    return response


def custom_info_box(message):
    """ Function for HTML Styling: Generates pink box desplaying message (LLM's response) """
    html_code = f"""
    <div style='background-color: pink; padding: 10px; border-radius: 5px;'>
        <span style='color: black;'>{message}</span>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)


def main():
    """ Main """
    # Set page title
    st.set_page_config(
        page_title="All About Bihan")

    # HTML Styling for page
    page_bg_img = """
    <style>
    /* Image Background */
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("https://i.postimg.cc/QdcN6fmv/watercolor-sugar-cotton-clouds-background-52683-80661.jpg");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: local;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    /* Attribution Message */
    .attribution-message {{
        position: fixed;
        bottom: 10px;
        left: 10px;
        font-size: 10px;
        color: white;
        font-family: Cursive;
    }}
    /* Search Bar */
    .stTextInput > div > div > input {
        background-color: pink;
        color: magenta; /* Set text color */
        border-color: magenta; /* Set border color */
        font-family: Cursive;
        font-size: 20px; /* Set font size */
    }
    /* Typing Bar */
    .stTextInput > div > div > div > input {
        background-color: white;
        color: magenta;
    }
    /* Placeholder Text in Search Bar */
    .stTextInput > div > div > input::placeholder {
        color: magenta; /* Set placeholder text color */
        font-size: 30px; /* Set placeholder font size */
    }
    /* "Searching..." Text */
    .searching-text {
        color: magenta;
        font-family: Cursive;
    }
    /* st.info Box */
    .stInfo {
        background-color: pink;
    }
    /* st.Button */
    .stButton {
            color: pink;
            margin-top: -85px; /* Adjust the top margin to move the button down */
            margin-left: -75px; /* Adjust the left margin to move the button right */
        }
    </style>
    """
    # Markdown entire page styling
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Add background image
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://i.postimg.cc/QdcN6fmv/watercolor-sugar-cotton-clouds-background-52683-80661.jpg");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """
    # Markdown background image styling
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Add search engine heading above search bar
    html_string = '<h1><center><p style="font-family: Cursive; font-size: 25px; color: Magenta;" id="heading">  🎀🪞🩰🦢🕯️˖𓍢ִ໋🌷͙֒✧˚.🎀༘⋆꧁ B I H A N ꧂˚˖𓍢ִ໋🌷͙֒✧˚.🎀༘⋆🕯️🦢🩰🪞🎀  </p></center></h1>'
    # Markdown header styling
    st.markdown(html_string, unsafe_allow_html=True)

    # Add default text input for search bar
    message = st.text_input("", "🎀 Who Is Bihan...🎀")
    # Style and markdown default text input
    st.markdown(
        """
        <style>
        div.stButton {
            font-size: 30px;
            color: white;
            background-color: pink:
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Add search button
    st.button('⌕')
    # Style and markdown search button
    st.markdown(
        """
        <style>
        div.stButton {
            color: pink;
            margin-top: -20px; /* Adjust the top margin to move the button down */
            margin-left: -10px; /* Adjust the left margin to move the button right */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Conditions for if message is entered
    if message:
        # Write text that LLM is searching for answer ("Searching...")
        st.write("<span class='searching-text'>ᶻ 𝗓 𐰁ᶻ 𝗓 𐰁S Searching... ᶻ 𝗓 𐰁ᶻ 𝗓 𐰁</span>", unsafe_allow_html=True)

        # Call function to generate LLM response from message, assign to variable result
        result = generate_response(message)

        # Assign result to variable RESULT, which adds extra decoration to message string
        RESULT = "\n「 🃜🃚🃖🃁🂭🂺🃜🃚🃖🃁🂭🂺🃜🃚🃖🃁🂭🂺🃜🃚🃖🃁「 ✦ 🎀↓↓↓↓🎀 ✦ 」🂭🂺🃜🃚🃖🃁🂭🂺🃜🃚🃖🃁🂭🂺🃜🃚🃖🃁🂭🂺🃜🃚🃖🃁\n\n\n" + str(result) + "\n\n🃜🃚🃖🃁🂭🂺🃜🃚🃖🃁🂭🂺🃜🃚🃖🃁🂭🂺🃜🃚🃖🃁🂭🂺🃜🃚🃖🃁🂭🂺🃜🃚🃖🃁🂭🂺🃜🃚🃖🃁🂭🂺🃜🃚🃖🃁🂭🂺🃜🃚🃖🃁🂭🂺🃜🃜🃚🃖🃁」\n"

        # Write "RESULT" before desplaying the resulting response
        st.write("<span class='searching-text'>🎀 RESULT 🎀</span>", unsafe_allow_html=True)
        # Display LLM's response in the custom info box
        custom_info_box(RESULT)

    # Add attribution message to credit background image's original artist (always give credit!)
    credit_string = """<div class="attribution-message">
        <a href="https://www.freepik.com/free-vector/watercolor-sugar-cotton-clouds-background_22378664.htm#query=pink%20pinterest%20wallpaper&position=41&from_view=search&track=ais&uuid=fb9dc042-f248-4686-b1f6-a7a023a19dcf">Image by pikisuperstar</a> on Freepik
    </div>"""
    # Markdown attribution message
    st.markdown(credit_string, unsafe_allow_html=True)


# Run main()
if __name__ == '__main__':
    main()
