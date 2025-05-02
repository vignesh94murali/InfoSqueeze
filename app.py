import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain.schema import Document 
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.outputs import LLMResult
from requests.exceptions import HTTPError
import urllib.parse
from langchain_huggingface import HuggingFaceEndpoint


## sstreamlit APP
st.set_page_config(page_title="InfoSqueeze", page_icon="🦜")
st.title("InfoSqueeze")
st.subheader('Summarize URL')



## Get the HuggingFace API Key and url(YT or website)to be summarized
with st.sidebar:
    hf_api_key=st.text_input("HuggingFace API Token",value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

## Gemma Model USsing Groq API
# llm =ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, 
                          task="text-generation",
                          max_length=150, 
                          temperature=0.7, 
                          token=hf_api_key)

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can be a YT video url or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    try:
                        # Manual fetch
                        parsed_url = urllib.parse.urlparse(generic_url)
                        video_id = urllib.parse.parse_qs(parsed_url.query).get('v', [None])[0]
                        if not video_id:
                            raise ValueError("Invalid YouTube URL. Couldn't extract video ID.")
                        transcript = YouTubeTranscriptApi.get_transcript(video_id=video_id)
                        text = " ".join([entry['text'] for entry in transcript])
                        docs = [Document(page_content=text)]
                    except Exception as yt_error:
                        st.warning(f"Manual caption fetch failed: {yt_error}")
                        st.info("Trying fallback method using YoutubeLoader...")
                        try:
                            loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                            docs = loader.load()
                        except Exception as fallback_error:
                            st.error(f"Fallback method also failed: {fallback_error}")
                            docs = None
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs=loader.load()

                if not docs or not docs[0].page_content.strip():
                    st.error("Failed to load content from the provided URL. The video might not have captions or the website content could not be extracted.")

                else:
                    ## Chain For Summarization
                    chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                    result=chain.invoke(docs)
                    output_summary = result['output_text']

                st.success(output_summary)
        except HTTPError as http_err:
            st.error(f"HTTP error occurred: {http_err}")
        except Exception as e:
            st.error(f"Exception occurred: {e}")
                    