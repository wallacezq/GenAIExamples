import os

from embedding.vector_stores import db
import time
import torch

import torch
import streamlit as st
#from transformers import AutoTokenizer
#from transformers import LlamaTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoProcessor, set_seed

import argparse

from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
import threading
from utils import config_reader as reader
#HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
#from embedding.extract_vl_embedding import VLEmbeddingExtractor as VL
from embedding.generate_store_embeddings import setup_meanclip_model 
#from embedding.video_llama.common.config import Config
#from embedding.video_llama.common.registry import registry
#from embedding.video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
from decord import VideoReader, cpu
import numpy as np
from PIL import Image

SAVED_MODEL_PATH = "./save_quantized_model"
LVM_MODEL_PATH = "openbmb/MiniCPM-V-2_6"

instructions = [
    """ Identify the person [with specific features / seen at a specific location / performing a specific action] in the provided data based on the video content. 
    Describe in detail the relevant actions of the individuals mentioned in the question. 
    Provide full details of their actions being performed and roles. Focus on the individual and the actions being performed.
    Exclude information about their age and items on the shelf that are not directly observable. 
    Do not mention items on the shelf that are not  visible. \
    Exclude information about the background and surrounding details.
    Ensure all information is distinct, accurate, and directly observable. 
    Do not repeat actions of individuals and do not mention anything about other persons not visible in the video.
    Mention actions and roles once only.
    """,
    
    """Analyze the provided data to recognize and describe the activities performed by individuals.
    Specify the type of activity and any relevant contextual details, 
    Do not give repetitions, always give distinct and accurate information only.""",
    
    """Determine the interactions between individuals and items in the provided data. 
    Describe the nature of the interaction between individuals and the items involved. 
    Provide full details of their relevant actions and roles. Focus on the individuals and the action being performed by them.
    Exclude information about their age and items on the shelf that are not directly observable. 
    Exclude information about the background and surrounding details.
    Ensure all information is distinct, accurate, and directly observable. 
    Do not repeat actions of individuals and do not mention anything about other persons not visible in the video.
    Do not mention  items on the shelf that are not observable. \
    """,
    
    """Analyze the provided data to answer queries based on specific time intervals.
    Provide detailed information corresponding to the specified time frames,
    Do not give repetitions, always give distinct and accurate information only.""",
    
    """Identify individuals based on their appearance as described in the provided data.
     Provide details about their identity and actions,
     Do not give repetitions, always give distinct and accurate information only.""",
    
    """Answer questions related to events and activities that occurred on a specific day.
    Provide a detailed account of the events,
    Do not give repetitions, always give distinct and accurate information only."""
]

# Embeddings
HFembeddings = HuggingFaceEmbeddings(model_kwargs = {'device': 'cpu'})

hf_db = FAISS.from_texts(instructions, HFembeddings)

def get_context(query, hf_db=hf_db):
    context = hf_db.similarity_search(query)
    return [i.page_content for i in context]

if 'config' not in st.session_state.keys():
    st.session_state.config = reader.read_config('docs/config.yaml')

config = st.session_state.config
device = "cpu" #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = config['model_path']
video_dir = config['videos']
# Read MeanCLIP
if not os.path.exists(os.path.join(config['meta_output_dir'], "metadata.json")):
    from embedding.generate_store_embeddings import main
    vs = main()
st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

logo_path = 'intel-logo-0.png'
st.logo(logo_path)
st.title("Advanced Video Retrieval System")

title_alignment="""
<style>
h1 {
  text-align: center
}

video.stVideo {
    width: 200px;
    height: 500px;      
}
</style>
"""
st.markdown(title_alignment, unsafe_allow_html=True)

@st.cache_resource       
def load_models():
    print("loading in model")
    processor = None
    if not os.path.exists(SAVED_MODEL_PATH):
       model = AutoModel.from_pretrained(LVM_MODEL_PATH, 
                                      load_in_low_bit="sym_int4",
                                      optimize_model=True,
                                      trust_remote_code=True,
                                      use_cache=True,
                                      modules_to_not_convert=["vpm", "resampler"],
                                      model_hub='huggingface')

       tokenizer = AutoTokenizer.from_pretrained(LVM_MODEL_PATH,
                                              trust_remote_code=True)
    else:
        model = AutoModel.load_low_bit(SAVED_MODEL_PATH, 
                                       optimize_model=True,
                                       trust_remote_code=True,
                                       use_cache=True,
                                       modules_to_not_convert=["vpm", "resampler"])
        tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL_PATH,
                                                  trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(LVM_MODEL_PATH,
                                                trust_remote_code=True)                                                  
    model.eval()
    
    if not os.path.exists(SAVED_MODEL_PATH):
        processor = AutoProcessor.from_pretrained(LVM_MODEL_PATH,
                                                trust_remote_code=True)
        model.save_low_bit(SAVED_MODEL_PATH)
        tokenizer.save_pretrained(SAVED_MODEL_PATH)
        processor.save_pretrained(SAVED_MODEL_PATH)
        
    video_minicpm = model.half().to('xpu')
    
    return video_minicpm, tokenizer, processor

video_minicpm, tokenizer, minicpm_processor = load_models()

print("-"*30)
print("initializing model")
chat_messages = []

def chat_reset(chat_state, img_list):
    print("-"*30)
    print("resetting chatState")
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return chat_state, img_list
    
def load_video(video_path, start_time=0, duration=-1, n_frms=8, height=-1, width=-1, sampling="uniform", return_msg = False, augment = False):
    #decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_path, height=height, width=width)
    fps = vr.get_avg_fps()
    vlen = len(vr)
    start, end = int(fps*start_time), min(vlen, int(fps*(start_time+duration))) if duration != -1 else vlen
    dur = min(vlen/fps,duration) if duration != -1 else vlen/fps

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, (end-start)/n_frms).astype(int).tolist()
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(dur*fps // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(dur*fps // 2, dur*fps), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(indices).asnumpy()
    # print(type(temp_frms))
    # Perform Augmentation
    if augment:
        vra = VideoRandomAugment()
        temp_frms = vra(temp_frms)
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    # " " should be added in the start and end
    #msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    msg = f"The video is {dur:.2f} seconds long at {fps:.2f} FPS and {len(indices)} frames are sampled. "
    return frms, msg
    
def load_video_PIL(video_path, start_time=0, duration=-1, n_frms=8, height=-1, width=-1, sampling="uniform", return_msg = False, augment = False):
    #decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_path, height=height, width=width)
    fps = vr.get_avg_fps()
    vlen = len(vr)
    start, end = int(fps*start_time), min(vlen, int(fps*(start_time+duration))) if duration != -1 else vlen
    dur = min(vlen/fps,duration) if duration != -1 else vlen/fps

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, (end-start)/n_frms).astype(int).tolist()
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(dur*fps // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(dur*fps // 2, dur*fps), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    frames = vr.get_batch(indices).asnumpy()

    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))

    if not return_msg:
        return frames

    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    # " " should be added in the start and end
    #msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    msg = f"The video is {dur:.2f} seconds long at {fps:.2f} FPS and {len(indices)} frames are sampled. "
    return frames, msg      

class VideoLLM(LLM):
    model: Optional[object] = None
    tokenizer: Optional[object] = None
    processor: Optional[object] = None
    
    @torch.inference_mode()
    def _call(
            self, 
            video_path,
            text_input,
            messages, # dict object that hold chat history
            start_time,
            duration,
            #f_stream,  # Add streamer as an argument (Boolean)
            **kwargs
            
        ):
        
        params={}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution >  448*448
        
        for k,v in kwargs.items():
          params[k]=v

        if video_path or video_path != "":                
           video_frames, msg = load_video_PIL(
                video_path=video_path, start_time=start_time, duration=duration,
                n_frms=8,
                height=-1,
                width=-1,
                sampling ="uniform", return_msg = True
           )
           print(f"message: {msg}")
        else:
           video_frames=[]
           msg=""
        
        global chat_messages
        chat_messages = messages # initialize chat_message with old chat history
        
        chat_messages.append({'role': 'user', 'content': video_frames + [text_input]})
        
        if self.processor:
           processor = self.processor
           
        res = self.model.chat(
            image=None,
            msgs=chat_messages,
            tokenizer=self.tokenizer,
            **params
        )
        
        return res
        #chat = None 
        #chat.upload_video_without_audio(video_path, start_time, duration)
        #chat.ask(text_input)#, chat_state)
        #answer = chat.answer(chat_state, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9, repetition_penalty=1.0, length_penalty=1, temperature=0.1, max_length=2000, keep_conv_hist=True, streamer=streamer)
        #answer = chat.answer(max_new_tokens=150, num_beams=1, min_length=1, top_p=0.9, repetition_penalty=1.0, length_penalty=1, temperature=0.02, max_length=2000, keep_conv_hist=True, streamer=streamer)

    #def stream_res(self, video_path, text_input, chat, start_time, duration):
    #    #thread = threading.Thread(target=self._call, args=(video_path, text_input, chat, chat_state, img_list, streamer))  # Pass streamer to _call
    #    thread = threading.Thread(target=self._call, args=(video_path, text_input, chat, start_time, duration, streamer))  # Pass streamer to _call
    #    thread.start()
    #    
    #    for text in streamer:
    #        yield text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return model_path # {"name_of_model": model_path}

    @property
    def _llm_type(self) -> str:
        return "custom"
    
def get_top_doc(results, qcnt):
    if qcnt < len(results):
        print("video retrieval done")
        return results[qcnt]
    return None

def play_video(x, offset, duration):
    if x is not None:
        video_file = open(x, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes, start_time=int(offset))
        
        #video_bytes = extract_clip_with_opencv(x, offset, duration)
        #print("len-video_bytes:", len(video_bytes))
        #st.video(video_bytes, start_time=0)

if 'llm' not in st.session_state.keys():
    with st.spinner('Loading Models . . .'):
        time.sleep(1)
        if config['embeddings']['type'] == "frame":
            print("Loading CustomLLM . . .")
            st.session_state['llm'] = CustomLLM()
        elif config['embeddings']['type'] == "video":
            print("Loading VideoLLM . . .")
            st.session_state['llm'] = VideoLLM(model=video_minicpm, tokenizer=tokenizer, processor=minicpm_processor)
        else:
            print("ERROR: line 240")
        
if 'vs' not in st.session_state.keys():
    with st.spinner('Preparing RAG pipeline'):
        time.sleep(1)
        host = st.session_state.config['vector_db']['host']
        port = int(st.session_state.config['vector_db']['port'])
        selected_db = st.session_state.config['vector_db']['choice_of_db']
        try:
            st.session_state['vs'] = vs
        except:
            if config['embeddings']['type'] == "frame":
                st.session_state['vs'] = db.VS(host, port, selected_db)
            elif config['embeddings']['type'] == "video":
                import json
                meanclip_cfg_json = json.load(open(config['meanclip_cfg_path'], 'r'))
                meanclip_cfg = argparse.Namespace(**meanclip_cfg_json)
                model, _ = setup_meanclip_model(meanclip_cfg, device="cpu")
                st.session_state['vs'] = db.VideoVS(host, port, selected_db, model) 

        if st.session_state.vs.client == None:
            print ('Error while connecting to vector DBs')
        
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        
def clear_chat_history():
    global chat_messages
    chat_messages=[]
    st.session_state.example_video = 'Enter Text'
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    #chat.clear()
        
def RAG(prompt):
    
    with st.status("Querying database . . . ", expanded=True) as status:
        st.write('Retrieving top-3 clips') #1 text doc and 
        results = st.session_state.vs.MultiModalRetrieval(prompt, top_k = 3) #n_texts = 1, n_images = 3)
        status.update(label="Retrieved top matching clip!", state="complete", expanded=False)
    print("---___---")
    print (f'\tRAG prompt={prompt}')
    print("---___---")
      
    result = get_top_doc(results, st.session_state["qcnt"])
    if result == None:
        return None
    try:
        top_doc, score = result
    except:
        top_doc = result
    print('TOP DOC = ', top_doc.metadata['video'])
    print("PLAYBACK OFFSET = ", top_doc.metadata['timestamp'])
    
    return top_doc

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if 'prevprompt' not in st.session_state.keys():
    st.session_state['prevprompt'] = ''
    print("Setting prevprompt to None")
if 'prompt' not in st.session_state.keys():
    st.session_state['prompt'] = ''
if 'qcnt' not in st.session_state.keys():
    st.session_state['qcnt'] = 0

def handle_message():
    print("-"*30)
    print("starting message handling")
    global chat_messages
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        # Handle user messages here
        with st.chat_message("assistant"):
            placeholder = st.empty()
            start = time.time()
            prompt = st.session_state['prompt']
            
            if prompt == 'Find similar videos':
                prompt = st.session_state['prevprompt']
                st.session_state['qcnt'] += 1
            else:
                st.session_state['qcnt'] = 0
                st.session_state['prevprompt'] = prompt
            top_doc = RAG(prompt)
            if top_doc == None:
                full_response = f"No more relevant videos found. Select a different query. \n\n"
                placeholder.markdown(full_response)
                end = time.time()
            else:
                video_name, playback_offset, video_path = top_doc.metadata['video'], int(top_doc.metadata['timestamp']), top_doc.metadata['video_path']
                with col2:
                    play_video(video_path, playback_offset, config['clip_duration'])

                full_response = ''
                full_response = f"Top retrieved clip is **{os.path.basename(video_name)}** at timestamp {playback_offset} -> {playback_offset//60:02d}:{playback_offset%60:02d} \n\n"
                instruction = f"Instruction: {get_context(prompt)[0]}\nQuestion: {prompt}"
                #instruction = f"Instruction: Describe the video content according to the user's question only if it includes the answer for the user's query. Otherwise, generate exactly:\'No related videos found in the database.\' and stop generating.\n User's question: {prompt}"
                #for new_text in st.session_state.llm.stream_res(formatted_prompt):
                generator = st.session_state.llm._call( video_path, 
                                                        instruction,
                                                        chat_messages,
                                                        playback_offset,
                                                        config['clip_duration'],
                                                        max_new_tokens=256,
                                                        sampling=True,
                                                        stream=True,
                                                        system_prompt=config['system_prompt']
                                                      )
                                                            
                if torch.xpu.is_available():
                  torch.xpu.synchronize()
                                                                              
                for new_text in generator:
                    full_response += new_text
                    placeholder.markdown(full_response)

                end = time.time()
                full_response += f'\n\n🚀 Generated in {(end - start):.4f} seconds.'
                #chat_state, img_list = chat_reset(chat_state, img_list)
                #chat.clear()
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        print("-"*30)
        st.session_state.messages.append(message)
      
def display_messages():
    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

col1, col2 = st.columns([2, 1])

with col1:
    st.selectbox(
        'Example Prompts',
        (
            'Enter Text', 
            'Find similar videos', 
            'Man wearing glasses', 
            'People reading item description',
            'Man holding red shopping basket',
            'Was there any person wearing a blue shirt seen today?',
            'Was there any person wearing a blue shirt seen in the last 6 hours?',
            'Was there any person wearing a blue shirt seen last Sunday?',
            'Was a person wearing glasses seen in the last 30 minutes?',
            'Was a person wearing glasses seen in the last 72 hours?',
        ),
        key='example_video'
    )

    st.write('You selected:', st.session_state.example_video)

if st.session_state.example_video == 'Enter Text':
    if prompt := st.chat_input(disabled=False):
        st.session_state['prompt'] = prompt
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        if prompt == 'Find similar videos':            
            st.session_state.messages.append({"role": "assistant", "content": "Not supported"})
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
else:
    prompt = st.session_state.example_video
    st.session_state['prompt'] = prompt
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.chat_input(disabled=True)
    if prompt == 'Find similar videos':
        st.session_state.messages.append({"role": "user", "content": prompt+': '+st.session_state['prevprompt']})
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

with col1:
    display_messages()
    handle_message()
