import os

import re

import json

import tqdm

import torch

import time

import argparse

import transformers

from transformers import AutoTokenizer

import openai

from transformers import StoppingCriteria ,StoppingCriteriaList

import tiktoken

from typing import Any ,Dict ,List ,Optional ,Sequence ,Tuple

import sys

sys .path .append ("src")

from .utils import RetrievalSystem ,DocExtracter

from .template import *



from .config import config



openai .api_type =openai .api_type or os .getenv ("OPENAI_API_TYPE")or config .get ("api_type")

openai .api_version =openai .api_version or os .getenv ("OPENAI_API_VERSION")or config .get ("api_version")

openai .api_key =openai .api_key or os .getenv ('OPENAI_API_KEY')or config ["api_key"]



if openai .__version__ .startswith ("0"):

    openai .api_base =openai .api_base or os .getenv ("OPENAI_API_BASE")or config .get ("api_base")

    if openai .api_type =="azure":

        openai_client =lambda **x :openai .ChatCompletion .create (**{'engine'if k =='model'else k :v for k ,v in x .items ()})["choices"][0 ]["message"]["content"]

    else :

        openai_client =lambda **x :openai .ChatCompletion .create (**x )["choices"][0 ]["message"]["content"]

else :

    if openai .api_type =="azure":

        openai .azure_endpoint =openai .azure_endpoint or os .getenv ("OPENAI_ENDPOINT")or config .get ("azure_endpoint")

        openai_client =lambda **x :openai .AzureOpenAI (

        api_version =openai .api_version ,

        azure_endpoint =openai .azure_endpoint ,

        api_key =openai .api_key ,

        ).chat .completions .create (**x ).choices [0 ].message .content

    else :

        openai_client =lambda **x :openai .OpenAI (

        api_key =openai .api_key ,

        ).chat .completions .create (**x ).choices [0 ].message .content





def _normalize_space (text :Any )->str :

    return re .sub (r"\s+"," ",str (text or "")).strip ()





def _render_options_map (options :Optional [Dict [str ,str ]])->str :

    if not options :

        return ""

    lines :List [str ]=[]

    for key in sorted (options .keys ()):

        value =_normalize_space (options [key ])

        if value :

            lines .append (f"{key}. {value}")

    return "\n".join (lines )





def _build_retrieval_query (question :str ,options :Optional [Dict [str ,str ]],include_options :bool =True )->str :

    question =_normalize_space (question )

    if (not include_options )or (not options ):

        return question

    options_text =_render_options_map (options )

    if not options_text :

        return question

    return f"{question}\n\nOptions:\n{options_text}"





def _snippet_identity (snippet :Dict [str ,Any ])->str :

    title =_normalize_space (snippet .get ("title",""))

    content =_normalize_space (snippet .get ("content",""))

    return f"{title}\n{content}".lower ()





def _dedupe_snippets (

snippets :Sequence [Dict [str ,Any ]],

scores :Optional [Sequence [float ]]=None ,

)->Tuple [List [Dict [str ,Any ]],List [float ]]:

    deduped_snippets :List [Dict [str ,Any ]]=[]

    deduped_scores :List [float ]=[]

    seen =set ()

    score_list =list (scores or [])

    for idx ,snippet in enumerate (snippets or []):

        if not isinstance (snippet ,dict ):

            continue

        key =_snippet_identity (snippet )

        if (not key )or (key in seen ):

            continue

        seen .add (key )

        deduped_snippets .append (snippet )

        if idx <len (score_list ):

            try :

                deduped_scores .append (float (score_list [idx ]))

            except Exception :

                pass

    return deduped_snippets ,deduped_scores





def _select_prompt_snippets (

snippets :Sequence [Dict [str ,Any ]],

*,

prompt_snippet_limit :int =8 ,

prompt_char_budget :int =6000 ,

)->List [Dict [str ,Any ]]:

    selected :List [Dict [str ,Any ]]=[]

    total_chars =0

    limit =max (0 ,int (prompt_snippet_limit ))

    budget =max (0 ,int (prompt_char_budget ))

    for snippet in snippets or []:

        if limit and len (selected )>=limit :

            break

        title =_normalize_space (snippet .get ("title",""))

        content =_normalize_space (snippet .get ("content",""))

        if not title and not content :

            continue

        block_chars =len (title )+len (content )+32

        if budget and selected and (total_chars +block_chars )>budget :

            break

        selected .append (snippet )

        total_chars +=block_chars

    return selected





def _is_weak_retrieval (scores :Sequence [float ],retriever_name :str ,rrf_k :int )->bool :

    if not scores :

        return False

    try :

        top_score =float (scores [0 ])

    except Exception :

        return False

    if "RRF"in str (retriever_name or ""):

        single_retriever_top1 =1.0 /(float (rrf_k )+1.0 )

        return top_score <=(single_retriever_top1 *1.10 )

    return False





_TOKEN_RE =re .compile (r"[A-Za-z][A-Za-z0-9_-]{2,}")

_STOPWORDS ={

"about","after","again","against","among","also","because","before","being","between",

"both","could","does","doing","during","each","from","have","having","into","itself",

"just","more","most","other","over","same","such","than","that","their","there",

"these","they","this","those","through","under","very","what","when","where","which",

"while","with","would","your","patient","patients","following","question","choice",

"choices","option","options","best","most","least","true","false","except","incorrect",

"correct","regarding","management","diagnosis","treatment","cause","causes","associated",

}





def _keyword_tokens (text :str )->List [str ]:

    tokens :List [str ]=[]

    seen =set ()

    for token in _TOKEN_RE .findall (str (text or "").lower ()):

        if token in _STOPWORDS :

            continue

        if token in seen :

            continue

        seen .add (token )

        tokens .append (token )

    return tokens





def _rerank_snippets_lightweight (

snippets :Sequence [Dict [str ,Any ]],

scores :Sequence [float ],

*,

question :str ,

options :Optional [Dict [str ,str ]],

top_n :int =24 ,

preserve_top_n :int =2 ,

lexical_weight :float =0.85 ,

option_weight :float =0.35 ,

title_weight :float =0.20 ,

)->Tuple [List [Dict [str ,Any ]],List [float ],Dict [str ,Any ]]:

    snippet_list =list (snippets or [])

    score_list =[float (x )for x in list (scores or [])[:len (snippet_list )]]

    if (not snippet_list )or (top_n <=1 ):

        return snippet_list ,score_list ,{"enabled":False ,"kept_with_overlap":0 ,"reranked_candidates":0 }



    question_terms =set (_keyword_tokens (question ))

    option_terms =set (_keyword_tokens (_render_options_map (options or {})))

    if not question_terms and not option_terms :

        return snippet_list ,score_list ,{"enabled":False ,"kept_with_overlap":0 ,"reranked_candidates":0 }



    rerank_limit =max (1 ,min (int (top_n ),len (snippet_list )))

    combined_query_terms =question_terms |option_terms

    ranked_items =[]

    overlap_positive =0



    for rank_idx in range (rerank_limit ):

        snippet =snippet_list [rank_idx ]

        title_text =_normalize_space (snippet .get ("title",""))

        content_text =_normalize_space (snippet .get ("content",""))

        title_terms =set (_keyword_tokens (title_text ))

        content_terms =set (_keyword_tokens (content_text ))

        snippet_terms =title_terms |content_terms



        q_hits =len (question_terms &snippet_terms )

        o_hits =len (option_terms &snippet_terms )

        t_hits =len (combined_query_terms &title_terms )

        overlap_positive +=1 if (q_hits +o_hits +t_hits )>0 else 0



        q_norm =q_hits /max (1.0 ,min (8.0 ,float (len (question_terms )or 1 )))

        o_norm =o_hits /max (1.0 ,min (8.0 ,float (len (option_terms )or 1 )))

        t_norm =t_hits /max (1.0 ,min (5.0 ,float (len (combined_query_terms )or 1 )))

        base_rank =1.0 /float (rank_idx +1 )

        combined =base_rank +(lexical_weight *q_norm )+(option_weight *o_norm )+(title_weight *t_norm )



        ranked_items .append (

        {

        "rank_idx":rank_idx ,

        "combined":combined ,

        "q_hits":q_hits ,

        "o_hits":o_hits ,

        "t_hits":t_hits ,

        }

        )



    ranked_items .sort (

    key =lambda item :(

    item ["combined"],

    item ["q_hits"]+item ["o_hits"]+item ["t_hits"],

    -item ["rank_idx"],

    ),

    reverse =True ,

    )



    selected_indices :List [int ]=[]

    for item in ranked_items :

        idx =int (item ["rank_idx"])

        if idx not in selected_indices :

            selected_indices .append (idx )

    for idx in range (min (int (preserve_top_n ),rerank_limit )):

        if idx not in selected_indices :

            selected_indices .append (idx )



    selected_indices .extend (range (rerank_limit ,len (snippet_list )))



    reranked_snippets =[snippet_list [idx ]for idx in selected_indices ]

    reranked_scores =[score_list [idx ]for idx in selected_indices if idx <len (score_list )]

    return reranked_snippets ,reranked_scores ,{

    "enabled":True ,

    "kept_with_overlap":int (overlap_positive ),

    "reranked_candidates":int (rerank_limit ),

    "question_terms":int (len (question_terms )),

    "option_terms":int (len (option_terms )),

    }



class MedRAG :



    def __init__ (self ,llm_name ="OpenAI/gpt-3.5-turbo-16k",rag =True ,follow_up =False ,retriever_name ="MedCPT",

    corpus_name ="Textbooks",db_dir ="./corpus",cache_dir =None ,corpus_cache =False ,HNSW =False ,

    device =None ):

        self .llm_name =llm_name

        self .rag =rag

        self .retriever_name =retriever_name

        self .corpus_name =corpus_name

        self .db_dir =db_dir

        self .cache_dir =cache_dir

        self .docExt =None

        self .last_answer_meta :Dict [str ,Any ]={}





        if rag :

            self .retrieval_system =RetrievalSystem (self .retriever_name ,self .corpus_name ,self .db_dir ,

            cache =corpus_cache ,HNSW =HNSW )

        else :

            self .retrieval_system =None





        self .templates ={"cot_system":general_cot_system ,"cot_prompt":general_cot ,

        "medrag_system":general_medrag_system ,"medrag_prompt":general_medrag }



        self .max_length =8192

        self .context_length =7168





        if self .llm_name .split ('/')[0 ].lower ()=="openai":

            self .model =self .llm_name .split ('/')[-1 ]

            if "gpt-3.5"in self .model or "gpt-35"in self .model :

                self .max_length =16384

                self .context_length =15000

            elif "gpt-4"in self .model :

                self .max_length =32768

                self .context_length =30000

            self .tokenizer =tiktoken .get_encoding ("cl100k_base")



        elif "gemini"in self .llm_name .lower ():

            import google .generativeai as genai

            genai .configure (api_key =os .environ ['GOOGLE_API_KEY'])

            self .model =genai .GenerativeModel (

            model_name =self .llm_name .split ('/')[-1 ],

            generation_config ={

            "temperature":0 ,

            "max_output_tokens":2048 ,

            }

            )

            if "1.5"in self .llm_name .lower ():

                self .max_length =1048576

                self .context_length =1040384

            else :

                self .max_length =30720

                self .context_length =28672

            self .tokenizer =tiktoken .get_encoding ("cl100k_base")



        else :



            self .max_length =2048

            self .context_length =1024

            self .tokenizer =AutoTokenizer .from_pretrained (self .llm_name ,cache_dir =self .cache_dir )



            if "mixtral"in llm_name .lower ():

                self .tokenizer .chat_template =open ('./templates/mistral-instruct.jinja').read ().replace ('    ',

                '').replace (

                '\n','')

                self .max_length =32768

                self .context_length =30000

            elif "llama-2"in llm_name .lower ():

                self .max_length =4096

                self .context_length =3072

            elif "llama-3"in llm_name .lower ():

                self .max_length =8192

                self .context_length =7168

                if ".1"in llm_name or ".2"in llm_name :

                    self .max_length =131072

                    self .context_length =128000

            elif "meditron-70b"in llm_name .lower ():

                self .tokenizer .chat_template =open ('./templates/meditron.jinja').read ().replace ('    ','').replace (

                '\n','')

                self .max_length =4096

                self .context_length =3072

                self .templates ["cot_prompt"]=meditron_cot

                self .templates ["medrag_prompt"]=meditron_medrag

            elif "pmc_llama"in llm_name .lower ():

                self .tokenizer .chat_template =open ('./templates/pmc_llama.jinja').read ().replace ('    ','').replace (

                '\n','')

                self .max_length =2048

                self .context_length =1024





            pipeline_kwargs ={

            "model":self .llm_name ,



            "torch_dtype":torch .bfloat16 ,

            "model_kwargs":{"cache_dir":self .cache_dir },

            }





            if device is not None :



                target_device =0

                if isinstance (device ,str )and "cuda:"in device :

                    try :

                        target_device =int (device .split (":")[-1 ])

                    except ValueError :

                        target_device =0

                elif isinstance (device ,int ):

                    target_device =device



                pipeline_kwargs ["device"]=target_device

                print (f"[MedRAG] Force loading model on GPU {target_device}")

            else :

                pipeline_kwargs ["device_map"]="auto"



            self .model =transformers .pipeline (

            "text-generation",

            **pipeline_kwargs

            )



        self .follow_up =follow_up

        if self .rag and self .follow_up :

            self .answer =self .i_medrag_answer

            self .templates ["medrag_system"]=simple_medrag_system

            self .templates ["medrag_prompt"]=simple_medrag_prompt

            self .templates ["i_medrag_system"]=i_medrag_system

            self .templates ["follow_up_ask"]=follow_up_instruction_ask

            self .templates ["follow_up_answer"]=follow_up_instruction_answer

        else :

            self .answer =self .medrag_answer



    def custom_stop (self ,stop_str ,input_len =0 ):

        stopping_criteria =StoppingCriteriaList ([CustomStoppingCriteria (stop_str ,self .tokenizer ,input_len )])

        return stopping_criteria



    def generate (self ,messages ,**kwargs ):


        if "openai"in self .llm_name .lower ():

            ans =openai_client (

            model =self .model ,

            messages =messages ,

            temperature =0.0 ,

            **kwargs

            )

        elif "gemini"in self .llm_name .lower ():

            response =self .model .generate_content (messages [0 ]["content"]+'\n\n'+messages [1 ]["content"],**kwargs )

            ans =response .candidates [0 ].content .parts [0 ].text

        else :

            stopping_criteria =None



            prompt =self .tokenizer .apply_chat_template (messages ,tokenize =False ,add_generation_prompt =True )







            input_ids =self .tokenizer .encode (prompt ,add_special_tokens =False )

            input_len =len (input_ids )





            try :

                gen_tokens =int (kwargs .get ("max_new_tokens",512 )or 512 )

            except Exception :

                gen_tokens =512



            model_limit =self .max_length





            if input_len +gen_tokens >model_limit :

                print (

                f"[MedRAG Warning] Input length ({input_len}) + Gen ({gen_tokens}) > Limit ({model_limit}). Truncating input.")



                keep_len =model_limit -gen_tokens -50

                if keep_len >0 :



                    input_ids =input_ids [-keep_len :]



                    prompt =self .tokenizer .decode (input_ids ,skip_special_tokens =True )





            if "meditron"in self .llm_name .lower ():

                stopping_criteria =self .custom_stop (["###","User:","\n\n\n"],input_len =len (

                self .tokenizer .encode (prompt ,add_special_tokens =True )))







            generate_kwargs ={

            "do_sample":False ,

            "eos_token_id":self .tokenizer .eos_token_id ,

            "pad_token_id":self .tokenizer .eos_token_id ,

            "max_new_tokens":gen_tokens ,

            "truncation":True ,

            "stopping_criteria":stopping_criteria ,

            }



            if "llama-3"in self .llm_name .lower ():

                generate_kwargs ["eos_token_id"]=[self .tokenizer .eos_token_id ,

                self .tokenizer .convert_tokens_to_ids ("<|eot_id|>")]





            generate_kwargs .update (kwargs )





            response =self .model (prompt ,**generate_kwargs )





            ans =response [0 ]["generated_text"][len (prompt ):]



        return ans



    def medrag_answer (self ,question ,options =None ,k =32 ,rrf_k =100 ,save_dir =None ,snippets =None ,snippets_ids =None ,**kwargs ):




        answer_kwargs =dict (kwargs )

        include_options_in_query =bool (answer_kwargs .pop ("include_options_in_query",True ))

        fallback_to_cot_on_weak_retrieval =bool (answer_kwargs .pop ("fallback_to_cot_on_weak_retrieval",False ))

        prompt_snippet_limit =int (answer_kwargs .pop ("prompt_snippet_limit",8 )or 0 )

        prompt_char_budget =int (answer_kwargs .pop ("prompt_char_budget",6000 )or 0 )

        lightweight_rerank =bool (answer_kwargs .pop ("lightweight_rerank",True ))

        rerank_top_n =int (answer_kwargs .pop ("rerank_top_n",24 )or 0 )

        rerank_preserve_top_n =int (answer_kwargs .pop ("rerank_preserve_top_n",2 )or 0 )

        snippet_scores =answer_kwargs .pop ("snippet_scores",None )



        options_map =options if isinstance (options ,dict )else None

        options_text =_render_options_map (options_map )

        retrieval_query =_build_retrieval_query (question ,options_map ,include_options =include_options_in_query )





        if self .rag :

            if snippets is not None :

                retrieved_snippets =snippets [:k ]

                try :

                    scores =[float (x )for x in list (snippet_scores or [])[:len (retrieved_snippets )]]

                except Exception :

                    scores =[]

            elif snippets_ids is not None :

                if self .docExt is None :

                    self .docExt =DocExtracter (db_dir =self .db_dir ,cache =True ,corpus_name =self .corpus_name )

                retrieved_snippets =self .docExt .extract (snippets_ids [:k ])

                scores =[]

            else :

                assert self .retrieval_system is not None

                retrieved_snippets ,scores =self .retrieval_system .retrieve (retrieval_query ,k =k ,rrf_k =rrf_k )



            retrieved_snippets ,scores =_dedupe_snippets (retrieved_snippets ,scores )

            rerank_meta ={"enabled":False }

            if lightweight_rerank :

                retrieved_snippets ,scores ,rerank_meta =_rerank_snippets_lightweight (

                retrieved_snippets ,

                scores ,

                question =question ,

                options =options_map ,

                top_n =rerank_top_n ,

                preserve_top_n =rerank_preserve_top_n ,

                )

            prompt_snippets =_select_prompt_snippets (

            retrieved_snippets ,

            prompt_snippet_limit =prompt_snippet_limit ,

            prompt_char_budget =prompt_char_budget ,

            )

            weak_retrieval =(len (prompt_snippets )==0 )or _is_weak_retrieval (scores ,self .retriever_name ,rrf_k )

            use_cot_fallback =bool (fallback_to_cot_on_weak_retrieval and weak_retrieval )



            contexts =[

            "Document [{:d}] (Title: {:s}) {:s}".format (idx ,prompt_snippets [idx ]["title"],prompt_snippets [idx ]["content"])

            for idx in range (len (prompt_snippets ))

            ]

            if len (contexts )==0 :

                contexts =[""]

            if "openai"in self .llm_name .lower ():

                contexts =[self .tokenizer .decode (self .tokenizer .encode ("\n".join (contexts ))[:self .context_length ])]

            elif "gemini"in self .llm_name .lower ():

                contexts =[self .tokenizer .decode (self .tokenizer .encode ("\n".join (contexts ))[:self .context_length ])]

            else :

                contexts =[self .tokenizer .decode (self .tokenizer .encode ("\n".join (contexts ),add_special_tokens =False )[:self .context_length ])]

        else :

            retrieved_snippets =[]

            scores =[]

            contexts =[]

            prompt_snippets =[]

            weak_retrieval =False

            use_cot_fallback =False

            rerank_meta ={"enabled":False }



        self .last_answer_meta ={

        "retrieval_query":retrieval_query ,

        "retrieved_snippet_count":len (retrieved_snippets ),

        "prompt_snippet_count":len (prompt_snippets ),

        "weak_retrieval":bool (weak_retrieval ),

        "used_cot_fallback":bool (use_cot_fallback ),

        "top_score":(float (scores [0 ])if scores else None ),

        "rerank":rerank_meta ,

        }



        if save_dir is not None and not os .path .exists (save_dir ):

            os .makedirs (save_dir )





        answers =[]

        if (not self .rag )or use_cot_fallback :

            prompt_cot =self .templates ["cot_prompt"].render (question =question ,options =options_text )

            messages =[

            {"role":"system","content":self .templates ["cot_system"]},

            {"role":"user","content":prompt_cot }

            ]

            ans =self .generate (messages ,**answer_kwargs )

            answers .append (re .sub ("\s+"," ",ans ))

        else :

            for context in contexts :

                prompt_medrag =self .templates ["medrag_prompt"].render (context =context ,question =question ,options =options_text )

                messages =[

                {"role":"system","content":self .templates ["medrag_system"]},

                {"role":"user","content":prompt_medrag }

                ]

                ans =self .generate (messages ,**answer_kwargs )

                answers .append (re .sub ("\s+"," ",ans ))



        if save_dir is not None :

            with open (os .path .join (save_dir ,"snippets.json"),'w')as f :

                json .dump (retrieved_snippets ,f ,indent =4 )

            with open (os .path .join (save_dir ,"response.json"),'w')as f :

                json .dump (answers ,f ,indent =4 )



        return answers [0 ]if len (answers )==1 else answers ,retrieved_snippets ,scores



    def i_medrag_answer (self ,question ,options =None ,k =32 ,rrf_k =100 ,save_path =None ,n_rounds =4 ,n_queries =3 ,qa_cache_path =None ,**kwargs ):

        if options is not None :

            options ='\n'.join ([key +". "+options [key ]for key in sorted (options .keys ())])

        else :

            options =''

        QUESTION_PROMPT =f"Here is the question:\n{question}\n\n{options}"



        context =""

        qa_cache =[]

        if qa_cache_path is not None and os .path .exists (qa_cache_path ):

            qa_cache =eval (open (qa_cache_path ,'r').read ())[:n_rounds ]

            if len (qa_cache )>0 :

                context =qa_cache [-1 ]

            n_rounds =n_rounds -len (qa_cache )

        last_context =None





        max_iterations =n_rounds +3

        saved_messages =[{"role":"system","content":self .templates ["i_medrag_system"]}]



        for i in range (max_iterations ):

            if i <n_rounds :

                if context =="":

                    messages =[

                    {

                    "role":"system",

                    "content":self .templates ["i_medrag_system"],

                    },

                    {

                    "role":"user",

                    "content":f"{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",

                    },

                    ]

                else :

                    messages =[

                    {

                    "role":"system",

                    "content":self .templates ["i_medrag_system"],

                    },

                    {

                    "role":"user",

                    "content":f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",

                    },

                    ]

            elif context !=last_context :

                messages =[

                {

                "role":"system",

                "content":self .templates ["i_medrag_system"],

                },

                {

                "role":"user",

                "content":f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",

                },

                ]

            elif len (messages )==1 :

                messages =[

                {

                "role":"system",

                "content":self .templates ["i_medrag_system"],

                },

                {

                "role":"user",

                "content":f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",

                },

                ]

            saved_messages .append (messages [-1 ])

            if save_path :

                with open (save_path ,'w')as f :

                    json .dump ([p if type (p )==dict else p .model_dump ()for p in saved_messages ],f ,indent =4 )

            last_context =context

            last_content =self .generate (messages ,**kwargs )

            response_message ={"role":"assistant","content":last_content }

            saved_messages .append (response_message )

            if save_path :

                with open (save_path ,'w')as f :

                    json .dump ([p if type (p )==dict else p .model_dump ()for p in saved_messages ],f ,indent =4 )

            if i >=n_rounds and ("## Answer"in last_content or "answer is"in last_content .lower ()):

                messages .append (response_message )

                messages .append (

                {

                "role":"user",

                "content":"Output the answer in JSON: {'answer': your_answer (A/B/C/D)}"if options else "Output the answer in JSON: {'answer': your_answer}",

                }

                )

                saved_messages .append (messages [-1 ])

                answer_content =self .generate (messages ,**kwargs )

                answer_message ={"role":"assistant","content":answer_content }

                messages .append (answer_message )

                saved_messages .append (messages [-1 ])

                if save_path :

                    with open (save_path ,'w')as f :

                        json .dump ([p if type (p )==dict else p .model_dump ()for p in saved_messages ],f ,indent =4 )

                return messages [-1 ]["content"],messages

            elif "## Queries"in last_content :

                messages =messages [:-1 ]

                if last_content .split ("## Queries")[-1 ].strip ()=="":

                    print ("Empty queries. Continue with next iteration.")

                    continue

                try :

                    action_str =self .generate ([

                    {

                    "role":"user",

                    "content":f"Parse the following passage and extract the queries as a list: {last_content}.\n\nPresent the queries as they are. DO NOT merge or break down queries. Output the list of queries in JSON format: {{\"output\": [\"query 1\", ..., \"query N\"]}}",

                    }

                    ],**kwargs )

                    action_str =re .search (r"output\": (\[.*\])",action_str ,re .DOTALL ).group (1 )

                    action_list =[re .sub (r'^\d+\.\s*','',s .strip ())for s in eval (action_str )]

                except Exception as E :

                    print ("Error parsing action list. Continue with next iteration.")

                    error_class =E .__class__ .__name__

                    error =f"{error_class}: {str(E)}"

                    print (error )

                    if save_path :

                        with open (save_path +".error",'a')as f :

                            f .write (f"{error}\n")

                    continue

                for question in action_list :

                    if question .strip ()=="":

                        continue

                    try :

                        rag_result =self .medrag_answer (question ,k =k ,rrf_k =rrf_k ,**kwargs )[0 ]

                        context +=f"\n\nQuery: {question}\nAnswer: {rag_result}"

                        context =context .strip ()

                    except Exception as E :

                        error_class =E .__class__ .__name__

                        error =f"{error_class}: {str(E)}"

                        print (error )

                        if save_path :

                            with open (save_path +".error",'a')as f :

                                f .write (f"{error}\n")

                qa_cache .append (context )

                if qa_cache_path :

                    with open (qa_cache_path ,'w')as f :

                        json .dump (qa_cache ,f ,indent =4 )

            else :

                messages .append (response_message )

                print ("No queries or answer. Continue with next iteration.")

                continue

        return messages [-1 ]["content"],messages



class CustomStoppingCriteria (StoppingCriteria ):

    def __init__ (self ,stop_words ,tokenizer ,input_len =0 ):

        super ().__init__ ()

        self .tokenizer =tokenizer

        self .stops_words =stop_words

        self .input_len =input_len



    def __call__ (self ,input_ids :torch .LongTensor ,scores :torch .FloatTensor ):

        tokens =self .tokenizer .decode (input_ids [0 ][self .input_len :])

        return any (stop in tokens for stop in self .stops_words )

