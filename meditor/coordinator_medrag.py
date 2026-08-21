




from __future__ import annotations



from typing import Any ,Dict ,List ,Optional ,Tuple



import json

import os

import re

from datetime import datetime



MedRAG =None

RetrievalSystem =None

collect_index_statuses =None

medrag_utils =None

from .import medqa

from .agents import ask_once ,make_agent



def _load_medrag_components ()->None :

    global MedRAG ,RetrievalSystem ,collect_index_statuses ,medrag_utils

    if MedRAG is not None :

        return

    try :

        from .medrag_src .medrag import MedRAG as _MedRAG

        from .medrag_src .utils import RetrievalSystem as _RetrievalSystem ,collect_index_statuses as _collect_index_statuses

        from .medrag_src import utils as _medrag_utils

    except ModuleNotFoundError as exc :

        raise RuntimeError (

        "Local MedRAG retrieval requires the optional retrieval dependencies. "

        "Install them with pip install -e '.[retrieval]', or configure a remote RAG endpoint."

        )from exc

    MedRAG =_MedRAG

    RetrievalSystem =_RetrievalSystem

    collect_index_statuses =_collect_index_statuses

    medrag_utils =_medrag_utils



try :

    from .custom_rag .bm25_index import BM25Searcher

except Exception :

    BM25Searcher =None

try :

    from .custom_rag .dense_index import DenseSearcher

except Exception :

    DenseSearcher =None

try :

    from .custom_rag .retriever import HybridRetriever

except Exception :

    HybridRetriever =None





_MODE_TO_CORPUS ={

"rag_textbooks":"Textbooks",

"rag_pubmed":"PubMed",

"rag_wikipedia":"Wikipedia",

"rag_mednosp":"MedNoSP",

"rag_medcorp":"MedCorp",

}



_DATASET_TO_MODE ={

"pubmedqa":"rag_pubmed",

"medqa":"rag_mednosp",

"medmcqa":"rag_textbooks",

"mmlu_med":"rag_textbooks",

}



_CUSTOM_CORPORA ={

"MedNoSP":["pubmed","textbooks","wikipedia"],

}





def _short (s :str ,n :int )->str :

    s =s or ""

    return (s [:n ]+"...")if len (s )>n else s





def _now ()->str :

    return datetime .now ().strftime ("%Y-%m-%d %H:%M:%S")





_BAD_LINES =(

re .compile (r"\bPreferred\b",re .I ),

re .compile (r"\[RESULT\]",re .I ),

re .compile (r"\bFinal\s*Answer\b",re .I ),

re .compile (r"\bAnswer\s*[:：]",re .I ),

re .compile (r"answer_choice",re .I ),

re .compile (r"\bwinner\b",re .I ),

)





def _sanitize_coord_output (text :str )->str :

    if not text :

        return ""

    lines :List [str ]=[]

    for ln in str (text ).splitlines ():

        if any (p .search (ln )for p in _BAD_LINES ):

            continue

        if re .fullmatch (r"\s*[A-Z]\s*",ln .strip ()):

            continue

        lines .append (ln .rstrip ())

    return "\n".join (lines ).strip ()





def _append_jsonl (path :str ,obj :Dict [str ,Any ])->None :

    if not path :

        return

    os .makedirs (os .path .dirname (path ),exist_ok =True )

    with open (path ,"a",encoding ="utf-8")as f :

        f .write (json .dumps (obj ,ensure_ascii =False )+"\n")





def _log_path_for_action (path :str ,meta :Optional [Dict [str ,Any ]])->str :

    if not path :

        return path

    action =str ((meta or {}).get ("action","")or "").strip ()

    if not action :

        return path

    stem ,ext =os .path .splitext (path )

    safe_action =re .sub (r"[^A-Za-z0-9_.-]+","_",action ).strip ("._")

    if not safe_action :

        return path

    return f"{stem}.{safe_action}{ext or '.jsonl'}"





def _norm_text (text :str )->str :

    return re .sub (r"\s+"," ",(text or "").strip ().lower ())





def _dedupe_pairs (pairs :List [Tuple [str ,float ]])->List [Tuple [str ,float ]]:

    out :List [Tuple [str ,float ]]=[]

    seen =set ()

    for text ,score in pairs :

        key =_norm_text (text )

        if (not key )or (key in seen ):

            continue

        seen .add (key )

        out .append ((text ,score ))

    return out





def _register_custom_corpora ()->None :

    _load_medrag_components ()

    for name ,members in _CUSTOM_CORPORA .items ():

        medrag_utils .corpus_names [name ]=list (members )





class MedRAGCoordinator :

    def __init__ (

    self ,

    mode :str ="rag_textbooks",

    llm_name :str ="",

    k :int =32 ,

    corpus_dir :str ="",

    device_id :int =2 ,

    log_path :str ="",

    retriever_name :str ="RRF-2",

    build_missing_indexes :bool =False ,

    preflight_indexes :bool =True ,

    custom_rag_textbooks_corpus_dir :str ="",

    custom_rag_textbooks_bm25_dir :str ="",

    custom_rag_textbooks_dense_dir :str ="",

    custom_rag_textbooks_dense_device :str ="",

    custom_rag_textbooks_fusion :str ="rrf",

    custom_rag_textbooks_query_mode :str ="question_plus_options",

    custom_rag_textbooks_top_k :int =32 ,

    custom_rag_textbooks_sparse_k :int =32 ,

    custom_rag_textbooks_dense_k :int =32 ,

    custom_rag_textbooks_rrf_k :int =60 ,

    summary_base_url :str ="",

    summary_model :str ="",

    summary_api_key :str ="EMPTY",

    remote_rag_base_url :str ="",

    remote_rag_model :str ="",

    remote_rag_api_key :str ="EMPTY",

    ):

        llm_name =str (llm_name or os .getenv ("MEDRAG_LLM_NAME","")).strip ()

        self .mode =mode

        self .k =int (k )

        self .log_path =log_path

        self .corpus_dir =str (corpus_dir or os .getenv ("MEDRAG_CORPUS_DIR","./corpus")).strip ()

        self .device_str =f"cuda:{int(device_id)}"

        self .requested_retriever_name =str (retriever_name or "MedCPT")

        self .build_missing_indexes =bool (build_missing_indexes )

        self .preflight_indexes =bool (preflight_indexes )

        self .custom_rag_textbooks_corpus_dir =str (custom_rag_textbooks_corpus_dir or "").strip ()

        self .custom_rag_textbooks_bm25_dir =str (custom_rag_textbooks_bm25_dir or "").strip ()

        self .custom_rag_textbooks_dense_dir =str (custom_rag_textbooks_dense_dir or "").strip ()

        self .custom_rag_textbooks_dense_device =str (custom_rag_textbooks_dense_device or "").strip ()

        self .custom_rag_textbooks_fusion =str (custom_rag_textbooks_fusion or "rrf").strip ()

        self .custom_rag_textbooks_query_mode =str (custom_rag_textbooks_query_mode or "question_plus_options").strip ()

        self .custom_rag_textbooks_top_k =int (custom_rag_textbooks_top_k )

        self .custom_rag_textbooks_sparse_k =int (custom_rag_textbooks_sparse_k )

        self .custom_rag_textbooks_dense_k =int (custom_rag_textbooks_dense_k )

        self .custom_rag_textbooks_rrf_k =int (custom_rag_textbooks_rrf_k )

        self .summary_base_url =str (summary_base_url or "").strip ()

        self .summary_model =str (summary_model or "").strip ()

        self .summary_api_key =str (summary_api_key or "EMPTY").strip ()

        self .remote_rag_base_url =str (remote_rag_base_url or "").strip ()

        self .remote_rag_model =str (remote_rag_model or "").strip ()

        self .remote_rag_api_key =str (remote_rag_api_key or "EMPTY").strip ()





        self .base_mode =mode if mode in _MODE_TO_CORPUS else "rag_textbooks"

        if self .base_mode =="rag_auto":

            self .base_mode ="rag_textbooks"

        self .base_corpus_name =_MODE_TO_CORPUS .get (self .base_mode ,"Textbooks")

        self .base_retriever_name =self .requested_retriever_name

        self ._retrieval_systems :Dict [str ,RetrievalSystem ]={}

        self ._retriever_names_by_mode :Dict [str ,str ]={}

        self ._preflight_rows :List [Dict [str ,Any ]]=[]

        self ._retrieval_errors :List [Dict [str ,Any ]]=[]

        self ._custom_textbooks_retriever =None

        self ._custom_textbooks_status :Dict [str ,Any ]={}

        self .summary_agent =(

        make_agent (

        "RAGCoordinator",

        self .summary_base_url ,

        self .summary_model ,

        temperature =0.0 ,

        system_message =(

        "You are a retrieval coordinator. Summarize evidence only. "

        "Do NOT answer the question. Do NOT output any final option letter."

        ),

        )

        if self .summary_base_url and self .summary_model

        else None

        )

        if self .summary_agent is not None :

            try :

                self .summary_agent .llm_config ["config_list"][0 ]["api_key"]=self .summary_api_key or "EMPTY"

            except Exception :

                pass

        self .remote_rag_agent =(

        make_agent (

        "RemoteRAGCoordinator",

        self .remote_rag_base_url ,

        self .remote_rag_model ,

        temperature =0.0 ,

        system_message =(

        "You are a retrieval-augmented medical evidence coordinator. "

        "Use your retrieval backend and summarize evidence only."

        ),

        )

        if self .remote_rag_base_url and self .remote_rag_model

        else None

        )

        if self .remote_rag_agent is not None :

            try :

                self .remote_rag_agent .llm_config ["config_list"][0 ]["api_key"]=self .remote_rag_api_key or "EMPTY"

            except Exception :

                pass

        if not self ._use_custom_textbooks_backend ()and self .remote_rag_agent is None :

            _register_custom_corpora ()



        if self .preflight_indexes and not self ._use_custom_textbooks_backend ()and self .remote_rag_agent is None :

            self ._preflight_required_indexes ()











        self .medrag =None if (

        self ._use_custom_textbooks_backend ()

        or self .summary_agent is not None

        or self .remote_rag_agent is not None

        )else self ._build_medrag (llm_name )



    def _required_modes (self )->List [str ]:

        if self .mode =="rag_auto":

            ordered :List [str ]=[]

            for item in list (_DATASET_TO_MODE .values ())+["rag_mednosp"]:

                if item in _MODE_TO_CORPUS and item not in ordered :

                    ordered .append (item )

            return ordered

        mode =self .mode if self .mode in _MODE_TO_CORPUS else "rag_textbooks"

        return [mode ]



    def _preflight_required_indexes (self )->None :

        rows :List [Dict [str ,Any ]]=[]

        seen =set ()

        for mode in self ._required_modes ():

            corpus_name =_MODE_TO_CORPUS [mode ]

            for row in collect_index_statuses (

            retriever_name =self .requested_retriever_name ,

            corpus_name =corpus_name ,

            db_dir =self .corpus_dir ,

            ):

                key =(

                row .get ("member_corpus"),

                row .get ("retriever_kind"),

                row .get ("query_model"),

                row .get ("article_model"),

                row .get ("expected_dir"),

                )

                if key in seen :

                    continue

                seen .add (key )

                row =dict (row )

                row ["mode"]=mode

                rows .append (row )

        self ._preflight_rows =rows

        missing =[row for row in rows if not row .get ("complete")]

        if missing and not self .build_missing_indexes :

            raise RuntimeError (

            "MedRAGCoordinator preflight found missing or incomplete retrieval indexes. "

            "Either build them first or enable on-demand building. "

            f"Details: {json.dumps(missing, ensure_ascii=False)}"

            )



    def _use_custom_textbooks_backend (self )->bool :

        return any (

        [

        self .custom_rag_textbooks_corpus_dir ,

        self .custom_rag_textbooks_bm25_dir ,

        self .custom_rag_textbooks_dense_dir ,

        ]

        )



    def _custom_textbooks_corpus_name (self )->str :

        value =self .custom_rag_textbooks_corpus_dir .rstrip ("/\\")

        if value :

            return os .path .basename (value )

        return "custom_textbooks"



    def _get_custom_textbooks_retriever (self )->HybridRetriever :

        if HybridRetriever is None :

            raise RuntimeError (

            "Custom textbooks RAG components are unavailable. "

            "Reinstall MEDitor or use the MedRAG/remote RAG backend."

            )

        if self ._custom_textbooks_retriever is not None :

            return self ._custom_textbooks_retriever



        corpus_dir =self .custom_rag_textbooks_corpus_dir

        bm25_dir =self .custom_rag_textbooks_bm25_dir

        dense_dir =self .custom_rag_textbooks_dense_dir

        if corpus_dir :

            if not bm25_dir :

                bm25_dir =os .path .join (corpus_dir ,"indexes","bm25")

            if not dense_dir :

                dense_dir =os .path .join (corpus_dir ,"indexes","medcpt")



        if bm25_dir and os .path .isdir (bm25_dir )and BM25Searcher is None :

            raise RuntimeError ("BM25 index support is unavailable.")

        if dense_dir and os .path .isdir (dense_dir )and DenseSearcher is None :

            raise RuntimeError (

            "Dense index support requires the retrieval dependencies. "

            "Install them with pip install -e '.[retrieval]'."

            )

        bm25 =BM25Searcher (bm25_dir )if bm25_dir and os .path .isdir (bm25_dir )else None

        dense =(

        DenseSearcher (dense_dir ,device =(self .custom_rag_textbooks_dense_device or None ))

        if dense_dir and os .path .isdir (dense_dir )

        else None

        )



        if bm25 is None and dense is None :

            raise RuntimeError (

            "custom textbooks RAG is enabled but no retrieval index was loadable. "

            f"Checked bm25={bm25_dir!r} dense={dense_dir!r}"

            )



        self ._custom_textbooks_status ={

        "corpus_dir":corpus_dir ,

        "bm25_dir":bm25_dir ,

        "dense_dir":dense_dir ,

        "bm25_loaded":bool (bm25 is not None ),

        "dense_loaded":bool (dense is not None ),

        "fusion":self .custom_rag_textbooks_fusion ,

        "query_mode":self .custom_rag_textbooks_query_mode ,

        "top_k":self .custom_rag_textbooks_top_k ,

        "sparse_k":self .custom_rag_textbooks_sparse_k ,

        "dense_k":self .custom_rag_textbooks_dense_k ,

        "rrf_k":self .custom_rag_textbooks_rrf_k ,

        }

        self ._custom_textbooks_retriever =HybridRetriever (bm25 =bm25 ,dense =dense )

        return self ._custom_textbooks_retriever



    def _build_medrag (self ,llm_name :str )->MedRAG :

        use_internal_retrieval =not self ._use_custom_textbooks_backend ()

        try :

            return MedRAG (

            llm_name =llm_name ,

            rag =use_internal_retrieval ,

            retriever_name =self .base_retriever_name ,

            corpus_name =self .base_corpus_name ,

            db_dir =(self .corpus_dir or "./corpus"),

            corpus_cache =False ,

            HNSW =True ,

            device =self .device_str ,

            )

        except Exception :

            if self .base_retriever_name =="MedCPT":

                raise

            self .base_retriever_name ="MedCPT"

            return MedRAG (

            llm_name =llm_name ,

            rag =use_internal_retrieval ,

            retriever_name =self .base_retriever_name ,

            corpus_name =self .base_corpus_name ,

            db_dir =(self .corpus_dir or "./corpus"),

            corpus_cache =False ,

            HNSW =True ,

            device =self .device_str ,

            )



    def _resolve_mode (self ,sample :Optional [Dict [str ,Any ]])->str :

        if self .mode !="rag_auto":

            return self .mode if self .mode in _MODE_TO_CORPUS else "rag_textbooks"



        dataset =str ((sample or {}).get ("dataset")or "").strip ().lower ()

        if dataset in _DATASET_TO_MODE :

            return _DATASET_TO_MODE [dataset ]

        return "rag_mednosp"



    def _get_retrieval_system (self ,mode :str )->Tuple [RetrievalSystem ,str ,str ]:

        mode =mode if mode in _MODE_TO_CORPUS else "rag_textbooks"

        corpus_name =_MODE_TO_CORPUS [mode ]



        if self .medrag is not None and mode ==self .base_mode and getattr (self .medrag ,"retrieval_system",None )is not None :

            return self .medrag .retrieval_system ,self .base_retriever_name ,corpus_name



        if mode in self ._retrieval_systems :

            return self ._retrieval_systems [mode ],self ._retriever_names_by_mode [mode ],corpus_name



        retriever_name =self .requested_retriever_name

        try :

            system =RetrievalSystem (

            retriever_name =retriever_name ,

            corpus_name =corpus_name ,

            db_dir =self .corpus_dir ,

            HNSW =True ,

            cache =False ,

            )

        except Exception :

            if retriever_name =="MedCPT":

                raise

            retriever_name ="MedCPT"

            system =RetrievalSystem (

            retriever_name =retriever_name ,

            corpus_name =corpus_name ,

            db_dir =self .corpus_dir ,

            HNSW =True ,

            cache =False ,

            )



        self ._retrieval_systems [mode ]=system

        self ._retriever_names_by_mode [mode ]=retriever_name

        return system ,retriever_name ,corpus_name



    def retrieve (

    self ,

    query :str ,

    k :Optional [int ]=None ,

    sample :Optional [Dict [str ,Any ]]=None ,

    )->Tuple [List [str ],List [float ],str ,str ,str ]:

        resolved_mode =self ._resolve_mode (sample )

        corpus_name =_MODE_TO_CORPUS .get (resolved_mode ,"Textbooks")

        retriever_name =self .base_retriever_name

        top_k =int (k or self .k )



        if resolved_mode =="rag_textbooks"and self ._use_custom_textbooks_backend ():

            try :

                retriever =self ._get_custom_textbooks_retriever ()

                hits ,_trace =retriever .retrieve (

                question =query ,

                options =None ,

                top_k =max (1 ,int (self .custom_rag_textbooks_top_k or top_k )),

                sparse_k =max (1 ,int (self .custom_rag_textbooks_sparse_k )),

                dense_k =max (1 ,int (self .custom_rag_textbooks_dense_k )),

                query_mode =self .custom_rag_textbooks_query_mode ,

                fusion =self .custom_rag_textbooks_fusion ,

                rrf_k =max (1 ,int (self .custom_rag_textbooks_rrf_k )),

                )

                snippets =[]

                scores =[]

                for hit in hits [:top_k ]:

                    title =str (getattr (hit ,"title","")or "").strip ()

                    text =str (getattr (hit ,"text","")or "").strip ()

                    snippets .append (f"{title}\n{text}".strip ()if title else text )

                    scores .append (float (getattr (hit ,"score",0.0 )or 0.0 ))

                return (

                snippets ,

                scores ,

                resolved_mode ,

                f"custom_textbooks:{self.custom_rag_textbooks_fusion}",

                self ._custom_textbooks_corpus_name (),

                )

            except Exception as err :

                self ._retrieval_errors .append (

                {

                "time":_now (),

                "mode_resolved":resolved_mode ,

                "corpus":self ._custom_textbooks_corpus_name (),

                "retriever":"custom_textbooks",

                "k":top_k ,

                "query":_short (query ,800 ),

                "error":repr (err ),

                }

                )

                return [],[],resolved_mode ,"custom_textbooks:error",self ._custom_textbooks_corpus_name ()



        try :

            retrieval_system ,retriever_name ,corpus_name =self ._get_retrieval_system (resolved_mode )

            snippets ,scores =retrieval_system .retrieve (query ,k =top_k )

        except Exception as err :

            self ._retrieval_errors .append (

            {

            "time":_now (),

            "mode_resolved":resolved_mode ,

            "corpus":corpus_name ,

            "retriever":retriever_name ,

            "k":top_k ,

            "query":_short (query ,800 ),

            "error":repr (err ),

            }

            )

            return [],[],resolved_mode ,retriever_name ,corpus_name



        snips :List [str ]=[]

        for x in snippets or []:

            if isinstance (x ,str ):

                snips .append (x )

            elif isinstance (x ,dict ):

                snips .append (str (x .get ("text")or x .get ("content")or x ))

            else :

                snips .append (str (x ))



        scs :List [float ]=[]

        for s in (scores or []):

            try :

                scs .append (float (s ))

            except Exception :

                continue



        return snips ,scs ,resolved_mode ,retriever_name ,corpus_name



    def _build_remote_rag_summary_for_judge (

    self ,

    sample :Dict [str ,Any ],

    a_pick :str ,

    b_pick :str ,

    a_rationale :str ,

    b_rationale :str ,

    summary_chars :int ,

    meta :Optional [Dict [str ,Any ]]=None ,

    )->Tuple [str ,Dict [str ,Any ]]:

        question =(sample .get ("question")or "").strip ()

        options =sample .get ("options")or {}

        if not isinstance (options ,dict ):

            options ={"A":str (options )}



        def _opt (letter :str )->str :

            try :

                return (options .get (letter )or "").strip ()

            except Exception :

                return ""



        prompt =(

        "You are the RAG coordinator in a medical multiple-choice workflow.\n"

        "Use your retrieval backend/corpus before answering. Your job is NOT to solve the final question directly; "

        "your job is to provide evidence that helps a later judge compare two candidate expert answers.\n\n"

        "Rules:\n"

        "- Do NOT output a final option letter as the answer.\n"

        "- Do NOT choose Expert A or Expert B as the final winner.\n"

        "- Ground claims in retrieved medical evidence.\n"

        "- Use citation markers [A1], [A2], ... for evidence relevant to Candidate A and [B1], [B2], ... for Candidate B.\n"

        "- If evidence is weak or missing, say so explicitly.\n\n"

        f"[Question]\n{_short(question, 2200)}\n\n"

        f"[Options]\n{_short(medqa.render_options(options), 2200)}\n\n"

        f"[Candidate A]\nOption: {a_pick or '-'}\nOption text: {_short(_opt(a_pick), 600)}\n"

        f"Expert rationale: {_short((a_rationale or '').strip(), 900)}\n\n"

        f"[Candidate B]\nOption: {b_pick or '-'}\nOption text: {_short(_opt(b_pick), 600)}\n"

        f"Expert rationale: {_short((b_rationale or '').strip(), 900)}\n\n"

        "Output exactly these sections, with no final answer section:\n"

        "[Key Evidence]\n"

        "- <3-8 concise evidence bullets with [A*] or [B*] citations>\n"

        "[Evidence favoring Candidate A]\n"

        "- <0-5 bullets, each with [A*], or 'none'>\n"

        "[Evidence favoring Candidate B]\n"

        "- <0-5 bullets, each with [B*], or 'none'>\n"

        "[Conflicts / Uncertainties]\n"

        "- <1-5 bullets>\n"

        "[Retrieval Note]\n"

        "- <what source/corpus was searched, or whether evidence was weak>\n"

        )



        coord_summary =""

        retrieval_errors :List [Dict [str ,Any ]]=[]

        if self .remote_rag_agent is None :

            retrieval_errors .append ({"time":_now (),"error":"remote_rag_agent_not_configured"})

        else :

            messages =[

            {

            "role":"system",

            "content":(

            "You are a medical RAG coordinator. Retrieve evidence and summarize it. "

            "Do not output a final answer option."

            ),

            },

            {"role":"user","content":prompt },

            ]

            out =ask_once (self .remote_rag_agent ,messages ,max_tokens =768 ,temperature =0.0 )

            coord_summary =_short (_sanitize_coord_output (str (out or "").strip ()),int (summary_chars ))

            if coord_summary .startswith ("[ERROR]"):

                retrieval_errors .append ({"time":_now (),"error":coord_summary })

                coord_summary =""



        if (not coord_summary )or (len (coord_summary )<80 ):

            coord_summary =(

            "[Key Evidence]\n"

            "- No usable remote RAG evidence was returned.\n"

            "[Evidence favoring Candidate A]\n- none\n"

            "[Evidence favoring Candidate B]\n- none\n"

            "[Conflicts / Uncertainties]\n- Evidence is insufficient to adjudicate.\n"

            "[Retrieval Note]\n- Remote RAG call failed or returned an empty response.\n"

            )



        coord_summary =_short (_sanitize_coord_output (coord_summary ),int (summary_chars ))

        coord_info :Dict [str ,Any ]={

        "mode_requested":self .mode ,

        "mode_resolved":"remote_rag_http",

        "corpus":"remote_rag",

        "retriever_requested":"remote_rag_http",

        "retriever":"remote_rag_http",

        "remote_rag_base_url":self .remote_rag_base_url ,

        "remote_rag_model":self .remote_rag_model ,

        "k":self .k ,

        "query":_short (prompt ,1200 ),

        "top_score":None ,

        "topA":None ,

        "topB":None ,

        "top_score_A":None ,

        "top_score_B":None ,

        "scores_A":[],

        "scores_B":[],

        "scores":{"A":[],"B":[]},

        "snippets_A":[],

        "snippets_B":[],

        "snippets":{"A":[],"B":[]},

        "retrieval_errors":retrieval_errors ,

        "low_conf":bool (retrieval_errors or "No usable remote RAG evidence"in coord_summary ),

        "remote_rag":True ,

        }

        _append_jsonl (

        _log_path_for_action (self .log_path ,meta ),

        {

        "time":_now (),

        "meta":meta or {},

        "mode_requested":self .mode ,

        "mode_resolved":"remote_rag_http",

        "corpus":"remote_rag",

        "retriever":"remote_rag_http",

        "remote_rag_base_url":self .remote_rag_base_url ,

        "remote_rag_model":self .remote_rag_model ,

        "coord_summary":coord_summary ,

        "retrieval_errors":retrieval_errors ,

        },

        )

        return coord_summary ,coord_info



    def build_coord_summary_for_judge (

    self ,

    sample :Dict [str ,Any ],

    a_pick :str ,

    b_pick :str ,

    a_rationale :str ,

    b_rationale :str ,

    min_score :float =-18.7 ,

    max_snippets :int =6 ,

    snippet_chars :int =420 ,

    summary_chars :int =3000 ,

    score_margin :float =4.0 ,

    meta :Optional [Dict [str ,Any ]]=None ,

    )->Tuple [str ,Dict [str ,Any ]]:

        question =(sample .get ("question")or "").strip ()

        options =sample .get ("options")or {}

        if not isinstance (options ,dict ):

            options ={"A":str (options )}



        if self .remote_rag_agent is not None :

            return self ._build_remote_rag_summary_for_judge (

            sample =sample ,

            a_pick =a_pick ,

            b_pick =b_pick ,

            a_rationale =a_rationale ,

            b_rationale =b_rationale ,

            summary_chars =summary_chars ,

            meta =meta ,

            )



        def _opt (letter :str )->str :

            try :

                return (options .get (letter )or "").strip ()

            except Exception :

                return ""



        a_pick =(a_pick or "").strip ()

        b_pick =(b_pick or "").strip ()



        optA =_opt (a_pick )

        optB =_opt (b_pick )

        qA =f"{question}\n\nCandidate ({a_pick}): {optA}"

        qB =f"{question}\n\nCandidate ({b_pick}): {optB}"



        self ._retrieval_errors =[]

        snA ,scA ,resolved_mode ,retriever_used ,corpus_name =self .retrieve (qA ,k =self .k ,sample =sample )

        snB ,scB ,_resolved_mode_b ,_retriever_used_b ,_corpus_name_b =self .retrieve (

        qB ,k =self .k ,sample =sample

        )

        retrieval_errors =list (self ._retrieval_errors )



        pairsA :List [Tuple [str ,float ]]=[]

        for i in range (min (len (snA or []),len (scA or []))):

            try :

                pairsA .append ((str (snA [i ]),float (scA [i ])))

            except Exception :

                continue



        pairsB :List [Tuple [str ,float ]]=[]

        for i in range (min (len (snB or []),len (scB or []))):

            try :

                pairsB .append ((str (snB [i ]),float (scB [i ])))

            except Exception :

                continue



        pairsA =_dedupe_pairs (pairsA )

        pairsB =_dedupe_pairs (pairsB )



        def _filter_and_pack (

        pairs :List [Tuple [str ,float ]],

        )->Tuple [List [str ],List [float ],Optional [float ],float ]:

            local_top =max ([s for _ ,s in pairs ],default =None )

            local_thr =float (min_score )if local_top is None else max (float (min_score ),float (local_top )-float (score_margin ))

            kept =[(t ,s )for (t ,s )in pairs if s >=local_thr ]

            kept .sort (key =lambda x :x [1 ],reverse =True )

            kept =kept [:int (max_snippets )]

            sn_short =[_short (t ,int (snippet_chars ))for (t ,_ )in kept ]

            sc_short =[float (s )for (_ ,s )in kept ]

            return sn_short ,sc_short ,local_top ,float (local_thr )



        snA_short ,scA_short ,topA ,thrA =_filter_and_pack (pairsA )

        snB_short ,scB_short ,topB ,thrB =_filter_and_pack (pairsB )



        top_candidates =[x for x in [topA ,topB ]if x is not None ]

        top_score =max (top_candidates )if top_candidates else None

        low_conf =(

        ((topA is None or float (topA )<float (min_score ))and (topB is None or float (topB )<float (min_score )))

        or (len (snA_short )+len (snB_short )==0 )

        or bool (retrieval_errors )

        )



        coord_info :Dict [str ,Any ]={

        "mode_requested":self .mode ,

        "mode_resolved":resolved_mode ,

        "corpus":corpus_name ,

        "retriever_requested":self .requested_retriever_name ,

        "retriever":retriever_used ,

        "k":self .k ,

        "query_A":_short (qA ,800 ),

        "query_B":_short (qB ,800 ),

        "top_score":top_score ,

        "topA":topA ,

        "topB":topB ,

        "top_score_A":topA ,

        "top_score_B":topB ,

        "min_score":float (min_score ),

        "score_margin":float (score_margin ),

        "threshold_A":float (thrA ),

        "threshold_B":float (thrB ),

        "scores_A":scA_short ,

        "scores_B":scB_short ,

        "scores":{"A":scA_short ,"B":scB_short },

        "snippets_A":snA_short ,

        "snippets_B":snB_short ,

        "snippets":{"A":snA_short ,"B":snB_short },

        "retrieval_errors":retrieval_errors ,

        "low_conf":bool (low_conf ),

        }



        prompt =(

        "You are a retrieval coordinator for medical multiple-choice questions.\n"

        "Your ONLY job is to summarize retrieved evidence and map it to the experts' claims.\n"

        "DO NOT answer the question. DO NOT choose between Expert A/B.\n"

        "DO NOT output any option letter as a final decision. Do NOT output 'Preferred' or '[RESULT]'.\n"

        "Ground every bullet in the retrieved snippets using citations like [A1], [B2], ...\n\n"

        f"[Question]\n{_short(question, 1800)}\n\n"

        f"[Options]\n{_short(medqa.render_options(options), 2200)}\n\n"

        f"[Expert A Pick]\n{a_pick or '-'}\n"

        f"[Expert A Rationale]\n{_short((a_rationale or '').strip(), 900)}\n\n"

        f"[Expert B Pick]\n{b_pick or '-'}\n"

        f"[Expert B Rationale]\n{_short((b_rationale or '').strip(), 900)}\n\n"

        f"[Retrieved Snippets for Candidate A ({a_pick})] (kept {len(snA_short)})\n"

        +("\n".join ([f"[A{i + 1}] {t}"for i ,t in enumerate (snA_short )])if snA_short else "none")

        +"\n\n"

        f"[Retrieved Snippets for Candidate B ({b_pick})] (kept {len(snB_short)})\n"

        +("\n".join ([f"[B{i + 1}] {t}"for i ,t in enumerate (snB_short )])if snB_short else "none")

        +"\n\n"

        "Output format (no extra sections):\n"

        "[Key Evidence]\n- <3-8 bullets with citations like [A1] / [B2]>\n"

        "[Evidence favoring Candidate A]\n- <0-5 bullets with [A*] citations or 'none'>\n"

        "[Evidence favoring Candidate B]\n- <0-5 bullets with [B*] citations or 'none'>\n"

        "[Conflicts / Uncertainties]\n- <1-5 bullets>\n"

        "[Retrieval Note]\n- <mention if evidence is weak/missing>\n"

        )

        if low_conf :

            prompt +="\nIMPORTANT: Retrieval confidence is LOW. Explicitly say evidence may be weak.\n"



        if self .summary_agent is not None :

            messages =[

            {

            "role":"system",

            "content":(

            "You are a retrieval coordinator. Summarize evidence only. "

            "Do NOT answer the question. Do NOT output any final option letter."

            ),

            },

            {"role":"user","content":prompt },

            ]

            out =ask_once (self .summary_agent ,messages ,max_tokens =768 ,temperature =0.0 )

            coord_summary =_short (_sanitize_coord_output (str (out or "").strip ()),int (summary_chars ))

            if coord_summary .startswith ("[ERROR]"):

                coord_summary =""

        elif self .medrag is None :

            coord_summary =""

        else :

            try :

                messages =[

                {

                "role":"system",

                "content":(

                "You are a retrieval coordinator. Summarize evidence only. "

                "Do NOT answer the question. Do NOT output any final option letter."

                ),

                },

                {"role":"user","content":prompt },

                ]

                out =self .medrag .generate (messages ,max_new_tokens =768 )

                coord_summary =_short (_sanitize_coord_output (str (out or "").strip ()),int (summary_chars ))

            except Exception as e :

                coord_summary =f"[Coordinator ERROR]{e}"



        if (not coord_summary )or (len (coord_summary )<120 ):

            bullets :List [str ]=[]

            for i ,t in enumerate (snA_short [:4 ]):

                bullets .append (f"- {t} [A{i + 1}]")

            for i ,t in enumerate (snB_short [:4 ]):

                bullets .append (f"- {t} [B{i + 1}]")

            if not bullets :

                bullets =["- No usable snippets were retrieved."]



            coord_summary =(

            "[Key Evidence]\n"

            +"\n".join (bullets [:8 ])

            +"\n[Evidence favoring Candidate A]\n- none\n"

            +"[Evidence favoring Candidate B]\n- none\n"

            +"[Conflicts / Uncertainties]\n- Evidence is insufficient to adjudicate.\n"

            +"[Retrieval Note]\n- Retrieval may be weak or empty.\n"

            )

            coord_summary =_short (_sanitize_coord_output (coord_summary ),int (summary_chars ))



        _append_jsonl (

        _log_path_for_action (self .log_path ,meta ),

        {

        "time":_now (),

        "meta":meta or {},

        "mode_requested":self .mode ,

        "mode_resolved":resolved_mode ,

        "corpus":corpus_name ,

        "retriever_requested":self .requested_retriever_name ,

        "retriever":retriever_used ,

        "k":self .k ,

        "top_score":top_score ,

        "topA":topA ,

        "topB":topB ,

        "min_score":float (min_score ),

        "score_margin":float (score_margin ),

        "threshold_A":float (thrA ),

        "threshold_B":float (thrB ),

        "low_conf":bool (low_conf ),

        "retrieval_errors":retrieval_errors ,

        "query":{"A":_short (qA ,1200 ),"B":_short (qB ,1200 )},

        "scores":{"A":scA_short ,"B":scB_short },

        "snippets":{"A":snA_short ,"B":snB_short },

        },

        )



        return coord_summary ,coord_info

