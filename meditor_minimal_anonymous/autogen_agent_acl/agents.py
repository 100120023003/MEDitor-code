




from __future__ import annotations

from typing import Any ,Dict ,List ,Optional ,Tuple



import json

import urllib .request

import urllib .error



try :

    import autogen as _pyautogen

except Exception :

    _pyautogen =None





class _ConfigOnlyAgent :




    def __init__ (self ,name :str ,system_message :str ,llm_config :Dict [str ,Any ]):

        self .name =name

        self .system_message =system_message or ""

        self .llm_config =llm_config



    def reset (self )->None :

        return None





def make_agent (

name :str ,

base_url :str ,

model :str ,

temperature :float =0.0 ,

top_p :float =1.0 ,

system_message :str ="",

):


    llm_config ={

    "config_list":[

    {

    "model":model ,



    "base_url":base_url ,

    "api_key":"EMPTY",

    "temperature":float (temperature ),

    "top_p":float (top_p ),

    }

    ],

    "cache_seed":None ,

    }



    if _pyautogen is not None and hasattr (_pyautogen ,"ConversableAgent"):

        agent =_pyautogen .ConversableAgent (

        name =name ,

        system_message =system_message or "",

        llm_config =llm_config ,

        human_input_mode ="NEVER",

        )

        return agent



    return _ConfigOnlyAgent (

    name =name ,

    system_message =system_message or "",

    llm_config =llm_config ,

    )





def get_conn_from_agent (agent )->Tuple [str ,str ]:

    cfg =agent .llm_config .get ("config_list",[{}])[0 ]

    base =cfg .get ("base_url")or cfg .get ("api_base")or ""

    return base ,cfg .get ("model","")





def _clear_agent_state (agent :Any )->None :


    if hasattr (agent ,"reset")and callable (getattr (agent ,"reset")):

        try :

            agent .reset ()

        except Exception :

            pass



    for attr in ("chat_messages","_oai_messages","oai_messages","_messages","messages"):

        if hasattr (agent ,attr ):

            try :

                obj =getattr (agent ,attr )

                if isinstance (obj ,dict ):

                    obj .clear ()

                elif isinstance (obj ,list ):

                    obj .clear ()

            except Exception :

                pass





def _join_chat_url (base_url :str )->str :

    base =(base_url or "").rstrip ("/")



    return f"{base}/chat/completions"





def _post_json (url :str ,payload :Dict [str ,Any ],api_key :str ="EMPTY",timeout_s :int =300 )->Tuple [int ,str ]:

    headers ={

    "Content-Type":"application/json",

    }

    if api_key and api_key !="EMPTY":

        headers ["Authorization"]=f"Bearer {api_key}"



    data =json .dumps (payload ,ensure_ascii =False ).encode ("utf-8")

    req =urllib .request .Request (url ,data =data ,headers =headers ,method ="POST")



    try :

        with urllib .request .urlopen (req ,timeout =timeout_s )as resp :

            body =resp .read ().decode ("utf-8",errors ="replace")

            return int (getattr (resp ,"status",200 )),body

    except urllib .error .HTTPError as e :

        try :

            body =e .read ().decode ("utf-8",errors ="replace")

        except Exception :

            body =str (e )

        return int (getattr (e ,"code",500 )),body

    except Exception as e :

        return 599 ,str (e )





def _extract_content (resp_json :Dict [str ,Any ])->str :

    choices =resp_json .get ("choices")or []

    if not choices :

        return ""

    c0 =choices [0 ]or {}



    if isinstance (c0 ,dict )and "message"in c0 and isinstance (c0 ["message"],dict ):

        return str (c0 ["message"].get ("content")or "")



    if isinstance (c0 ,dict )and "text"in c0 :

        return str (c0 .get ("text")or "")

    return ""





def ask_once (

agent ,

messages :List [Dict [str ,str ]],

max_tokens :int =256 ,

temperature :float =0.0 ,

stop :Optional [List [str ]]=None ,

extra_body :Optional [Dict [str ,Any ]]=None ,

)->str :


    _clear_agent_state (agent )



    cfg =agent .llm_config .get ("config_list",[{}])[0 ]

    base_url =cfg .get ("base_url")or cfg .get ("api_base")or ""

    model =cfg .get ("model")or ""

    api_key =cfg .get ("api_key")or "EMPTY"



    url =_join_chat_url (base_url )



    payload :Dict [str ,Any ]={

    "model":model ,

    "messages":messages ,

    "temperature":float (temperature ),

    "max_tokens":int (max_tokens ),

    }

    if stop :

        payload ["stop"]=stop





    payload1 =dict (payload )

    if extra_body :

        payload1 .update (extra_body )



    try :

        status ,body =_post_json (url ,payload1 ,api_key =api_key ,timeout_s =300 )

        if status !=200 :



            if extra_body :

                status2 ,body2 =_post_json (url ,payload ,api_key =api_key ,timeout_s =300 )

                if status2 ==200 :

                    data2 =json .loads (body2 )

                    return _extract_content (data2 )

                return f"[ERROR][HTTP {status2}] {body2}"

            return f"[ERROR][HTTP {status}] {body}"



        data =json .loads (body )

        out =_extract_content (data )

        return out or ""

    except Exception as e :

        return f"[ERROR]{e}"

    finally :

        _clear_agent_state (agent )

