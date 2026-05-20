

from __future__ import annotations

from typing import Dict ,Any ,List ,Optional ,Tuple

from collections import Counter

from .agents import ask_once

from .import medqa





def _short (s :str ,n :int )->str :

    s =s or ""

    return (s [:n ]+"…")if len (s )>n else s





def build_vote_messages (sample :Dict [str ,Any ],Ls :List [str ])->List [Dict [str ,str ]]:



    q =_short (sample .get ("question",""),1200 )

    opts =_short (medqa .render_options (sample ["options"]),900 )



    allowed =", ".join (Ls )

    return [

    {

    "role":"system",

    "content":(

    "You are answering a multiple-choice question.\n"

    f"Allowed choices: {allowed}.\n"

    "You MUST output exactly one capital letter (A/B/C/D), then newline, then <END>.\n"

    "Do NOT output any reasoning, explanation, markdown, or extra words.\n"

    "Examples:\n"

    "A\n<END>\n"

    "D\n<END>\n"

    ),

    },

    {

    "role":"user",

    "content":(

    f"[Question]\n{q}\n\n"

    f"[Options]\n{opts}\n\n"

    "Output ONLY the letter."

    ),

    },

    ]



def run_votes (

agent ,

sample :Dict [str ,Any ],

Ls :List [str ],

repeats :int ,

vote_tokens :int ,

temp :float ,

)->Tuple [List [Optional [str ]],List [str ]]:


    letters :List [Optional [str ]]=[]

    raws :List [str ]=[]

    error_flag =False





    mt =max (8 ,min (vote_tokens ,64 ))



    for _ in range (repeats ):

        ...

        msgs =build_vote_messages (sample ,Ls )

        txt =ask_once (

        agent ,

        msgs ,

        max_tokens =mt ,

        temperature =temp ,

        stop =["<END>"],

        )

        raw =txt or ""

        lab =medqa .parse_label (raw )if raw and not raw .startswith ("[ERROR]")else None





        if lab is None and raw and (not raw .startswith ("[ERROR]")):

            txt2 =ask_once (

            agent ,

            msgs ,

            max_tokens =max (mt ,96 ),

            temperature =0.0 ,

            stop =["<END>"],

            )

            raw2 =txt2 or ""

            lab2 =medqa .parse_label (raw2 )if raw2 and not raw2 .startswith ("[ERROR]")else None

            if lab2 is not None :

                raw ,lab =raw2 ,lab2



        letters .append (lab )

        raws .append (raw )



        if raw and raw .startswith ("[ERROR]"):

            error_flag =True



    return letters ,raws



def run_rationale (agent ,sample :Dict [str ,Any ],Ls :List [str ],

max_tokens :int )->Tuple [str ,Optional [str ]]:


    msgs =[

    {"role":"system","content":medqa .sys_expert_rationale (Ls )},

    {"role":"user","content":medqa .build_user_expert_rationale (sample )},

    ]

    txt =ask_once (

    agent ,

    msgs ,

    max_tokens =max_tokens ,

    temperature =0.0 ,

    stop =["<END>"],

    )



    txt =medqa .take_until_answer (txt )





    lab =medqa .parse_label_from_answer (txt )or medqa .parse_label (txt )

    return txt ,lab





def majority (votes :List [Optional [str ]],labels =("A","B","C","D"))->Optional [str ]:

    cnt =Counter (v for v in votes if v in labels )

    if not cnt :

        return None

    best =sorted (cnt .items (),key =lambda x :(-x [1 ],x [0 ]))

    return best [0 ][0 ]





def vote_topcount (votes :List [Optional [str ]],maj :Optional [str ])->int :

    if not votes or not maj :

        return 0

    c =Counter (v for v in votes if v )

    return c .get (maj ,0 )





def choose_by_margin (a_maj ,b_maj ,a_top ,b_top ,Ls :List [str ])->Optional [str ]:

    if a_maj and b_maj :

        if a_top >b_top :

            return a_maj

        if b_top >a_top :

            return b_maj

        return a_maj or b_maj or (Ls [0 ]if Ls else None )

    return a_maj or b_maj or (Ls [0 ]if Ls else None )

