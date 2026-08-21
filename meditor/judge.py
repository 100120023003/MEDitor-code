




from __future__ import annotations



from typing import Dict ,Any ,List ,Tuple ,Optional

import re

import json

import traceback

from collections import Counter



from .agents import ask_once



META_BEGIN ="__JUDGE_META__"

META_END ="__END_META__"



def _pack_meta (meta :Dict [str ,Any ],body :str )->str :

    try :

        head =json .dumps (meta ,ensure_ascii =False )

    except Exception :

        head ="{}"

    return f"{META_BEGIN}\n{head}\n{META_END}\n{body or ''}"



def extract_judge_meta (debug_text :str )->Tuple [Dict [str ,Any ],str ]:


    if not (debug_text or "").startswith (META_BEGIN ):

        return {},debug_text or ""

    try :

        parts =(debug_text or "").splitlines ()

        if len (parts )>=3 and parts [0 ].strip ()==META_BEGIN and parts [2 ].strip ()==META_END :

            meta =json .loads (parts [1 ])

            body ="\n".join (parts [3 :])if len (parts )>3 else ""

            return (meta if isinstance (meta ,dict )else {}),body

    except Exception :

        pass

    return {},debug_text or ""



from .import medqa



MAX_CTX_CHARS =6500



DEFAULT_RUBRIC =(

"1) Evidence alignment: key claims should be supported by the Coordinator evidence or the question.\n"

"2) Medical correctness: penalize contradictions, hallucinated medical facts, and guideline conflicts.\n"

"3) MCQ discipline: the chosen option must be clearly justified; penalize vague or non-committal answers.\n"

"4) Internal consistency: coherent reasoning, no self-contradiction.\n"

"5) Calibration: prefer answers that acknowledge uncertainty when evidence is weak.\n"

)





def _short (s :str ,n :int )->str :

    s =s or ""

    return (s [:n ]+"…")if len (s )>n else s









_RESULT_RE =re .compile (r"\[RESULT\]\s*\(?\s*([A-E])\s*\)?",re .I )

_JSON_RE =re .compile (r"\{.*\}",re .S )



def parse_result_ab (text :str )->Optional [str ]:


    if not text :

        return None

    s =str (text ).strip ()

    if not s :

        return None





    m =_RESULT_RE .search (s )

    if m :

        letter =m .group (1 ).upper ()

        if letter in ("A","B"):

            return letter

        return None





    m =re .match (r"^\s*([AB])\s*$",s ,flags =re .IGNORECASE )

    if m :

        return m .group (1 ).upper ()





    m =re .search (r"\b(?:winner|pick|choice)\s*[:：]\s*([AB])\b",s ,flags =re .IGNORECASE )

    if m :

        return m .group (1 ).upper ()





    ex =re .findall (r"\bexpert\s*([AB])\b",s ,flags =re .IGNORECASE )

    if ex :

        return ex [-1 ].upper ()





    ab =re .findall (r"\b([AB])\b",s ,flags =re .IGNORECASE )

    return ab [-1 ].upper ()if ab else None





def _extract_first_json (text :str )->Optional [Dict [str ,Any ]]:

    if not text :

        return None

    m =_JSON_RE .search (text )

    if not m :

        return None

    try :

        obj =json .loads (m .group (0 ))

        return obj if isinstance (obj ,dict )else None

    except Exception :

        return None





def parse_judge_scored (text :str )->Tuple [str ,Optional [float ],Optional [float ],bool ]:


    obj =_extract_first_json (text or "")

    if obj is not None :

        winner =str (obj .get ("winner")or "").strip ().upper ()

        score_a =obj .get ("score_A",None )

        score_b =obj .get ("score_B",None )



        try :

            score_a =float (score_a )if score_a is not None else None

        except Exception :

            score_a =None

        try :

            score_b =float (score_b )if score_b is not None else None

        except Exception :

            score_b =None



        if winner in ("A","B"):

            return winner ,score_a ,score_b ,False

        if winner in ("ABSTAIN","NONE","UNCERTAIN"):

            return "",score_a ,score_b ,True



    w =parse_result_ab (text or "")

    return (w ,None ,None ,False )if w else ("",None ,None ,False )









def short_critic_line (agent ,sample :Dict [str ,Any ],pick :str ,Ls :List [str ],max_tokens :int )->str :

    if not pick :

        return ""

    msgs =[

    {

    "role":"system",

    "content":(

    "Provide ONE short line: 'Choice: <LETTER>; Evidence: <10-20 words>'. "

    "Do not add explanations beyond this line. End with <END>."

    ),

    },

    {

    "role":"user",

    "content":(

    f"[Question]\n{_short(sample.get('question',''), 2200)}\n\n"

    f"[Options]\n{_short(medqa.render_options(sample.get('options') or {}), 1800)}\n\n"

    f"You must defend choice {pick}. Provide the line now.\n<END>"

    ),

    },

    ]

    out =ask_once (agent ,msgs ,max_tokens =max_tokens ,temperature =0.0 ,stop =["<END>"])

    return _short (out or "",256 )









def build_judge_messages (

sample :Dict [str ,Any ],

Ls :List [str ],

a_pick :str ,

b_pick :str ,

lastA :str ,

lastB :str ,

criticA :str ,

criticB :str ,

coord_summary :str ="",

rubric_text :str ="",

)->List [Dict [str ,str ]]:

    lastA_short =_short ((lastA or "").strip (),900 )

    lastB_short =_short ((lastB or "").strip (),900 )



    criticA_short =_short ((criticA or "").strip (),320 )

    criticB_short =_short ((criticB or "").strip (),320 )



    coord_summary_short =_short ((coord_summary or "").strip (),2400 )



    rubric =(rubric_text or "").strip ()or DEFAULT_RUBRIC



    user =(

    "### Instruction\n"

    "You are a judge comparing Expert A vs Expert B.\n"

    "CRITICAL:\n"

    "1) Do NOT solve the question yourself.\n"

    "2) Do NOT propose any new option letter.\n"

    "3) Your output winner is EXPERT label A or B (Expert A/Expert B), NOT the answer option letters.\n"

    "Decide which expert is MORE RELIABLE given their reasoning and the Coordinator evidence.\n\n"

    f"[Question]\n{_short(sample.get('question',''), 2200)}\n\n"

    f"[Options]\n{_short(medqa.render_options(sample.get('options') or {}), 2200)}\n\n"

    f"[Options letters are] {', '.join(Ls)}  (These are answer choices, not the winner format.)\n\n"

    "[Experts' Picks]\n"

    f"Expert A picked option: {a_pick or '-'}\n"

    f"Expert B picked option: {b_pick or '-'}\n\n"

    "[Experts' Reasoning / Debate Last Turn]\n"

    "Expert A reasoning:\n"

    f"{lastA_short or '-'}\n\n"

    "Expert B reasoning:\n"

    f"{lastB_short or '-'}\n\n"

    )



    if criticA_short or criticB_short :

        user +="[Optional Critic Lines]\n"

        if criticA_short :

            user +=f"Expert A critic: {criticA_short}\n"

        if criticB_short :

            user +=f"Expert B critic: {criticB_short}\n"

        user +="\n"



    if coord_summary_short :

        user +=f"[Coordinator Evidence (RAG)]\n{coord_summary_short}\n\n"



    user +=(

    "[Score Rubric]\n"

    f"{rubric}\n\n"

    "### Output Format\n"

    "Return ONE JSON object, then <END>. No extra text.\n"

    "JSON schema:\n"

    "{\n"

    '  "score_A": <0-10>,\n'

    '  "score_B": <0-10>,\n'

    '  "winner": "A" | "B" | "ABSTAIN",\n'

    '  "why": "<ONE short sentence; MUST include at least one evidence cite like [A3] or [B2]>"\n'

    "Rules:\n"

    "- You MUST cite evidence markers in why (e.g., [A1], [B2]).\n"

    "- If you cannot provide at least one valid cite, set winner=ABSTAIN.\n"

    "- If evidence is insufficient or both are similar, set winner=ABSTAIN.\n"

    "4) Your 'why' MUST include at least one cite token like [A1] or [B2], otherwise ABSTAIN.\n"

    "- Winner A/B refers to EXPERT labels, NOT option letters.\n"

    "<END>"

    )



    user =_short (user ,MAX_CTX_CHARS )



    sys =(

    "You are a medical pairwise judge. Compare Expert A vs Expert B only. "

    "Do NOT solve the question directly and do NOT introduce any new option. "

    "Use the Coordinator Evidence (RAG) as the primary reference when available; "

    "otherwise rely on medical correctness and internal consistency. "

    "You may abstain if evidence is insufficient. "

    "Keep output short and follow the required JSON Output Format."

    )

    return [

    {"role":"system","content":sys },

    {"role":"user","content":user },

    ]





def _repair_winner (Judge ,max_tokens :int =16 )->str :


    msgs =[

    {"role":"system","content":"You must output a single JSON object with winner only."},

    {"role":"user","content":'{"winner":"A"} or {"winner":"B"} or {"winner":"ABSTAIN"}\n<END>'},

    ]

    out =ask_once (Judge ,msgs ,max_tokens =max_tokens ,temperature =0.0 ,stop =["<END>"])

    return out or ""











def judge_once_scored (

Judge ,

sample :Dict [str ,Any ],

Ls :List [str ],

a_pick :str ,

b_pick :str ,

lastA :str ,

lastB :str ,

criticA :str ,

criticB :str ,

coord_summary :str ,

rubric_text :str ,

judge_max :int ,

delta_abstain :float =1.5 ,

temperature :float =0.2 ,

)->Tuple [str ,str ]:


    msgs =build_judge_messages (

    sample =sample ,

    Ls =Ls ,

    a_pick =a_pick ,

    b_pick =b_pick ,

    lastA =lastA ,

    lastB =lastB ,

    criticA =criticA ,

    criticB =criticB ,

    coord_summary =coord_summary ,

    rubric_text =rubric_text ,

    )

    mt =max (24 ,min (int (judge_max ),160 ))

    out =ask_once (Judge ,msgs ,max_tokens =mt ,temperature =temperature ,stop =["<END>"])or ""



    w ,sa ,sb ,abst =parse_judge_scored (out )





    if w and (sa is None or sb is None ):

        rep_msgs =[

        {"role":"system","content":"Rewrite the previous output into ONE valid JSON object with numeric score_A and score_B (0-10) and winner A/B/ABSTAIN. No extra text. End with <END>."},

        {"role":"user","content":f"Previous output:\n{out}\n\nReturn JSON now.\n<END>"},

        ]

        rep =ask_once (Judge ,rep_msgs ,max_tokens =80 ,temperature =0.0 ,stop =["<END>"])or ""

        w2 ,sa2 ,sb2 ,abst2 =parse_judge_scored (rep )



        if w2 and (sa2 is not None )and (sb2 is not None ):

            out =out +"\n\n-----\n\n"+rep

            w ,sa ,sb ,abst =w2 ,sa2 ,sb2 ,abst2



    reason ="ok"

    repaired =False





    if (sa is not None )and (sb is not None )and abs (sa -sb )<float (delta_abstain ):

        reason ="low_margin"

        meta ={

        "reason":reason ,

        "repaired":repaired ,

        "score_A":sa ,

        "score_B":sb ,

        "delta_abstain":float (delta_abstain ),

        "temperature":float (temperature ),

        "winner":"ABSTAIN",

        }

        return "",_pack_meta (meta ,out )



    if abst :

        reason ="abstain"

        meta ={

        "reason":reason ,

        "repaired":repaired ,

        "score_A":sa ,

        "score_B":sb ,

        "delta_abstain":float (delta_abstain ),

        "temperature":float (temperature ),

        "winner":"ABSTAIN",

        }

        return "",_pack_meta (meta ,out )



    if not w :



        rep =_repair_winner (Judge ,max_tokens =24 )or ""

        repaired =True

        out2 =out +"\n\n-----\n\n"+rep

        w2 ,sa2 ,sb2 ,abst2 =parse_judge_scored (rep )



        if (sa2 is not None )and (sb2 is not None )and abs (sa2 -sb2 )<float (delta_abstain ):

            reason ="low_margin"

            meta ={

            "reason":reason ,

            "repaired":repaired ,

            "score_A":sa2 ,

            "score_B":sb2 ,

            "delta_abstain":float (delta_abstain ),

            "temperature":float (temperature ),

            "winner":"ABSTAIN",

            }

            return "",_pack_meta (meta ,out2 )



        if abst2 or (not w2 ):

            reason ="repaired_invalid"

            meta ={

            "reason":reason ,

            "repaired":repaired ,

            "score_A":sa2 ,

            "score_B":sb2 ,

            "delta_abstain":float (delta_abstain ),

            "temperature":float (temperature ),

            "winner":"ABSTAIN",

            }

            return "",_pack_meta (meta ,out2 )



        reason ="repaired_ok"

        meta ={

        "reason":reason ,

        "repaired":repaired ,

        "score_A":sa2 ,

        "score_B":sb2 ,

        "delta_abstain":float (delta_abstain ),

        "temperature":float (temperature ),

        "winner":w2 ,

        }

        return (w2 or ""),_pack_meta (meta ,out2 )



    meta ={

    "reason":reason ,

    "repaired":repaired ,

    "score_A":sa ,

    "score_B":sb ,

    "delta_abstain":float (delta_abstain ),

    "temperature":float (temperature ),

    "winner":w ,

    }

    return w ,_pack_meta (meta ,out )





def judge_vote_bidirectional (

Judge ,

sample :Dict [str ,Any ],

Ls :List [str ],

a_pick :str ,

b_pick :str ,

lastA :str ,

lastB :str ,

criticA :str ,

criticB :str ,

coord_summary :str ,

rubric_text :str ,

judge_max :int ,

delta_abstain :float =1.5 ,

default_winner :str ="B",

)->Tuple [str ,str ]:


    try :

        w1 ,out1 =judge_once_scored (

        Judge ,sample ,Ls ,

        a_pick ,b_pick ,lastA ,lastB ,criticA ,criticB ,

        coord_summary ,rubric_text ,judge_max ,

        delta_abstain =delta_abstain ,

        temperature =0.2 ,

        )



        w2_swapped ,out2 =judge_once_scored (

        Judge ,sample ,Ls ,

        b_pick ,a_pick ,lastB ,lastA ,criticB ,criticA ,

        coord_summary ,rubric_text ,judge_max ,

        delta_abstain =delta_abstain ,

        temperature =0.2 ,

        )





        if w2_swapped =="A":

            w2 ="B"

        elif w2_swapped =="B":

            w2 ="A"

        else :

            w2 =""



        debug ="[PASS1]\n"+out1 +"\n\n[PASS2_SWAP]\n"+out2



        if w1 and w2 and (w1 ==w2 ):

            meta ={"mode":"bidirectional","final":w1 ,"pass1":w1 ,"pass2":w2 }

            return w1 ,_pack_meta (meta ,debug )



        meta ={"mode":"bidirectional","final":"ABSTAIN","pass1":w1 ,"pass2":w2 }

        return "",_pack_meta (meta ,debug )

    except Exception as e :

        return "",f"[ERROR]{e}\n{traceback.format_exc()}"





def judge_vote (

Judge ,

sample :Dict [str ,Any ],

Ls :List [str ],

a_pick :str ,

b_pick :str ,

lastA :str ,

lastB :str ,

criticA :str ,

criticB :str ,

coord_summary :str ,

rubric_text :str ,

judge_max :int ,

self_consistency :int =1 ,

delta_abstain :float =1.5 ,

)->Tuple [str ,str ]:


    outs :List [str ]=[]

    winners :List [str ]=[]

    try :

        n_rounds =max (int (self_consistency ),1 )

        for i in range (n_rounds ):

            w ,out =judge_once_scored (

            Judge ,sample ,Ls ,

            a_pick ,b_pick ,lastA ,lastB ,criticA ,criticB ,

            coord_summary ,rubric_text ,judge_max ,

            delta_abstain =float (delta_abstain ),

            temperature =0.2 if (i %2 ==0 )else 0.0 ,

            )

            outs .append (out )

            if w in ("A","B"):

                winners .append (w )



        if winners :

            c =Counter (winners )

            max_count =max (c .values ())

            top =[k for k ,v in c .items ()if v ==max_count ]

            final ="B"if "B"in top else "A"

            debug_body ="\n\n-----\n\n".join (outs )

            meta ={"mode":"self_consistency","rounds":n_rounds ,"delta_abstain":float (delta_abstain ),"valid_votes":winners ,"final":final }

            return final ,_pack_meta (meta ,debug_body )



        debug_body ="\n\n-----\n\n".join (outs )

        meta ={"mode":"self_consistency","rounds":n_rounds ,"delta_abstain":float (delta_abstain ),"valid_votes":winners ,"final":"ABSTAIN"}

        return "",_pack_meta (meta ,debug_body )

    except Exception as e :

        debug_body =f"[ERROR]{e}\n{traceback.format_exc()}\n\n-----\n\n"+"\n\n-----\n\n".join (outs )

        meta ={"mode":"self_consistency","delta_abstain":float (delta_abstain ),"error":str (e )}

        return "",_pack_meta (meta ,debug_body )

