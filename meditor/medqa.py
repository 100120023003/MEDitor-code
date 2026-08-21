




from __future__ import annotations



import json

import re

from typing import Any ,Dict ,List ,Optional





LABELS =list ("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

_RE_FINAL =re .compile (r"(?:Final\s*(?:Answer|Response)|Answer)\s*[:：]?\s*([A-Z])",re .IGNORECASE )

_RE_STANDALONE =re .compile (r"^\s*([A-Z])\s*(?:<END>)?\s*$",re .IGNORECASE )





def parse_label (text :str )->Optional [str ]:

    if not text :

        return None

    m =_RE_FINAL .search (text )

    if m :

        c =m .group (1 ).upper ()

        return c if c in LABELS else None

    m =_RE_STANDALONE .match (text )

    if m :

        c =m .group (1 ).upper ()

        return c if c in LABELS else None

    return None





def _to_options (obj :Any )->Dict [str ,str ]:

    if isinstance (obj ,dict ):

        return {k :str (obj [k ])for k in LABELS if k in obj }

    if isinstance (obj ,list ):

        return {LABELS [i ]:str (o )for i ,o in enumerate (obj [:len (LABELS )])}

    return {}





def load (path :str )->List [Dict [str ,Any ]]:

    samples :List [Dict [str ,Any ]]=[]

    with open (path ,"r",encoding ="utf-8")as f :

        for i ,line in enumerate (f ):

            line =line .strip ()

            if not line :

                continue

            obj =json .loads (line )

            opts =_to_options (obj .get ("options")or obj .get ("choices")or obj .get ("answers")or [])

            labels =list (opts .keys ())or LABELS [:4 ]



            gold :Optional [str ]=None

            for k in ("gold_letter","gold","label","answer","answer_idx","label_idx"):

                v =obj .get (k )

                if isinstance (v ,str )and v .strip ().upper ()in labels :

                    gold =v .strip ().upper ()

                    break

                if isinstance (v ,int )and 0 <=v <len (labels ):

                    gold =labels [v ]

                    break

                if isinstance (v ,str )and v .isdigit ():

                    vi =int (v )

                    if 0 <=vi <len (labels ):

                        gold =labels [vi ]

                        break



            if gold is None :

                ans_text =obj .get ("answer")

                if isinstance (ans_text ,str )and opts :

                    norm =ans_text .strip ()

                    for k ,v in opts .items ():

                        if str (v ).strip ()==norm :

                            gold =k

                            break



            samples .append (

            {

            "id":obj .get ("id",i ),

            "question":obj .get ("question")or obj .get ("prompt")or obj .get ("Q")or "",

            "options":opts ,

            "gold":gold ,

            }

            )

    return samples





def render_options (opts :Dict [str ,str ])->str :

    return "\n".join (f"{k}. {v}"for k ,v in opts .items ())





def sys_expert_letter (Ls :List [str ])->str :

    choices =", ".join (Ls )

    return (

    "You are an expert physician answering a multiple-choice medical question. "

    f"Choose exactly one option from: {choices}. "

    "End with 'Answer: <LETTER><END>'."

    )





def sys_expert_rationale (Ls :List [str ])->str :

    choices =", ".join (Ls )

    return (

    "You are an expert physician. Give concise clinical reasoning, then choose "

    f"exactly one option from: {choices}. End with 'Answer: <LETTER><END>'."

    )





def sys_judge_prometheus (Ls :List [str ])->str :

    choices =", ".join (Ls )

    return (

    "You are an impartial medical exam judge. Compare the candidate answers "

    f"and choose exactly one option from: {choices}. End with 'Answer: <LETTER><END>'."

    )





def build_user_expert_vote (sample :Dict [str ,Any ])->str :

    return (

    f"Question:\n{sample.get('question', '')}\n\n"

    f"Options:\n{render_options(sample.get('options', {}))}\n\n"

    "Return only your final choice."

    )





def build_user_expert_rationale (sample :Dict [str ,Any ])->str :

    return (

    f"Question:\n{sample.get('question', '')}\n\n"

    f"Options:\n{render_options(sample.get('options', {}))}\n\n"

    "Give concise reasoning and then your final choice."

    )





def build_user_judge (sample :Dict [str ,Any ],a_text :str ,b_text :str )->str :

    return (

    f"Question:\n{sample.get('question', '')}\n\n"

    f"Options:\n{render_options(sample.get('options', {}))}\n\n"

    f"Candidate A:\n{a_text}\n\n"

    f"Candidate B:\n{b_text}\n\n"

    "Choose the better final answer."

    )





def take_until_answer (text :str )->str :

    if not text :

        return ""

    m =re .search (r"(.*?Answer\s*[:：]?\s*[A-Z]\s*(?:<END>)?)",text ,flags =re .I |re .S )

    return m .group (1 ).strip ()if m else text .strip ()





def parse_label_from_answer (text :str ,labels =("A","B","C","D"))->str |None :

    if not text :

        return None

    m =re .search (r"Answer\s*[:：]?\s*([A-Za-z])",text )

    if not m :

        return None

    c =m .group (1 ).upper ()

    return c if c in labels else None

