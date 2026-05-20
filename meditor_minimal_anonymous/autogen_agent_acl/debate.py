




from __future__ import annotations



from typing import Any ,Dict ,List ,Tuple

import re



from .agents import ask_once ,get_conn_from_agent ,make_agent

from .import medqa





ANSWER_RE =re .compile (r"\b(?:Answer|Final\s*Answer)\s*[:：]?\s*([A-Z])\b",re .IGNORECASE )





def _short (text :str ,limit :int )->str :

    text =""if text is None else str (text )

    return text [:limit ]+"..."if len (text )>limit else text





def _build_turn_prompt (sample :Dict [str ,Any ],labels :List [str ],coord_summary :str ,opponent_last :str )->str :

    prompt =(

    f"[Question]\n{_short(sample.get('question', ''), 2200)}\n\n"

    f"[Options]\n{_short(medqa.render_options(sample.get('options') or {}), 2200)}\n\n"

    f"Allowed choices: {', '.join(labels)}.\n\n"

    )

    if coord_summary :

        prompt +=f"[Coordinator Evidence (RAG Summary)]\n{_short(coord_summary, 2200)}\n\n"

    if opponent_last :

        prompt +=f"[Opponent Last Message]\n{_short(opponent_last, 1000)}\n\n"

    prompt +=(

    "Task:\n"

    "- Give 3-6 concise bullets focused on key medical discriminators.\n"

    "- Use Coordinator evidence when available, without fabricating citations.\n"

    "- End with exactly one line: Answer: <LETTER>\n"

    "<END>"

    )

    return prompt





def run_debate_groupchat (

AgentA ,

AgentB ,

sample :Dict [str ,Any ],

Ls :List [str ],

rounds :int ,

a_rationale :str ="",

b_rationale :str ="",

coord_summary :str ="",

max_tokens :int =192 ,

temperature :float =0.2 ,

)->Tuple [str ,str ,str ]:

    base_a ,model_a =get_conn_from_agent (AgentA )

    base_b ,model_b =get_conn_from_agent (AgentB )



    sys_a =(

    "You are Expert A in a medical multiple-choice debate. Be concise, clinical, "

    "and end with exactly one line: Answer: <LETTER>."

    )

    sys_b =(

    "You are Expert B in a medical multiple-choice debate. Be concise, clinical, "

    "and end with exactly one line: Answer: <LETTER>."

    )

    if a_rationale :

        sys_a +=f"\nEarlier rationale: {_short(a_rationale, 300)}"

    if b_rationale :

        sys_b +=f"\nEarlier rationale: {_short(b_rationale, 300)}"



    agent_a =make_agent ("DebateA",base_a ,model_a ,temperature =temperature ,top_p =1.0 ,system_message =sys_a )

    agent_b =make_agent ("DebateB",base_b ,model_b ,temperature =temperature ,top_p =1.0 ,system_message =sys_b )



    transcript :List [str ]=[]

    last_a =a_rationale or ""

    last_b =b_rationale or ""

    opponent_for_a =last_b

    opponent_for_b =last_a



    for idx in range (max (1 ,int (rounds ))):

        out_a =ask_once (

        agent_a ,

        [

        {"role":"system","content":sys_a },

        {"role":"user","content":_build_turn_prompt (sample ,Ls ,coord_summary ,opponent_for_a )},

        ],

        max_tokens =max_tokens ,

        temperature =temperature ,

        stop =["<END>"],

        )

        last_a =out_a or ""

        transcript .append (f"**Expert A (round {idx + 1})**\n```\n{last_a}\n```\n")

        opponent_for_b =last_a



        out_b =ask_once (

        agent_b ,

        [

        {"role":"system","content":sys_b },

        {"role":"user","content":_build_turn_prompt (sample ,Ls ,coord_summary ,opponent_for_b )},

        ],

        max_tokens =max_tokens ,

        temperature =temperature ,

        stop =["<END>"],

        )

        last_b =out_b or ""

        transcript .append (f"**Expert B (round {idx + 1})**\n```\n{last_b}\n```\n")

        opponent_for_a =last_b



        if last_a .startswith ("[ERROR]")and last_b .startswith ("[ERROR]"):

            break



    return "\n".join (transcript ),last_a ,last_b





def run_debate_fallback (ExpertA ,ExpertB ,sample ,Ls ,rounds :int )->Tuple [str ,str ,str ]:

    return "[Fallback Debate Skipped]","",""

