

import re ,json

from typing import Dict ,Any ,List ,Tuple ,Optional



_OPT_RE =re .compile (r"^([A-E])[\.\)]\s*(.*)\s*$")



def allowed_by_task_type (task_type :str )->List [str ]:

    t =(task_type or "").lower ().strip ()

    if t =="mcq3":

        return ["A","B","C"]

    if t =="mcq4":

        return ["A","B","C","D"]

    if t =="mcq5":

        return ["A","B","C","D","E"]

    return ["A","B","C","D"]



def load_jsonl (path :str )->List [Dict [str ,Any ]]:

    out =[]

    with open (path ,"r",encoding ="utf-8")as f :

        for line in f :

            line =line .strip ()

            if line :

                out .append (json .loads (line ))

    return out



def load_labels (test_labels :str )->Dict [str ,str ]:

    tbl :Dict [str ,str ]={}

    for r in load_jsonl (test_labels ):

        uid =r .get ("uid",r .get ("id",r .get ("idx")))

        gold =r .get ("gold")or r .get ("target")or r .get ("label")

        if uid is None or gold is None :

            continue

        tbl [str (uid )]=str (gold ).strip ().upper ()

    return tbl



def _strip_return_only (prompt :str )->str :

    if not prompt :

        return ""



    for key in ["Return only:","Return Only:","RETURN ONLY:"]:

        p =prompt .find (key )

        if p !=-1 :

            return prompt [:p ].rstrip ()

    return prompt .rstrip ()



def parse_prompt_to_question_options (prompt :str )->Tuple [str ,Dict [str ,str ]]:


    prompt =_strip_return_only (prompt or "")

    lines =[ln .rstrip ()for ln in prompt .splitlines ()]





    opt_idx =None

    for i ,ln in enumerate (lines ):

        if ln .strip ().lower ().startswith ("options"):

            opt_idx =i

            break



    if opt_idx is None :



        return prompt .strip (),{}





    q_lines =lines [:opt_idx ]



    if q_lines and q_lines [0 ].strip ().lower ().startswith ("question"):

        q_lines =q_lines [1 :]

    question ="\n".join ([x for x in q_lines ]).strip ()





    opts :Dict [str ,str ]={}

    cur_k :Optional [str ]=None

    cur_buf :List [str ]=[]



    def flush ():

        nonlocal cur_k ,cur_buf

        if cur_k :

            opts [cur_k ]="\n".join ([x for x in cur_buf ]).strip ()

        cur_k ,cur_buf =None ,[]



    for ln in lines [opt_idx +1 :]:

        s =ln .strip ()

        if not s :



            if cur_k :

                cur_buf .append ("")

            continue

        m =_OPT_RE .match (s )

        if m :

            flush ()

            cur_k =m .group (1 ).upper ()

            cur_buf =[m .group (2 )]

            continue



        if cur_k :

            cur_buf .append (s )



    flush ()

    return question ,opts



def load_unified (test_inputs :str ,test_labels :str )->List [Dict [str ,Any ]]:

    labels =load_labels (test_labels )

    rows =load_jsonl (test_inputs )

    out :List [Dict [str ,Any ]]=[]

    for i ,r in enumerate (rows ):

        uid =str (r .get ("uid",r .get ("id",i )))

        dataset =(r .get ("dataset")or "").lower ()

        task_type =r .get ("task_type")or "mcq4"

        Ls =allowed_by_task_type (task_type )



        prompt =r .get ("prompt")or ""

        q ,opts =parse_prompt_to_question_options (prompt )



        out .append ({

        "id":uid ,

        "uid":uid ,

        "dataset":dataset ,

        "task_type":task_type ,

        "Ls":Ls ,

        "question":q ,

        "options":opts ,

        "gold":labels .get (uid ,None ),

        "raw_prompt":prompt ,

        "messages":r .get ("messages",None ),

        })

    return out

