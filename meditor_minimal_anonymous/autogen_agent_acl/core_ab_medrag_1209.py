




from __future__ import annotations

from typing import List ,Dict ,Any ,Optional ,Tuple



import os

import json

import argparse

import re

from collections import Counter



from .coordinator_medrag import MedRAGCoordinator

from .import medqa

from .utils import ensure_dir ,write_text ,now_str

from .agents import make_agent ,ask_once

from .voting import run_votes ,majority

from .debate import run_debate_groupchat ,run_debate_fallback

from .judge import judge_vote ,judge_vote_bidirectional ,short_critic_line ,extract_judge_meta

from .router import DefaultRouter



_DEBATE_ANSWER_RE =re .compile (r"\b(?:Answer|Final\s*Answer)\s*[:：]\s*([A-Z])\b",re .IGNORECASE )





def _extract_pick_from_debate (text :str ,Ls :List [str ])->str :


    if not text :

        return ""

    m =_DEBATE_ANSWER_RE .search (str (text ))

    if not m :

        return ""

    c =m .group (1 ).upper ()

    return c if c in set (Ls )else ""





ANSWER_RE_ANY =re .compile (r"\b(?:Answer|Final\s*Answer|Pick)\s*[:：]\s*([A-Z])\b",re .IGNORECASE )









_JMETA_RE =re .compile (r"__JUDGE_META__\s*\n(\{.*?\})\s*\n__END_META__",re .S )

_CITE_RE =re .compile (r"\[(?:A|B)\d+\]")





def _judge_max_gap (judge_raw :str )->Optional [float ]:


    if not judge_raw :

        return None

    gaps :List [float ]=[]

    for m in _JMETA_RE .finditer (judge_raw ):

        try :

            meta =json .loads (m .group (1 ))

            if str (meta .get ("winner","")).upper ()!="A":

                continue

            sa =meta .get ("score_A",None )

            sb =meta .get ("score_B",None )

            if (sa is not None )and (sb is not None ):

                gaps .append (abs (float (sa )-float (sb )))

        except Exception :

            pass

    return min (gaps )if gaps else None



def _judge_has_cites (judge_raw :str )->bool :


    return bool (_CITE_RE .search (judge_raw or ""))





def _agent_failed (raws :List [Optional [str ]])->bool :

    non_empty =[r for r in raws if r ]

    if not non_empty :

        return False

    return all (

    r .startswith ("[ERROR]")or r .startswith ("[SKIP_AFTER_ERROR]")

    for r in non_empty

    )





def _short_text (s :str ,n :int =2000 )->str :

    s =""if s is None else str (s )

    return (s [:n ]+"…")if len (s )>n else s







_SEC_A =re .compile (r"\[Evidence favoring Candidate A\](.*?)(?:\n\[|$)",re .S |re .I )

_SEC_B =re .compile (r"\[Evidence favoring Candidate B\](.*?)(?:\n\[|$)",re .S |re .I )





def _count_evidence_bullets (section_text :str )->int :

    if not section_text :

        return 0

    lines =[ln .strip ()for ln in section_text .splitlines ()if ln .strip ()]

    if len (lines )==1 and lines [0 ].lower ().startswith ("none"):

        return 0

    return sum (1 for ln in lines if ln .startswith ("-"))





def _coord_bullet_counts (coord_summary :str )->Tuple [int ,int ]:


    if not coord_summary :

        return 0 ,0

    mA =_SEC_A .search (coord_summary )

    mB =_SEC_B .search (coord_summary )

    a_cnt =_count_evidence_bullets (mA .group (1 )if mA else "")

    b_cnt =_count_evidence_bullets (mB .group (1 )if mB else "")

    return a_cnt ,b_cnt





def coord_rule_decision (coord_summary :str ,min_gap :int =2 )->str :


    if not coord_summary :

        return ""

    mA =_SEC_A .search (coord_summary )

    mB =_SEC_B .search (coord_summary )

    a_cnt =_count_evidence_bullets (mA .group (1 )if mA else "")

    b_cnt =_count_evidence_bullets (mB .group (1 )if mB else "")

    if (a_cnt >=int (min_gap ))and (b_cnt ==0 ):

        return "A"

    if (b_cnt >=int (min_gap ))and (a_cnt ==0 ):

        return "B"

    return ""





def _coord_top_scores (coord_info :Optional [Dict [str ,Any ]])->Tuple [Optional [float ],Optional [float ]]:

    if not isinstance (coord_info ,dict ):

        return None ,None

    try :

        topA =coord_info .get ("topA",coord_info .get ("top_score_A",None ))

        topB =coord_info .get ("topB",coord_info .get ("top_score_B",None ))

        if topA is None :

            scores_a =(coord_info .get ("scores")or {}).get ("A")or coord_info .get ("scores_A")or []

            topA =max ([float (x )for x in scores_a ],default =None )

        if topB is None :

            scores_b =(coord_info .get ("scores")or {}).get ("B")or coord_info .get ("scores_B")or []

            topB =max ([float (x )for x in scores_b ],default =None )

        return (

        float (topA )if topA is not None else None ,

        float (topB )if topB is not None else None ,

        )

    except Exception :

        return None ,None





def _coord_score_rule_winner (

coord_info :Optional [Dict [str ,Any ]],

*,

min_score :float ,

gap :float ,

)->str :

    topA ,topB =_coord_top_scores (coord_info )

    if (topA is not None )and (topB is not None ):

        if (topA -topB )>=float (gap )and topA >=float (min_score ):

            return "A"

        if (topB -topA )>=float (gap )and topB >=float (min_score ):

            return "B"

        return ""

    if (topA is not None )and topA >=float (min_score ):

        return "A"

    if (topB is not None )and topB >=float (min_score ):

        return "B"

    return ""





def _extract_pick (text :str ,Ls :List [str ])->Optional [str ]:

    if not text :

        return None

    m =ANSWER_RE_ANY .search (str (text ))

    if not m :

        return None

    x =m .group (1 ).upper ()

    return x if x in Ls else None





def _brief_rationale (agent ,sample :Dict [str ,Any ],pick :str ,Ls :List [str ],max_tokens :int =192 )->str :


    if not pick :

        return ""

    q =_short_text (sample .get ("question",""),2200 )

    opts =_short_text (medqa .render_options (sample .get ("options")or {}),1800 )

    msgs =[

    {"role":"system","content":(

    "You are a senior physician writing a standardized rationale for a medical MCQ. "

    "Be concise and evidence-oriented. Do NOT repeat the full question/options. "

    "Write EXACTLY 4 bullets in this order:\n"

    "- Key clinical facts (1 sentence)\n"

    "- Mechanism/diagnostic principle (1 sentence)\n"

    "- Why the chosen option fits (1 sentence)\n"

    "- Why the strongest alternative(s) are less likely (1 sentence)\n"

    "Then output a final line: Answer: <LETTER>\n"

    "End with <END>."

    )},

    {"role":"user","content":(

    f"[Question]\n{q}\n\n"

    f"[Options]\n{opts}\n\n"

    f"You must defend choice {pick}.\n"

    "Output now.\n<END>"

    )},

    ]

    out =ask_once (agent ,msgs ,max_tokens =max_tokens ,temperature =0.0 ,stop =["<END>"])

    return _short_text (out or "",1200 )



def _solve_final (

Solver ,

sample :Dict [str ,Any ],

Ls :List [str ],

coord_summary :str ,

a_text :str ,

b_text :str ,

max_tokens :int =256 ,

)->Optional [str ]:


    if Solver is None :

        return None

    q =_short_text (sample .get ("question",""),2400 )

    opts =_short_text (medqa .render_options (sample .get ("options")or {}),1800 )

    sysS =medqa .sys_judge_prometheus (Ls )



    msgs =[

    {"role":"system","content":sysS },

    {"role":"user","content":(

    f"[Question]\n{q}\n\n"

    f"[Options]\n{opts}\n\n"

    f"[Expert A]\n{(a_text or '').strip()}\n\n"

    f"[Expert B]\n{(b_text or '').strip()}\n\n"

    f"[Coordinator Evidence]\n{(coord_summary or '').strip()}\n\n"

    "Output only as:\nAnswer: <LETTER>\n<END>"

    )},

    ]

    out =ask_once (Solver ,msgs ,max_tokens =int (max_tokens ),temperature =0.0 ,stop =["<END>"])

    out2 =medqa .take_until_answer (out or "")

    lab =medqa .parse_label_from_answer (out2 ,labels =tuple (Ls ))or medqa .parse_label (out2 )

    return lab if (lab in set (Ls ))else None





def _recover_agreement_with_solver (

Solver ,

medrag_coord ,

sample :Dict [str ,Any ],

Ls :List [str ],

agreed_pick :str ,

a_text :str ,

b_text :str ,

*,

solver_max :int ,

medrag_min_score :float ,

medrag_max_snippets :int ,

medrag_score_margin :float ,

meta :Optional [Dict [str ,Any ]]=None ,

)->Tuple [Optional [str ],str ,Optional [Dict [str ,Any ]]]:


    coord_summary =""

    coord_info :Optional [Dict [str ,Any ]]=None



    if medrag_coord is not None and agreed_pick :

        try :

            coord_summary ,coord_info =medrag_coord .build_coord_summary_for_judge (

            sample ,

            a_pick =agreed_pick ,

            b_pick =agreed_pick ,

            a_rationale =a_text ,

            b_rationale =b_text ,

            min_score =float (medrag_min_score ),

            max_snippets =int (medrag_max_snippets ),

            score_margin =float (medrag_score_margin ),

            meta =meta or {},

            )

        except Exception as e :

            coord_summary =f"[ERROR][AgreementCoordinator]{e}"

            coord_info =None



    pred =_solve_final (

    Solver ,

    sample ,

    Ls ,

    coord_summary ,

    a_text ,

    b_text ,

    max_tokens =int (solver_max ),

    )

    return pred ,coord_summary ,coord_info





def judge_vote_bidirectional_sc (

Judge ,sample ,Ls ,

a_pick :str ,b_pick :str ,

lastA :str ,lastB :str ,

criticA :str ,criticB :str ,

coord_summary :str ,

rubric_text :str ,

judge_max :int ,

delta_abstain :float ,

default_winner :str ,

n :int =1 ,

)->Tuple [str ,str ]:


    n =max (1 ,int (n ))

    votes :List [str ]=[]

    raws :List [str ]=[]



    for _ in range (n ):

        w ,raw =judge_vote_bidirectional (

        Judge ,sample ,Ls ,

        a_pick =a_pick ,b_pick =b_pick ,

        lastA =lastA ,lastB =lastB ,

        criticA =criticA ,criticB =criticB ,

        coord_summary =coord_summary ,

        rubric_text =rubric_text ,

        judge_max =judge_max ,

        delta_abstain =delta_abstain ,

        default_winner =default_winner ,

        )

        votes .append (w if w in ("A","B")else "ABSTAIN")

        raws .append (raw or "")



    c =Counter (votes )

    if c ["A"]>c ["B"]and c ["A"]>c ["ABSTAIN"]:

        final ="A"

    elif c ["B"]>c ["A"]and c ["B"]>c ["ABSTAIN"]:

        final ="B"

    else :

        final =""



    debug ={"sc_n":n ,"sc_votes":dict (c ),"sc_final":final or "ABSTAIN"}

    packed ="\n\n".join (raws )+"\n\n__SC_DEBUG__\n"+json .dumps (debug ,ensure_ascii =False )

    return final ,packed



















































def _build_case_md (

idx :int ,

sample :Dict [str ,Any ],

Ls :List [str ],

a_votes :List [Optional [str ]],

b_votes :List [Optional [str ]],

a_seed :Optional [str ],

b_seed :Optional [str ],

a_pick_final :Optional [str ],

b_pick_final :Optional [str ],

a_rat_txt :str ,

b_rat_txt :str ,

debate_md :str ,

j_raw :str ,

pred :Optional [str ],

gold :Optional [str ],

acc :float ,

)->str :

    sid =sample .get ("id",f"{idx}")

    lines :List [str ]=[]

    lines .append (f"# Case {idx} — id={sid}")

    lines .append ("")

    lines .append (f"- Time: {now_str()}")

    lines .append (f"- Labels: {', '.join(Ls)}")

    lines .append (f"- Gold: {gold or '-'}")

    lines .append (f"- Pred: {pred or '-'}")

    lines .append (f"- Acc so far: {acc:.3f}")

    lines .append ("")

    lines .append ("## Question")

    lines .append ("```")

    lines .append ((sample .get ("question")or "").rstrip ())

    lines .append ("```")

    lines .append ("")

    lines .append ("## Options")

    lines .append ("```")

    lines .append (medqa .render_options (sample ["options"]))

    lines .append ("```")

    lines .append ("")

    lines .append ("## Picks")

    lines .append ("```")

    lines .append (f"A seed={a_seed or '-'}  A final={a_pick_final or '-'}")

    lines .append (f"B seed={b_seed or '-'}  B final={b_pick_final or '-'}")

    lines .append ("```")

    lines .append ("")

    lines .append ("## Votes")

    lines .append ("```")

    lines .append (f"A votes: {a_votes}")

    lines .append (f"B votes: {b_votes}")

    lines .append ("```")

    lines .append ("")

    if a_rat_txt :

        lines .append ("## Rationale A")

        lines .append ("```")

        lines .append (a_rat_txt )

        lines .append ("```")

        lines .append ("")

    if b_rat_txt :

        lines .append ("## Rationale B")

        lines .append ("```")

        lines .append (b_rat_txt )

        lines .append ("```")

        lines .append ("")

    if debate_md :

        lines .append ("## Debate")

        lines .append (debate_md )

        lines .append ("")

    if j_raw :

        lines .append ("## Judge Output")

        lines .append ("```")

        lines .append (j_raw )

        lines .append ("```")

        lines .append ("")

    lines .append ("## Decision")

    lines .append ("```")

    lines .append (f"Pred: {pred or '-'}   Gold: {gold or '-'}")

    lines .append ("```")

    lines .append ("")

    return "\n".join (lines )





def _load_baseline (path :str )->Dict [str ,str ]:


    if not path :

        return {}

    if not os .path .isfile (path ):

        print (f"[WARN] baseline preds not found: {path}",flush =True )

        return {}



    valid =set ("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    table :Dict [str ,str ]={}

    with open (path ,"r",encoding ="utf-8")as f :

        for line in f :

            line =line .strip ()

            if not line :

                continue

            try :

                obj =json .loads (line )

            except Exception :

                continue



            key =obj .get ("uid",None )

            if key is None :

                key =obj .get ("id",None )

            if key is None :

                key =obj .get ("sid",None )

            if key is None :

                key =obj .get ("idx",None )



            lab =(

            obj .get ("pred")

            or obj .get ("prediction")

            or obj .get ("pred_letter")

            or obj .get ("answer")

            or obj .get ("answer_choice")

            or obj .get ("label")

            or obj .get ("output")

            )

            if lab is None :

                idx_lab =obj .get ("pred_idx",obj .get ("answer_idx",None ))

                if isinstance (idx_lab ,int )and 0 <=idx_lab <26 :

                    lab =chr (ord ("A")+idx_lab )



            if key is None or not isinstance (lab ,str ):

                continue



            lab_text =lab .strip ().upper ()

            m_lab =re .search (r"(?:ANSWER|FINAL\s*ANSWER|PRED(?:ICTION)?|LABEL)\s*[:：]\s*([A-Z])\b",lab_text )

            if not m_lab :

                m_lab =re .fullmatch (r"\s*([A-Z])\s*",lab_text )

            if not m_lab :

                m_lab =re .search (r"^\s*([A-Z])[\.\):：]",lab_text )

            if not m_lab :

                continue

            lab =m_lab .group (1 ).upper ()

            if lab not in valid :

                continue

            table [str (key )]=lab



    print (f"[INFO] loaded baseline from {path}, {len(table)} entries.",flush =True )

    return table



def main (argv :Optional [List [str ]]=None ):

    ap =argparse .ArgumentParser ()



    ap .add_argument ("--a-baseline-preds",type =str ,default ="")

    ap .add_argument ("--b-baseline-preds",type =str ,default ="")



    ap .add_argument ("--data",required =True )



    ap .add_argument ("--labels",type =str ,default ="",help ="Unified benchmark labels jsonl (optional).")

    ap .add_argument ("--run-dir",required =True )



    ap .add_argument ("--a-base",required =True )

    ap .add_argument ("--a-model",required =True )

    ap .add_argument ("--b-base",required =True )

    ap .add_argument ("--b-model",required =True )



    ap .add_argument ("--j-base",default ="")

    ap .add_argument ("--j-model",default ="")





    ap .add_argument ("--solver-base",type =str ,default ="",help ="Optional final solver base_url")

    ap .add_argument ("--solver-model",type =str ,default ="",help ="Optional final solver model name")

    ap .add_argument ("--solver-max",type =int ,default =256 )

    ap .add_argument ("--solver-only-on-abstain",action ="store_true",help ="Only run solver when judge ABSTAINS/fallback")

    ap .add_argument ("--solver-on-agree",action ="store_true",

    help ="Run the open solver on agreement cases to measure agreement blind spots and recovery.")



    ap .add_argument ("--a-repeats",type =int ,default =10 )

    ap .add_argument ("--a-temp",type =float ,default =0.7 )

    ap .add_argument ("--b-repeats",type =int ,default =5 )

    ap .add_argument ("--b-temp",type =float ,default =0.3 )



    ap .add_argument ("--vote-tokens",type =int ,default =12 )

    ap .add_argument ("--max-samples",type =int ,default =None )



    ap .add_argument ("--debate-rounds",type =int ,default =3 )



    ap .add_argument ("--disable-debate",action ="store_true",

    help ="Ablation: disable debate even when --debate-rounds > 0.")

    ap .add_argument ("--coord-rule-min-gap",type =int ,default =2 ,

    help ="Conservative override: min evidence bullets required to pick one side by Coordinator rule.")

    ap .add_argument ("--no-bidirectional-judge",action ="store_true",

    help ="Disable bidirectional judging (positional-bias mitigation).")

    ap .add_argument ("--judge-delta-abstain",type =float ,default =1.5 ,

    help ="If Judge outputs scores and |scoreA-scoreB| < delta, abstain and defer to solver/evidence fallback.")



    ap .add_argument ("--judge-flip-min-gap",type =float ,default =3.0 ,

    help ="Allow flipping to A only if max(|score_A-score_B|) >= this threshold.")

    ap .add_argument ("--judge-require-cites",action ="store_true",

    help ="Require evidence marker cites like [A12]/[B3] in judge output to allow flipping to A.")

    ap .add_argument ("--flip-require-coord-evidence",action ="store_true",

    help ="Only allow flips to A when coordinator evidence strongly favors A.")

    ap .add_argument ("--flip-min-a-bullets",type =int ,default =2 ,

    help ="Min number of bullets in [Evidence favoring Candidate A] to allow flip.")

    ap .add_argument ("--flip-max-b-bullets",type =int ,default =0 ,

    help ="Max number of bullets in [Evidence favoring Candidate B] allowed when flipping to A.")



    ap .add_argument ("--no-llm-judge",action ="store_true")

    ap .add_argument ("--judge-max",type =int ,default =128 )

    ap .add_argument ("--judge-self-consistency",type =int ,default =1 )



    ap .add_argument ("--judge-rubric-file",type =str ,default ="")

    ap .add_argument ("--judge-rubric-inline",type =str ,default ="")



    ap .add_argument ("--use-critics",action ="store_true")

    ap .add_argument ("--critic-tokens",type =int ,default =96 )

    ap .add_argument ("--enable-coord-rules",action ="store_true",

    help ="Enable coordinator-based rule override before LLM judge.")



    ap .add_argument ("--use-medrag-judge",action ="store_true")

    ap .add_argument ("--medrag-device-id",type =int ,default =3 )

    ap .add_argument ("--medrag-mode",type =str ,default ="rag_auto",

    choices =["cot_only","rag_auto","rag_textbooks","rag_pubmed","rag_wikipedia","rag_mednosp","rag_medcorp"])

    ap .add_argument ("--medrag-llm-name",type =str ,default =os .getenv ("MEDRAG_LLM_NAME",""))

    ap .add_argument ("--medrag-k",type =int ,default =32 )

    ap .add_argument ("--medrag-retriever",type =str ,default ="RRF-2")

    ap .add_argument (

    "--medrag-corpus-dir",

    type =str ,

    default =os .getenv ("MEDRAG_CORPUS_DIR","./corpus"),

    )

    ap .add_argument ("--medrag-min-score",type =float ,default =-18.7 )

    ap .add_argument ("--medrag-score-gap",type =float ,default =4.0 )

    ap .add_argument ("--medrag-score-margin",type =float ,default =4.0 )

    ap .add_argument ("--medrag-max-snippets",type =int ,default =6 )

    ap .add_argument ("--custom-rag-textbooks-corpus-dir",type =str ,default =os .getenv ("CUSTOM_RAG_TEXTBOOKS_CORPUS_DIR",""))

    ap .add_argument ("--custom-rag-textbooks-bm25-dir",type =str ,default =os .getenv ("CUSTOM_RAG_TEXTBOOKS_BM25_DIR",""))

    ap .add_argument ("--custom-rag-textbooks-dense-dir",type =str ,default =os .getenv ("CUSTOM_RAG_TEXTBOOKS_DENSE_DIR",""))

    ap .add_argument ("--custom-rag-textbooks-dense-device",type =str ,default =os .getenv ("CUSTOM_RAG_TEXTBOOKS_DENSE_DEVICE",""))

    ap .add_argument ("--custom-rag-textbooks-fusion",type =str ,default =os .getenv ("CUSTOM_RAG_TEXTBOOKS_FUSION","rrf"),

    choices =["rrf","sparse"])

    ap .add_argument ("--custom-rag-textbooks-query-mode",type =str ,

    default =os .getenv ("CUSTOM_RAG_TEXTBOOKS_QUERY_MODE","question_plus_options"),

    choices =["question_only","question_plus_options"])

    ap .add_argument ("--custom-rag-textbooks-top-k",type =int ,default =int (os .getenv ("CUSTOM_RAG_TEXTBOOKS_TOP_K","32")))

    ap .add_argument ("--custom-rag-textbooks-sparse-k",type =int ,default =int (os .getenv ("CUSTOM_RAG_TEXTBOOKS_SPARSE_K","32")))

    ap .add_argument ("--custom-rag-textbooks-dense-k",type =int ,default =int (os .getenv ("CUSTOM_RAG_TEXTBOOKS_DENSE_K","32")))

    ap .add_argument ("--custom-rag-textbooks-rrf-k",type =int ,default =int (os .getenv ("CUSTOM_RAG_TEXTBOOKS_RRF_K","60")))

    ap .add_argument ("--rag-summary-base",type =str ,default =os .getenv ("RAG_API_BASE",""))

    ap .add_argument ("--rag-summary-model",type =str ,default =os .getenv ("RAG_MODEL_NAME",""))

    ap .add_argument ("--rag-summary-api-key",type =str ,default =os .getenv ("RAG_API_KEY",os .getenv ("OPENAI_API_KEY","EMPTY")))

    ap .add_argument ("--remote-rag-base",type =str ,default =os .getenv ("REMOTE_RAG_BASE",""))

    ap .add_argument ("--remote-rag-model",type =str ,default =os .getenv ("REMOTE_RAG_MODEL",""))

    ap .add_argument ("--remote-rag-api-key",type =str ,default =os .getenv ("REMOTE_RAG_API_KEY",os .getenv ("OPENAI_API_KEY","EMPTY")))



    ap .add_argument ("--jsonl-text-chars",type =int ,default =2000 )

    ap .add_argument ("--md-all",action ="store_true")



    args =ap .parse_args (argv )



    router =DefaultRouter ()



    ensure_dir (args .run_dir )

    md_dir =os .path .join (args .run_dir ,"md")

    ensure_dir (md_dir )





    rubric_text :str =""

    if args .judge_rubric_inline :

        rubric_text =str (args .judge_rubric_inline ).strip ()

    elif args .judge_rubric_file and os .path .isfile (args .judge_rubric_file ):

        rubric_text =open (args .judge_rubric_file ,"r",encoding ="utf-8").read ().strip ()





    a_baseline =_load_baseline (args .a_baseline_preds )

    b_baseline =_load_baseline (args .b_baseline_preds )



    def _lookup_baseline (tbl :Dict [str ,str ],idx :int ,sid :Any )->Optional [str ]:


        if not tbl :

            return None



        if sid is not None :

            k =str (sid )

            if k in tbl :

                return tbl [k ]



        k2 =str (idx )

        if k2 in tbl :

            return tbl [k2 ]



        if isinstance (sid ,str )and sid .isdigit ():

            return tbl .get (str (int (sid )))



        return None



    if args .labels :

        from .bench_io import load_unified

        samples =load_unified (args .data ,args .labels )

    else :

        samples =medqa .load (args .data )

    if args .max_samples is not None :

        samples =samples [:args .max_samples ]

    total =len (samples )



    sysA =(

    "You are Expert A, a senior physician answering USMLE-style multiple-choice questions. "

    "Be concise. End with 'Answer: <LETTER>'."

    )

    sysB =(

    "You are Expert B, a senior physician answering USMLE-style multiple-choice questions. "

    "Be concise. End with 'Answer: <LETTER>'."

    )

    sysJ =(

    "You are a STRICT pairwise evaluator, not a solver. "

    "You must NOT answer the question yourself. "

    "You must choose which candidate (A or B) is better supported by the PROVIDED evidence and rationales. "

    "You must treat a_pick and b_pick as fixed; do NOT propose any new option letter. "

    "If evidence is insufficient or both are weak, output ABSTAIN.\n\n"

    "Return ONE JSON object with keys score_A, score_B, winner, why, then <END>. "

    "winner must be \"A\", \"B\", or \"ABSTAIN\"."

    )



    ExpertA =make_agent ("ExpertA",args .a_base ,args .a_model ,temperature =args .a_temp ,system_message =sysA )

    ExpertB =make_agent ("ExpertB",args .b_base ,args .b_model ,temperature =args .b_temp ,system_message =sysB )



    def _get_agent_by_model_tag (model_tag :str ):


        tag =(model_tag or "").strip ()

        if not tag :

            raise ValueError ("empty model_tag")



        if tag ==args .a_model :

            return ExpertA ,args .a_temp

        if tag ==args .b_model :

            return ExpertB ,args .b_temp



        raise ValueError (f"Unknown model_tag={tag}. Only have: {args.a_model}, {args.b_model}")



    Judge =None

    if (not args .no_llm_judge )and args .j_base and args .j_model :

        j_temp =0.2 if (args .judge_self_consistency and args .judge_self_consistency >1 )else 0.0

        Judge =make_agent ("Judge",args .j_base ,args .j_model ,temperature =j_temp ,system_message =sysJ )



    Solver =None

    if args .solver_base and args .solver_model :



        Solver =make_agent ("Solver",args .solver_base ,args .solver_model ,temperature =0.0 ,system_message ="")



    medrag_coord :Optional [MedRAGCoordinator ]=None

    if args .use_medrag_judge :

        medrag_log =os .path .join (args .run_dir ,"medrag","coord_retrieval.jsonl")

        device_id =int (os .getenv ("MEDRAG_DEVICE_ID",str (args .medrag_device_id )))

        medrag_coord =MedRAGCoordinator (

        mode =args .medrag_mode ,

        llm_name =args .medrag_llm_name ,

        k =args .medrag_k ,

        corpus_dir =args .medrag_corpus_dir ,

        device_id =device_id ,

        log_path =medrag_log ,

        retriever_name =args .medrag_retriever ,

        custom_rag_textbooks_corpus_dir =args .custom_rag_textbooks_corpus_dir ,

        custom_rag_textbooks_bm25_dir =args .custom_rag_textbooks_bm25_dir ,

        custom_rag_textbooks_dense_dir =args .custom_rag_textbooks_dense_dir ,

        custom_rag_textbooks_dense_device =args .custom_rag_textbooks_dense_device ,

        custom_rag_textbooks_fusion =args .custom_rag_textbooks_fusion ,

        custom_rag_textbooks_query_mode =args .custom_rag_textbooks_query_mode ,

        custom_rag_textbooks_top_k =args .custom_rag_textbooks_top_k ,

        custom_rag_textbooks_sparse_k =args .custom_rag_textbooks_sparse_k ,

        custom_rag_textbooks_dense_k =args .custom_rag_textbooks_dense_k ,

        custom_rag_textbooks_rrf_k =args .custom_rag_textbooks_rrf_k ,

        summary_base_url =args .rag_summary_base ,

        summary_model =args .rag_summary_model ,

        summary_api_key =args .rag_summary_api_key ,

        remote_rag_base_url =args .remote_rag_base ,

        remote_rag_model =args .remote_rag_model ,

        remote_rag_api_key =args .remote_rag_api_key ,

        )

        print (

        f"[*] MedRAGCoordinator: mode={args.medrag_mode} retriever={args.medrag_retriever} "

        f"k={args.medrag_k} device_id={device_id}"

        )

        if args .custom_rag_textbooks_corpus_dir or args .custom_rag_textbooks_bm25_dir or args .custom_rag_textbooks_dense_dir :

            print (

            "[*] Custom textbooks RAG enabled: "

            f"corpus_dir={args.custom_rag_textbooks_corpus_dir or '<unset>'} "

            f"bm25_dir={args.custom_rag_textbooks_bm25_dir or '<auto>'} "

            f"dense_dir={args.custom_rag_textbooks_dense_dir or '<auto>'} "

            f"fusion={args.custom_rag_textbooks_fusion}",

            flush =True ,

            )

        if args .rag_summary_base and args .rag_summary_model :

            print (

            f"[*] RAG coordinator summary endpoint: base={args.rag_summary_base} model={args.rag_summary_model}",

            flush =True ,

            )

        if args .remote_rag_base and args .remote_rag_model :

            print (

            f"[*] Remote full RAG endpoint: base={args.remote_rag_base} model={args.remote_rag_model}",

            flush =True ,

            )

        print (f"[*] MedRAG log => {medrag_log}",flush =True )



    preds_path =os .path .join (args .run_dir ,"preds.jsonl")

    judge_trace_path =os .path .join (args .run_dir ,"judge_traces.jsonl")

    correct =0



    coord_used_n =0

    coord_summary_chars :List [int ]=[]

    coord_too_short_n =0



    with open (preds_path ,"w",encoding ="utf-8")as f_out :

        for idx ,sample in enumerate (samples ):

            sid =sample .get ("id",idx )

            gold =sample .get ("gold")

            Ls =sample .get ("Ls")or medqa .LABELS



            baseA =_lookup_baseline (a_baseline ,idx ,sid )

            baseB =_lookup_baseline (b_baseline ,idx ,sid )

            if (baseA is not None )and (baseA not in set (Ls )):

                baseA =None

            if (baseB is not None )and (baseB not in set (Ls )):

                baseB =None

































            pre =router .decide_pre_vote (sample =sample ,idx =idx ,sid =sid ,baseA =baseA ,baseB =baseB )





            if pre .decision =="BASELINE_AGREE"and pre .pred :

                pred =pre .pred

                used_agreement_recovery =False

                agreement_recovery_pred =""

                agreement_recovery_changed =False

                agreement_recovery_coord_info =None

                agreement_recovery_summary =""



                if args .solver_on_agree and (Solver is not None ):

                    a_rat_txt =_brief_rationale (ExpertA ,sample ,baseA or "",Ls ,max_tokens =256 )if baseA else ""

                    b_rat_txt =_brief_rationale (ExpertB ,sample ,baseB or "",Ls ,max_tokens =256 )if baseB else ""

                    rec_pred ,rec_summary ,rec_coord_info =_recover_agreement_with_solver (

                    Solver ,

                    medrag_coord ,

                    sample ,

                    Ls ,

                    agreed_pick =pre .pred ,

                    a_text =a_rat_txt ,

                    b_text =b_rat_txt ,

                    solver_max =args .solver_max ,

                    medrag_min_score =args .medrag_min_score ,

                    medrag_max_snippets =args .medrag_max_snippets ,

                    medrag_score_margin =args .medrag_score_margin ,

                    meta ={"idx":idx ,"id":sid ,"path":"baseline_agree"},

                    )

                    used_agreement_recovery =True

                    agreement_recovery_pred =rec_pred or ""

                    agreement_recovery_changed =bool (rec_pred and rec_pred in set (Ls )and rec_pred !=pre .pred )

                    agreement_recovery_coord_info =rec_coord_info

                    agreement_recovery_summary =rec_summary or ""

                    if rec_pred and rec_pred in set (Ls ):

                        pred =rec_pred



                    if agreement_recovery_summary :

                        coord_used_n +=1

                        _clen =len (agreement_recovery_summary or "")

                        coord_summary_chars .append (_clen )

                        if _clen <160 :

                            coord_too_short_n +=1



                correct_flag =int (bool (gold and pred and gold ==pred ))

                correct +=correct_flag

                acc =correct /(idx +1 )



                rec ={

                "idx":idx ,"id":sid ,"gold":gold ,"pred":pred ,"correct":bool (correct_flag ),

                "used_baseline_agree":True ,

                "baselineA":baseA ,"baselineB":baseB ,

                "A_seed":baseA ,"B_seed":baseB ,

                "A_final":baseA ,"B_final":baseB ,

                "used_debate":False ,"used_judge":False ,

                "coord_info":agreement_recovery_coord_info ,"coord_summary":agreement_recovery_summary ,

                "judge_winner":None ,"judge_debug":"",

                "judge_meta":{},"judge_body":"","judge_raw_full":"",

                "used_agreement_recovery":used_agreement_recovery ,

                "agreement_recovery_pred":agreement_recovery_pred ,

                "agreement_recovery_changed":agreement_recovery_changed ,

                }

                f_out .write (json .dumps (rec ,ensure_ascii =False )+"\n")



                if args .md_all :

                    md_txt =_build_case_md (

                    idx ,sample ,Ls ,

                    a_votes =[],b_votes =[],

                    a_seed =baseA ,b_seed =baseB ,

                    a_pick_final =baseA ,b_pick_final =baseB ,

                    a_rat_txt ="",b_rat_txt ="",

                    debate_md ="",j_raw ="",

                    pred =pred ,gold =gold ,acc =acc ,

                    )

                    write_text (os .path .join (md_dir ,f"case_{idx:06d}_{sid}.md"),md_txt )



                print (f"[{idx + 1}/{total}] acc={acc:.3f} BASE={pred} (baseline agree)",flush =True )

                continue



            baseline_disagree =(pre .decision =="BASELINE_DISAGREE")



            a_votes :List [Optional [str ]]=[]

            b_votes :List [Optional [str ]]=[]

            a_failed =False

            b_failed =False



            if baseline_disagree :

                a_seed =pre .a_seed

                b_seed =pre .b_seed



                a_rat_txt =_brief_rationale (ExpertA ,sample ,a_seed or "",Ls ,max_tokens =256 )

                b_rat_txt =_brief_rationale (ExpertB ,sample ,b_seed or "",Ls ,max_tokens =256 )

            else :



                a_votes ,a_raws =run_votes (

                ExpertA ,sample ,Ls ,

                repeats =args .a_repeats ,

                vote_tokens =args .vote_tokens ,

                temp =args .a_temp ,

                )

                b_votes ,b_raws =run_votes (

                ExpertB ,sample ,Ls ,

                repeats =args .b_repeats ,

                vote_tokens =args .vote_tokens ,

                temp =args .b_temp ,

                )

                a_failed =_agent_failed (a_raws )

                b_failed =_agent_failed (b_raws )

                a_seed =majority ([x for x in a_votes if x ])

                b_seed =majority ([x for x in b_votes if x ])





                a_rat_txt =_brief_rationale (ExpertA ,sample ,a_seed or "",Ls ,max_tokens =256 )if a_seed else ""

                b_rat_txt =_brief_rationale (ExpertB ,sample ,b_seed or "",Ls ,max_tokens =256 )if b_seed else ""



            post =router .decide_post_seed (

            a_seed =a_seed ,

            b_seed =b_seed ,

            a_failed =a_failed ,

            b_failed =b_failed ,

            disable_debate =args .disable_debate ,

            debate_rounds =args .debate_rounds ,

            coord_enabled =(medrag_coord is not None ),

            judge_enabled =(Judge is not None ),

            )





            if post .decision =="SEED_AGREE"and post .pred :

                pred =post .pred

                used_debate =False

                used_judge =False

                debate_md =""

                j_raw =""

                coord_summary =""

                coord_info =None

                judge_winner =None

                judge_raw =""

                j_meta :Dict [str ,Any ]={}

                j_body =""

                a_final =a_seed

                b_final =b_seed

                used_agreement_recovery =False

                agreement_recovery_pred =""

                agreement_recovery_changed =False



                if args .solver_on_agree and (Solver is not None ):

                    rec_pred ,rec_summary ,rec_coord_info =_recover_agreement_with_solver (

                    Solver ,

                    medrag_coord ,

                    sample ,

                    Ls ,

                    agreed_pick =post .pred ,

                    a_text =a_rat_txt ,

                    b_text =b_rat_txt ,

                    solver_max =args .solver_max ,

                    medrag_min_score =args .medrag_min_score ,

                    medrag_max_snippets =args .medrag_max_snippets ,

                    medrag_score_margin =args .medrag_score_margin ,

                    meta ={"idx":idx ,"id":sid ,"path":"seed_agree"},

                    )

                    used_agreement_recovery =True

                    agreement_recovery_pred =rec_pred or ""

                    agreement_recovery_changed =bool (rec_pred and rec_pred in set (Ls )and rec_pred !=post .pred )

                    if rec_pred and rec_pred in set (Ls ):

                        pred =rec_pred

                    coord_summary =rec_summary or ""

                    coord_info =rec_coord_info



                    if coord_summary :

                        coord_used_n +=1

                        _clen =len (coord_summary or "")

                        coord_summary_chars .append (_clen )

                        if _clen <160 :

                            coord_too_short_n +=1

            else :



                used_debate =False

                used_judge =False

                debate_md =""

                j_raw =""

                coord_summary =""

                coord_info =None

                judge_winner =None

                judge_raw =""

                j_meta :Dict [str ,Any ]={}

                j_body =""

                used_agreement_recovery =False

                agreement_recovery_pred =""

                agreement_recovery_changed =False







                coord_summary =""

                coord_info =None



                if post .need_coord and (medrag_coord is not None ):

                    try :

                        coord_summary ,coord_info =medrag_coord .build_coord_summary_for_judge (

                        sample ,

                        a_pick =a_seed or "",

                        b_pick =b_seed or "",

                        a_rationale =a_rat_txt ,

                        b_rationale =b_rat_txt ,

                        min_score =args .medrag_min_score ,

                        max_snippets =args .medrag_max_snippets ,

                        score_margin =args .medrag_score_margin ,

                        meta ={"idx":idx ,"id":sid },

                        )

                    except Exception as e :

                        coord_summary =f"[ERROR][Coordinator]{e}"

                        coord_info =None





                    coord_used_n +=1

                    _clen =len (coord_summary or "")

                    coord_summary_chars .append (_clen )

                    if _clen <160 :

                        coord_too_short_n +=1





                lastA =a_rat_txt

                lastB =b_rat_txt

                if post .need_debate :

                    try :

                        debate_md ,lastA ,lastB =run_debate_groupchat (

                        ExpertA ,

                        ExpertB ,

                        sample ,

                        Ls ,

                        args .debate_rounds ,

                        a_rationale =a_rat_txt ,

                        b_rationale =b_rat_txt ,

                        coord_summary =coord_summary ,

                        )

                    except Exception as e :

                        try :

                            debate_md ,lastA_fb ,lastB_fb =run_debate_fallback (

                            ExpertA ,ExpertB ,sample ,Ls ,args .debate_rounds

                            )



                            lastA =lastA_fb .strip ()or a_rat_txt

                            lastB =lastB_fb .strip ()or b_rat_txt



                        except Exception as e2 :

                            debate_md =f"[ERROR]{e}\n[ERROR_FALLBACK]{e2}"

                            lastA ,lastB ="",""

                    used_debate =True





                a_final =_extract_pick (lastA ,Ls )or a_seed

                b_final =_extract_pick (lastB ,Ls )or b_seed





                if a_final and b_final and (a_final ==b_final ):

                    pred =a_final

                    used_judge =False

                    judge_winner =None

                    j_raw =""

                else :



                    criticA =""

                    criticB =""

                    if args .use_critics :

                        try :

                            criticA =short_critic_line (ExpertA ,sample ,a_final or "",Ls ,

                            max_tokens =args .critic_tokens )

                        except Exception as e :

                            criticA =f"[ERROR][criticA]{e}"

                        try :

                            criticB =short_critic_line (ExpertB ,sample ,b_final or "",Ls ,

                            max_tokens =args .critic_tokens )

                        except Exception as e :

                            criticB =f"[ERROR][criticB]{e}"













                    rule_w =""



                    if args .enable_coord_rules :



                        try :

                            if isinstance (coord_info ,dict ):

                                topA ,topB =_coord_top_scores (coord_info )

                                coord_info ["topA"]=topA

                                coord_info ["topB"]=topB

                                rule_w =_coord_score_rule_winner (

                                coord_info ,

                                min_score =float (args .medrag_min_score ),

                                gap =float (args .medrag_score_gap ),

                                )

                        except Exception :

                            rule_w =""





                        if not rule_w :

                            rule_w =coord_rule_decision (coord_summary ,min_gap =args .coord_rule_min_gap )





                    if args .enable_coord_rules and rule_w =="A"and (a_final in Ls ):

                        pred =a_final

                        used_judge =False

                        judge_winner ="RULE_A"

                        try :

                            j_raw =json .dumps ({

                            "winner":"RULE_A",

                            "coord_info":coord_info ,

                            "coord_summary":_short_text (coord_summary ,args .jsonl_text_chars ),

                            },ensure_ascii =False ,indent =2 )

                        except Exception :

                            j_raw =f"[winner]RULE_A\n\n[coord]\n{coord_summary}"



                    elif args .enable_coord_rules and rule_w =="B"and (b_final in Ls ):

                        pred =b_final

                        used_judge =False

                        judge_winner ="RULE_B"

                        try :

                            j_raw =json .dumps ({

                            "winner":"RULE_B",

                            "coord_info":coord_info ,

                            "coord_summary":_short_text (coord_summary ,args .jsonl_text_chars ),

                            },ensure_ascii =False ,indent =2 )

                        except Exception :

                            j_raw =f"[winner]RULE_B\n\n[coord]\n{coord_summary}"



                    else :



                        if post .need_judge and (Judge is not None ):

                            used_judge =True

                            try :

                                if args .no_bidirectional_judge :

                                    winner ,judge_raw =judge_vote (

                                    Judge ,sample ,Ls ,

                                    a_pick =a_final or "",

                                    b_pick =b_final or "",

                                    lastA =lastA or "",

                                    lastB =lastB or "",

                                    criticA =criticA ,

                                    criticB =criticB ,

                                    coord_summary =coord_summary ,

                                    rubric_text =rubric_text ,

                                    judge_max =args .judge_max ,

                                    self_consistency =args .judge_self_consistency ,

                                    delta_abstain =args .judge_delta_abstain ,

                                    )

                                else :

                                    winner ,judge_raw =judge_vote_bidirectional_sc (

                                    Judge ,sample ,Ls ,

                                    a_pick =a_final or "",

                                    b_pick =b_final or "",

                                    lastA =lastA or "",

                                    lastB =lastB or "",

                                    criticA =criticA ,

                                    criticB =criticB ,

                                    coord_summary =coord_summary ,

                                    rubric_text =rubric_text ,

                                    judge_max =args .judge_max ,

                                    delta_abstain =args .judge_delta_abstain ,

                                    default_winner ="B",

                                    n =args .judge_self_consistency ,

                                    )

                            except Exception as e :

                                winner =""

                                judge_raw =f"[ERROR]{e}"



                            gap =_judge_max_gap (judge_raw )

                            allow_flip_A =True

                            if (gap is None )or (gap <float (args .judge_flip_min_gap )):

                                allow_flip_A =False

                            if args .judge_require_cites and (not _judge_has_cites (judge_raw )):

                                allow_flip_A =False

                            if args .flip_require_coord_evidence :

                                a_cnt ,b_cnt =_coord_bullet_counts (coord_summary or "")



                                low_conf =bool (isinstance (coord_info ,dict )and coord_info .get ("low_conf",False ))

                                if low_conf :

                                            allow_flip_A =False

                                if a_cnt <int (args .flip_min_a_bullets ):

                                            allow_flip_A =False

                                if b_cnt >int (args .flip_max_b_bullets ):

                                            allow_flip_A =False

                            coord_rule_fallback =_coord_score_rule_winner (

                            coord_info ,

                            min_score =float (args .medrag_min_score ),

                            gap =float (args .medrag_score_gap ),

                            )



                            if winner =="A"and (a_final in Ls )and allow_flip_A :

                                pred =a_final

                                judge_winner ="A"

                            elif winner =="B"and (b_final in Ls ):

                                pred =b_final

                                judge_winner ="B"

                            else :

                                pred =None

                                judge_winner ="ABSTAIN"

                                if winner =="A"and (a_final in Ls )and (not allow_flip_A ):

                                    judge_winner ="A_GATED"



                                if coord_rule_fallback =="A"and (a_final in Ls ):

                                    pred =a_final

                                    judge_winner ="COORD_RULE_A"

                                elif coord_rule_fallback =="B"and (b_final in Ls ):

                                    pred =b_final

                                    judge_winner ="COORD_RULE_B"

                                elif Solver is not None :

                                    if (

                                    (not args .solver_only_on_abstain )

                                    or str (judge_winner ).startswith ("ABSTAIN")

                                    or judge_winner =="A_GATED"

                                    ):

                                        sol =_solve_final (

                                        Solver ,

                                        sample ,

                                        Ls ,

                                        coord_summary or "",

                                        a_rat_txt ,

                                        b_rat_txt ,

                                        max_tokens =args .solver_max ,

                                        )

                                        if sol in set (Ls ):

                                            pred =sol

                                            judge_winner ="SOLVER"



                                if pred is None :

                                    if (a_final in Ls )and (b_final not in Ls ):

                                        pred =a_final

                                        judge_winner ="SINGLE_VALID_A"

                                    elif (b_final in Ls )and (a_final not in Ls ):

                                        pred =b_final

                                        judge_winner ="SINGLE_VALID_B"

                                    else :

                                        pred =a_final if (a_final in Ls )else (

                                        b_final if (b_final in Ls )else (Ls [0 ]if Ls else None )

                                        )

                                        judge_winner ="ABSTAIN_FALLBACK_A_FIRST"





                            try :

                                j_meta ,j_body =extract_judge_meta (judge_raw )

                            except Exception :

                                j_meta ,j_body ={},(judge_raw or "")



                            if baseline_disagree :

                                try :

                                    with open (judge_trace_path ,"a",encoding ="utf-8")as tf :

                                        tf .write (json .dumps ({

                                        "idx":idx ,

                                        "id":sid ,

                                        "gold":gold ,

                                        "baseA":baseA ,

                                        "baseB":baseB ,

                                        "a_final":a_final ,

                                        "b_final":b_final ,

                                        "pred":pred ,

                                        "judge_winner":judge_winner ,

                                        "judge_meta":j_meta ,

                                        "judge_body":j_body ,

                                        "judge_raw_full":judge_raw ,

                                        "coord_top_score":(coord_info or {}).get ("top_score")if isinstance (

                                        coord_info ,dict )else None ,

                                        "coord_summary_chars":len (coord_summary or ""),

                                        },ensure_ascii =False )+"\n")

                                except Exception :

                                    pass



                            try :

                                j_raw =json .dumps ({

                                "winner":judge_winner ,

                                "coord_info":coord_info ,

                                "coord_summary":_short_text (coord_summary ,args .jsonl_text_chars ),

                                "judge_meta":j_meta ,

                                "judge_body":_short_text (j_body ,args .jsonl_text_chars ),

                                "judge_raw_full":_short_text (judge_raw ,args .jsonl_text_chars ),

                                },ensure_ascii =False ,indent =2 )

                            except Exception :

                                j_raw =f"[winner]{judge_winner}\n\n[coord]\n{coord_summary}\n\n[judge]\n{judge_raw}"

                        else :

                            coord_rule_fallback =_coord_score_rule_winner (

                            coord_info ,

                            min_score =float (args .medrag_min_score ),

                            gap =float (args .medrag_score_gap ),

                            )

                            if coord_rule_fallback =="A"and (a_final in Ls ):

                                pred =a_final

                                judge_winner ="COORD_RULE_A_NO_JUDGE"

                            elif coord_rule_fallback =="B"and (b_final in Ls ):

                                pred =b_final

                                judge_winner ="COORD_RULE_B_NO_JUDGE"

                            else :

                                pred =a_final if (a_final in Ls )else (

                                b_final if (b_final in Ls )else (Ls [0 ]if Ls else None ))

                                judge_winner ="NO_JUDGE_FALLBACK_A_FIRST"



            correct_flag =int (bool (gold and pred and gold ==pred ))

            correct +=correct_flag

            acc =correct /(idx +1 )



            rec ={

            "idx":idx ,

            "id":sid ,

            "gold":gold ,

            "pred":pred ,

            "correct":bool (correct_flag ),



            "baselineA":baseA ,

            "baselineB":baseB ,

            "baseline_disagree":baseline_disagree ,



            "A_votes":a_votes ,

            "B_votes":b_votes ,

            "A_seed":a_seed ,

            "B_seed":b_seed ,

            "A_final":a_final ,

            "B_final":b_final ,



            "A_rationale":_short_text (a_rat_txt ,args .jsonl_text_chars ),

            "B_rationale":_short_text (b_rat_txt ,args .jsonl_text_chars ),



            "used_debate":used_debate ,

            "debate_md":_short_text (debate_md ,args .jsonl_text_chars ),

            "used_agreement_recovery":used_agreement_recovery ,

            "agreement_recovery_pred":agreement_recovery_pred ,

            "agreement_recovery_changed":agreement_recovery_changed ,



            "coord_summary":_short_text (coord_summary ,args .jsonl_text_chars ),

            "coord_info":coord_info ,

            "coord_summary_chars":len (coord_summary or ""),

            "coord_top_score":(coord_info or {}).get ("top_score")if isinstance (coord_info ,dict )else None ,



            "used_judge":used_judge ,

            "judge_winner":judge_winner ,

            "judge_meta":j_meta if isinstance (j_meta ,dict )else {},

            "judge_body":_short_text (j_body or "",args .jsonl_text_chars ),

            "judge_raw_full":_short_text (judge_raw or "",args .jsonl_text_chars ),

            "judge_debug":_short_text (j_raw ,args .jsonl_text_chars ),

            }

            f_out .write (json .dumps (rec ,ensure_ascii =False )+"\n")



            need_md =args .md_all or (not correct_flag )or (a_seed !=b_seed )

            if need_md :

                md_txt =_build_case_md (

                idx ,sample ,Ls ,

                a_votes =a_votes ,b_votes =b_votes ,

                a_seed =a_seed ,b_seed =b_seed ,

                a_pick_final =a_final ,b_pick_final =b_final ,

                a_rat_txt =a_rat_txt ,b_rat_txt =b_rat_txt ,

                debate_md =debate_md ,j_raw =j_raw ,

                pred =pred ,gold =gold ,acc =acc ,

                )

                write_text (os .path .join (md_dir ,f"case_{idx:06d}_{sid}.md"),md_txt )



            print (

            f"[{idx + 1}/{total}] acc={acc:.3f} "

            f"A={a_seed or '-'}->{a_final or '-'} "

            f"B={b_seed or '-'}->{b_final or '-'} "

            f"pred={pred or '-'} used_judge={used_judge} disagree={bool(a_seed and b_seed and a_seed != b_seed)}",

            flush =True ,

            )



    summary_lines =[

    f"Total samples: {total}",

    f"Correct: {correct}",

    f"Accuracy: {correct / total if total else 0.0:.4f}",

    ]



    if coord_used_n >0 and coord_summary_chars :

        xs =sorted (coord_summary_chars )



        def _pct (p :float )->int :

            if not xs :

                return 0

            i =int (round ((p /100.0 )*(len (xs )-1 )))

            i =max (0 ,min (i ,len (xs )-1 ))

            return int (xs [i ])



        summary_lines +=[

        f"Coordinator used: {coord_used_n}",

        f"Coordinator summary chars: median={_pct(50)}, p10={_pct(10)}, p90={_pct(90)}",

        f"Coordinator too-short (<160 chars): {coord_too_short_n}",

        ]



    write_text (os .path .join (args .run_dir ,"run_summary.md"),"\n".join (summary_lines ))

    print ("\n".join (summary_lines ))





if __name__ =="__main__":

    main ()

