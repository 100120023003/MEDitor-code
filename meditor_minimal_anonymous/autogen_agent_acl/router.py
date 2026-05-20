




from __future__ import annotations



from dataclasses import dataclass

from enum import Enum

from typing import Any ,Dict ,Optional





class TaskType (str ,Enum ):




    TEXT_QA ="text_qa"

    VQA ="vqa"





@dataclass (frozen =True )

class PreVoteDecision :




    decision :str

    pred :Optional [str ]=None

    a_seed :Optional [str ]=None

    b_seed :Optional [str ]=None

    reason :str =""





@dataclass (frozen =True )

class PostSeedDecision :




    decision :str

    pred :Optional [str ]=None

    need_coord :bool =False

    need_debate :bool =False

    need_judge :bool =False

    reason :str =""





class Router :




    def decide_task_type (self ,sample :Dict [str ,Any ])->TaskType :

        raise NotImplementedError



    def decide_pre_vote (

    self ,

    *,

    sample :Dict [str ,Any ],

    idx :int ,

    sid :Any ,

    baseA :Optional [str ],

    baseB :Optional [str ],

    )->PreVoteDecision :

        raise NotImplementedError



    def decide_post_seed (

    self ,

    *,

    a_seed :Optional [str ],

    b_seed :Optional [str ],

    a_failed :bool ,

    b_failed :bool ,

    disable_debate :bool ,

    debate_rounds :int ,

    coord_enabled :bool ,

    judge_enabled :bool ,

    )->PostSeedDecision :

        raise NotImplementedError





class DefaultRouter (Router ):




    def decide_task_type (self ,sample :Dict [str ,Any ])->TaskType :

        if any (k in sample for k in ("image","image_path","pixel_values")):

            return TaskType .VQA

        return TaskType .TEXT_QA



    def decide_pre_vote (

    self ,

    *,

    sample :Dict [str ,Any ],

    idx :int ,

    sid :Any ,

    baseA :Optional [str ],

    baseB :Optional [str ],

    )->PreVoteDecision :



        if (baseA is not None )and (baseB is not None )and (baseA ==baseB ):

            return PreVoteDecision (

            decision ="BASELINE_AGREE",

            pred =baseA ,

            a_seed =baseA ,

            b_seed =baseB ,

            reason ="baseline_agree",

            )





        if (baseA is not None )and (baseB is not None )and (baseA !=baseB ):

            return PreVoteDecision (

            decision ="BASELINE_DISAGREE",

            pred =None ,

            a_seed =baseA ,

            b_seed =baseB ,

            reason ="baseline_disagree",

            )





        return PreVoteDecision (decision ="NEED_VOTE",reason ="need_vote")



    def decide_post_seed (

    self ,

    *,

    a_seed :Optional [str ],

    b_seed :Optional [str ],

    a_failed :bool ,

    b_failed :bool ,

    disable_debate :bool ,

    debate_rounds :int ,

    coord_enabled :bool ,

    judge_enabled :bool ,

    )->PostSeedDecision :



        if a_seed and b_seed and (a_seed ==b_seed ):

            return PostSeedDecision (

            decision ="SEED_AGREE",

            pred =a_seed ,

            need_coord =False ,

            need_debate =False ,

            need_judge =False ,

            reason ="seed_agree",

            )





        need_debate =(

        (not disable_debate )

        and (int (debate_rounds )>0 )

        and (not a_failed )

        and (not b_failed )

        and bool (a_seed and b_seed )

        )

        return PostSeedDecision (

        decision ="DISAGREE",

        pred =None ,

        need_coord =bool (coord_enabled ),

        need_debate =bool (need_debate ),

        need_judge =bool (judge_enabled ),

        reason ="disagree",

        )

