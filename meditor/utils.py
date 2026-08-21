

from __future__ import annotations

import os ,json ,time ,urllib .request

from datetime import datetime

from typing import Dict ,Any ,Optional



def now_str ()->str :

    return datetime .now ().strftime ("%Y-%m-%d %H:%M:%S")



def ensure_dir (p :str ):

    os .makedirs (p ,exist_ok =True )



def write_text (path :str ,txt :str ,mode :str ="w"):

    ensure_dir (os .path .dirname (path ))

    with open (path ,mode ,encoding ="utf-8")as f :

        f .write (txt )



def md_escape (s :str )->str :

    return (s or "").replace ("\r","")



def _normalize_base_url (u :str )->str :

    if not u :return u

    u =u .rstrip ("/")

    while u .endswith ("/v1"):

        u =u [:-3 ].rstrip ("/")

    return u +"/v1"



def http_get (url ,timeout =5.0 )->Optional [Dict [str ,Any ]]:

    try :

        with urllib .request .urlopen (url ,timeout =timeout )as r :

            return json .loads (r .read ().decode ("utf-8"))

    except Exception :

        return None



def wait_ready (base :str ,timeout_s :int =600 ):

    base =_normalize_base_url (base )

    t0 =time .time ()

    while time .time ()-t0 <timeout_s :

        info =http_get (base .rstrip ("/")+"/models",timeout =2.0 )

        if info and isinstance (info .get ("data"),list )and info ["data"]:

            return info

        time .sleep (2 )

    raise RuntimeError (f"Timeout waiting for {base}")



def get_model_id (base :str )->str :

    info =wait_ready (base )

    return info ["data"][0 ]["id"]

