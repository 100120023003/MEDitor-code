import bisect

from sentence_transformers .models import Transformer ,Pooling

from sentence_transformers import SentenceTransformer

import os

import faiss

import json

import re

import subprocess

import sys

import torch

import tqdm

import numpy as np



corpus_names ={

"PubMed":["pubmed"],

"Textbooks":["textbooks"],

"StatPearls":["statpearls"],

"Wikipedia":["wikipedia"],

"MedText":["textbooks","statpearls"],

"MedCorp":["pubmed","textbooks","statpearls","wikipedia"],

}





def _index_dir_has_files (path ):

    if not os .path .isdir (path ):

        return False

    with os .scandir (path )as entries :

        for _ in entries :

            return True

    return False





def _normalize_index_token (text ):

    raw =os .path .basename (str (text or "").strip ().replace ("\\","/").rstrip ("/"))

    raw =re .sub (r"[^A-Za-z0-9]+","_",raw ).strip ("_").lower ()

    if not raw :

        return ""

    simplified =re .sub (r"(?:_)?(?:query|article)(?:_)?encoder$","",raw ).strip ("_")

    simplified =re .sub (r"(?:_)?(?:query|article)$","",simplified ).strip ("_")

    return simplified or raw





def _sanitize_path_part (text ):

    text =str (text or "").strip ().replace ("\\","/").rstrip ("/")

    if not text :

        return "retriever"

    return re .sub (r"[^A-Za-z0-9._-]+","_",text )





def _derive_article_model_name (query_model ):

    query_model =str (query_model or "").strip ()

    if not query_model :

        return query_model

    replacements =[

    ("Query-Encoder","Article-Encoder"),

    ("QUERY","ARTICLE"),

    ("query","article"),

    ("Query","Article"),

    ]

    for src ,dst in replacements :

        if src in query_model :

            return query_model .replace (src ,dst )

    return query_model





def _medcpt_query_model ():

    return (

    os .getenv ("MEDCPT_QUERY_ENCODER")

    or os .getenv ("MEDCPT_QUERY_MODEL")

    or "ncbi/MedCPT-Query-Encoder"

    )





def _medcpt_article_model (query_model ):

    return (

    os .getenv ("MEDCPT_ARTICLE_ENCODER")

    or os .getenv ("MEDCPT_ARTICLE_MODEL")

    or _derive_article_model_name (query_model )

    )





def _dense_spec (name ,query_model ,article_model =None ):

    article_model =article_model or query_model

    return {

    "kind":"dense",

    "name":str (name ),

    "query_model":str (query_model ),

    "article_model":str (article_model ),

    "index_name":str (article_model ),

    }





def _sparse_spec (name ):

    return {

    "kind":"sparse",

    "name":str (name ),

    "query_model":str (name ),

    "article_model":str (name ),

    "index_name":str (name ),

    }





def _normalize_retriever_spec (spec ):

    if isinstance (spec ,dict ):

        out =dict (spec )

        query_model =str (out .get ("query_model")or out .get ("name")or "")

        kind =str (out .get ("kind")or ("sparse"if query_model .lower ()=="bm25"else "dense")).lower ()

        out ["kind"]=kind

        out ["name"]=str (out .get ("name")or query_model or "retriever")

        if kind =="sparse":

            out ["query_model"]=query_model or out ["name"]

            out ["article_model"]=str (out .get ("article_model")or out ["query_model"])

            out ["index_name"]=str (out .get ("index_name")or out ["name"])

        else :

            out ["query_model"]=query_model

            out ["article_model"]=str (out .get ("article_model")or _derive_article_model_name (query_model ))

            out ["index_name"]=str (out .get ("index_name")or out ["article_model"])

        return out

    if isinstance (spec ,str ):

        if spec .lower ()=="bm25":

            return _sparse_spec (spec )

        return _dense_spec (spec ,spec ,_derive_article_model_name (spec ))

    raise TypeError (f"Unsupported retriever spec: {spec!r}")





def _spec_is_sparse (spec ):

    return str (spec .get ("kind")or "").lower ()=="sparse"





def _spec_uses_specter (spec ):

    text =" ".join (

    [

    str (spec .get ("name")or ""),

    str (spec .get ("query_model")or ""),

    str (spec .get ("article_model")or ""),

    ]

    ).lower ()

    return "specter"in text





def _spec_index_parts (spec ):

    raw =str (

    spec .get ("index_name")

    or spec .get ("article_model")

    or spec .get ("query_model")

    or spec .get ("name")

    or "retriever"

    ).strip ()

    raw =raw .replace ("\\","/").rstrip ("/")

    if os .path .isabs (raw ):

        raw =os .path .basename (raw )

    parts =[p for p in raw .split ("/")if p and p not in (".","..")]

    if not parts :

        parts =["retriever"]

    return [_sanitize_path_part (p )for p in parts ]





def _expected_index_dir (db_dir ,corpus_name ,spec ):

    return os .path .join (db_dir ,corpus_name ,"index",*_spec_index_parts (spec ))





def _legacy_index_dir_candidates (db_dir ,corpus_name ,spec ):

    seen =set ()

    raw_values =[

    spec .get ("name"),

    spec .get ("index_name"),

    spec .get ("article_model"),

    spec .get ("query_model"),

    ]

    for value in raw_values :

        raw =str (value or "").strip ().replace ("\\","/").rstrip ("/")

        if not raw :

            continue

        for item in [raw ,os .path .basename (raw )]:

            token =_normalize_index_token (item )

            if not token :

                continue

            for candidate in [

            os .path .join (db_dir ,corpus_name ,"index",token ),

            os .path .join (db_dir ,f"{corpus_name}_{token}"),

            ]:

                if candidate in seen :

                    continue

                seen .add (candidate )

                yield candidate





def _index_dir_complete (path ,spec ):

    if _spec_is_sparse (spec ):

        return _index_dir_has_files (path )

    return os .path .isfile (os .path .join (path ,"faiss.index"))and os .path .isfile (os .path .join (path ,"metadatas.jsonl"))





def _index_dir_present_artifacts (path ,spec ):

    if _spec_is_sparse (spec ):

        if not os .path .isdir (path ):

            return []

        return sorted ([entry .name for entry in os .scandir (path )])[:8 ]

    names =[]

    for name in ["faiss.index","metadatas.jsonl","metadata_ranges.json","embedding"]:

        if os .path .exists (os .path .join (path ,name )):

            names .append (name )

    return names





def resolve_index_dir (db_dir ,corpus_name ,spec ):

    expected_dir =_expected_index_dir (db_dir ,corpus_name ,spec )

    if _index_dir_complete (expected_dir ,spec ):

        return expected_dir ,expected_dir ,"expected"

    for candidate in _legacy_index_dir_candidates (db_dir ,corpus_name ,spec ):

        if candidate ==expected_dir :

            continue

        if _index_dir_complete (candidate ,spec ):

            return candidate ,expected_dir ,"legacy"

    return expected_dir ,expected_dir ,"missing"





def collect_index_statuses (retriever_name ,corpus_name ,db_dir ="./corpus"):

    assert corpus_name in corpus_names

    assert retriever_name in retriever_names

    rows =[]

    retriever_specs =[_normalize_retriever_spec (spec )for spec in retriever_names [retriever_name ]]

    for spec in retriever_specs :

        for member_corpus in corpus_names [corpus_name ]:

            resolved_dir ,expected_dir ,resolved_from =resolve_index_dir (db_dir ,member_corpus ,spec )

            candidate_dirs =[expected_dir ]

            for candidate in _legacy_index_dir_candidates (db_dir ,member_corpus ,spec ):

                if candidate not in candidate_dirs :

                    candidate_dirs .append (candidate )

            rows .append (

            {

            "corpus":corpus_name ,

            "member_corpus":member_corpus ,

            "retriever":spec .get ("name"),

            "retriever_kind":spec .get ("kind"),

            "query_model":spec .get ("query_model"),

            "article_model":spec .get ("article_model"),

            "expected_dir":expected_dir ,

            "resolved_dir":resolved_dir ,

            "resolved_from":resolved_from ,

            "complete":_index_dir_complete (resolved_dir ,spec ),

            "required_artifacts":(

            ["faiss.index","metadatas.jsonl"]if not _spec_is_sparse (spec )else ["<lucene index files>"]

            ),

            "present_artifacts":_index_dir_present_artifacts (

            resolved_dir if os .path .exists (resolved_dir )else expected_dir ,

            spec ,

            ),

            "candidate_dirs":candidate_dirs ,

            }

            )

    return rows





def _statpearls_script_path ():

    return os .path .join (os .path .dirname (__file__ ),"data","statpearls.py")





_DEFAULT_MEDCPT_QUERY =_medcpt_query_model ()

_DEFAULT_MEDCPT_ARTICLE =_medcpt_article_model (_DEFAULT_MEDCPT_QUERY )





retriever_names ={

"BM25":[_sparse_spec ("bm25")],

"Contriever":[_dense_spec ("Contriever","facebook/contriever")],

"SPECTER":[_dense_spec ("SPECTER","allenai/specter")],

"MedCPT":[_dense_spec ("MedCPT",_DEFAULT_MEDCPT_QUERY ,_DEFAULT_MEDCPT_ARTICLE )],

"RRF-2":[

_sparse_spec ("bm25"),

_dense_spec ("MedCPT",_DEFAULT_MEDCPT_QUERY ,_DEFAULT_MEDCPT_ARTICLE ),

],

"RRF-4":[

_sparse_spec ("bm25"),

_dense_spec ("Contriever","facebook/contriever"),

_dense_spec ("SPECTER","allenai/specter"),

_dense_spec ("MedCPT",_DEFAULT_MEDCPT_QUERY ,_DEFAULT_MEDCPT_ARTICLE ),

],

}



retriever_names ={

"BM25":[_sparse_spec ("bm25")],

"Contriever":[_dense_spec ("Contriever","facebook/contriever")],

"SPECTER":[_dense_spec ("SPECTER","allenai/specter")],

"MedCPT":[_dense_spec ("MedCPT",_DEFAULT_MEDCPT_QUERY ,_DEFAULT_MEDCPT_ARTICLE )],

"RRF-2":[

_sparse_spec ("bm25"),

_dense_spec ("MedCPT",_DEFAULT_MEDCPT_QUERY ,_DEFAULT_MEDCPT_ARTICLE ),

],

"RRF-4":[

_sparse_spec ("bm25"),

_dense_spec ("Contriever","facebook/contriever"),

_dense_spec ("SPECTER","allenai/specter"),

_dense_spec ("MedCPT",_DEFAULT_MEDCPT_QUERY ,_DEFAULT_MEDCPT_ARTICLE ),

],

}





def ends_with_ending_punctuation (s ):

    ending_punctuation =('.','?','!')

    return any (s .endswith (char )for char in ending_punctuation )



def concat (title ,content ):

    if ends_with_ending_punctuation (title .strip ()):

        return title .strip ()+" "+content .strip ()

    else :

        return title .strip ()+". "+content .strip ()



class CustomizeSentenceTransformer (SentenceTransformer ):



    def _load_auto_model (self ,model_name_or_path ,*args ,**kwargs ):


        print ("No sentence-transformers model found with name {}. Creating a new one with CLS pooling.".format (model_name_or_path ))

        token =kwargs .get ('token',None )

        cache_folder =kwargs .get ('cache_folder',None )

        revision =kwargs .get ('revision',None )

        trust_remote_code =kwargs .get ('trust_remote_code',False )

        if 'token'in kwargs or 'cache_folder'in kwargs or 'revision'in kwargs or 'trust_remote_code'in kwargs :

            transformer_model =Transformer (

            model_name_or_path ,

            cache_dir =cache_folder ,

            model_args ={"token":token ,"trust_remote_code":trust_remote_code ,"revision":revision },

            tokenizer_args ={"token":token ,"trust_remote_code":trust_remote_code ,"revision":revision },

            )

        else :

            transformer_model =Transformer (model_name_or_path )

        pooling_model =Pooling (transformer_model .get_word_embedding_dimension (),'cls')

        return [transformer_model ,pooling_model ]





def embed (chunk_dir ,index_dir ,model_name ,**kwarg ):



    save_dir =os .path .join (index_dir ,"embedding")

    encode_kwarg =dict (kwarg )

    embed_file_batch_size =int (encode_kwarg .pop ("embed_file_batch_size",os .getenv ("MEDRAG_EMBED_FILE_BATCH_SIZE","2048")))

    if embed_file_batch_size <=0 :

        raise ValueError ("embed_file_batch_size must be a positive integer")

    if "batch_size"not in encode_kwarg and os .getenv ("MEDRAG_ENCODE_BATCH_SIZE"):

        encode_kwarg ["batch_size"]=int (os .getenv ("MEDRAG_ENCODE_BATCH_SIZE"))



    if "contriever"in model_name :

        model =SentenceTransformer (model_name ,device ="cuda"if torch .cuda .is_available ()else "cpu")

    else :

        model =CustomizeSentenceTransformer (model_name ,device ="cuda"if torch .cuda .is_available ()else "cpu")



    model .eval ()



    fnames =sorted ([fname for fname in os .listdir (chunk_dir )if fname .endswith (".jsonl")])



    if not os .path .exists (save_dir ):

        os .makedirs (save_dir )



    with torch .no_grad ():

        for fname in tqdm .tqdm (fnames ):

            fpath =os .path .join (chunk_dir ,fname )

            source_name =fname .replace (".jsonl","")

            legacy_save_path =os .path .join (save_dir ,source_name +".npy")

            done_path =os .path .join (save_dir ,source_name +".done")

            if os .path .exists (legacy_save_path )or os .path .exists (done_path ):

                continue

            stale_prefix =source_name +"__part"

            for existing_name in list (os .listdir (save_dir )):

                if existing_name .startswith (stale_prefix ):

                    os .remove (os .path .join (save_dir ,existing_name ))



            def _to_model_inputs (items ):

                if "specter"in model_name .lower ():

                    return [model .tokenizer .sep_token .join ([item ["title"],item ["content"]])for item in items ]

                if "contriever"in model_name .lower ():

                    return [". ".join ([item ["title"],item ["content"]]).replace ('..','.').replace ("?.","?")for item in items ]

                if "medcpt"in model_name .lower ():

                    return [[item ["title"],item ["content"]]for item in items ]

                return [concat (item ["title"],item ["content"])for item in items ]



            pending_items =[]

            pending_line_numbers =[]

            shard_idx =0

            h_dim =None



            def _flush_pending ():

                nonlocal pending_items ,pending_line_numbers ,shard_idx ,h_dim

                if not pending_items :

                    return

                start_line =pending_line_numbers [0 ]

                shard_base =f"{source_name}__part{shard_idx:06d}"

                shard_path =os .path .join (save_dir ,shard_base +".npy")

                meta_path =os .path .join (save_dir ,shard_base +".meta.json")

                model_inputs =_to_model_inputs (pending_items )

                try :

                    embed_chunks =model .encode (model_inputs ,**encode_kwarg )

                except MemoryError as err :

                    raise MemoryError (

                    f"Out of memory while embedding {fpath} near line {start_line}. "

                    f"Try lowering MEDRAG_EMBED_FILE_BATCH_SIZE or MEDRAG_ENCODE_BATCH_SIZE."

                    )from err

                np .save (shard_path ,embed_chunks )

                with open (meta_path ,"w",encoding ="utf-8")as f :

                    json .dump (

                    {

                    "source":source_name ,

                    "start":int (start_line ),

                    "count":int (len (embed_chunks )),

                    },

                    f ,

                    ensure_ascii =False ,

                    )

                if h_dim is None :

                    h_dim =int (embed_chunks .shape [-1 ])

                shard_idx +=1

                pending_items =[]

                pending_line_numbers =[]



            with open (fpath ,"r",encoding ="utf-8")as f :

                for line_no ,line in enumerate (f ):

                    line =line .strip ()

                    if not line :

                        continue

                    pending_items .append (json .loads (line ))

                    pending_line_numbers .append (line_no )

                    if len (pending_items )>=embed_file_batch_size :

                        _flush_pending ()



            _flush_pending ()

            if h_dim is None :

                with open (done_path ,"w",encoding ="utf-8")as f :

                    f .write ("0\n")

                continue



            with open (done_path ,"w",encoding ="utf-8")as f :

                f .write (f"{shard_idx}\n")



        embed_chunks =model .encode ([""],**encode_kwarg )

    return embed_chunks .shape [-1 ]



def construct_index (index_dir ,model_name ,h_dim =768 ,HNSW =False ,M =32 ):



    with open (os .path .join (index_dir ,"metadatas.jsonl"),'w')as f :

        f .write ("")

    metadata_ranges =[]



    if HNSW :

        M =M

        if "specter"in model_name .lower ():

            index =faiss .IndexHNSWFlat (h_dim ,M )

        else :

            index =faiss .IndexHNSWFlat (h_dim ,M )

            index .metric_type =faiss .METRIC_INNER_PRODUCT

    else :

        if "specter"in model_name .lower ():

            index =faiss .IndexFlatL2 (h_dim )

        else :

            index =faiss .IndexFlatIP (h_dim )



    embedding_dir =os .path .join (index_dir ,"embedding")

    embedding_files =sorted ([fname for fname in os .listdir (embedding_dir )if fname .endswith (".npy")])

    with open (os .path .join (index_dir ,"metadatas.jsonl"),'a+',encoding ="utf-8")as meta_f :

        for fname in tqdm .tqdm (embedding_files ):

            curr_embed =np .load (os .path .join (embedding_dir ,fname ))

            index .add (curr_embed )

            shard_meta_path =os .path .join (embedding_dir ,fname .replace (".npy",".meta.json"))

            if os .path .exists (shard_meta_path ):

                with open (shard_meta_path ,"r",encoding ="utf-8")as f :

                    shard_meta =json .load (f )

                source_name =str (shard_meta .get ("source")or fname .replace (".npy",""))

                source_start =int (shard_meta .get ("start",0 ))

            else :

                source_name =fname .replace (".npy","")

                source_start =0

            row_start =int (index .ntotal -len (curr_embed ))

            row_end =int (index .ntotal )

            metadata_ranges .append (

            {

            "row_start":row_start ,

            "row_end":row_end ,

            "source":source_name ,

            "source_start":source_start ,

            }

            )

            meta_f .write (

            "\n".join (

            [

            json .dumps ({'index':source_start +i ,'source':source_name })

            for i in range (len (curr_embed ))

            ]

            )+'\n'

            )



    faiss .write_index (index ,os .path .join (index_dir ,"faiss.index"))

    with open (os .path .join (index_dir ,"metadata_ranges.json"),"w",encoding ="utf-8")as f :

        json .dump (metadata_ranges ,f ,ensure_ascii =False ,indent =2 )

    return index





class Retriever :



    def __init__ (self ,retriever_name ="ncbi/MedCPT-Query-Encoder",corpus_name ="textbooks",db_dir ="./corpus",HNSW =False ,**kwarg ):

        self .retriever_spec =_normalize_retriever_spec (retriever_name )

        self .retriever_name =self .retriever_spec ["query_model"]

        self .article_retriever_name =self .retriever_spec ["article_model"]

        self .corpus_name =corpus_name



        self .db_dir =db_dir

        if not os .path .exists (self .db_dir ):

            os .makedirs (self .db_dir )

        self .chunk_dir =os .path .join (self .db_dir ,self .corpus_name ,"chunk")

        if not os .path .exists (self .chunk_dir ):

            print ("Cloning the {:s} corpus from Huggingface...".format (self .corpus_name ))

            os .system ("git clone https://huggingface.co/datasets/MedRAG/{:s} {:s}".format (corpus_name ,os .path .join (self .db_dir ,self .corpus_name )))

            if self .corpus_name =="statpearls":

                print ("Downloading the statpearls corpus from NCBI bookshelf...")

                os .system ("wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P {:s}".format (os .path .join (self .db_dir ,self .corpus_name )))

                os .system ("tar -xzvf {:s} -C {:s}".format (os .path .join (db_dir ,self .corpus_name ,"statpearls_NBK430685.tar.gz"),os .path .join (self .db_dir ,self .corpus_name )))

                print ("Chunking the statpearls corpus...")

                statpearls_script =_statpearls_script_path ()

                if not os .path .exists (statpearls_script ):

                    raise FileNotFoundError ("StatPearls chunking script not found: {:s}".format (statpearls_script ))

                subprocess .check_call ([sys .executable ,statpearls_script ],cwd =os .path .dirname (__file__ ))

        self .expected_index_dir =_expected_index_dir (self .db_dir ,self .corpus_name ,self .retriever_spec )

        self .index_dir ,_unused_expected ,self .index_dir_source =resolve_index_dir (

        self .db_dir ,

        self .corpus_name ,

        self .retriever_spec ,

        )

        self .metadatas =None

        self .metadata_ranges =None

        self .metadata_range_starts =None

        if self .index_dir_source =="legacy":

            print (

            "[Compat] Using legacy index directory for {:s} with {:s}: {:s}".format (

            self .corpus_name ,

            self .retriever_spec ["name"],

            self .index_dir ,

            )

            )

        if _spec_is_sparse (self .retriever_spec ):

            from pyserini .search .lucene import LuceneSearcher

            self .embedding_function =None

            if _index_dir_complete (self .index_dir ,self .retriever_spec ):

                self .index =LuceneSearcher (os .path .join (self .index_dir ))

            else :

                self .index_dir =self .expected_index_dir

                os .system ("python -m pyserini.index.lucene --collection JsonCollection --input {:s} --index {:s} --generator DefaultLuceneDocumentGenerator --threads 16".format (self .chunk_dir ,self .index_dir ))

                self .index =LuceneSearcher (os .path .join (self .index_dir ))

        else :

            if _index_dir_complete (self .index_dir ,self .retriever_spec ):

                self .index =faiss .read_index (os .path .join (self .index_dir ,"faiss.index"))

                self ._load_metadata ()

            else :

                self .index_dir =self .expected_index_dir

                print ("[In progress] Embedding the {:s} corpus with the {:s} retriever...".format (self .corpus_name ,self .article_retriever_name ))

                h_dim =embed (chunk_dir =self .chunk_dir ,index_dir =self .index_dir ,model_name =self .article_retriever_name ,**kwarg )



                print ("[In progress] Embedding finished! The dimension of the embeddings is {:d}.".format (h_dim ))

                self .index =construct_index (index_dir =self .index_dir ,model_name =self .article_retriever_name ,h_dim =h_dim ,HNSW =HNSW )

                print ("[Finished] Corpus indexing finished!")

                self ._load_metadata ()

            if "contriever"in self .retriever_name .lower ():

                self .embedding_function =SentenceTransformer (self .retriever_name ,device ="cuda"if torch .cuda .is_available ()else "cpu")

            else :

                self .embedding_function =CustomizeSentenceTransformer (self .retriever_name ,device ="cuda"if torch .cuda .is_available ()else "cpu")

            self .embedding_function .eval ()



    def _load_metadata (self ):

        metadata_ranges_path =os .path .join (self .index_dir ,"metadata_ranges.json")

        if os .path .exists (metadata_ranges_path ):

            with open (metadata_ranges_path ,"r",encoding ="utf-8")as f :

                self .metadata_ranges =json .load (f )

            self .metadata_range_starts =[int (item ["row_start"])for item in self .metadata_ranges ]

            self .metadatas =None

            return



        metadata_path =os .path .join (self .index_dir ,"metadatas.jsonl")

        ranges =[]

        dense_row_idx =0

        with open (metadata_path ,"r",encoding ="utf-8")as f :

            for line in f :

                line =line .strip ()

                if not line :

                    continue

                meta =json .loads (line )

                source =str (meta ["source"])

                source_index =int (meta ["index"])

                if ranges :

                    prev =ranges [-1 ]

                    prev_next_source_index =int (prev ["source_start"])+(int (prev ["row_end"])-int (prev ["row_start"]))

                    if source ==prev ["source"]and source_index ==prev_next_source_index :

                        prev ["row_end"]=dense_row_idx +1

                    else :

                        ranges .append (

                        {

                        "row_start":dense_row_idx ,

                        "row_end":dense_row_idx +1 ,

                        "source":source ,

                        "source_start":source_index ,

                        }

                        )

                else :

                    ranges .append (

                    {

                    "row_start":dense_row_idx ,

                    "row_end":dense_row_idx +1 ,

                    "source":source ,

                    "source_start":source_index ,

                    }

                    )

                dense_row_idx +=1



        self .metadata_ranges =ranges

        self .metadata_range_starts =[int (item ["row_start"])for item in self .metadata_ranges ]

        self .metadatas =None

        if self .metadata_ranges :

            try :

                with open (metadata_ranges_path ,"w",encoding ="utf-8")as f :

                    json .dump (self .metadata_ranges ,f ,ensure_ascii =False ,indent =2 )

            except Exception :

                pass



    def _metadata_for_row (self ,row_idx ):

        row_idx =int (row_idx )

        if self .metadata_ranges is None :

            return self .metadatas [row_idx ]

        pos =bisect .bisect_right (self .metadata_range_starts ,row_idx )-1

        if pos <0 :

            raise IndexError ("Metadata row out of range: {:d}".format (row_idx ))

        item =self .metadata_ranges [pos ]

        if row_idx >=int (item ["row_end"]):

            raise IndexError ("Metadata row out of range: {:d}".format (row_idx ))

        return {

        "source":item ["source"],

        "index":int (item ["source_start"])+(row_idx -int (item ["row_start"])),

        }



    def get_relevant_documents (self ,question ,k =32 ,id_only =False ,**kwarg ):

        assert type (question )==str

        question =[question ]



        if "bm25"in self .retriever_name .lower ():

            res_ =[[]]

            hits =self .index .search (question [0 ],k =k )

            res_ [0 ].append (np .array ([h .score for h in hits ]))

            ids =[h .docid for h in hits ]

            indices =[{"source":'_'.join (h .docid .split ('_')[:-1 ]),"index":eval (h .docid .split ('_')[-1 ])}for h in hits ]

        else :

            with torch .no_grad ():

                query_embed =self .embedding_function .encode (question ,**kwarg )

            res_ =self .index .search (query_embed ,k =k )

            indices =[self ._metadata_for_row (i )for i in res_ [1 ][0 ]]

            ids =['_'.join ([item ["source"],str (item ["index"])])for item in indices ]



        scores =res_ [0 ][0 ].tolist ()



        if id_only :

            return [{"id":i }for i in ids ],scores

        else :

            return self .idx2txt (indices ),scores



    def idx2txt (self ,indices ):


        import os

        import json

        from collections import defaultdict





        jsonl_files =[f for f in os .listdir (self .chunk_dir )if f .endswith (".jsonl")]

        if not jsonl_files :

            raise FileNotFoundError (f"No .jsonl files found under {self.chunk_dir}")



        jsonl_set =set (jsonl_files )





        path2idx_list =defaultdict (list )





        key_order =[]



        for meta in indices :

            src =meta .get ("source")

            idx =meta .get ("index",0 )





            if src is not None and f"{src}.jsonl"in jsonl_set :

                fname =f"{src}.jsonl"

            else :



                fname =jsonl_files [0 ]



            path =os .path .join (self .chunk_dir ,fname )



            path2idx_list [path ].append (idx )

            key_order .append ((path ,idx ))





        path2idx2doc ={}



        for path ,idx_list in path2idx_list .items ():



            wanted =sorted (set (idx_list ))

            wanted_set =set (wanted )

            idx2doc ={}



            with open (path ,"r",encoding ="utf-8")as f :

                for lineno ,line in enumerate (f ):

                    if lineno >wanted [-1 ]:

                        break

                    if lineno in wanted_set :

                        try :

                            idx2doc [lineno ]=json .loads (line )

                        except Exception :



                            continue



            path2idx2doc [path ]=idx2doc





        results =[]

        for path ,idx in key_order :

            doc =path2idx2doc .get (path ,{}).get (idx )

            if doc is not None :

                results .append (doc )



        return results





class RetrievalSystem :



    def __init__ (self ,retriever_name ="MedCPT",corpus_name ="Textbooks",db_dir ="./corpus",HNSW =False ,cache =False ):

        self .retriever_name =retriever_name

        self .corpus_name =corpus_name

        assert self .corpus_name in corpus_names

        assert self .retriever_name in retriever_names

        self .retriever_specs =[_normalize_retriever_spec (spec )for spec in retriever_names [self .retriever_name ]]

        self .retrievers =[]

        for retriever in self .retriever_specs :

            self .retrievers .append ([])

            for corpus in corpus_names [self .corpus_name ]:

                self .retrievers [-1 ].append (Retriever (retriever ,corpus ,db_dir ,HNSW =HNSW ))

        self .cache =cache

        if self .cache :

            self .docExt =DocExtracter (cache =True ,corpus_name =self .corpus_name ,db_dir =db_dir )

        else :

            self .docExt =None



    def retrieve (self ,question ,k =32 ,rrf_k =100 ,id_only =False ):


        assert type (question )==str



        output_id_only =id_only

        if self .cache :

            id_only =True



        texts =[]

        scores =[]



        if "RRF"in self .retriever_name :

            k_ =max (k *2 ,100 )

        else :

            k_ =k

        for i in range (len (self .retriever_specs )):

            texts .append ([])

            scores .append ([])

            for j in range (len (corpus_names [self .corpus_name ])):

                t ,s =self .retrievers [i ][j ].get_relevant_documents (question ,k =k_ ,id_only =id_only )

                texts [-1 ].append (t )

                scores [-1 ].append (s )

        texts ,scores =self .merge (texts ,scores ,k =k ,rrf_k =rrf_k )

        if self .cache :

            texts =self .docExt .extract (texts )

        return texts ,scores



    def merge (self ,texts ,scores ,k =32 ,rrf_k =100 ):


        RRF_dict ={}

        for i in range (len (self .retriever_specs )):

            texts_all ,scores_all =None ,None

            for j in range (len (corpus_names [self .corpus_name ])):

                if texts_all is None :

                    texts_all =texts [i ][j ]

                    scores_all =scores [i ][j ]

                else :

                    texts_all =texts_all +texts [i ][j ]

                    scores_all =scores_all +scores [i ][j ]

            if _spec_uses_specter (self .retriever_specs [i ]):

                sorted_index =np .array (scores_all ).argsort ()

            else :

                sorted_index =np .array (scores_all ).argsort ()[::-1 ]

            texts [i ]=[texts_all [i ]for i in sorted_index ]

            scores [i ]=[scores_all [i ]for i in sorted_index ]

            for j ,item in enumerate (texts [i ]):

                if item ["id"]in RRF_dict :

                    RRF_dict [item ["id"]]["score"]+=1 /(rrf_k +j +1 )

                    RRF_dict [item ["id"]]["count"]+=1

                else :

                    RRF_dict [item ["id"]]={

                    "id":item ["id"],

                    "title":item .get ("title",""),

                    "content":item .get ("content",""),

                    "score":1 /(rrf_k +j +1 ),

                    "count":1

                    }

        RRF_list =sorted (RRF_dict .items (),key =lambda x :x [1 ]["score"],reverse =True )

        if len (texts )==1 :

            texts =texts [0 ][:k ]

            scores =scores [0 ][:k ]

        else :

            texts =[dict ((key ,item [1 ][key ])for key in ("id","title","content"))for item in RRF_list [:k ]]

            scores =[item [1 ]["score"]for item in RRF_list [:k ]]

        return texts ,scores





class DocExtracter :




    def __init__ (self ,corpus_name :str ,db_dir :str ="./corpus",cache :bool =False ):

        import os

        import json



        self .corpus_name =corpus_name

        self .db_dir =db_dir

        self .cache =cache





        corpus_dir =os .path .join (self .db_dir ,corpus_name .lower ())

        chunk_dir =os .path .join (corpus_dir ,"chunk")



        if not os .path .isdir (chunk_dir ):

            raise FileNotFoundError (

            f"[DocExtracter] chunk 目录不存在: {chunk_dir}，请确认你已经在这里生成了 jsonl 文件"

            )



        self .dict ={}





        for fname in sorted (os .listdir (chunk_dir )):

            if not fname .endswith (".jsonl"):

                continue

            fpath =os .path .join (chunk_dir ,fname )

            base =fname .replace (".jsonl","")



            with open (fpath ,"r",encoding ="utf-8")as f :

                for idx ,line in enumerate (f ):

                    line =line .strip ()

                    if not line :

                        continue

                    data =json .loads (line )





                    orig_id =data .get ("id")

                    if orig_id is not None :

                        data ["orig_id"]=orig_id





                    sid =f"{base}_{idx}"



                    self .dict [sid ]=data



        if not self .dict :

            raise RuntimeError (

            f"[DocExtracter] 在 {chunk_dir} 里没有读到任何 snippet，请检查 jsonl 是否生成成功"

            )





        print (f"[DocExtracter] Loaded {len(self.dict)} snippets from {chunk_dir}")



    def extract (self ,items ):


        if items is None :

            return []



        results =[]

        for i in items :



            if isinstance (i ,dict )and "content"in i :

                results .append (i )

                continue





            if isinstance (i ,str ):

                sid =i

            elif isinstance (i ,dict ):

                sid =i .get ("id")

            else :



                continue



            if sid is None :

                continue



            doc =self .dict .get (sid )

            if doc is None :







                continue



            results .append (doc )



        return results

