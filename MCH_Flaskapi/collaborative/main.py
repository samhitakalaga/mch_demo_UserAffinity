import os
import pickle
import numpy as np
import pandas as pd
import argparse
import joblib
import datetime
from io import StringIO
import nltk
import json

from useraffinity.useraffinity_model_artwork import Useraffinityartwork
from collab.artwork_collab_training import Collaborative_Artwork


def create_collab_artwork(utilitymatrix_artwork,all_artwork_profile,ovr_artwork_profile):

    collab_artwork = Collaborative_Artwork(utilitymatrix_artwork,all_artwork_profile,ovr_artwork_profile)
    collab_artwork.fit(utilitymatrix_artwork,all_artwork_profile,ovr_artwork_profile)
    return collab_artwork.result, collab_artwork.upe, collab_artwork.map_k, collab_artwork.user_vec_df,collab_artwork.pred_df_with_feature

def train_collab_model(utilitymatrix_artwork, all_artwork_profile, ovr_artwork_profile):  
    """
    
    Artwork collab model Training
    
    """
    
    print('\n********Artwork collaborative model Training started********')
    nltk.download('all')
    try:
        collab_artwork, upe, map_k, user_vec_df, pred_df_with_feature = create_collab_artwork(utilitymatrix_artwork
                                                                         , all_artwork_profile, ovr_artwork_profile)
        print('\n********Artwork collaborative model Training Completed Successfully********')
    except Exception as e:
        collab_artwork = None
        upe = None
        map_k = None
        user_vec_df = None
        pred_df_with_feature=None
        print(f"\nArtwork collab model training failed with Error: {e}")


    return collab_artwork, upe, map_k, user_vec_df, pred_df_with_feature


def driver(utilitymatrix_artwork,all_artwork_profile,ovr_artwork_profile):

    # Collab model training
    collab_artwork, upe, map_k, user_vec_df, pred_df_with_feature = train_collab_model(utilitymatrix_artwork
                                                                 , all_artwork_profile,ovr_artwork_profile)
    return collab_artwork, upe, map_k, user_vec_df, pred_df_with_feature



if __name__ == '__main__':
    utilitymatrix_artwork=pd.read_csv(r"C:\Users\smurugan\Desktop\MCH_Flaskapi\robert_actual_feature_case2.csv")
    all_artwork_profile=pd.read_csv(r"C:\Users\smurugan\Desktop\mch_latest\demo\content_complete_dump.csv")
    ovr_artwork_profile=pd.read_csv(r"C:\Users\smurugan\Desktop\mch_latest\demo\content_latest_OVR.csv")

    # train model
    collab_artwork, upe, map_k, user_vec_df,pred_df_with_feature = driver(utilitymatrix_artwork,all_artwork_profile,ovr_artwork_profile)
