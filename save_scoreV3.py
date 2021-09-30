# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:00:13 2021

@author: Maurice
"""

from score import score
import argparse
import datetime
import os, csv

from wki_utilities import Database








def save_scoreV3(output_dir,data_set,run_count_team=0,run_time=0,model_name='test_model',is_binary_classifier=True,test_dir='../test/',teamname='KISMED',training_name='train'):
    
    # TODO runtime
    
    current_time = datetime.datetime.now()
    F1,F1_mult,Conf_Matrix = score(args.test_dir)
    
    db = Database()
    
    team_id = db.get_team_id(teamname)
    nr_runs = db.get_nr_runs(team_id)
    model_id = db.put_model(team_id,model_name,is_binary_classifier,parameter_dict=None)
    db.put_scored_entry(data_set,team_id,run_count_team,F1,F1_mult,model_id,run_time,Conf_Matrix)
    
    
   ##################### alt ################################################ 
    
    filename =   output_dir + teamname + '.csv'  
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as scores_file:
            scores_writer = csv.writer(scores_file, delimiter=';')
            scores_writer.writerow(['team_name','date_time','training','dataset','F1_score','multilabel_score'])
    
    with open(filename, mode='+a', newline='') as scores_file:
        scores_writer = csv.writer(scores_file, delimiter=';')
        
        scores_writer.writerow([teamname,str(current_time),training_name, args.data_set, F1, F1_mult] )
    if not os.path.exists(args.output_dir + 'Confusion/'):
        os.mkdir(args.output_dir + 'Confusion/')
    with open(args.output_dir + 'Confusion/' +teamname+str(current_time.date())+'_'+ str(current_time.hour)+'_'+str(current_time.minute) + '.csv',mode='w',newline='') as matrix_file:
        matrix_writer = csv.writer(matrix_file, delimiter=';')
        true_names = list(Conf_Matrix.keys())
        pred_names = list(Conf_Matrix[true_names[0]].keys())
        matrix_writer.writerow([''] + pred_names)
        for tn in true_names:
            matrix_writer.writerow([tn] + list(Conf_Matrix[tn].values()))






if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Score based on Predictions')
    parser.add_argument('output_dir', action='store',type=str)
    parser.add_argument('data_set', action='store',type=int)
    parser.add_argument('--test_dir', action='store',type=str,default='../test/')
    parser.add_argument('--teamname', action='store',type=str,default='KISMED')
    parser.add_argument('--training_name', action='store',type=str,default='train')
    
    
    

    args = parser.parse_args()
    
    save_scoreV3(args.output_dir,args.data_set,args.test_dir,args.team_name,args.training_name)
        
    

        