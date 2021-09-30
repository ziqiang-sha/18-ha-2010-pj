# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 23:18:11 2021

@author: Maurice
"""

import mysql.connector
from score import score
import datetime
import json

def get_dt_str():
    current_time = datetime.datetime.now()
    time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    return time_str
def sec2time_str(seconds):
    return datetime.time(seconds//3600,(seconds//60)%60,seconds%60).strftime('%H:%M:%S')

class Database:
    def __init__(self,config_file='database_config.json'):
        with open(config_file,"r") as fp:
            database_config = json.load(fp)
          
        #USER = "test_user_mypc"
        #PASSWORD = "test"
        #HOST="130.83.56.68"
        #PORT="3309"
        
        self.mydb = mysql.connector.connect(
            host=database_config["HOST"],
            port=database_config["PORT"],
            user=database_config["USER"],
            password=database_config["PASSWORD"],
            database="wki_main")  #TODO remove hardcode
        print("Database Opened")
        self.database_config=database_config  
    
    def __del__(self):
        self.mydb.close()
        print('Database Closed')
        
    def print_databases(self):
        self.cursor.execute("SHOW DATABASES")
        for x in self.cursor:
            print(x)
            
    def put_scored_entry(self,dataset_nr,team_nr,run_count_team,f1_score,multi_score,model_nr,run_time,confusion_matrix=None):
        '''
    
        Parameters
        ----------
        dataset_nr : INT from db
        team_nr : INT from db
        run_count_team : INT from db check  if +1
        f1_score : float
        multi_score : float
        model_nr : INT from db (maybe new model)
        run_time : INT in seconds
        confusion_matrix : dict, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        cursor = self.mydb.cursor()
        time_str = get_dt_str()
        run_time_str = sec2time_str(run_time)
        
        confusion_nr=self.put_confusion_matrix(confusion_matrix)
        team_name = self.get_team_name(team_nr)
        if team_name==None:
            print("WARNING! Team does not exist! Setting team_nr to NULL")
            team_nr=None
        
        sql = "INSERT INTO `wki_main`.`wki_scored_runs`(`dataset_nr`,`team_nr`,`run_count_team`,\
        `datetime`,`f1_score`,`multi_score`,`model_nr`,`confusion_matrix`,`run_time`)\
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        val = (dataset_nr,team_nr,run_count_team,time_str,f1_score,multi_score,model_nr,confusion_nr,run_time_str)
        cursor.execute(sql,val)
        entry_id = cursor.lastrowid
        self.mydb.commit()
        cursor.close()
        print("Scored entry inserted for Team",team_name)
        
        return entry_id
        
    def put_confusion_matrix(self,confusion_matrix):
        if confusion_matrix != None:
            cursor = self.mydb.cursor()
            content = list()
            true_names = list(confusion_matrix.keys())
            for tn in true_names:
                content.extend(list(confusion_matrix[tn].values()))
            content = tuple(content)
            sql = "INSERT INTO `wki_main`.`wki_confusion_tables`(`Nn`,`Na`,`No`,`Np`,`An`,`Aa`,`Ao`,`Ap`,`On`,`Oa`,`Oo`,`Op`,`Pn`,`Pa`,`Po`,`Pp`) \
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            cursor.execute(sql,content)
            table_id = cursor.lastrowid
            self.mydb.commit()
            cursor.close()
            return table_id
        return None
    
    def put_unscored_entry(self,dataset_nr,team_nr,model_nr,run_time,output_file):
        
        cursor = self.mydb.cursor()
        
        time_str = get_dt_str()
        team_name = self.get_team_name(team_nr)
        if team_name==None:
            print("WARNING! Team does not exist! Setting team_nr to NULL")
            team_nr=None
        
        sql = "INSERT INTO `wki_main`.`wki_unscored_runs`(`dataset_nr`,`team_nr`,`datetime`,`model_nr`,`run_time`,`output_file`) \
            VALUES(%s,%s,%s,%s,%s,%s)"
        val = (dataset_nr,team_nr,time_str,model_nr,sec2time_str(run_time),output_file)    
        cursor.execute(sql,val)
        entry_id = cursor.lastrowid
        self.mydb.commit()
        print("Unscored entry inserted for Team",team_name)
        cursor.close()
        return entry_id
        
        
    def put_model(self,team_nr,model_name,is_binary_classifier,parameter_dict):
        
        parameters = json.dumps(parameter_dict,indent=4)
        
        team_name = self.get_team_name(team_nr)
        if team_name==None:
            print("WARNING! Team does not exist! Setting team_nr to NULL")
            team_nr=None
        
        cursor = self.mydb.cursor()
        
        sql = "INSERT INTO `wki_main`.`wki_models`(`team_nr`,`model_name`,`is_binary_classifier`,`parameters`) \
            VALUES(%s,%s,%s,%s)"
        val = (team_nr,model_name,is_binary_classifier,parameters)
        cursor.execute(sql,val)
        model_id=cursor.lastrowid
        self.mydb.commit()
        print("model added to database")
        cursor.close()
        return model_id
        
        
    def get_team_id(self,team_name):
        
        cursor = self.mydb.cursor(prepared=True)
        #sql = ("SELECT `wki_main`.`wki_teams`.`team_id` FROM `wki_main`.`wki_teams`"
        #       " WHERE `wki_teams`.`team_name` = '%s'") 
        sql = ("SELECT `wki_main`.`wki_teams`.`team_id` FROM `wki_main`.`wki_teams`"
               " WHERE `wki_teams`.`team_name` = '%s'") %(team_name)
        cursor.execute(sql)
        (team_id,) = cursor.fetchone()
        cursor.close()
        return team_id
    
    def get_team_name(self,team_id):
        cursor = self.mydb.cursor(prepared=True)
        #sql = ("SELECT `wki_main`.`wki_teams`.`team_id` FROM `wki_main`.`wki_teams`"
        #       " WHERE `wki_teams`.`team_name` = '%s'") 
        sql = ("SELECT `wki_main`.`wki_teams`.`team_name` FROM `wki_main`.`wki_teams`"
               " WHERE `wki_teams`.`team_id` = %s") %(team_id)
        cursor.execute(sql)
        (team_name,) = cursor.fetchone()
        cursor.close()
        return team_name
        
    
    def get_dataset_folder(self,data_set_id):
        
        cursor = self.mydb.cursor(prepared=True)
        
        sql = ("SELECT `wki_main`.`wki_datasets`.`folder_name` FROM `wki_main`.`wki_datasets`"
               " WHERE `wki_datasets`.`dataset_id` = %s") %(data_set_id)
        cursor.execute(sql)
        (dataset_folder,) = cursor.fetchone()
        cursor.close()
        return dataset_folder
    
    def get_nr_runs(self,team_id):
        cursor = self.mydb.cursor(prepared=True)
     
        sql = ("SELECT MAX(`wki_scored_runs`.`run_count_team`) AS `runs` FROM (`wki_scored_runs` JOIN `wki_teams` ON ((`wki_teams`.`team_id` = `wki_scored_runs`.`team_nr`)))"
               " WHERE `wki_teams`.`team_id` = %s") %(team_id)
        cursor.execute(sql)
        (nr_runs,) = cursor.fetchone()
        cursor.close()
        return nr_runs
    
    
        
        