"""
@author: Nicola Lombardi - HW Baseband - Telit Cinterion
SCRIPT     DOCUMENTATION  (RELEASE 1.0) - SUBROUTINE FOR PERTURBATION TASK & Statistics
"""
import os
import pandas as pd

def display_array(list_resp):
  for el in list_resp:
    print(el)

def display_dict(d):
  """
  :param d: dict
  :return: null
  """
  for t in d.items():
    print(t)

def display_formatted_dict(d):
  """
  :param d: self.__dict__
  :return: null
  """
  for key, val in d.items():  # for each tuple
    print(f"{key}: {val}")

def describe_obj(d):
  """
  :param d: self.__dict__
  :return: dict converted into macro string
  """
  print(" sono qui _____________")
  return '\n'.join(f"{key}: {val}" for key, val in d.items())


def display_status_msg(string):
  print("*****************************************************************************************")
  print(f"\t\t\t\t\t\t\t*****{string}****")
  print("*****************************************************************************************")

def adjust_path(p='C:\\Users\\NicolaLo\\OneDrive - Telit Communications PLC\\Documents\\Python Scripts\\Machine_Learning_training\\classifier/mnist_data.csv'):
  return p.replace("\n", "/")

def adjust_path_level(p, level=1):
  # delete inconsistent char
  p = os.path.normpath(p)
  folders = p.split(os.path.sep) # num_levels = len(folders)
  if level > 0 and level <= len(folders):
    new_path = os.path.join(*folders[:-level])
    return new_path
  else:
    print("Chosen level is NaN.")
    return p


dataset_name = 'mnist_data.csv'

def path_processing():
  absolute_path = os.path.abspath(os.path.dirname(dataset_name)) # obtain path
  #print(f"1) Absolute path -> {absolute_path}")
  absolute_path = adjust_path(absolute_path) # adjust path
  #print(f"2) Absolute adjusted path -> {absolute_path}")
  absolute_path = adjust_path_level(absolute_path,1) # adjust level
  #print(f"3) Level adjusted path -> {absolute_path}")
  correct_path = absolute_path + "" + dataset_name # add dataset name
  #print(f"4) Path for {dataset_name}: absolute correct path -> {correct_path}")
  return correct_path

def generate_dataframe(csv_dataset_name):
  df = pd.read_csv(csv_dataset_name)
  print(df)
  return df


def obtain_size(df):
  return df.shape


if __name__ == '__main__':
  #print(adjust_path())
  #p = path_processing()
  # Load the MNIST digit data.
  df = generate_dataframe(dataset_name)
  # to review
  #X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
  r, c = obtain_size(df=df)
  print(f"Dataframe rows -> {r}\nDataframe cols -> {c}")
  print(type(df))

