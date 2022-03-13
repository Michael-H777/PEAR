import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()

df=pd.read_csv("class_result.csv")
dfoutput=pd.read_csv('class_result_nutrition.csv')
calories=[]
fats=[]
carbs=[]
proteins=[]
summarys=[]
mores=[]

def search(class_name):
   url="http://www.fatsecret.com/calories-nutrition/search?q=%s" % class_name
   driver.get(url)
   time.sleep(1)
   firstsearch = driver.find_element(by=By.XPATH, value='/html/body/div[2]/div/div/div[2]/table/tbody/tr/td[1]/div/table[1]/tbody/tr[1]/td/a')
   driver.get(firstsearch.get_attribute('href'))
   time.sleep(1)
   calories=driver.find_element(by=By.XPATH, value='/html/body/div[2]/div/div/div[2]/table/tbody/tr/td[1]/div/table/tbody/tr/td[3]/div/table[1]/tbody/tr/td[1]/div[2]').text
   fat=driver.find_element(by=By.XPATH, value='/html/body/div[2]/div/div/div[2]/table/tbody/tr/td[1]/div/table/tbody/tr/td[3]/div/table[1]/tbody/tr/td[3]/div[2]').text
   carbs=driver.find_element(by=By.XPATH, value='/html/body/div[2]/div/div/div[2]/table/tbody/tr/td[1]/div/table/tbody/tr/td[3]/div/table[1]/tbody/tr/td[5]/div[2]').text
   protein=driver.find_element(by=By.XPATH, value='/html/body/div[2]/div/div/div[2]/table/tbody/tr/td[1]/div/table/tbody/tr/td[3]/div/table[1]/tbody/tr/td[7]/div[2]').text
   summary=driver.find_element(by=By.XPATH, value='/html/body/div[2]/div/div/div[2]/table/tbody/tr/td[1]/div/table/tbody/tr/td[3]/div/table[2]/tbody').text
   more=driver.find_element(by=By.XPATH, value='/html/body/div[2]/div/div/div[2]/table/tbody/tr/td[1]/div/table/tbody/tr/td[1]').get_attribute('innerHTML')
   return [calories,fat,carbs,protein,summary,more]

i=0
for index, row in dfoutput.iterrows():
   class_name=row['class_name']
   if str(row['calories'])!='nan':
      calories.append(row['calories'])
      fats.append(row['fat'])
      carbs.append(row['carbs'])
      proteins.append(row['protein'])
      summarys.append(row['summary'])
      mores.append(row['more'])
      continue
   try:
      [calorie,fat,carb,protein,summary,more]=search(class_name)
   except:
      calorie=""
      fat=""
      carb=""
      protein=""
      summary=""
      more=""
   calories.append(calorie)
   fats.append(fat)
   carbs.append(carb)
   proteins.append(protein)
   summarys.append(summary)
   mores.append(more)
   i+=1
   if i%30==0:
      print(i)
      df['calories'] = pd.Series(calories, index = df.index[:len(calories)])
      df['fat'] = pd.Series(fats, index = df.index[:len(fats)])
      df['carbs'] = pd.Series(carbs, index = df.index[:len(carbs)])
      df['protein'] = pd.Series(proteins, index = df.index[:len(proteins)])
      df['summary'] = pd.Series(summarys, index = df.index[:len(summarys)])
      df['more'] = pd.Series(mores, index = df.index[:len(mores)])
      df.to_csv('class_result_nutrition.csv',index=False)
df['calories'] = pd.Series(calories, index = df.index[:len(calories)])
df['fat'] = pd.Series(fats, index = df.index[:len(fats)])
df['carbs'] = pd.Series(carbs, index = df.index[:len(carbs)])
df['protein'] = pd.Series(proteins, index = df.index[:len(proteins)])
df['summary'] = pd.Series(summarys, index = df.index[:len(summarys)])
df['more'] = pd.Series(mores, index = df.index[:len(mores)])
df.to_csv('class_result_nutrition.csv',index=False)
   


