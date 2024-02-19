import glob
import main
import re
#小椋先輩の手話判別と比べる
def compare_shuwa():
    print("a")
    files = glob.glob("./data/バナナ/*")
    sorted_file_names = sorted(files, key=extract_number)

    for i, file in enumerate(sorted_file_names):

        
        print(file)
        app = main.WebCam(False, file, 'banana')
        app.root.mainloop()
   

def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else None

# 数字に基づいてファイル名をソートZZ
    

      
        

compare_shuwa()