import glob
import main
import re
# file_name = "./test/こんばんは/15.MP4"
def compare_shuwa():
    print("a")
    files = glob.glob("./test/こんばんは/*")
    sorted_file_names = sorted(files, key=extract_number)

    for i, file in enumerate(sorted_file_names):

        
        print(file)
        app = main.WebCam(True, file, 'a')
        app.root.mainloop()
   

def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else None


compare_shuwa()
# app = main.WebCam(True, file_name, "a")
# app.root.mainloop()
