import LSA64_split
import main
import gc
from memory_profiler import profile
#訓練データを登録する関数
def setTrain():
    wordList = ["Opaque", "Red", "Green", "Yellow", "Bright", "LightBlue", "Colors", "Pink",
                "Women", "Enemy", "Son", "Man", "Away", "Drawer", "Born", "Learn", "Call",
                "Skimmer", "Bitter", "SweetMilk", "Milk", "Water", "Food", "Argentina",
                "Uruguay", "Country", "LastName", "Where", "Mock", "Birthday", "Breakfast",
                "Photo", "Hungry", "Map", "Coin", "Music", "Ship", "None", "Name", "Patience",
                "perfume", "Deaf", "Trap", "Rice", "Barbecue", "Candy", "ChewingGum", "Spaghetti", 
                "Yogurt", "Accept", "Thanks", "ShutDown", "Appear", "ToLand", "Catch", "Help", 
                "Dance", "Bathe", "Buy", "Copy", "Run", "Realize", "Give", "Find"]
    
    #train, test = LSA64_split.DatasetSplit()
    """
    メモリ不足の関係上、一度にファイルパスを呼ぶことは良くない 
    そこで、LSA64のファイルパスの特性を活かして001_010_XXXのファイルパス以外を取得するコードにする。
    """
    obj = main.WebCam()
    for i in range(1): # 64
        for j in range(1): #8
            for k in range(5):
                
                path = "./data/all/" + str(i+1).zfill(3) + "_" + str(j+1).zfill(3) + "_" + str(k+1).zfill(3) + ".mp4"
                print(path)
                obj.video_loop(path, wordList[i], False)
                print(i*60+j * 8 + k)
                #メモリの量に自信があるのならどうぞ
                #obj.holistic_manager.reset()
                gc.collect()
        print(i, wordList[i])

setTrain()