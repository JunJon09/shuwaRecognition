import LSA64_split
import main
import gc

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
    
    train, test = LSA64_split.DatasetSplit()
    
    for i, t in enumerate(train):
        if i%40 == 0:
            word = wordList[int(i/40)]
            print(word)
        print(i)
        path = "/Users/jonmac/jon/研究/手話/手話単語判別/shuwa/data/all/" + t
        app = main.WebCam(False, path, word)
        del app
        gc.collect()
        

    print(len(train), len(train)/64)

setTrain()