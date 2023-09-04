import numpy as np
import math

WORKSPACE_LIMITS = np.asarray([[0.276, 0.724], [-0.224, 0.224], [-0.0001, 0.4]])

# task
LANG_TEMPLATES = ["give me the {keyword}", # label
                "I need a {keyword}", # general label
                "grasp a {keyword} object", # shape or color
                "I want a {keyword} object", # shape or color
                "get something to {keyword}", # function
                ]

LABEL = ["tomato soup can", "banana", "red mug", "power drill", "strawberry", "apple", "lemon", 
        "peach", "pear", "orange", "knife", "flat screwdriver", "racquetball", "cup", "toy airplane",
        "dabao sod", "toothpaste", "darlie box", "dabao facewash", "pantene", "head shoulders", "tape"]

GENERAL_LABEL = ["fruit", "container", "toy", "cup"]
COLOR_SHAPE = ["yellow", "red", "round"]
FUNCTION = ["eat", "drink", "play", "hold other things"]
LABEL_DIR_MAP = ["002", "005", "007", "008", "011", "012", "013",
                "014", "015", "016", "018", "020", "021", "022", "024", 
                "038", "041", "058", "061", "062", "066", "070"]

KEYWORD_DIR_MAP = {"fruit": ["005", "011", "012", "013", "014", "015", "016", "017"],
                    "container": ["006", "007", "022"],
                    "toy": ["024", "026", "027", "028", "029", "030", "031",
                            "075", "076", "077", "078", "079", "080", "081", "082", "083", 
                            "084", "085", "086", "087"],
                    "cup": ["022"],
                    "yellow": ["005", "013", "028", "031"],
                    "red": ["011", "012"],
                    "round": ["016", "017", "021"],
                    "box": ["039"],
                    "eat": ["005", "011", "012", "013", "014", "015", "016", "017"], 
                    "drink": ["057"],
                    "play": ["024", "026", "027", "028", "029", "030", "031"],
                    "hold other things": ["006", "007", "022"]}

UNSEEN_LABEL = ["black marker", "bleach cleanser", "blue moon", "gelatin box", "magic clean", "pink tea box", "red marker", 
                "remote controller", "repellent", "shampoo", "small clamp", "soap dish", "suger", "suger", "two color hammer",
                "yellow bowl", "yellow cup"]

UNSEEN_LABEL_DIR_MAP = ["black_marker", "bleach_cleanser", "blue_moon", "gelatin_box", "magic_clean", "pink_tea_box", "red_marker", 
                "remote_controller_1", "repellent", "shampoo", "small_clamp", "soap_dish", "suger_1", "suger_2", "two_color_hammer",
                "yellow_bowl", "yellow_cup"]

UNSEEN_GENERAL_LABEL = ["suger", "container"]
UNSEEN_COLOR_SHAPE = ["yellow", "red"]
UNSEEN_FUNCTION = ["clean"]
UNSEEN_KEYWORD_DIR_MAP = {"suger": ["suger_1", "suger_2"],
                    "container": ["soap_dish", "yellow_cup"],
                    "yellow": ["yellow_bowl", "yellow_cup"],
                    "red": ["red_marker"],
                    "clean": ["bleach_cleanser", "blue_moon", "magic_clean", "shampoo"]}

# image
PIXEL_SIZE = 0.002
IMAGE_SIZE = 224