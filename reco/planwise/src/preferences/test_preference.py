from preference_extractor import pearl_extract_preferences_single
# Single input text from the user.
input_text = (
        "I hate clubs and dancing but love coffe shops, and I enjoy a casual stroll in nature. "
        "My favorite thing is to visit must-see places and museums when visiting a city. "
        "I'm also a big fan of going to the beach and swimming, although I'm not keen on going to the gym."
    )
prefs = pearl_extract_preferences_single(input_text, exemplar_weight=0.3)
print("Extracted Preferences:")
for cat, score in prefs.items():
    print(f"{cat}: {score:.1f}")