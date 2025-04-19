import re
from typing import Dict, List
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# ---------------------------
# Load spaCy English model and VADER analyzer.
# ---------------------------
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = SentimentIntensityAnalyzer()

# ---------------------------
# Extended and Exhaustive Category-Keyword Mapping
# ---------------------------
category_keywords: Dict[str, List[str]] = {
    "resorts": [
        "resort", "resorts", "spa resort", "holiday resort", "luxury retreat", "wellness retreat"
    ],
    "burger/pizza shops": [
        "burger", "burgers", "cheeseburger", "cheeseburgers", "pizza", "pizzas", "pizzeria", "fast food", "pizza slice", "slices"
    ],
    "hotels/other lodgings": [
        "hotel", "hotels", "motel", "motels", "lodging", "lodgings", "bnb", "guest house", "guesthouses", "accommodation", "accommodations"
    ],
    "juice bars": [
        "juice bar", "juice bars", "smoothie bar", "smoothie bars", "fresh juice", "cold press juice", "protein shake", "protein shakes"
    ],
    "beauty & spas": [
        "spa", "spas", "massage", "massages", "salon", "salons", "facial", "facials", "beauty treatment", "manicure", "manicures", "pedicure", "pedicures"
    ],
    "gardens": [
        "botanical garden", "botanical gardens", "flower garden", "flower gardens", "public garden", "public gardens", "greenhouse"
    ],
    "amusement parks": [
        "theme park", "theme parks", "roller coaster", "roller coasters", "amusement park", "amusement parks", "fairground", "carnival rides", "carnival"
    ],
    "farmer market": [
        "farmer's market", "farmers market", "fresh market", "organic market", "produce market"
    ],
    "market": [
        "market", "markets", "bazaar", "bazaars", "flea market", "local vendors", "traders", "shopping area"
    ],
    "music halls": [
        "concert hall", "concert halls", "live music venue", "music hall", "music halls", "orchestra hall"
    ],
    "nature": [
        "nature", "natural environment", "wilderness", "scenery", "landscape", "wildlife", "outdoors"
    ],
    "tourist attractions": [
        "tourist attraction", "tourist attractions", "landmark", "landmarks", "monument", "monuments", "must-see", "sightseeing spot", "historic place"
    ],
    "beaches": [
        "beach", "beaches", "coast", "coasts", "seaside", "shoreline", "oceanfront"
    ],
    "parks": [
        "park", "parks", "green space", "green spaces", "recreational area", "walking trail", "stroll", "strolls", "jog", "jogs", "jogging"
    ],
    "theatres": [
        "theatre", "theatres", "stage show", "live theatre", "drama", "performance"
    ],
    "museums": [
        "museum", "museums", "exhibition", "exhibitions", "artifacts", "cultural exhibit", "natural history"
    ],
    "malls": [
        "mall", "malls", "shopping mall", "shopping malls", "retail center", "plaza", "commercial complex"
    ],
    "restaurants": [
        "restaurant", "restaurants", "eatery", "dining spot", "dining", "eat out", "fine dining", "casual dining", "cuisine", "meal", "meals"
    ],
    "pubs/bars": [
        "bar", "bars", "pub", "pubs", "brewery", "taproom", "drinking place"
    ],
    "local services": [
        "tailor", "repair service", "laundry service", "dry cleaner", "locksmith", "cleaner"
    ],
    "art galleries": [
        "art gallery", "art galleries", "exhibition space", "visual art", "modern art", "painting", "paintings", "sculpture"
    ],
    "dance clubs": [
        "dance club", "dance clubs", "nightclub", "nightclubs", "club", "clubs", "disco", "rave party", "club scene"
    ],
    "swimming pools": [
        "swimming pool", "swimming pools", "lap pool", "public pool", "indoor pool", "recreational pool"
    ],
    "bakeries": [
        "bakery", "bakeries", "pastry shop", "pastry shops", "bread store", "cake shop", "baked goods", "dessert", "cupcake"
    ],
    "cafes": [
        "cafe", "cafes", "coffee shop", "coffee shops", "coffe shop", "coffe shops", "espresso", "latte", "brew", "coffeehouse", "coffeehouses"
    ],
    "view points": [
        "viewpoint", "viewpoints", "scenic overlook", "observation deck", "panoramic view", "lookout", "overlook"
    ],
    "monuments": [
        "monument", "monuments", "historic statue", "memorial site", "war memorial"
    ],
    "zoo": [
        "zoo", "zoos", "wildlife park", "animal sanctuary", "zoological garden"
    ],
    "supermarket": [
        "supermarket", "supermarkets", "grocery store", "grocery stores", "food mart", "food market", "groceries"
    ]
}

# ---------------------------
# Utility: Map VADER compound sentiment to a 1.0-5.0 rating.
# ---------------------------
def score_from_sentiment(sent_score: float) -> float:
    if sent_score >= 0.5:
        return 5.0
    elif sent_score >= 0.2:
        return 4.0
    elif sent_score <= -0.5:
        return 1.0
    elif sent_score <= -0.2:
        return 2.0
    else:
        return 3.0

# ---------------------------
# Clause splitter using a broader set of discourse markers.
# ---------------------------
def split_into_clauses(sentence: str) -> List[str]:
    """
    Splits a sentence into clauses using common discourse markers and punctuation.
    """
    delimiter_pattern = r'\b(?:but|however|although|yet|still)\b|[;,]'
    clauses = re.split(delimiter_pattern, sentence)
    return [clause.strip() for clause in clauses if clause.strip()]

# ---------------------------
# Direct extraction function: clause-level sentiment analysis.
# ---------------------------
def extract_preferences_direct(text: str) -> Dict[str, float]:
    """
    Processes input text to extract category preferences using clause-level sentiment analysis.
    Returns a dictionary mapping each category to an average sentiment-derived score.
    """
    doc = nlp(text.lower())
    category_to_scores: Dict[str, List[float]] = {}
    for sent in doc.sents:
        clauses = split_into_clauses(sent.text)
        for clause in clauses:
            if not clause:
                continue
            clause_sentiment = sentiment_analyzer.polarity_scores(clause)['compound']
            clause_score = score_from_sentiment(clause_sentiment)
            for category, keywords in category_keywords.items():
                for keyword in keywords:
                    if re.search(rf"\b{re.escape(keyword)}\b", clause):
                        category_to_scores.setdefault(category, []).append(clause_score)
                        break
    final_scores = {cat: float(np.mean(scores)) for cat, scores in category_to_scores.items()}
    return final_scores

# ---------------------------
# Simulated Dynamic Exemplar Retrieval
# ---------------------------
# A small set of exemplar utterances with annotated preferences.
EXEMPLARS = [
    {
        "utterance": "I love nature and beaches, and enjoy the outdoors.",
        "preferences": {"nature": 5.0, "beaches": 5.0}
    },
    {
        "utterance": "I hate clubs and dancing, I never go to bars.",
        "preferences": {"dance clubs": 1.0, "pubs/bars": 1.0}
    },
    {
        "utterance": "I prefer cozy coffee shops and quiet cafes.",
        "preferences": {"cafes": 5.0}
    },
    {
        "utterance": "I enjoy strolling in parks and spending time in gardens.",
        "preferences": {"parks": 5.0, "gardens": 5.0}
    }
]

def retrieve_exemplar(summary: str, threshold: float = 0.6) -> Dict[str, float]:
    """
    Retrieves the exemplar whose utterance is most similar to the summary.
    Returns the exemplar's annotated preferences if the similarity exceeds the threshold; otherwise, returns an empty dict.
    """
    summary_doc = nlp(summary)
    best_sim = 0.0
    best_exemplar = None
    for exemplar in EXEMPLARS:
        exemplar_doc = nlp(exemplar["utterance"].lower())
        sim = summary_doc.similarity(exemplar_doc)
        if sim > best_sim:
            best_sim = sim
            best_exemplar = exemplar
    if best_sim >= threshold and best_exemplar is not None:
        return best_exemplar["preferences"]
    return {}

# ---------------------------
# Main PEARL-style Extraction Function for Single User Input
# ---------------------------
def pearl_extract_preferences_single(input_text: str, exemplar_weight: float = 0.3) -> Dict[str, float]:
    """
    Extracts customer preferences from a single input string.
    Mimics the PEARL architecture:
      - Uses the input as the contextualized utterance.
      - Dynamically retrieves an exemplar (if similar enough).
      - Performs direct clause-level extraction.
      - Combines exemplar preferences with extraction via weighted averaging.
    
    Args:
        input_text (str): The user's single input string.
        exemplar_weight (float): Weight for exemplar preferences (0 to 1). Direct extraction weight will be (1 - exemplar_weight).
    
    Returns:
        Dict[str, float]: Aggregated preference scores per category.
    """
    # Use the input text directly as the summary.
    summary = input_text.strip()
    
    # Retrieve exemplar preferences.
    exemplar_prefs = retrieve_exemplar(summary)
    
    # Extract preferences directly from the summary.
    extracted_prefs = extract_preferences_direct(summary)
    
    # Combine preferences using a weighted average.
    combined_prefs = {}
    all_categories = set(extracted_prefs.keys()) | set(exemplar_prefs.keys())
    for category in all_categories:
        ext_score = extracted_prefs.get(category)
        ex_score = exemplar_prefs.get(category)
        if ext_score is not None and ex_score is not None:
            combined = (1 - exemplar_weight) * ext_score + exemplar_weight * ex_score
        elif ext_score is not None:
            combined = ext_score
        else:
            combined = ex_score
        combined_prefs[category] = combined
    return combined_prefs



    