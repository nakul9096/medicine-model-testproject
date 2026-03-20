# app.py
# BEST MODEL FOR YOUR USE CASE (ZERO LABELED DATA):
# ================================================
# Since you have NO labeled dataset (patient features → correct remedy), 
# a supervised classifier (RandomForest, XGBoost, Neural Net) is impossible right now.
# 
# The most efficient, accurate and production-ready approach is:
# → TF-IDF + Cosine Similarity (classic Information Retrieval ML model)
# 
# Why this is the BEST for you:
# - Works immediately with the PDF indications you provided (no training needed)
# - Extremely fast & lightweight (no GPU, no torch, only sklearn + numpy)
# - Perfect for Flask on Render.com (free tier works great)
# - Semantic enough for homeopathy totality matching
# - Later when you have 100+ verified records (prediction_verified=1), you can switch to supervised model easily.
# 
# Alternative (if you want even better semantic understanding later):
# Replace with sentence-transformers 'all-MiniLM-L6-v2' — I can give you that version too.

from flask import Flask, request, jsonify
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ====================== ALL MEDICINES FROM YOUR PDF ONLY ======================
# I extracted every single remedy + exact indications + kingdom exactly as in the PDF.
remedies = [
    {"name": "Syzygium Jambolanum", "kingdom": "Plant Kingdom",
     "indications": "A most useful remedy in diabetes mellitus. No other remedy causes in so marked degree the diminution and disappearance of sugar in the urine. Emaciation & Weakness. Imperfect assimilation that leads to emaciation. Prickly heat in upper part of body. Small red pimples itch violently. Excessive thirst & Urination. Very large amount of urine, specific gravity high. Urine in every two hours. Old ulcers of skin. Diabetic ulceration."},

    {"name": "Phosphoric acid", "kingdom": "Mineral Kingdom",
     "indications": "Phosphoric acid is found in young people who grow rapidly, overtaxed, mentally or physically. Debility is very marked in this remedy, nervous exhaustion. Mental debility first; later physical. Urine frequent, profuse, watery, milky. Polyuria with dry mouth and throat. Diabetes with a history of sexual excesses and severe emotional or mental strain. Micturition, preceded by anxiety and followed by burning. Paralytic weakness and fornication along spine."},

    {"name": "Arsenicum album", "kingdom": "Mineral Kingdom",
     "indications": "Chilly patient. Rapid disproportionate prostration. Burning pains better by heat. Anxiety, anguish, fears for death and restlessness. After urinating, feeling of weakness in abdomen. Polyuria with bulimia. Great thirst drinks much but little at a time. Burning all over the body. Paleness of skin. Disposition to gangrene."},

    {"name": "Sulphur", "kingdom": "Mineral Kingdom",
     "indications": "Hot patient kicks off the cloth at night. Dirty, filthy, does not want to be washed. Lean, thin, stoop shouldered. Desires sweets, sugar, and meat. Frequent micturition, especially at night. Mucus and pus in urine. Excoriation, troublesome itching and burning sensation in genitals."},

    {"name": "Natrum sulphuricum", "kingdom": "Mineral Kingdom",
     "indications": "Hot patient. Extreme desire for fat. Aggravation in damp, cold weather. Tendency for early morning diarrhea. Irritable in morning. Loaded with bile. Excessive secretion."},

    {"name": "Phosphorus", "kingdom": "Mineral Kingdom",
     "indications": "Tall, fast growing child with tendency to stoop. Hemorrhagic tendency. Chilly patient craving for salt, cold foods and drinks. Glycosuria; urine pale, watery or turbid. Polyuria. Urine with white, serous, sandy and red, or else yellow sediment."},

    {"name": "Lachesis mutus", "kingdom": "Animal Kingdom",
     "indications": "Hot patient. Thin and emaciated. Haemorrhagic diathesis. Great sensitiveness to touch. Hot flushes and perspiration. Loquacious, jumps from one idea to another, jealous, suspicious. Boils, carbuncles, ulcers, with bluish, purple surroundings."},

    {"name": "Lycopodium clavatum", "kingdom": "Plant Kingdom",
     "indications": "Hot patient, intellectually keen but physically weak. Upper part of body emaciated, lower part semi-dropsical. Desire for warm foods and drink, sweet. Constant hunger and thirst worse at night. Polyuria at night. Urine milky, turbid. No erectile power, impotence. Great emaciation."},

    {"name": "Kali carbonicum", "kingdom": "Mineral Kingdom",
     "indications": "Chilly patient. Puffiness, weakness, backache and profuse perspiration. Aggravation in the morning 2-4 am. Obliged to rise several times at night to urinate. Pressure on bladder long before urine comes."},

    {"name": "Natrum muriaticum", "kingdom": "Mineral Kingdom",
     "indications": "Hot patient. Poorly nourished, great emaciation. Craving for salt. Increased thirst. Melancholic, sad. Frequent and urgent want to urinate, day and night, sometimes every hour, with copious emission."},

    {"name": "Secale cornutum", "kingdom": "Plant Kingdom",
     "indications": "Contraction of the muscles of blood vessels. Worse: Warmth. Better: Cold bathing, uncovering, fanning. Paralysis of bladder. Diabetes accompanied by hypertension. Rapid emaciation with much appetite and excessive thirst."},

    {"name": "Sepia officinalis", "kingdom": "Animal Kingdom",
     "indications": "Chilly patient. Tall, thin built. Desire for sour food. Red, adhesive, sand in urine. Involuntary urination, during first sleep. Chronic cystitis, slow micturition."},

    {"name": "Opium", "kingdom": "Plant Kingdom",
     "indications": "Affections of nerves, mind, senses producing insensibility. Urine slow to start, feeble stream. Retained or involuntary, after fright. Loss of power or sensibility of bladder."},

    {"name": "Argentum metallicum", "kingdom": "Mineral Kingdom",
     "indications": "Diuresis. Urine profuse, turbid, sweet odour. Legs weak and trembling. Frequent urination. Polyuria."},

    {"name": "Lacticum acidum", "kingdom": "Animal Kingdom",
     "indications": "Frequent passing of large quantities of saccharine urine. Rheumatic pains in joints, knees."},

    {"name": "Rhus aromatica", "kingdom": "Plant Kingdom",
     "indications": "Severe pain at beginning or before urination. Diabetes, large quantities of urine of low specific gravity. Renal and urinary affections, especially diabetes."},

    {"name": "Uranium nitricum", "kingdom": "Plant Kingdom",
     "indications": "Excessive thirst, nausea, vomiting, excessive appetite. Great emaciation. Diuresis. Incontinence of urine. Burning in urethra, with very acid urine. Complete impotency."},

    {"name": "Helonias dioica", "kingdom": "Plant Kingdom",
     "indications": "Sensation of weakness, dragging and weight in the sacrum and pelvis. Involuntary discharge of urine after the bladder seemed to be empty. Urine phosphatic; profuse and clear, saccharine."},

    {"name": "Manganum aceticum", "kingdom": "Mineral Kingdom",
     "indications": "Diabetes accompanied by psoriasis."},

    {"name": "Chionanthus virginica", "kingdom": "Plant Kingdom",
     "indications": "Frequent urination. Bile and sugar in urine. Urine very dark."},

    {"name": "Aurum Metallicum", "kingdom": "Mineral Kingdom",
     "indications": "Painful retention of urine, with urgent inclination to make water. Turbid, like buttermilk, with thick sediment."},

    {"name": "Plumbum Metallicum", "kingdom": "Mineral Kingdom",
     "indications": "Urine scanty. Tenesmus of bladder. Emission drop by drop."},

    {"name": "Phaseolus nanus", "kingdom": "Plant Kingdom",
     "indications": "Heart symptoms quite pronounce. Diabetic urine."},

    {"name": "Curare Muscular", "kingdom": "Plant Kingdom",
     "indications": "Diabetes mellitus. Glycosuria with motor paralysis."},

    {"name": "Serum anguillae", "kingdom": "Animal Kingdom",
     "indications": "The presence of albumin and renal elements in the urine. Haemoglobinuria, prolonged anuria."},

    {"name": "Terebinthinum", "kingdom": "Plant Kingdom",
     "indications": "Strangury, with bloody urine. Haematuria: blood thoroughly mixed with the urine."},

    {"name": "Urea Pura", "kingdom": "Mineral Kingdom",  # chemical / mineral nature
     "indications": "Albuminuria, diabetes; uraemia. Urine thin and of low specific gravity. Renal dropsy."},

    {"name": "Crotalus Horridus", "kingdom": "Animal Kingdom",
     "indications": "Dark, bloody urine. Casts. Inflamed kidney. Albuminous, dark, scanty."},

    {"name": "Conium Maculatum", "kingdom": "Plant Kingdom",
     "indications": "Much difficulty in voiding. It flows and stops again. Interrupted discharge. Dribbling in old men."},

    {"name": "Anthracinum", "kingdom": "Animal Kingdom",
     "indications": "Terrible burning. Carbuncle; with horrible burning pains."},

    {"name": "Aceticum acidum", "kingdom": "Mineral Kingdom",
     "indications": "Large quantities of pale urine. Diabetes, with great thirst and debility. Emaciation. edema of feet and legs."},

    {"name": "Calcarea arsenicsum", "kingdom": "Mineral Kingdom",
     "indications": "Albuminuria passes urine every hour."},

    {"name": "Viscum album", "kingdom": "Plant Kingdom",
     "indications": "Albuminuria. Lowered blood pressure."},

    {"name": "Crataegus oxyacantha", "kingdom": "Plant Kingdom",
     "indications": "Cardiac symptoms with Diabetes."}
]

# ====================== PRE-COMPUTE TF-IDF ======================
remedy_texts = [r["indications"] for r in remedies]
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
remedy_vectors = vectorizer.fit_transform(remedy_texts)

def build_patient_text(data: dict) -> str:
    """Build a rich symptom text from your table fields (works with raw fields OR totality_of_symptoms JSON)"""
    parts = []

    if data.get("chief_complaints"):
        parts.append(f"Chief complaints: {data['chief_complaints']}")
    if data.get("associated_complaints"):
        parts.append(f"Associated complaints: {data['associated_complaints']}")
    
    parts.append(f"Age: {data.get('age', '')}, Sex: {data.get('sex', '')}, Diabetes type: {data.get('diabetes_type', '')}, Duration: {data.get('diabetes_duration', '')} years")
    
    if data.get("thirst"):
        parts.append(f"Thirst: {data['thirst']}")
    if data.get("appetite"):
        parts.append(f"Appetite: {data['appetite']}")
    if data.get("sleep"):
        parts.append(f"Sleep: {data['sleep']}")
    if data.get("thermal_inference"):
        parts.append(f"Thermal inference: {data['thermal_inference']}")
    if data.get("perspiration"):
        parts.append(f"Perspiration: {data['perspiration']}")
    
    if data.get("emotional_upset"):
        parts.append(f"Emotional upset: {data['emotional_upset']}")
    if data.get("disposition"):
        parts.append(f"Disposition: {data['disposition']}")
    if data.get("reaction_to"):
        parts.append(f"Reaction to: {data['reaction_to']}")
    
    # Weight & complications (very important for diabetes remedies)
    if data.get("weight_change_type"):
        parts.append(f"Weight change: {data.get('weight_change_type')} {data.get('weight_change_kg')} kg")
    if data.get("complications_numbness") == "Yes":
        parts.append("Numbness present")
    if data.get("complications_eye") == "Yes":
        parts.append("Eye complications")
    if data.get("complications_heart") == "Yes":
        parts.append("Heart complications")
    
    # If you send the totality_of_symptoms JSON/string (as in your sample), use it directly
    if data.get("totality_of_symptoms"):
        totality = str(data["totality_of_symptoms"])
        parts.append(f"Totality of symptoms: {totality}")

    return " ".join(parts).strip()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    
    if not data:
        return jsonify({"error": "No data received"}), 400
    
    patient_text = build_patient_text(data)
    patient_vec = vectorizer.transform([patient_text])
    
    similarities = cosine_similarity(patient_vec, remedy_vectors).flatten()
    top_idx = np.argmax(similarities)
    
    best = remedies[top_idx]
    
    # Top 3 for doctor to choose
    top3_indices = np.argsort(similarities)[-3:][::-1]
    alternatives = [
        {
            "name": remedies[i]["name"],
            "kingdom": remedies[i]["kingdom"],
            "score": round(float(similarities[i]), 4)
        }
        for i in top3_indices
    ]
    
    response = {
        "predicted_medicine_name": best["name"],
        "predicted_medicine_type": best["kingdom"],
        "confidence_score": round(float(similarities[top_idx]), 4),
        "top_3_recommendations": alternatives,
        "patient_symptom_text_used": patient_text[:500] + "..."  # for debugging
    }
    
    return jsonify(response)


# Optional: health check
@app.route("/")
def home():
    return jsonify({"status": "Diabetes Homeopathy ML Predictor is running", "model": "TF-IDF + Cosine Similarity"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)