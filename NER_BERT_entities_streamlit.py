import streamlit as st
from transformers import pipeline
import requests
import time
import pandas as pd
import json
import folium
from streamlit_folium import st_folium
from io import StringIO

# Streamlit configuration
st.set_page_config(page_title="BERT NER + Geocoding", layout="centered")

# Load NER pipeline
@st.cache_resource
def load_ner_pipeline():
    model_name = "dslim/bert-base-NER"
    return pipeline("ner", model=model_name, tokenizer=model_name, aggregation_strategy="simple")

ner_pipeline = load_ner_pipeline()

# Simple geocoding function using OpenStreetMap Nominatim
def geocode_place(place_name):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': place_name,
            'format': 'json',
            'limit': 1,
            'addressdetails': 1
        }
        headers = {'User-Agent': 'BERT-NER-Geocoder'}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        time.sleep(0.5)  # polite delay
        if response.status_code == 200 and response.json():
            result = response.json()[0]
            return {
                'latitude': float(result['lat']),
                'longitude': float(result['lon']),
                'display_name': result['display_name']
            }
    except Exception as e:
        return {"error": str(e)}
    return {}

# Highlight entities in text
def highlight_text(text, entities):
    entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    for ent in entities:
        start, end = ent['start'], ent['end']
        original = text[start:end]
        span = f"<span style='background-color: #FFFF00;' title='{ent['type']}'>{original}</span>"
        text = text[:start] + span + text[end:]
    return text

# UI
st.title("Named Entity Recognition with Geocoding")
text_input = st.text_area("Enter text for entity recognition and location lookup:", height=250)

if "entities" not in st.session_state:
    st.session_state.entities = []
    st.session_state.text_input = ""
    st.session_state.highlighted_html = ""

process = st.button("Extract Entities")

if process:
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        st.session_state.text_input = text_input
        with st.spinner("Extracting entities..."):
            results = ner_pipeline(text_input)
        entities = []
        seen = set()
        for ent in results:
            ent_text = ent['word'].replace("##", "")
            key = (ent_text, ent['entity_group'], ent['start'])
            if key not in seen and ent['score'] > 0.6:
                seen.add(key)
                entity_info = {
                    'text': ent_text,
                    'type': ent['entity_group'],
                    'start': ent['start'],
                    'end': ent['end'],
                    'score': round(ent['score'], 3)
                }
                if ent['entity_group'] in ['LOC', 'GPE']:
                    geo = geocode_place(ent_text)
                    entity_info.update(geo)
                entities.append(entity_info)
        st.session_state.entities = entities
        st.session_state.highlighted_html = highlight_text(text_input, entities)

entities = st.session_state.entities
text_input = st.session_state.text_input
if entities:
    st.success(f"Found {len(entities)} entities.")
    st.markdown("### Highlighted Text")
    st.markdown(st.session_state.highlighted_html, unsafe_allow_html=True)

    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Extracting entities..."):
            results = ner_pipeline(text_input)

        # Process results
        entities = []
        seen = set()
        for ent in results:
            ent_text = ent['word'].replace("##", "")
            key = (ent_text, ent['entity_group'], ent['start'])
            if key not in seen and ent['score'] > 0.6:
                seen.add(key)
                entity_info = {
                    'text': ent_text,
                    'type': ent['entity_group'],
                    'start': ent['start'],
                    'end': ent['end'],
                    'score': round(ent['score'], 3)
                }
                if ent['entity_group'] in ['LOC', 'GPE']:
                    geo = geocode_place(ent_text)
                    entity_info.update(geo)
                entities.append(entity_info)

        st.success(f"Found {len(entities)} entities.")

        # Display highlighted text
        st.markdown("### Highlighted Text")
        st.markdown(highlight_text(text_input, entities), unsafe_allow_html=True)

        # Display map
        st.markdown("### Map of Locations")
        geo_entities = [e for e in entities if 'latitude' in e]
        if geo_entities:
            m = folium.Map(location=[geo_entities[0]['latitude'], geo_entities[0]['longitude']], zoom_start=2)
            for e in geo_entities:
                folium.Marker(
                    location=[e['latitude'], e['longitude']],
                    popup=e['text'],
                    tooltip=e.get('display_name', e['text'])
                ).add_to(m)
            st_folium(m, width=700)
        else:
            st.info("No geocoded entities found.")

        # Display table
        st.markdown("### Extracted Entities")
        df = pd.DataFrame(entities)
        st.dataframe(df)

        # Export options
        st.markdown("### Export Data")
        csv_data = df.to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="entities.csv", mime="text/csv")

        json_data = json.dumps(entities, indent=2)
        st.download_button("Download JSON", data=json_data, file_name="entities.json", mime="application/json")


        # JSON-LD export
        jsonld_data = {
            "@context": "http://schema.org/",
            "@type": "TextDigitalDocument",
            "text": text_input,
            "dateCreated": pd.Timestamp.now().isoformat(),
            "title": "Entity Extraction",
            "entities": []
        }

        for e in entities:
            ent = {
                "@type": "Thing",
                "name": e["text"],
                "additionalType": e["type"],
                "startOffset": e["start"],
                "endOffset": e["end"],
                "confidence": e["score"]
            }
            if "latitude" in e and "longitude" in e:
                ent["geo"] = {
                    "@type": "GeoCoordinates",
                    "latitude": e["latitude"],
                    "longitude": e["longitude"],
                    "name": e.get("display_name", e["text"])
                }
            jsonld_data["entities"].append(ent)

        jsonld_str = json.dumps(jsonld_data, indent=2)
        st.download_button("Download JSON-LD", data=jsonld_str, file_name="entities.jsonld", mime="application/ld+json")

        # HTML export with highlights
        highlighted_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset='utf-8'>
            <title>Entity Highlighting</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 2em; background: #fdfdfd; }}
                .entity {{ background-color: yellow; padding: 0.2em 0.4em; margin: 0 0.1em; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <h2>Entity Highlighted Text</h2>
            <p>{highlight_text(text_input, entities)}</p>
        </body>
        </html>
        """
        st.download_button("Download Highlighted HTML", data=highlighted_html, file_name="highlighted_entities.html", mime="text/html")
