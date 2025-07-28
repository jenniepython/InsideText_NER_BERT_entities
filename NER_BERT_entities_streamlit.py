def render_sidebar(self):
        """Render the sidebar with model information."""
        st.sidebar.subheader("Model & Analysis Information")
        st.sidebar.info("""
        **NER Model**: BERT-based NER (dslim/bert-base-NER)
        
        **Contextual Analysis**: Intelligent rule-based linguistic analysis
        - Semantic feature extraction
        - Contextual pattern recognition
        - Domain-specific categorization
        
        **Linking**: Wikidata, Wikipedia, Britannica, OpenStreetMap
        
        **Features**: 
        - Smart semantic categorization
        - Entity relationship analysis
        - Contextual confidence scoring
        - Geographic context detection
        - No additional downloads required
        """)
        
        st.sidebar.subheader("Entity Types & Categories")
        st.sidebar.info("""
        **BERT NER Types**:
        - PERSON, ORGANIZATION, LOCATION, GPE, MISC
        
        **Smart Categories**:
        - mythological, historical_ruler, military_leader
        - ancient_place, trade_center, cultural_site
        - educational, religious, commercial
        
        **Context Analysis**:
        - Temporal markers (historical references)
        - Authority markers (power/leadership)
        - Geographic markers (location references)
        - Cultural markers (civilization/society)
        - Military markers (conflict/war)
        - Commercial markers (trade/business)
        """)
        
        st.sidebar.subheader("Analysis Approach")
        st.sidebar.success("""
        ✅ **Fully Self-Contained**
        - No API calls required
        - No additional model downloads
        - Fast rule-based analysis
        - Reliable contextual insights
        - Works entirely within Streamlit
        """)

    def render_header(self):
        """Render the application header with logo."""
        # Display logo if it exists
        try:
            logo_path = "logo.png"
            if os.path.exists(logo_path):
                st.image(logo_path, width=300)
            else:
                st.info("💡 Place your logo.png file in the same directory as this app to display it here")
        except Exception as e:
            st.warning(f"Could not load logo: {e}")        
        
        # Add some spacing after logo
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main title and description
        st.header("From Text to Linked Data using BERT NER + Smart Context")
        st.markdown("**Extract and link named entities using BERT NER with intelligent rule-based contextual analysis**")
        
        # Create a simple process diagram
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #E0D7C0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="background-color: #C4C3A2; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Input Text</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="background-color: #9fd2cd; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>BERT NER (dslim/bert-base-NER)</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="background-color: #EFCA89; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Smart Contextual Analysis</strong><br><small>Rule-based linguistic features</small>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="text-align: center;">
                    <strong>Link to Knowledge Bases:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #C3B5AC; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Wikidata</strong><br><small>Structured knowledge</small>
                    </div>
                    <div style="background-color: #C4A998; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                        <strong>Wikipedia/Britannica</strong><br><small>Encyclopedia articles</small>
                    </div>
                    <div style="background-color: #BF7B69; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Geocoding</strong><br><small>Coordinates & locations</small>
                    </div>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="text-align: center;">
                    <strong>Enhanced with Intelligence:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #D4C5B9; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Semantic Categories</strong><br><small>mythological, historical_ruler, etc.</small>
                    </div>
                    <div style="background-color: #CCBEAA; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Contextual Features</strong><br><small>temporal, authority, cultural markers</small>
                    </div>
                    <div style="background-color: #DDD3C0; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Smart Keywords</strong><br><small>significant terms & relationships</small>
                    </div>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="text-align: center;">
                    <strong>Export Formats:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #E8E1D4; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em; border: 2px solid #EFCA89;">
                         <strong>Rich JSON-LD</strong><br><small>Structured data</small>
                    </div>
                    <div style="background-color: #E8E1D4; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em; border: 2px solid #C3B5AC;">
                         <strong>Interactive HTML</strong><br><small>Visual analysis</small>
                    </div>
                    <div style="background-color: #E8E1D4; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em; border: 2px solid #BF7B69;">
                         <strong>GeoJSON</strong><br><small>Mapping data</small>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)    def __init__(self):
        """Initialise the BERTEntityLinker."""
        
        # Color scheme for different entity types in HTML output
        self.colors = {
            'PERSON': '#BF7B69',          # F&B Red earth        
            'ORGANIZATION': '#9fd2cd',    # F&B Blue groun#!/usr/bin/env python3
"""
Streamlit Entity Linker Application with BERT-based NER Only

A web interface for entity extraction and linking using only the dslim/bert-base-NER model
with contextual enhancement and comprehensive linking to external knowledge bases.

Author: Based on NER_spaCy_streamlit.py - Modified to use only BERT NER with context
Version: 2.2 - Cleaned to use only dslim/bert-base-NER with contextual analysis
"""

import streamlit as st

# Configure Streamlit page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="From Text to Linked Data using BERT NER",
    layout="centered",  
    initial_sidebar_state="collapsed" 
)

# Authentication is REQUIRED - do not run app without proper login
try:
    import streamlit_authenticator as stauth
    import yaml
    from yaml.loader import SafeLoader
    import os
    
    # Check if config file exists
    if not os.path.exists('config.yaml'):
        st.error("Authentication required: config.yaml file not found!")
        st.info("Please ensure config.yaml is in the same directory as this app.")
        st.stop()
    
    # Load configuration
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    # Setup authentication
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    # Check if already authenticated via session state
    if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
        name = st.session_state['name']
        authenticator.logout("Logout", "sidebar")
        # Continue to app below...
    else:
        # Render login form
        try:
            # Try different login methods
            login_result = None
            try:
                login_result = authenticator.login(location='main')
            except TypeError:
                try:
                    login_result = authenticator.login('Login', 'main')
                except TypeError:
                    login_result = authenticator.login()
            
            # Handle the result
            if login_result is None:
                # Check session state for authentication result
                if 'authentication_status' in st.session_state:
                    auth_status = st.session_state['authentication_status']
                    if auth_status == False:
                        st.error("Username/password is incorrect")
                        st.info("Try username: demo_user with your password")
                    elif auth_status == None:
                        st.warning("Please enter your username and password")
                    elif auth_status == True:
                        st.rerun()  # Refresh to show authenticated state
                else:
                    st.warning("Please enter your username and password")
                st.stop()
            elif isinstance(login_result, tuple) and len(login_result) == 3:
                name, auth_status, username = login_result
                # Store in session state
                st.session_state['authentication_status'] = auth_status
                st.session_state['name'] = name
                st.session_state['username'] = username
                
                if auth_status == True:
                    st.rerun()  # Refresh to show authenticated state
                elif auth_status == False:
                    st.error("Username/password is incorrect")
                    st.stop()
            else:
                st.error(f"Unexpected login result format: {login_result}")
                st.stop()
                
        except Exception as login_error:
            st.error(f"Login method error: {login_error}")
            st.stop()
        
except ImportError:
    st.error("Authentication required: streamlit-authenticator not installed!")
    st.info("Please install streamlit-authenticator to access this application.")
    st.stop()
except Exception as e:
    st.error(f"Authentication error: {e}")
    st.info("Cannot proceed without proper authentication.")
    st.stop()

import sys
import subprocess

# Install packages if not available
try:
    import torch
    import transformers
    import numpy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers", "numpy"])
    import torch
    import transformers
    import numpy

import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io
import base64
from typing import List, Dict, Any, Optional
import sys
import os
import re
import time
import requests
import urllib.parse

class BERTEntityLinker:
    """
    Main class for BERT-based entity linking functionality.
    
    This class handles the complete pipeline from text processing to entity
    extraction using only the dslim/bert-base-NER model with contextual enhancement,
    validation, linking, and output generation.
    """
    
    def __init__(self):
        """Initialise the BERTEntityLinker."""
        
        # Color scheme for different entity types in HTML output
        self.colors = {
            'PERSON': '#BF7B69',          # F&B Red earth        
            'ORGANIZATION': '#9fd2cd',    # F&B Blue ground
            'GPE': '#C4C3A2',             # F&B Cooking apple green
            'LOCATION': '#EFCA89',        # F&B Yellow ground. 
            'FACILITY': '#C3B5AC',        # F&B Elephants breath
            'EVENT': '#C4A998',           # F&B Dead salmon
            'PRODUCT': '#CCBEAA',         # F&B Oxford stone
            'WORK_OF_ART': '#D4C5B9',     # F&B String
            'MISC': '#DDD3C0',            # F&B Old White
        }
        
        # Initialise model
        self.ner_pipeline = None
        self._load_model()

    def _load_model(self):
        """Load BERT model for entity extraction."""
        try:
            from transformers import pipeline
            
            # Load BERT NER model
            with st.spinner("Loading BERT NER model..."):
                try:
                    ner_model_name = "dslim/bert-base-NER"
                    self.ner_pipeline = pipeline(
                        "ner",
                        model=ner_model_name,
                        tokenizer=ner_model_name,
                        aggregation_strategy="simple"
                    )
                    st.success("dslim/bert-base-NER model loaded successfully")
                except Exception as e:
                    st.error(f"Failed to load NER model: {e}")
                    st.stop()
                    
        except ImportError:
            st.error("Required packages not installed. Please install:")
            st.code("pip install transformers torch")
            st.stop()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

    def _generate_contextual_analysis(self, text: str, entity_text: str, entity_type: str, start: int, end: int) -> Dict[str, Any]:
        """Generate contextual analysis using rule-based approaches."""
        context_info = {
            'sentence_context': self._extract_sentence_context({'text': entity_text, 'start': start, 'end': end}, text),
            'semantic_category': self._determine_semantic_category_from_context(entity_text, entity_type, text),
            'context_keywords': self._extract_context_keywords(entity_text, text),
            'entity_frequency': text.lower().count(entity_text.lower()),
            'surrounding_entities': []
        }
        
        return context_info

    def extract_entities(self, text: str):
        """Extract named entities from text using BERT model only."""
        entities = []
        
        if not self.ner_pipeline:
            st.error("NER model not loaded")
            return entities
        
        try:
            raw_entities = self.ner_pipeline(text)
            
            # Process BERT entities
            for ent in raw_entities:
                entity_type = self._map_entity_type(ent['entity_group'])
                
                # Filter out low-confidence entities
                if ent['score'] < 0.6:
                    continue
                
                # Clean up entity text
                entity_text = ent['word'].replace('##', '').strip()
                
                # Skip very short or invalid entities
                if len(entity_text) <= 1:
                    continue
                
                # Create entity dictionary
                entity = {
                    'text': entity_text,
                    'type': entity_type,
                    'start': ent['start'],
                    'end': ent['end'],
                    'confidence': ent['score'],
                    'original_label': ent['entity_group'],
                    'extraction_method': 'bert_transformer'
                }
                
                # Add contextual information
                context_info = self._generate_contextual_analysis(text, entity_text, entity_type, ent['start'], ent['end'])
                entity.update(context_info)
                
                # Validation
                if self._is_valid_entity(entity, text):
                    entities.append(entity)
                        
        except Exception as e:
            st.error(f"Entity extraction failed: {e}")
            return []
        
        # Remove overlapping entities
        entities = self._remove_overlapping_entities(entities)
        
        return entities

    def _map_entity_type(self, ner_label: str) -> str:
        """Map NER model labels to our standardised types."""
        mapping = {
            'PER': 'PERSON',
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION',
            'ORGANIZATION': 'ORGANIZATION',
            'LOC': 'LOCATION',
            'LOCATION': 'LOCATION',
            'GPE': 'GPE',
            'MISC': 'MISC'
        }
        return mapping.get(ner_label, 'MISC')

    def _is_valid_entity(self, entity: Dict[str, Any], text: str) -> bool:
        """Validate an entity using heuristics and context."""
        entity_text = entity['text'].strip()
        
        # Skip very short entities
        if len(entity_text) <= 1:
            return False
        
        # Skip common false positives
        false_positives = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        if entity_text.lower() in false_positives:
            return False
        
        # Skip entities that are mostly punctuation
        if len(re.sub(r'[^\w\s]', '', entity_text)) <= 1:
            return False
        
        # Additional validation based on entity type
        if entity['type'] == 'PERSON':
            return self._validate_person_entity(entity_text)
        elif entity['type'] in ['ORGANIZATION', 'LOCATION', 'GPE']:
            return self._validate_place_or_org_entity(entity_text)
        
        return True

    def _validate_person_entity(self, entity_text: str) -> bool:
        """Validate person entities."""
        # Should contain at least one capital letter
        if not any(c.isupper() for c in entity_text):
            return False
        
        # Should not be all caps (likely not a person name)
        if entity_text.isupper() and len(entity_text) > 3:
            return False
        
        # Should not contain numbers (for names)
        if any(c.isdigit() for c in entity_text):
            return False
        
        return True

    def _validate_place_or_org_entity(self, entity_text: str) -> bool:
        """Validate place or organisation entities."""
        # Should contain at least one capital letter
        if not any(c.isupper() for c in entity_text):
            return False
        
        # Should not be just punctuation
        if not any(c.isalnum() for c in entity_text):
            return False
        
        return True

    def _remove_overlapping_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping entities, keeping the highest confidence ones."""
        entities.sort(key=lambda x: x['start'])
        
        filtered = []
        for entity in entities:
            overlaps = False
            for existing in filtered[:]:
                # Check if entities overlap
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    # Keep the higher confidence entity
                    if entity.get('confidence', 0) > existing.get('confidence', 0):
                        filtered.remove(existing)
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered

    def _extract_sentence_context(self, entity: Dict[str, Any], text: str) -> str:
        """Extract the sentence containing the entity."""
        # Find sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if entity['text'] in sentence:
                return sentence.strip()
        
        # Fallback: extract context window around entity
        start = max(0, entity['start'] - 100)
        end = min(len(text), entity['end'] + 100)
        return text[start:end].strip()

    def _determine_semantic_category_from_context(self, entity_text: str, entity_type: str, text: str) -> str:
        """Determine semantic category using intelligent rule-based context analysis."""
        # Get context around the entity
        entity_pos = text.lower().find(entity_text.lower())
        if entity_pos == -1:
            return 'general'
        
        # Extract surrounding context (wider window for better analysis)
        start = max(0, entity_pos - 300)
        end = min(len(text), entity_pos + len(entity_text) + 300)
        context = text[start:end].lower()
        
        # Smart contextual analysis using linguistic patterns and co-occurrence
        category = self._analyze_context_patterns(entity_type, entity_text.lower(), context)
        
        return category
    
    def _analyze_context_patterns(self, entity_type: str, entity_text: str, context: str) -> str:
        """Analyze context patterns using intelligent rule-based approaches."""
        
        # Create contextual feature vectors based on linguistic patterns
        features = self._extract_contextual_features(context, entity_text)
        
        # Entity type-specific analysis
        if entity_type == 'PERSON':
            return self._categorize_person(entity_text, context, features)
        elif entity_type in ['LOCATION', 'GPE']:
            return self._categorize_location(entity_text, context, features)
        elif entity_type == 'ORGANIZATION':
            return self._categorize_organization(entity_text, context, features)
        else:
            return self._categorize_general(context, features)
    
    def _extract_contextual_features(self, context: str, entity_text: str) -> dict:
        """Extract contextual features using linguistic analysis."""
        features = {
            'temporal_markers': 0,
            'authority_markers': 0,
            'geographic_markers': 0,
            'cultural_markers': 0,
            'commercial_markers': 0,
            'military_markers': 0,
            'academic_markers': 0,
            'religious_markers': 0,
            'relational_markers': 0,
            'descriptive_density': 0
        }
        
        # Temporal markers (historical context)
        temporal_patterns = [
            r'\b(ancient|old|early|late|first|last|during|when|time|period|era|century|year|age)\b',
            r'\b(was|were|had|became|ruled|lived|died|born)\b',
            r'\b(then|now|before|after|since|until|while)\b'
        ]
        
        for pattern in temporal_patterns:
            features['temporal_markers'] += len(re.findall(pattern, context))
        
        # Authority/power markers
        authority_patterns = [
            r'\b(king|queen|emperor|ruler|leader|chief|head|master|lord|prince|princess)\b',
            r'\b(ruled|commanded|led|governed|controlled|owned|founded)\b',
            r'\b(power|authority|rule|reign|command|dominion|control)\b'
        ]
        
        for pattern in authority_patterns:
            features['authority_markers'] += len(re.findall(pattern, context))
        
        # Geographic markers
        geographic_patterns = [
            r'\b(city|town|village|place|region|area|land|country|nation|territory)\b',
            r'\b(sea|ocean|river|mountain|coast|island|port|harbor|shore)\b',
            r'\b(north|south|east|west|central|near|far|distant|located)\b'
        ]
        
        for pattern in geographic_patterns:
            features['geographic_markers'] += len(re.findall(pattern, context))
        
        # Cultural/civilization markers
        cultural_patterns = [
            r'\b(people|culture|civilization|society|custom|tradition|practice)\b',
            r'\b(temple|palace|building|monument|structure|architecture)\b',
            r'\b(art|craft|skill|knowledge|wisdom|learning)\b'
        ]
        
        for pattern in cultural_patterns:
            features['cultural_markers'] += len(re.findall(pattern, context))
        
        # Commercial/trade markers
        commercial_patterns = [
            r'\b(trade|trading|trader|merchant|business|commerce|market)\b',
            r'\b(goods|cargo|merchandise|wares|products|sell|buy|sold)\b',
            r'\b(ship|ships|voyage|journey|travel|carried|brought)\b'
        ]
        
        for pattern in commercial_patterns:
            features['commercial_markers'] += len(re.findall(pattern, context))
        
        # Military/conflict markers
        military_patterns = [
            r'\b(war|battle|fight|conflict|siege|attack|defense|victory|defeat)\b',
            r'\b(army|soldiers|warriors|forces|troops|military|weapon|weapons)\b',
            r'\b(conquered|captured|seized|fought|defended|invaded)\b'
        ]
        
        for pattern in military_patterns:
            features['military_markers'] += len(re.findall(pattern, context))
        
        # Academic/intellectual markers
        academic_patterns = [
            r'\b(learned|scholar|wise|knowledge|study|research|book|text)\b',
            r'\b(university|school|education|teaching|professor|student)\b',
            r'\b(theory|idea|concept|philosophy|science|discovery)\b'
        ]
        
        for pattern in academic_patterns:
            features['academic_markers'] += len(re.findall(pattern, context))
        
        # Religious/mythological markers
        religious_patterns = [
            r'\b(god|gods|goddess|divine|sacred|holy|blessed|prayer)\b',
            r'\b(temple|shrine|altar|worship|ritual|ceremony|sacrifice)\b',
            r'\b(myth|legend|story|tale|belief|faith|religion)\b'
        ]
        
        for pattern in religious_patterns:
            features['religious_markers'] += len(re.findall(pattern, context))
        
        # Relational markers (family, social connections)
        relational_patterns = [
            r'\b(father|mother|son|daughter|brother|sister|family|relative)\b',
            r'\b(friend|ally|enemy|rival|companion|associate|partner)\b',
            r'\b(married|wife|husband|children|offspring|heir|ancestor)\b'
        ]
        
        for pattern in relational_patterns:
            features['relational_markers'] += len(re.findall(pattern, context))
        
        # Descriptive density (how much descriptive language surrounds the entity)
        descriptive_patterns = [
            r'\b(great|famous|powerful|important|significant|notable|renowned)\b',
            r'\b(beautiful|magnificent|impressive|extraordinary|remarkable)\b',
            r'\b(large|small|huge|tiny|massive|enormous|vast|immense)\b'
        ]
        
        for pattern in descriptive_patterns:
            features['descriptive_density'] += len(re.findall(pattern, context))
        
        return features
    
    def _categorize_person(self, entity_text: str, context: str, features: dict) -> str:
        """Categorize person entities based on contextual features."""
        
        # Mythological/divine figures
        if features['religious_markers'] > 2 or any(term in context for term in ['god', 'goddess', 'divine', 'deity', 'myth', 'legend']):
            return 'mythological'
        
        # Historical rulers/political figures
        if features['authority_markers'] > 1 and features['temporal_markers'] > 1:
            return 'historical_ruler'
        
        # Military leaders
        if features['military_markers'] > 2 and features['authority_markers'] > 0:
            return 'military_leader'
        
        # Scholars/intellectual figures
        if features['academic_markers'] > 1:
            return 'scholar'
        
        # Trade/commercial figures
        if features['commercial_markers'] > 2:
            return 'merchant'
        
        # General historical figure
        if features['temporal_markers'] > 2:
            return 'historical_figure'
        
        # Family/relational context
        if features['relational_markers'] > 1:
            return 'family_member'
        
        return 'person'
    
    def _categorize_location(self, entity_text: str, context: str, features: dict) -> str:
        """Categorize location entities based on contextual features."""
        
        # Ancient/historical places
        if features['temporal_markers'] > 2 and features['cultural_markers'] > 1:
            return 'ancient_place'
        
        # Commercial centers/ports
        if features['commercial_markers'] > 2 and any(term in context for term in ['port', 'harbor', 'trade', 'merchant']):
            return 'trade_center'
        
        # Military/strategic locations
        if features['military_markers'] > 2:
            return 'strategic_location'
        
        # Religious/cultural sites
        if features['religious_markers'] > 1 or features['cultural_markers'] > 2:
            return 'cultural_site'
        
        # Natural geographic features
        if any(term in context for term in ['sea', 'ocean', 'river', 'mountain', 'coast', 'island']):
            return 'natural_feature'
        
        # Political entities
        if features['authority_markers'] > 1 and any(term in context for term in ['kingdom', 'empire', 'nation', 'country']):
            return 'political_entity'
        
        return 'place'
    
    def _categorize_organization(self, entity_text: str, context: str, features: dict) -> str:
        """Categorize organization entities based on contextual features."""
        
        # Educational institutions
        if features['academic_markers'] > 1:
            return 'educational'
        
        # Religious organizations
        if features['religious_markers'] > 1:
            return 'religious'
        
        # Military organizations
        if features['military_markers'] > 1:
            return 'military'
        
        # Commercial enterprises
        if features['commercial_markers'] > 1:
            return 'commercial'
        
        return 'organization'
    
    def _categorize_general(self, context: str, features: dict) -> str:
        """Categorize general entities based on dominant contextual features."""
        
        # Find the dominant feature
        feature_scores = {k: v for k, v in features.items() if v > 0}
        
        if not feature_scores:
            return 'general'
        
        dominant_feature = max(feature_scores, key=feature_scores.get)
        
        # Map dominant features to categories
        feature_to_category = {
            'temporal_markers': 'historical',
            'authority_markers': 'political',
            'geographic_markers': 'geographical',
            'cultural_markers': 'cultural',
            'commercial_markers': 'commercial',
            'military_markers': 'military',
            'academic_markers': 'academic',
            'religious_markers': 'religious',
            'relational_markers': 'social'
        }
        
        return feature_to_category.get(dominant_feature, 'general')

    def _extract_context_keywords(self, entity_text: str, text: str) -> List[str]:
        """Extract relevant keywords using intelligent linguistic analysis."""
        entity_pos = text.lower().find(entity_text.lower())
        if entity_pos == -1:
            return []
        
        # Extract surrounding text
        start = max(0, entity_pos - 200)
        end = min(len(text), entity_pos + len(entity_text) + 200)
        context = text[start:end]
        
        keywords = []
        
        # 1. Extract semantically significant terms using linguistic patterns
        significant_terms = self._extract_significant_terms(context, entity_text)
        keywords.extend(significant_terms)
        
        # 2. Extract relational terms (words that show relationships)
        relational_terms = self._extract_relational_terms(context, entity_text)
        keywords.extend(relational_terms)
        
        # 3. Extract domain indicators
        domain_terms = self._extract_domain_indicators(context)
        keywords.extend(domain_terms)
        
        # Remove duplicates while preserving order and relevance
        unique_keywords = []
        seen = set()
        for kw in keywords:
            if kw.lower() not in seen and len(kw) > 2:
                unique_keywords.append(kw)
                seen.add(kw.lower())
        
        return unique_keywords[:5]  # Top 5 most relevant
    
    def _extract_significant_terms(self, context: str, entity_text: str) -> List[str]:
        """Extract semantically significant terms from context."""
        terms = []
        
        # Find proper nouns (capitalized words not at sentence start)
        sentences = re.split(r'[.!?]+', context)
        for sentence in sentences:
            words = sentence.strip().split()
            for i, word in enumerate(words):
                if i > 0 and word and len(word) > 2:  # Not first word
                    clean_word = re.sub(r'[^\w]', '', word)
                    if clean_word and clean_word[0].isupper() and clean_word.isalpha():
                        if clean_word.lower() != entity_text.lower():  # Don't include the entity itself
                            terms.append(clean_word)
        
        return terms
    
    def _extract_relational_terms(self, context: str, entity_text: str) -> List[str]:
        """Extract terms that show relationships or important descriptors."""
        relational_patterns = [
            # Familial/social relationships
            r'\b(father|mother|son|daughter|brother|sister|wife|husband|king|queen|ruler|leader)\b',
            # Temporal relationships
            r'\b(ancient|old|early|first|last|former|during|when|after|before)\b',
            # Descriptive relationships
            r'\b(great|famous|powerful|mighty|sacred|holy|divine|wise|learned)\b',
            # Functional relationships
            r'\b(founded|built|ruled|led|commanded|conquered|created|established)\b'
        ]
        
        terms = []
        context_lower = context.lower()
        
        for pattern in relational_patterns:
            matches = re.findall(pattern, context_lower)
            terms.extend(matches)
        
        return terms
    
    def _extract_domain_indicators(self, context: str) -> List[str]:
        """Extract domain-specific indicators that provide thematic context."""
        domain_indicators = []
        context_lower = context.lower()
        
        # Define domain patterns with their labels
        domains = {
            'historical': r'\b(ancient|history|civilization|empire|kingdom|dynasty|era|period|classical)\b',
            'military': r'\b(war|battle|army|soldiers|conflict|siege|victory|defeat|conquest)\b',
            'religious': r'\b(god|gods|goddess|temple|sacred|holy|divine|worship|prayer)\b',
            'commercial': r'\b(trade|merchant|cargo|goods|market|commerce|sell|buy|ships)\b',
            'geographical': r'\b(sea|ocean|river|mountain|coast|island|port|harbor|city|land)\b',
            'cultural': r'\b(people|culture|custom|tradition|art|craft|learning|knowledge)\b',
            'political': r'\b(ruler|government|authority|power|law|policy|control|dominion)\b'
        }
        
        for domain, pattern in domains.items():
            if re.search(pattern, context_lower):
                domain_indicators.append(domain)
        
        return domain_indicators

    def _detect_geographical_context(self, text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Detect geographical context from the text to improve geocoding accuracy."""
        context_clues = []
        text_lower = text.lower()
        
        # Major locations mentioned in the text
        major_locations = {
            'uk': ['uk', 'united kingdom', 'britain', 'great britain', 'england', 'scotland', 'wales'],
            'usa': ['usa', 'united states', 'america', 'us ', 'united states of america'],
            'canada': ['canada'],
            'australia': ['australia'],
            'france': ['france'],
            'germany': ['germany'],
            'italy': ['italy'],
            'spain': ['spain'],
            'japan': ['japan'],
            'china': ['china'],
            'india': ['india'],
            'greece': ['greece', 'greek', 'hellas'],
            'egypt': ['egypt', 'egyptian'],
            'persia': ['persia', 'persian'],
            'phoenicia': ['phoenicia', 'phoenician'],
            'assyria': ['assyria', 'assyrian'],
            'london': ['london'],
            'new york': ['new york', 'nyc', 'manhattan'],
            'paris': ['paris'],
            'tokyo': ['tokyo'],
            'argos': ['argos'],
            'mediterranean': ['mediterranean', 'red sea']
        }
        
        # Check for explicit mentions
